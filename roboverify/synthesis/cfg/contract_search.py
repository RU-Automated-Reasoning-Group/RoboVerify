"""Adapt recorded CFG segments to finite contract-guided proposals.

The adapter reads observations, scope and specifications, never teacher action
labels. Nominal goal satisfaction is a proposal filter; the existing BMC, CEM,
PostScore and verification stages still determine candidate acceptance.
"""

from copy import deepcopy
from time import perf_counter

import numpy as np

from synthesis.api.instructions import Assign, Get, Skip
from synthesis.api.program import Program
from synthesis.cfg.bindings import require_closed, require_ids
from synthesis.cfg.refine import scene_at
from synthesis.cfg.search_feasibility import (
    CandidateFeasibility,
    check_scene_feasibility,
    resolve_entry_bindings,
)
from synthesis.predicates.scene import Scene, evaluate
from synthesis.util import on


def observation_at(segment, index):
    state = segment.trace.states[index]
    value = state.observation if isinstance(state, Scene) else state
    if value is None:
        raise ValueError(
            "Contract search requires recorded gripper/object observations"
        )
    value = np.asarray(value, dtype=float)
    if (
        value.ndim != 1
        or len(value) < 10 + 12 * segment.trace.num_blocks
        or not np.isfinite(value).all()
    ):
        raise ValueError(
            "Contract search requires finite flattened RoboVerify observations"
        )
    return value


def entry_holding(segment):
    """Only explicit metadata, a full start or observed open fingers imply state.

    Closed-finger interior observations do not prove which object is held. A
    caller can provide ``holding_by_index`` metadata for such boundaries, using
    a physical ID or None for empty. Unknown entries are not silently emptied.
    """
    metadata = segment.trace.metadata
    entries = metadata.get("holding_by_index", {})
    if segment.t_start in entries:
        return entries[segment.t_start]
    if str(segment.t_start) in entries:
        return entries[str(segment.t_start)]
    if segment.t_start == 0:
        return metadata.get("initial_holding", None)
    obs = observation_at(segment, segment.t_start)
    if float(np.sum(obs[3:5])) > 0.053:
        return None
    raise ValueError(
        "Interior contract entry has unknown holding state; record holding_by_index "
        "or provide an observation with open fingers"
    )


def table_reference_at(segment):
    value = segment.trace.metadata.get("table_reference")
    if value is None:
        return None
    frame = np.asarray(value, dtype=float)
    if frame.shape != (3,) or not np.isfinite(frame).all():
        raise ValueError("table_reference must be a finite runtime XYZ frame")
    return frame


def check_contract_feasibility(
    candidate, segments, *, timeout_ms=1000, fixed_offsets=False
):
    """BMC every recorded start, including scalar references and actual gripper."""
    if not segments:
        return CandidateFeasibility(False, "No demonstration segments", "unsupported")
    total = 0.0
    for index, segment in enumerate(segments):
        try:
            obs = observation_at(segment, segment.t_start)
            scene = scene_at(segment, segment.t_start)
            result = check_scene_feasibility(
                candidate,
                scene,
                gripper=obs[:3],
                initial_holding=entry_holding(segment),
                table_reference=table_reference_at(segment),
                fixed_offsets=fixed_offsets,
                timeout_ms=timeout_ms,
            )
        except (ValueError, KeyError, TypeError) as exc:
            return CandidateFeasibility(
                False, f"segment {index}: {exc}", "unsupported", total
            )
        total += result.seconds
        if not result:
            return CandidateFeasibility(
                False, f"segment {index}: {result.reason}", result.status, total
            )
    return CandidateFeasibility(
        True, "BMC feasible on every recorded entry", "sat", total
    )


class ContractCandidates:
    """One lazy bounded proposal search; iteration never reruns completed work."""

    def __init__(
        self,
        segments,
        post,
        initial_program,
        available_scope,
        *,
        max_length,
        budget,
        id_first=False,
        refinement_fallback=False,
    ):
        self.segments = tuple(segments)
        self.post = post
        self.initial_program = deepcopy(initial_program)
        self.available_scope = frozenset(available_scope)
        self.max_length = max_length
        self.budget = budget
        self.id_first = id_first
        self.refinement_fallback = refinement_fallback
        self.stats = {}
        self.status = "not_started"
        self.reason = ""
        self._programs = None
        self.problem = None
        prefix = []
        if not id_first:
            for instruction in self.initial_program.instructions:
                if not isinstance(instruction, (Get, Assign, Skip)):
                    break
                prefix.append(instruction)
        # No unevaluated random physical instruction should become an Exec
        # negative merely because the contract search exhausted its budget.
        self.fallback_program = Program(len(prefix), deepcopy(prefix))

    def _prepare(self):
        started = perf_counter()
        deadline = started + self.budget.contract_seconds

        def check_deadline():
            if perf_counter() >= deadline:
                raise DeadlineExceeded(
                    "Contract proposal deadline reached during preparation"
                )

        from synthesis.search.contract import (
            ContractConfig,
            ContractInputError,
            ContractProblem,
            DeadlineExceeded,
            SearchConfig,
            SearchExample,
            beam_search,
        )

        if not self.segments:
            raise ContractInputError("Contract search needs demonstration segments")
        sizes = {s.trace.num_blocks for s in self.segments}
        if len(sizes) != 1 or min(sizes) < 1:
            raise ContractInputError(
                "A contract search batch requires one positive block count"
            )
        num_blocks = next(iter(sizes))
        prefix = []
        if self.id_first:
            require_ids(self.initial_program.instructions, num_blocks)
            operands = tuple(range(num_blocks))
        else:
            for instruction in self.initial_program.instructions:
                if not isinstance(instruction, (Get, Assign, Skip)):
                    break
                prefix.append(instruction)
            operands = tuple(sorted(require_closed(prefix, self.available_scope)))
        examples, entries, bindings = [], [], []
        terminal_empty = []
        for segment in self.segments:
            check_deadline()
            entry = scene_at(segment, segment.t_start)
            scoped = resolve_entry_bindings(prefix, entry)
            recorded = []
            for index in range(segment.t_start, segment.t_end + 1):
                check_deadline()
                recorded.append(observation_at(segment, index))
            observations = np.stack(recorded)
            positions = np.stack(
                [observations[:, 10 + 12 * i : 13 + 12 * i] for i in range(num_blocks)],
                axis=1,
            )
            examples.append(
                SearchExample(
                    positions=positions[0],
                    gripper=observations[0, :3],
                    bindings=scoped,
                    gripper_path=observations[:, :3],
                    positions_path=positions,
                    initial_holding=entry_holding(segment),
                    table_reference=table_reference_at(segment),
                )
            )
            entries.append(entry)
            bindings.append(scoped)
            terminal_empty.append(float(np.sum(observations[-1, 3:5])) > 0.053)
        # The relational table can occur in specifications without an available
        # runtime coordinate frame. Do not invent its XY from a random block.
        if any(example.table_reference is None for example in examples):
            operands = tuple(
                name
                for name in operands
                if not any(mapping.get(name, name) == "tbl" for mapping in bindings)
            )

        def goal(state, index):
            current = {i: state.positions[index, i] for i in range(num_blocks)}
            if "tbl" in entries[index].positions:
                current["tbl"] = on.TABLE
            scene = Scene(current, bindings[index], entries[index].entry_positions)
            return (
                bool(self.post(scene))
                if callable(self.post)
                else evaluate(self.post, scene)
            )

        config = ContractConfig(
            observation_proposals=self.budget.contract_observation_proposals,
            observation_references=not self.id_first,
            require_empty_hand=all(terminal_empty),
            solve_offsets=getattr(self.budget, "contract_solve_offsets", False),
            solver_timeout_ms=getattr(self.budget, "contract_solver_timeout_ms", 250),
            solver_attempts=getattr(self.budget, "contract_solver_attempts", 24),
            offset_radius=getattr(self.budget, "contract_offset_radius", 0.02),
        )

        def symbolic_goal(positions, index):
            from synthesis.search.offset_solver import compile_endpoint

            return compile_endpoint(
                self.post,
                positions,
                bindings[index],
                entries[index].entry_positions,
                check_deadline=check_deadline,
            )

        self.problem = ContractProblem(
            examples,
            goal,
            allowed_operands=operands,
            config=config,
            deadline=deadline,
            symbolic_goal=None if callable(self.post) else symbolic_goal,
        )
        preparation_seconds = perf_counter() - started
        result = beam_search(
            self.problem,
            SearchConfig(
                max_length=self.max_length,
                beam_width=self.budget.contract_beam_width,
                max_expansions=self.budget.contract_max_expansions,
                seconds=self.budget.contract_seconds,
                memoize=self.budget.contract_memoize,
                diversity=self.budget.contract_diversity,
                pool_limit=self.budget.contract_pool_limit,
                deadline=deadline,
                partial_pool_limit=(
                    self.budget.contract_pool_limit if self.refinement_fallback else 0
                ),
            ),
        )
        self.stats = dict(result.stats)
        self.stats["preparation_seconds"] = preparation_seconds
        self.stats["proposal_seconds"] = perf_counter() - started
        self.status = result.status
        refinement_only = self.refinement_fallback and not result.candidates
        nodes = result.partial_candidates if refinement_only else result.candidates
        self.stats["refinement_candidates"] = len(nodes) if refinement_only else 0
        programs = []
        for node in nodes:
            program = node.to_program(named=not self.id_first, prefix=prefix)
            if refinement_only:
                program._contract_refinement_only = True
            if self.id_first:
                require_ids(program.instructions, num_blocks)
            else:
                require_closed(program.instructions, self.available_scope)
            programs.append(program)
        return tuple(programs)

    def __iter__(self):
        if self._programs is None:
            try:
                self._programs = self._prepare()
            except (ValueError, KeyError, TypeError) as exc:
                self._programs = ()
                self.status = "unsupported"
                self.reason = str(exc)
                self.stats = {"status": self.status, "reason": self.reason}
            except TimeoutError as exc:
                from synthesis.search.contract import DeadlineExceeded

                # The CLI's end-to-end alarm also raises TimeoutError. Only
                # consume this proposal search's own deadline exception.
                if not isinstance(exc, DeadlineExceeded):
                    raise
                self._programs = ()
                self.status = "deadline_exhausted"
                self.reason = str(exc)
                self.stats = {"status": self.status, "reason": self.reason}
        yield from self._programs


def make_contract_candidates(*args, **kwargs):
    return ContractCandidates(*args, **kwargs)
