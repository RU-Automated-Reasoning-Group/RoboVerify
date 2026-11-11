"""Bounded refinement of geometric milestones using primitive entry contracts.

No demonstration action labels are consumed. A proposed cut preserves the
milestone continuously, and admits a shared first action with successor bindings.
It is a search proposal, not evidence that an entire body implements its contract.
The synthesis driver still searches and physically checks that body.
"""

from copy import copy
from dataclasses import dataclass, replace
from time import perf_counter

import numpy as np

from synthesis.cfg.demos import DemoAssignment, DemoSegment
from synthesis.cfg.iterations import Iterations
from synthesis.cfg.refine import scene_at
from synthesis.predicates.scene import evaluate


@dataclass(frozen=True)
class BoundaryBudget:
    max_shift_steps: int = 80
    max_first_actions: int = 32
    max_proposals: int = 3
    seconds: float = 15.0

    def __post_init__(self):
        for name in ("max_shift_steps", "max_first_actions", "max_proposals"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not np.isfinite(self.seconds) or self.seconds <= 0:
            raise ValueError("Boundary budget must be positive and finite")


def point(row, index, *, bindings=None, end=None):
    return DemoSegment(
        row.demo_idx,
        index,
        index if end is None else end,
        row.trace,
        row.bindings if bindings is None else bindings,
        parent=row,
        entry_index=row.entry_index,
    )


def _check_deadline(deadline):
    if deadline is not None and perf_counter() >= deadline:
        from synthesis.search.contract import DeadlineExceeded

        raise DeadlineExceeded("Boundary proposal deadline reached")


def truth_window(row, predicate, stop, max_shift_steps, *, deadline=None):
    """The bounded truth component containing this milestone, never a later one."""
    start = row.t_end
    _check_deadline(deadline)
    if not evaluate(predicate, scene_at(row, start)):
        return None
    end = min(stop, len(row.trace.states) - 1, start + max_shift_steps)
    if end < start:
        return None
    for index in range(start + 1, end + 1):
        _check_deadline(deadline)
        if not evaluate(predicate, scene_at(row, index)):
            return start, index - 1
    return start, end


def _problem(rows, *, operands=None, actions=None, deadline=None):
    from synthesis.cfg.contract_search import (
        entry_holding,
        observation_at,
        table_reference_at,
    )
    from synthesis.search.contract import ContractConfig, ContractProblem, SearchExample

    examples = []
    for row in rows:
        recorded = []
        for t in range(row.t_start, row.t_end + 1):
            _check_deadline(deadline)
            recorded.append(observation_at(row, t))
        observations = np.stack(recorded)
        positions = np.stack(
            [
                observations[:, 10 + 12 * i : 13 + 12 * i]
                for i in range(row.trace.num_blocks)
            ],
            axis=1,
        )
        examples.append(
            SearchExample(
                positions[0],
                observations[0, :3],
                dict(row.bindings),
                observations[:, :3],
                positions,
                initial_holding=entry_holding(row),
                table_reference=table_reference_at(row),
            )
        )
    if operands is None:
        operands = tuple(sorted(set.intersection(*(set(e.bindings) for e in examples))))
    if any(e.table_reference is None for e in examples):
        operands = tuple(
            n
            for n in operands
            if not any(e.bindings.get(n, n) == "tbl" for e in examples)
        )
    return ContractProblem(
        examples,
        lambda state, index: False,
        allowed_operands=operands,
        config=ContractConfig(
            observation_references=not all(type(n) is int for n in operands)
        ),
        actions=actions,
        deadline=deadline,
    )


def propose_handoffs(
    rows,
    successors,
    predicate,
    *,
    budget=None,
    prototypes=None,
    operands=None,
    initial_required=False,
):
    """Propose endpoint vectors; None successors are terminal loop iterations.

    Bounds always leave the next segment's original endpoint available. Terminal
    iterations retain their remaining tail and impose no next-action constraint.
    Unknown held states fail closed, just as in the contract search adapter.
    """
    from synthesis.cfg.contract_search import entry_holding
    from synthesis.search.contract import DeadlineExceeded, Node

    def placement_milestone(term):
        return term.op == "ON" or (
            term.op == "and" and any(placement_milestone(child) for child in term.args)
        )

    require_released = placement_milestone(predicate)

    def completed(row, index):
        if not require_released:
            return True
        try:
            return entry_holding(point(row, index)) is None
        except (ValueError, KeyError, TypeError):
            return False

    budget = budget or BoundaryBudget()
    if not rows or len(rows) != len(successors):
        raise ValueError("Every milestone needs its corresponding successor")
    started = perf_counter()
    deadline = started + budget.seconds
    stats = {"status": "no_proposal", "actions_examined": 0, "proposals": []}
    try:
        windows = [
            truth_window(
                row,
                predicate,
                (
                    len(row.trace.states) - 1
                    if nxt is None
                    else min(
                        nxt.t_end - 1,
                        (
                            nxt.milestone_index
                            if nxt.milestone_index is not None
                            else nxt.t_end - 1
                        ),
                    )
                ),
                budget.max_shift_steps,
                deadline=deadline,
            )
            for row, nxt in zip(rows, successors)
        ]
        if any(window is None for window in windows):
            stats["reason"] = "No persistent milestone window"
            return [], stats
        if prototypes is None:
            prototypes = [
                point(nxt, begin, end=end)
                for nxt, (begin, end) in zip(successors, windows)
                if nxt is not None
            ]
        if not prototypes:
            cuts = tuple(
                next((t for t in range(end, begin - 1, -1) if completed(row, t)), None)
                for row, (begin, end) in zip(rows, windows)
            )
            return ([] if None in cuts else [cuts]), {
                **stats,
                "status": "terminal",
                "seconds": perf_counter() - started,
            }
        problem = _problem(prototypes, operands=operands, deadline=deadline)
        ranked, other = [], []
        for action in problem.actions:
            state = problem.transition(problem.initial, action)
            if state is None:
                if not initial_required:
                    other.append(action)
                continue
            ranked.append(
                Node(
                    state,
                    (action,),
                    np.concatenate(
                        [
                            problem.vector(problem.initial)[:, None],
                            problem.action_path(problem.initial, action, state),
                        ],
                        axis=1,
                    ),
                )
            )
        scores = problem.rank_many(ranked, [False] * len(ranked)) if ranked else []
        actions = [
            node.actions[0]
            for _, node in sorted(zip(scores, ranked), key=lambda pair: pair[0])
        ] + other
        cache, vectors = {}, []
        for action in actions[: budget.max_first_actions]:
            if perf_counter() >= deadline:
                raise DeadlineExceeded("Boundary proposal deadline reached")
            stats["actions_examined"] += 1
            cuts = []
            for i, (row, nxt, window) in enumerate(zip(rows, successors, windows)):
                begin, end = window
                if nxt is None:
                    chosen = next(
                        (t for t in range(end, begin - 1, -1) if completed(row, t)),
                        None,
                    )
                    if chosen is None:
                        break
                    cuts.append(chosen)
                    continue
                chosen = None
                for index in range(begin, end + 1):
                    if perf_counter() >= deadline:
                        raise DeadlineExceeded("Boundary proposal deadline reached")
                    if not completed(row, index):
                        continue
                    key = (i, index)
                    if key not in cache:
                        try:
                            cache[key] = _problem(
                                [point(nxt, index)],
                                operands=operands,
                                actions=(),
                                deadline=deadline,
                            )
                        except (ValueError, KeyError, TypeError):
                            cache[key] = None
                    instance = cache[key]
                    if (
                        instance is not None
                        and instance.transition(instance.initial, action) is not None
                    ):
                        chosen = index
                        break
                if chosen is None:
                    break
                cuts.append(chosen)
            vector = tuple(cuts)
            if len(cuts) == len(rows) and vector not in vectors:
                vectors.append(vector)
                stats["proposals"].append({"cuts": cuts, "action": action.row()})
                if len(vectors) >= budget.max_proposals:
                    break
        stats["status"] = "found" if vectors else "no_proposal"
        return vectors, stats
    except DeadlineExceeded as exc:
        stats.update(status="budget_exhausted", reason=str(exc))
        return [], stats
    except (ValueError, KeyError, TypeError) as exc:
        stats.update(status="unsupported", reason=str(exc))
        return [], stats
    finally:
        stats["seconds"] = perf_counter() - started


def refine_iterations(extracted, predicates, *, budget=None, choice=0):
    """Refine each template slot while preserving witnesses, updates and tails."""
    budget = budget or BoundaryBudget()
    groups = [[list(body) for body in it.bodies] for it in extracted]
    evidence = []
    for slot, predicate in enumerate(predicates):
        next_slot = (slot + 1) % len(predicates)
        rows, successors = [], []
        for group in groups:
            for i, body in enumerate(group):
                rows.append(body[slot])
                successors.append(
                    body[next_slot]
                    if next_slot
                    else (group[i + 1][0] if i + 1 < len(group) else None)
                )
        prototypes = [group[0][next_slot] for group in groups]
        vectors, report = propose_handoffs(
            rows,
            successors,
            predicate,
            budget=budget,
            prototypes=prototypes,
            initial_required=next_slot == 0,
        )
        evidence.append(report)
        if not vectors:
            return extracted, evidence
        if choice >= len(vectors):
            return None, evidence
        cuts = iter(vectors[choice])
        for group in groups:
            for i, body in enumerate(group):
                old = body[slot]
                end = next(cuts)
                body[slot] = replace(
                    old,
                    t_end=end,
                    milestone_index=(
                        old.milestone_index
                        if old.milestone_index is not None
                        else old.t_end
                    ),
                )
                if next_slot:
                    body[next_slot] = replace(body[next_slot], t_start=end)
                elif i + 1 < len(group):
                    group[i + 1][0] = replace(group[i + 1][0], t_start=end)
    result = []
    for original, group in zip(extracted, groups):
        end = group[-1][-1].t_end
        result.append(
            Iterations(
                [tuple(body) for body in group],
                replace(original.terminal, t_start=end, t_end=end),
            )
        )
    return result, evidence


def refine_failed_handoff(cfg, node_name, *, budget=None):
    """Retry a failed block from a later contract-admissible incoming cut.

    Only the adjoining two segments change. Predicate labels, witnesses, frozen
    entry geometry and the complete suffix are retained; complete CFG validation
    precedes the atomic commit. Previously attempted vectors are never repeated.
    """
    from synthesis.cfg.region import LoopRegion
    from synthesis.cfg.scope import scope
    from synthesis.cfg.validate import validate_cfg

    index = cfg.order.index(node_name)
    if index == 0:
        return False
    previous = cfg.order[index - 1]
    if any(isinstance(cfg.nodes[n].region, LoopRegion) for n in (previous, node_name)):
        return False
    rows, following = cfg.demos.for_node(previous), cfg.demos.for_node(node_name)
    if len(rows) != len(following) or any(
        a.trace is not b.trace or a.demo_idx != b.demo_idx or a.t_end != b.t_start
        for a, b in zip(rows, following)
    ):
        return False
    operands = (
        tuple(range(rows[0].trace.num_blocks))
        if cfg.synthesis_approach == "id-first"
        else tuple(sorted(scope(cfg)[node_name]))
    )
    vectors, report = propose_handoffs(
        rows,
        following,
        cfg.incoming(node_name)[0].label,
        budget=budget,
        operands=operands,
    )
    history = getattr(cfg, "_boundary_attempts", set())
    cfg._boundary_reports = [*getattr(cfg, "_boundary_reports", []), report]
    for cuts in vectors:
        key = (previous, node_name, cuts)
        if key in history or cuts == tuple(row.t_end for row in rows):
            continue
        history = history | {key}
        cfg._boundary_attempts = history
        trial = copy(cfg)
        trial.demos = DemoAssignment(dict(cfg.demos.segments))
        trial.demos.segments[previous] = [
            replace(
                row,
                t_end=cut,
                milestone_index=(
                    row.milestone_index
                    if row.milestone_index is not None
                    else row.t_end
                ),
            )
            for row, cut in zip(rows, cuts)
        ]
        trial.demos.segments[node_name] = [
            replace(row, t_start=cut) for row, cut in zip(following, cuts)
        ]
        if validate_cfg(trial):
            cfg.__dict__.update(trial.__dict__)
            return True
    return False
