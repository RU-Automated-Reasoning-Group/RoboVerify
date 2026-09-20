"""Algorithm 6: bounded, explicit synthesis/symbolic/motion feedback stages."""

import json
from copy import deepcopy
from dataclasses import dataclass, field

import z3
from synthesis.api.instructions import While
from synthesis.cfg.demo_validation import validate_demonstrations
from synthesis.cfg.graph import RelationalCFG
from synthesis.cfg.invariants import loop_learning_data
from synthesis.cfg.lower import lower
from synthesis.cfg.region import LoopRegion
from synthesis.cfg.synthesize import synthesize_cfg
from synthesis.cfg.verification import (
    propose_summaries,
    verify_cfg_motion,
    verify_cfg_symbolic,
)
from synthesis.inference_lib.demo_store import InvInference
from synthesis.predicates.term import to_z3
from synthesis.verification_lib.cegis import (
    MonotoneInvariantLearner,
    PenStore,
    _project,
)
from synthesis.verification_lib.counterexamples import (
    UnrealizableCounterexample,
    state_holds,
    symbolic_successor,
)


@dataclass
class VerifiedResult:
    cfg: object
    status: str
    symbolic: object = None
    motion: object = None
    request: object = None
    reason: str = ""
    history: list = field(default_factory=list)

    def __bool__(self):
        return self.status == "verified_model"


def loop_regions(cfg):
    for node in cfg.nodes.values():
        if isinstance(node.region, LoopRegion):
            yield node.region
            yield from loop_regions(node.region.body_cfg)


def loop_instructions(program):
    def walk(instructions, prefix=""):
        for i, instruction in enumerate(instructions):
            path = prefix + str(i)
            if isinstance(instruction, While):
                yield path, instruction
                yield from walk(instruction.body, path + ".")

    return list(walk(program.instructions))


def node_at(cfg, path):
    parts = path.split("/")
    for name in parts[:-1]:
        cfg = cfg.nodes[name].region.body_cfg
    return cfg, cfg.nodes[parts[-1]]


class CFGPenalty:
    def __init__(self, cfg, path, context, pen, options):
        self.cfg, self.path, self.context, self.pen, self.options = (
            cfg,
            path,
            context,
            pen,
            options,
        )

    def __call__(self, program):
        cfg = deepcopy(self.cfg)
        parent, node = node_at(cfg, self.path)
        node.region.physical = tuple(program.instructions)
        try:
            propose_summaries(cfg, self.context)
        except ValueError:
            return max(1, len(self.pen.for_block(self.path)))
        root, conditions = cfg, None
        parts = self.path.split("/")
        for name in parts[:-1]:
            loop = root.nodes[name].region
            conditions = [loop.invariant, to_z3(loop.guard, self.context)]
            root = loop.body_cfg
        failures = 0
        for example in self.pen.for_block(self.path):
            options = dict(
                self.options,
                initial_positions=example.mu_k,
                initial_arm=example.initial_arm,
            )
            result = verify_cfg_motion(
                root, self.context, entry_conditions=conditions, **options
            )
            failures += not bool(result)
        return failures


def replay_successors(state, instructions, *, limit=128):
    """Enumerate Get choices in the finite countermodel; preserve frozen geometry."""
    import itertools

    from synthesis.api.instructions import Get

    rows = [state]
    for instruction in instructions:
        if isinstance(instruction, Get):
            next_rows = []
            names = [str(v) for v in instruction.guard_exists_vars]
            for row in rows:
                for values in itertools.product(row.positions, repeat=len(names)):
                    chosen = deepcopy(row)
                    chosen.constants.update(zip(names, values))
                    if state_holds(instruction.instantiated_cond, chosen):
                        next_rows.append(chosen)
                    if len(next_rows) > limit:
                        raise UnrealizableCounterexample(
                            "Get replay branching budget exhausted"
                        )
            if not next_rows:
                raise UnrealizableCounterexample(
                    "No executable Get witness in counterexample"
                )
            rows = next_rows
        else:
            rows = [symbolic_successor(row, [instruction]) for row in rows]
    return rows


def verified_synthesis(
    cfg,
    realize,
    execute,
    context,
    *,
    quotient=None,
    language=None,
    max_refinements=10,
    symbolic_iterations=10,
    motion_iterations=10,
    learner="legacy",
    relations=None,
    variables=2,
    timeout_ms=5000,
    min_blocks=2,
    max_blocks=4,
    demo_provider=None,
    repair_motion=None,
    motion_options=None,
    logger=None
):
    """Return success only when both verifiers certify this exact candidate.

    demo_provider(request) returns new validated task segments or None. Missing
    demonstrations are exported as a concrete request; they are never invented.
    repair_motion(node, demos, post, penalty) may change the physical instruction
    structure while preserving the complete abstract program and bindings.
    """
    history, stores, pen = [], {}, PenStore()
    options = dict(motion_options or {})
    options.setdefault("timeout_ms", timeout_ms)
    result = VerifiedResult(cfg, "not_started", history=history)

    def event(stage, **fields):
        history.append(dict(stage=stage, **fields))
        if logger:
            logger.log_event(stage, json.dumps(fields), force=True)

    def synthesize():
        attempt = synthesize_cfg(
            cfg,
            realize,
            execute,
            quotient=quotient,
            language=language,
            max_refinements=max_refinements,
            logger=logger,
        )
        event("synthesis", status=attempt.status)
        if not attempt:
            result.status, result.reason = attempt.status, attempt.reason
            return False
        try:
            propose_summaries(cfg, context)
        except ValueError as exc:
            result.status, result.reason = "unsupported_summary", str(exc)
            return False
        return True

    if not hasattr(cfg, "_task_demos"):
        cfg._task_demos = list(cfg.demos.for_node(cfg.order[0]))
    if not synthesize():
        return result
    for iteration in range(symbolic_iterations + 1):
        result.symbolic = verify_cfg_symbolic(
            cfg,
            context,
            min_blocks=min_blocks,
            max_blocks=max_blocks,
            timeout_ms=timeout_ms,
        )
        event("symbolic", iteration=iteration, ok=bool(result.symbolic))
        if result.symbolic:
            break
        failure = result.symbolic.failure
        if failure.status != "invalid":
            result.status, result.reason = failure.status, failure.reason
            return result
        if iteration == symbolic_iterations:
            result.status = "symbolic_budget_exhausted"
            return result
        state = result.symbolic.loop_head_state
        if failure.vc.kind != "preserve":
            request = {
                "kind": failure.vc.kind,
                "loop_id": failure.vc.loop_id,
                "model": str(result.symbolic.model),
                "reason": result.symbolic.reason,
                "configuration": (
                    None
                    if state is None
                    else {
                        "positions": {
                            k: None if k == "tbl" else v
                            for k, v in state.positions.items()
                        },
                        "entry_positions": {
                            k: None if k == "tbl" else v
                            for k, v in state.entry_positions.items()
                        },
                        "constants": state.constants,
                    }
                ),
            }
            result.request = request
            if logger:
                logger.write_artifact(
                    "resynthesis_request.json", json.dumps(request, indent=2)
                )
            more = None if demo_provider is None else demo_provider(request)
            if not more:
                result.status = "needs_demonstrations"
                return result
            validation = validate_demonstrations(
                more, cfg.precondition, cfg.postcondition
            )
            if not validation:
                result.status, result.reason = "invalid_demonstrations", str(
                    validation.as_dict()
                )
                return result
            # Repartition from complete recordings; old internal cuts cannot be
            # assigned to a new environment without re-running refinement.
            original = getattr(cfg, "_task_demos", None)
            if original is None:
                result.status, result.reason = (
                    "unsupported_resynthesis",
                    "Complete task demonstrations were not retained",
                )
                return result
            all_demos = original + list(more)
            replacement = RelationalCFG.initial(
                all_demos, cfg.precondition, cfg.postcondition, cfg.initial_scope
            )
            cfg.__dict__.update(replacement.__dict__)
            cfg._task_demos = all_demos
            stores.clear()
            if not synthesize():
                return result
            continue
        if state is None:
            result.status, result.reason = (
                "unrealizable_counterexample",
                result.symbolic.reason,
            )
            return result
        program = lower(cfg, context)
        mapping = dict(
            zip((p for p, _ in loop_instructions(program)), loop_regions(cfg))
        )
        loop = mapping[failure.vc.loop_id]
        instruction = dict(loop_instructions(program))[failure.vc.loop_id]
        try:
            successors = replay_successors(state, instruction.body)
            successor = next(
                (
                    row
                    for row in successors
                    if not state_holds(loop.invariant, _project(row, context.use_tbl))
                ),
                None,
            )
        except UnrealizableCounterexample as exc:
            result.status, result.reason = "unsupported_replay", str(exc)
            return result
        if successor is None:
            result.status, result.reason = (
                "no_progress",
                "Counterexample successor does not enlarge the observed loop heads",
            )
            return result
        key = id(loop)
        if key not in stores:
            heads = list(loop.body_demos[0]) + list(loop.exit_demos)
            stores[key] = loop_learning_data(
                heads,
                loop.body_cfg.initial_scope,
                relations=relations,
                variables=variables,
            )
        store, vocabulary = stores[key]
        successor.loop_id = "loop"
        store.add(successor)
        candidate = (
            InvInference(store, "loop", vocabulary, context)[0]
            if learner == "legacy"
            else MonotoneInvariantLearner()(store, "loop", vocabulary, context)
        )
        if not all(
            state_holds(candidate, _project(row, context.use_tbl))
            for row in store.for_loop("loop")
        ):
            result.status = "learning_failed"
            return result
        solver = context.new_solver(timeout_ms)
        solver.add(loop.invariant, z3.Not(candidate))
        if solver.check() != z3.unsat:
            result.status = "nonmonotone_or_unknown"
            return result
        loop.invariant = candidate
        event("invariant_refined", iteration=iteration, loop_id=failure.vc.loop_id)
    signature = str(lower(cfg, context))
    for iteration in range(motion_iterations + 1):
        result.motion = verify_cfg_motion(cfg, context, **options)
        event("motion", iteration=iteration, ok=bool(result.motion))
        if result.motion:
            result.status = "verified_model"
            return result
        for example in result.motion.counterexamples:
            pen.add(example)
        if logger:
            pen.save(logger.artifact_dir() / "motion_penalties.json")
        if (
            iteration == motion_iterations
            or repair_motion is None
            or not result.motion.counterexamples
        ):
            result.status = "motion_unverified"
            return result
        for path in sorted({e.block_v for e in result.motion.counterexamples}):
            parent, node = node_at(cfg, path)
            candidate = repair_motion(
                node,
                parent.demos.for_node(node.name),
                parent.outgoing(node.name)[0].label,
                CFGPenalty(cfg, path, context, pen, options),
            )
            if candidate is None:
                result.status = "motion_budget_exhausted"
                return result
            node.region = candidate
        try:
            propose_summaries(cfg, context)
            if str(lower(cfg, context)) != signature:
                raise ValueError("Motion repair changed the abstract program")
        except ValueError as exc:
            result.status, result.reason = "invalid_motion_repair", str(exc)
            return result
    raise AssertionError("Unreachable verification exit")
