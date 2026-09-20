"""Verify the same structured CFG and physical primitives produced by synthesis.

Abstract summaries are candidates until motion checking establishes their full
WP effects. A predicate observed in a demo alone never establishes a summary.
Unsupported summary shapes remain explicit failures.
"""

from copy import deepcopy

import z3
from synthesis.api.instructions import (
    Assign,
    Get,
    MoveByName,
    PickByName,
    Put,
    ReleaseByName,
    Skip,
)
from synthesis.cfg.bindings import require_closed
from synthesis.cfg.lower import lower
from synthesis.cfg.region import BlockRegion, LoopRegion
from synthesis.predicates.term import to_z3
from synthesis.verification_lib.lowlevel_verification_lib import LowLevelContext
from synthesis.verification_lib.motion_verification import (
    MotionCheck,
    MotionContract,
    MotionVerificationResult,
    check_abstract_effects,
    check_contract_realization,
    check_frame_preservation,
)
from synthesis.verification_lib.primitive_motion import PrimitiveMotionProblem


def propose_summaries(cfg, context):
    """Attach a Put only to a structurally identified completed placement.

    Pure binding/arm operations have no relational object effect. Transfers may
    span adjacent blocks, but no relational observation/control boundary may
    intervene while their abstract Put is pending.
    """
    from synthesis.cfg.scope import scope

    held = None
    last_move = None
    for name in cfg.order:
        node, edge = cfg.nodes[name], cfg.incoming(name)[0]
        region = node.region
        if edge.binds and held is not None:
            raise ValueError("Get observes a partially completed abstract placement")
        if isinstance(region, LoopRegion):
            if held is not None:
                raise ValueError("A loop cannot split an abstract placement")
            if region.body_cfg is None:
                raise ValueError("Verification requires the recovered loop body CFG")
            propose_summaries(region.body_cfg, context)
            continue
        if not isinstance(region, BlockRegion):
            raise ValueError("Cannot verify unresolved blocks")
        available = scope(cfg)[name]
        region.bindings = require_closed(region.physical, available) - available
        summary = []
        post = cfg.outgoing(name)[0].label
        for i in region.physical:
            if isinstance(i, Get):
                if held is not None:
                    raise ValueError("Get inside an incomplete placement")
                summary.append(deepcopy(i))
            elif isinstance(i, Assign):
                if held is not None:
                    raise ValueError(
                        "Assignment inside an incomplete placement requires an explicit summary"
                    )
                summary.append(deepcopy(i))
            elif isinstance(i, PickByName):
                if held is not None:
                    raise ValueError("Pick while holding")
                held = i.grab_box_name
                last_move = None
            elif isinstance(i, ReleaseByName):
                if held is None or held != i.release_box_name:
                    raise ValueError("Release must refer to the currently held object")
                if post.op in ("ON", "ON_star") and post.args[0].value == held:
                    target = post.args[1].value
                elif last_move is not None:
                    # This is only a proposed abstraction. The motion stage must
                    # prove all its effects even when the edge is a global goal.
                    target = last_move.target_box_name_z
                    if target == held and context.use_tbl:
                        target = "tbl"
                else:
                    raise ValueError(
                        "Placement needs an explicit ON edge or a transport target"
                    )
                summary.append(Put(held, target))
                held = None
            elif isinstance(i, MoveByName):
                if held is not None:
                    last_move = i
            elif not isinstance(i, Skip):
                # Explicit established IR is allowed only if its physical verifier
                # supports it; the primitive pipeline refuses unknown instructions.
                raise ValueError(f"Unsupported synthesis primitive {type(i).__name__}")
        if region.symbolic is None:
            region.symbolic = tuple(summary or [Skip(0)])
        else:
            # Compare semantically relevant instruction text, not object repr.
            actual = tuple(str(i) for i in region.symbolic if not isinstance(i, Skip))
            proposed = tuple(str(i) for i in summary if not isinstance(i, Skip))
            if actual != proposed:
                raise ValueError(
                    "Declared summary differs from the structural placement candidate"
                )
    if held is not None:
        raise ValueError("Unfinished placement at CFG exit")
    return cfg


def _all_names(cfg):
    names = set(cfg.initial_scope)
    for edge in cfg.edges:
        names.update(edge.binds)
    for node in cfg.nodes.values():
        region = node.region
        if isinstance(region, LoopRegion):
            names.update(region.exists_vars)
            for a, b in (*region.init, *region.update):
                names.update((a, b))
            names.update(_all_names(region.body_cfg))
        elif isinstance(region, BlockRegion):
            for inst in region.physical:
                if isinstance(inst, Get):
                    names.update(str(v) for v in inst.guard_exists_vars)
                elif isinstance(inst, Assign):
                    names.update((inst.left, inst.right))
                else:
                    names.update(
                        o["val"] for o in inst.get_operand() if o["type"] == "BoxName"
                    )
    return names


def verify_cfg_motion(
    cfg,
    context,
    *,
    noise=None,
    timeout_ms=5000,
    initial_positions=None,
    initial_arm=None,
    table_surface_height=None,
    entry_conditions=None,
):
    """Thread geometry/arm/held state through blocks; fresh states at loop boundaries.

    Each loop body is checked from invariant + witnessed guard, and continuation
    starts from a fresh invariant + guard-false state. A failed/unknown premise or
    unsupported primitive is never a successful proof.
    """
    constants = sorted(_all_names(cfg))
    low = LowLevelContext(default_L=0.05, use_tbl=context.use_tbl)
    checks = []
    count = 0
    physical_names = [n for n in constants if n != "tbl"]
    if not physical_names:
        return MotionVerificationResult(
            [MotionCheck("coverage", "unsupported", reason="No object bindings")], noise
        )
    base = "b0" if "b0" in constants else physical_names[0]
    dummy = MotionContract(base, base, base, table_surface_height)

    def fresh(conditions, path, positions=None, arm=None):
        p = PrimitiveMotionProblem(
            low,
            conditions,
            constants,
            dummy,
            noise,
            path,
            timeout_ms,
            initial_positions=positions,
            initial_arm=arm,
            enforce_source=False,
        )
        p.check("initial_consistency")
        return p

    def append_new(p, offset, path):
        checks.extend(
            MotionCheck(
                path + "/" + c.obligation,
                c.status,
                c.elapsed_seconds,
                c.counterexample,
                c.reason,
            )
            for c in p.checks[offset:]
        )

    def walk(graph, p, prefix=""):
        nonlocal count
        pending = None
        for name in graph.order:
            path = prefix + name
            edge, region = graph.incoming(name)[0], graph.nodes[name].region
            p.block_v = path
            offset = len(p.checks)
            if edge.binds:
                names = sorted(edge.binds)
                p.execute(
                    [
                        Get(
                            names[0],
                            to_z3(edge.binding_condition, context),
                            [context.get_consts(n) for n in names],
                            guard_term=edge.binding_condition,
                        )
                    ]
                )
            if isinstance(region, LoopRegion):
                if p.held_name is not None:
                    checks.append(
                        MotionCheck(
                            path, "unsupported", reason="Held object at loop boundary"
                        )
                    )
                    return p
                p.execute([Assign(a, b) for a, b in region.init])
                append_new(p, offset, path)
                guard = to_z3(region.guard, context)
                inv = region.invariant
                inv = (
                    z3.And(*[c.expr if hasattr(c, "expr") else c for c in inv])
                    if isinstance(inv, list)
                    else inv
                )
                child = fresh([inv, guard], path)
                append_new(child, 0, path)
                walk(region.body_cfg, child, path + "/")
                exists_vars = [context.get_consts(n) for n in region.exists_vars]
                no_guard = (
                    z3.Not(z3.Exists(exists_vars, guard))
                    if exists_vars
                    else z3.Not(guard)
                )
                p = fresh([inv, no_guard], path + "/exit")
                append_new(p, 0, path + "/exit")
                continue
            count += 1
            for instruction in region.physical:
                if isinstance(instruction, PickByName):
                    pending = (dict(p.current), p.fields)
                p.execute([instruction])
            puts = [i for i in region.symbolic or () if isinstance(i, Put)]
            if puts:
                if len(puts) != 1 or pending is None:
                    checks.append(
                        MotionCheck(
                            path,
                            "unsupported",
                            reason="Expected one completed placement",
                        )
                    )
                    return p
                put = puts[0]
                contract = MotionContract(
                    p.bindings[put.upper_block],
                    p.bindings[put.base_block],
                    base,
                    table_surface_height,
                )
                old_initial = p.initial
                p.initial, p.effect_entry_fields = pending
                p.contract = contract
                check_contract_realization(p)
                check_frame_preservation(p)
                check_abstract_effects(p)
                p.initial = old_initial
                pending = None
            p.check("transition_consistency")
            append_new(p, offset, path)
        if p.held_name is not None:
            checks.append(
                MotionCheck(
                    prefix + "exit",
                    "unsupported",
                    reason="Unreleased object at CFG exit",
                )
            )
        return p

    try:
        p = fresh(
            (
                entry_conditions
                if entry_conditions is not None
                else [to_z3(cfg.precondition, context)]
            ),
            "entry",
            initial_positions,
            initial_arm,
        )
        append_new(p, 0, "entry")
        walk(cfg, p)
    except (ValueError, KeyError, TypeError) as exc:
        checks.append(MotionCheck("coverage", "unsupported", reason=str(exc)))
    return MotionVerificationResult(checks, noise, count)


def symbolic_problem(cfg, original_context, size=None):
    """Rebuild formulas in each finite/unbounded context without changing the CFG."""
    from uuid import uuid4

    from synthesis.verification_lib.highlevel_verification_lib import HighLevelContext

    context = HighLevelContext(
        mode="declare" if size is None else "enum",
        num_blocks=size,
        use_tbl=original_context.use_tbl,
        sort_name="Box" if size is None else "CFGBox_" + uuid4().hex,
    )
    clone = deepcopy(cfg)
    names = sorted(_all_names(cfg))

    def rewrite(graph):
        for node in graph.nodes.values():
            region = node.region
            if isinstance(region, LoopRegion):
                invariant = region.invariant
                if isinstance(invariant, list):
                    invariant = z3.And(
                        *(c.expr if hasattr(c, "expr") else c for c in invariant)
                    )
                region.invariant = context.spec_to_expr(
                    original_context.expr_to_spec(invariant), known_const_names=names
                )
                rewrite(region.body_cfg)

    rewrite(clone)
    return (
        lower(clone, context),
        to_z3(cfg.precondition, context),
        to_z3(cfg.postcondition, context),
        context,
    )


def verify_cfg_symbolic(cfg, context, *, min_blocks=2, max_blocks=4, timeout_ms=5000):
    from synthesis.verification_lib.symbolic_verify import symbolic_verify

    return symbolic_verify(
        lambda size: symbolic_problem(cfg, context, size),
        min_blocks=min_blocks,
        max_blocks=max_blocks,
        timeout_ms=timeout_ms,
        constants=tuple(sorted(_all_names(cfg))),
    )
