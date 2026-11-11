"""Bounded, shared float proposals for existing Move/Release structures.

These queries choose one XYZ vector for every example. They encode endpoint
geometry and supported release, then replay the unchanged nominal transition
contracts (including swept envelopes). A solver model is never accepted without
that replay. This is an incomplete proposal mechanism, not a motion proof.
"""

from dataclasses import dataclass
from itertools import product
from time import perf_counter

import numpy as np
import z3

from synthesis.util import on


def real(value):
    return z3.RealVal(str(float(value)))


def compile_endpoint(
    term, positions, bindings, entry_positions, *, check_deadline=None
):
    """Finite geometric reading of the maintained predicate language.

    Bindings are instruction-entry bindings. Frozen ON* reads the invocation's
    entry geometry, and the logical table retains its isolated semantics.
    """
    universe = tuple(entry_positions)

    def relation(kind, left, right, coordinates):
        if left == "tbl" or right == "tbl":
            return z3.BoolVal(left == right and kind != "Scattered")
        a, b = coordinates[left], coordinates[right]
        if kind in ("ON_star", "ON_star_zero"):
            half = real(on.BLOCK_LENGTH / 2)
            return z3.And(
                z3.Abs(a[0] - b[0]) < half, z3.Abs(a[1] - b[1]) < half, a[2] >= b[2]
            )
        if kind == "Higher":
            return on.higher_z3(a[2], b[2])
        if kind == "EqualHeight":
            return on.equal_height_z3(a[2], b[2])
        if kind == "Scattered":
            gap = real(2 * on.BLOCK_LENGTH)
            return z3.Or(z3.Abs(a[0] - b[0]) >= gap, z3.Abs(a[1] - b[1]) >= gap)
        raise ValueError(f"Unsupported endpoint relation: {kind}")

    frozen = {
        key: (on.TABLE if key == "tbl" else tuple(map(real, value)))
        for key, value in entry_positions.items()
    }

    def evaluate(node, aliases):
        if check_deadline is not None:
            check_deadline()
        if node.op == "ref":
            return aliases[node.value]
        if node.op == "id":
            if node.value not in universe:
                raise ValueError(f"Unknown endpoint object: {node.value}")
            return node.value
        if node.op == "bool":
            return z3.BoolVal(node.value)
        if node.op in ("exists", "forall"):
            clauses = [
                evaluate(node.args[0], dict(aliases, **dict(zip(node.value, ids))))
                for ids in product(universe, repeat=len(node.value))
            ]
            return z3.Or(clauses) if node.op == "exists" else z3.And(clauses)
        if node.op == "not":
            return z3.Not(evaluate(node.args[0], aliases))
        if node.op in ("and", "or"):
            values = [evaluate(arg, aliases) for arg in node.args]
            return z3.And(values) if node.op == "and" else z3.Or(values)
        if node.op == "implies":
            return z3.Implies(*(evaluate(arg, aliases) for arg in node.args))
        left, right = (evaluate(arg, aliases) for arg in node.args)
        if node.op == "eq":
            return z3.BoolVal(left == right)
        if node.op == "ON":
            return z3.And(
                left != right,
                relation("ON_star", left, right, positions),
                *[
                    z3.Not(
                        z3.And(
                            relation("ON_star", left, middle, positions),
                            relation("ON_star", middle, right, positions),
                        )
                    )
                    for middle in universe
                    if middle not in (left, right)
                ],
            )
        return relation(
            node.op, left, right, frozen if node.op == "ON_star_zero" else positions
        )

    return evaluate(term, dict(bindings))


def strengthen(formula, margin, negative=False):
    """Add margin to nonconstant inequalities, preserving Boolean structure.

    A positive margin implies the original predicate, including strict versus
    non-strict boundaries. Constant facts and reflexive relations impose no
    artificial zero-margin obstruction.
    """
    formula = z3.simplify(formula)
    if z3.is_true(formula) or z3.is_false(formula):
        return z3.Not(formula) if negative else formula
    if z3.is_not(formula):
        return strengthen(formula.arg(0), margin, not negative)
    if z3.is_implies(formula):
        return strengthen(
            z3.Or(z3.Not(formula.arg(0)), formula.arg(1)), margin, negative
        )
    if z3.is_and(formula) or z3.is_or(formula):
        conjunction = z3.is_and(formula) != negative
        terms = [strengthen(child, margin, negative) for child in formula.children()]
        return z3.And(terms) if conjunction else z3.Or(terms)
    kind = formula.decl().kind()
    if kind in (z3.Z3_OP_LE, z3.Z3_OP_LT, z3.Z3_OP_GE, z3.Z3_OP_GT):
        left, right = formula.children()
        less = kind in (z3.Z3_OP_LE, z3.Z3_OP_LT)
        strict = kind in (z3.Z3_OP_LT, z3.Z3_OP_GT)
        if negative:
            less, strict = not less, not strict
        gap = right - left if less else left - right
        return gap > margin if strict else gap >= margin
    # Object equalities have already become constants. Fail closed if a caller
    # supplies a symbolic formula outside this finite geometric language.
    raise ValueError(f"Unsupported numeric endpoint clause: {formula}")


@dataclass(frozen=True)
class OffsetProposal:
    status: str
    actions: tuple = ()
    states: tuple = ()
    margin: float | None = None
    reason: str = ""
    seconds: float = 0.0


def solve_terminal_offsets(problem, state, move, release, *, timeout_ms, radius):
    """Repair all XYZ offsets of a generated terminal Move, shared over examples.

    Operand selection, the preceding prefix, and the Release parameter remain
    part of the caller's enumerated structure. Unknown/UNSAT/replay rejection
    return no candidate. A bounded maximum-margin model is chosen before physics.
    """
    from synthesis.search.contract import Action

    started = perf_counter()
    if problem.symbolic_goal is None:
        return OffsetProposal("unsupported", reason="Typed endpoint formula required")
    if (
        move.kind != "Move"
        or release.kind != "Release"
        or any(h < 0 for h in state.holding)
    ):
        return OffsetProposal(
            "unsupported", reason="A held Move/Release extension is required"
        )
    if timeout_ms <= 0 or not np.isfinite(radius) or radius <= 0:
        raise ValueError("Positive solver timeout and finite offset radius required")
    problem.check_deadline()
    cfg = problem.config
    offsets = [z3.FreshReal(f"shared_move_offset_{axis}") for axis in range(3)]
    margin = z3.FreshReal("endpoint_margin")
    solver = z3.Optimize()
    remaining = timeout_ms
    if problem.deadline is not None:
        remaining = min(
            remaining, max(1, int((problem.deadline - perf_counter()) * 1000))
        )
    solver.set(timeout=int(remaining))
    solver.add(margin >= 0, margin <= real(cfg.geometry_tolerance))
    for offset, original in zip(offsets, move.offsets):
        solver.add(offset >= real(original - radius), offset <= real(original + radius))
    try:
        for index in range(problem.E):
            problem.check_deadline()
            holding = state.holding[index]
            target = [
                real(problem.reference(state, ref)[index, axis]) + offsets[axis]
                for axis, ref in enumerate(move.operands)
            ]
            coordinates = {
                obj: tuple(map(real, state.positions[index, obj]))
                for obj in range(problem.N)
            }
            coordinates[holding] = tuple(
                real(state.positions[index, holding, axis] - state.gripper[index, axis])
                + target[axis]
                for axis in range(3)
            )
            solver.add(
                target[2]
                >= real(problem.table_centres[index] - cfg.geometry_tolerance),
                target[2] <= real(cfg.max_height),
            )
            source = coordinates[holding]
            if cfg.require_supported_release:
                supported = [
                    z3.Abs(source[2] - real(problem.table_centres[index]))
                    <= real(cfg.geometry_tolerance)
                ]
                for obj, position in coordinates.items():
                    if obj != holding:
                        supported.append(
                            z3.And(
                                z3.Abs(source[0] - position[0])
                                < real(cfg.block_half_extent),
                                z3.Abs(source[1] - position[1])
                                < real(cfg.block_half_extent),
                                z3.Abs(
                                    source[2]
                                    - position[2]
                                    - real(2 * cfg.block_half_extent)
                                )
                                <= real(cfg.geometry_tolerance),
                            )
                        )
                solver.add(strengthen(z3.Or(supported), margin))
            ref = release.operands[0]
            if ref == "gripper":
                reference = target[2]
            elif ref == "Z_m":
                reference = coordinates[0][2]
                for obj in range(1, problem.N):
                    reference = z3.If(
                        coordinates[obj][2] > reference, coordinates[obj][2], reference
                    )
            else:
                bound = problem.examples[index].bindings.get(ref, ref)
                reference = (
                    real(problem.table_reference[index, 2])
                    if bound == "tbl"
                    else coordinates[bound][2]
                )
            retreat = reference + real(release.offsets[0])
            solver.add(
                retreat >= target[2] + real(0.02), retreat <= real(cfg.max_height)
            )
            solver.add(strengthen(problem.symbolic_goal(coordinates, index), margin))
        solver.maximize(margin)
        solver.minimize(
            z3.Sum(
                [
                    z3.Abs(offset - real(original))
                    for offset, original in zip(offsets, move.offsets)
                ]
            )
        )
        status = solver.check()
        elapsed = perf_counter() - started
        if status != z3.sat:
            return OffsetProposal(
                str(status),
                reason=(
                    solver.reason_unknown()
                    if status == z3.unknown
                    else "No shared endpoint offsets within radius"
                ),
                seconds=elapsed,
            )
        model = solver.model()

        def number(value):
            value = model.eval(value, model_completion=True)
            if not z3.is_rational_value(value):
                raise ValueError("Offset optimizer returned a non-rational model")
            return value.numerator_as_long() / value.denominator_as_long()

        solved = Action(
            "Move",
            move.operands,
            tuple(number(value) for value in offsets),
            "shared-offset-solver",
        )
        problem.check_deadline()
        moved = problem.transition(state, solved)
        terminal = None if moved is None else problem.transition(moved, release)
        if terminal is None or not problem.goal(terminal):
            return OffsetProposal(
                "rejected",
                reason="Exact nominal replay or endpoint check failed",
                seconds=perf_counter() - started,
            )
        return OffsetProposal(
            "sat",
            (solved, release),
            (moved, terminal),
            number(margin),
            seconds=perf_counter() - started,
        )
    except (KeyError, TypeError, ValueError) as exc:
        return OffsetProposal(
            "unsupported", reason=str(exc), seconds=perf_counter() - started
        )


def terminal_proposals(problem, node, successors, *, remaining_length):
    """Bounded solver repairs of nearby generated terminal operand structures."""
    cfg = problem.config
    if (
        not cfg.solve_offsets
        or remaining_length < 2
        or problem.symbolic_goal is None
        or any(h < 0 for h in node.state.holding)
    ):
        return ()
    moves = [(action, state) for action, state in successors if action.kind == "Move"]
    if not moves:
        return ()
    endpoints = np.stack([example.positions_path[-1] for example in problem.examples])
    # Consider placements near an observed endpoint first. Ranking proposes;
    # the requested logical goal remains a hard solver/replay constraint.
    moves.sort(
        key=lambda item: (
            float(np.mean((item[1].positions - endpoints) ** 2)),
            repr(item[0]),
        )
    )
    releases = [action for action in problem.actions if action.kind == "Release"]
    results = []
    for move, moved in moves[:4]:
        if np.max(np.abs(moved.positions - endpoints)) > max(
            0.04, 2 * cfg.offset_radius
        ):
            continue
        ranked = sorted(
            releases,
            key=lambda action: (
                float(
                    np.mean(
                        (
                            problem.reference(moved, action.operands[0])[:, 2]
                            + action.offsets[0]
                            - np.array(
                                [
                                    example.gripper_path[-1, 2]
                                    for example in problem.examples
                                ]
                            )
                        )
                        ** 2
                    )
                ),
                repr(action),
            ),
        )
        for release in ranked[:2]:
            if problem.counts["offset_solver_attempts"] >= cfg.solver_attempts:
                problem.counts["offset_solver_budget_exhausted"] = 1
                return tuple(results)
            signature = (problem.state_key(node.state), move, release)
            if signature in problem._offset_attempted:
                continue
            problem._offset_attempted.add(signature)
            problem.counts["offset_solver_attempts"] += 1
            proposal = solve_terminal_offsets(
                problem,
                node.state,
                move,
                release,
                timeout_ms=cfg.solver_timeout_ms,
                radius=cfg.offset_radius,
            )
            problem.counts[f"offset_solver_{proposal.status}"] += 1
            if proposal.status == "sat":
                results.append(proposal)
    return tuple(results)
