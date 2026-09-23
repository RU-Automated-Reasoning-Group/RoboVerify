"""Bounded initial-state witnesses, distinct from inductiveness countermodels."""

import itertools
from dataclasses import dataclass, field

import z3

from synthesis.api.instructions import Assign, Get, Put, Skip, While
from synthesis.api.program import wp


def first_choices(instruction, context):
    """Ordered finite bindings exactly as the runtime's first-witness policy."""
    variables = [context.get_consts(str(v)) for v in instruction.guard_exists_vars]
    previous = []
    for values in itertools.product(context.enum_blocks, repeat=len(variables)):
        substitutions = tuple(zip(variables, values))
        guard = z3.substitute(instruction.instantiated_cond, *substitutions)
        selected = z3.And(guard, z3.Not(z3.Or(*previous)))
        yield selected, substitutions
        previous.append(guard)


def finite_formula(expr, context):
    """Expand finite Box quantifiers with capture-safe de Bruijn substitution."""
    memo = {}

    def visit(node):
        # Retain ASTs: numeric Z3 IDs can be reused after temporary binders die.
        key = node
        if key in memo:
            return memo[key]
        if z3.is_quantifier(node):
            if any(node.var_sort(i) != context.BoxSort for i in range(node.num_vars())):
                raise ValueError("Witness queries support Box quantifiers only")
            rows = [
                visit(z3.substitute_vars(node.body(), *reversed(values)))
                for values in itertools.product(
                    context.enum_blocks, repeat=node.num_vars()
                )
            ]
            result = (z3.And if node.is_forall() else z3.Or)(*rows)
        elif z3.is_app(node) and node.num_args():
            result = node.decl()(*(visit(child) for child in node.children()))
        else:
            result = node
        result = z3.simplify(result)
        memo[key] = result
        return result

    return visit(expr)


def execution_preimage(instructions, target, context):
    """Shared placement WP, with finite deterministic Get binding for execution."""
    for instruction in reversed(instructions):
        if isinstance(instruction, Get):
            target = z3.Or(
                *[
                    z3.And(guard, z3.substitute(target, *bindings))
                    for guard, bindings in first_choices(instruction, context)
                ]
            )
        elif isinstance(instruction, (Assign, Put, Skip)):
            target = wp(instruction, target, context)
        else:
            raise ValueError(
                f"Unsupported bounded instruction: {type(instruction).__name__}"
            )
        target = finite_formula(target, context)
    return target


def missing_head_query(program, context, iterations):
    """Initial states reaching NOT invariant within the declared iteration bound.

    Includes the head before the first guard and every subsequent head, even when
    the guard is false. The adapter is responsible for initial/frozen geometry and
    the physical domain. No inductiveness countermodel is assumed reachable.
    """
    if context.mode != "enum" or iterations < 0:
        raise ValueError(
            "A finite context and nonnegative iteration bound are required"
        )
    loops = [
        (i, row) for i, row in enumerate(program.instructions) if isinstance(row, While)
    ]
    if len(loops) != 1 or any(isinstance(row, While) for row in loops[0][1].body):
        raise ValueError("Witness queries require one non-nested loop")
    index, loop = loops[0]
    missing = z3.Not(loop.invariant)
    target = finite_formula(missing, context)
    for _ in range(iterations):
        continuation = execution_preimage(loop.body, target, context)
        target = finite_formula(
            z3.Or(
                missing,
                *[
                    z3.And(guard, z3.substitute(continuation, *bindings))
                    for guard, bindings in first_choices(loop, context)
                ],
            ),
            context,
        )
    return execution_preimage(program.instructions[:index], target, context)


@dataclass
class WitnessQuery:
    solver: object
    coverage: object
    decode: object
    iterations: int
    context: object = None


@dataclass
class WitnessSearch:
    status: str
    attempts: list = field(default_factory=list)
    size: int | None = None
    witness: object = None
    query: object = None
    reason: str = ""


def find_witness(build_query, max_blocks, *, record=None):
    """Smallest witness in the adapter's bounded domain; unknown stops search."""
    attempts = []
    for size in range(1, max_blocks + 1):
        query = build_query(size)
        # Separate empty initial domains from lack of uncovered reachable heads.
        feasible = query.solver.check()
        row = dict(
            num_blocks=size, iterations=query.iterations, initial_domain=str(feasible)
        )
        if feasible == z3.unknown:
            row.update(status="unknown", reason=query.solver.reason_unknown())
        elif feasible == z3.unsat:
            row.update(status="empty_initial_domain")
        else:
            query.solver.add(query.coverage)
            answer = query.solver.check()
            row.update(status=str(answer))
            if answer == z3.unknown:
                row["reason"] = query.solver.reason_unknown()
        attempts.append(row)
        if record:
            record(size, query, row)
        if row["status"] == "unknown":
            return WitnessSearch("unknown", attempts, reason=row["reason"])
        if row["status"] == "sat":
            return WitnessSearch(
                "found", attempts, size, query.decode(query.solver.model()), query
            )
    return WitnessSearch(
        "no_reachable_counterexample",
        attempts,
        reason="No missing head found in the declared initial domains and execution bounds",
    )
