"""Shared-representative Put effects, with selectors outside the whole post.

Legality remains in program.put_precondition. The atom substitutions match
WitnessWP.lean and are simultaneous and capture avoiding, including under
arbitrary quantifiers. This module performs no EPR/model reduction.
"""

import z3

from synthesis.util.symbols import fresh_const
from synthesis.verification_lib.relational_axioms import (
    bottom,
    maximum,
    next_rep,
    physical,
    strict,
)


def _rewrite(post, source, target, row, context):
    x, y = z3.Var(0, context.BoxSort), z3.Var(1, context.BoxSort)
    if target is None:
        on = z3.If(z3.Or(x == source, y == source), x == y, context.ON_star(x, y))
        higher_in = z3.BoolVal(True)
        scattered = z3.If(
            z3.Or(x == source, y == source), x != y, context.Scattered(x, y)
        )
    else:
        on = z3.If(
            x == source,
            z3.Or(y == source, context.ON_star(target, y)),
            z3.And(y != source, context.ON_star(x, y)),
        )
        higher_in = strict(context, x, target)
        scattered = z3.If(
            x == source,
            z3.And(y != source, context.Scattered(target, y)),
            z3.If(y == source, context.Scattered(x, target), context.Scattered(x, y)),
        )
    higher = z3.If(
        x == source,
        z3.Or(y == source, row(y)),
        z3.If(y == source, higher_in, context.Higher(x, y)),
    )
    return z3.substitute_funs(
        post,
        *(
            (
                relation,
                z3.If(
                    z3.And(physical(context, x), physical(context, y)),
                    replacement,
                    relation(x, y),
                ),
            )
            for relation, replacement in (
                (context.ON_star, on),
                (context.Higher, higher),
                (context.Scattered, scattered),
            )
        ),
    )


def put_effect(instruction, post, context):
    context.validate_height_formula(post)
    pending, seen, needs_height = [post], set(), False
    while pending:
        node = pending.pop()
        if node.get_id() in seen:
            continue
        seen.add(node.get_id())
        if z3.is_quantifier(node):
            pending.append(node.body())
        elif z3.is_app(node):
            if node.decl() == context.Higher:
                needs_height = True
                break
            pending.extend(node.children())
    source = context.get_consts(instruction.upper_block)
    target = context.get_consts(instruction.base_block)
    representative = fresh_const(
        context.BoxSort, "height_rep", avoid=(post, source, target)
    )
    row = lambda obj: context.Higher(representative, obj)
    table_post = None
    if context.use_tbl:
        table_rewrite = _rewrite(post, source, None, row, context)
        table_post = (
            z3.ForAll(
                [representative],
                z3.Implies(
                    z3.And(
                        physical(context, representative),
                        bottom(context, representative, avoid=(post,)),
                    ),
                    table_rewrite,
                ),
            )
            if needs_height
            else table_rewrite
        )
    if instruction.base_block == "tbl":
        if not context.use_tbl:
            raise ValueError("Shared WP for table placement requires use_tbl=True")
        return table_post
    # If Q contains no Higher atoms, its replacement is independent of the
    # representative. Coverage makes the whole guarded rule equivalent to this
    # one rewrite; avoiding the redundant binders matters to the reduction bound.
    block_post = (
        z3.And(
            z3.Implies(
                maximum(context, target, avoid=(post,)),
                _rewrite(post, source, target, lambda _: z3.BoolVal(True), context),
            ),
            z3.ForAll(
                [representative],
                z3.Implies(
                    z3.And(
                        physical(context, representative),
                        next_rep(context, target, representative, avoid=(post,)),
                    ),
                    _rewrite(post, source, target, row, context),
                ),
            ),
        )
        if needs_height
        else _rewrite(post, source, target, row, context)
    )
    if not context.use_tbl:
        return block_post
    return z3.And(
        z3.Implies(target == context.get_consts("tbl"), table_post),
        z3.Implies(target != context.get_consts("tbl"), block_post),
    )
