"""Relational height selectors and the strengthening described in the axiom PDF.

All selectors quantify over physical blocks. They are macros, not new symbols.
The original ON*, frozen ON*, Scattered, and table axioms remain in the context.
"""

import z3

from synthesis.util.symbols import fresh_const


def physical(context, obj):
    return obj != context.get_consts("tbl") if context.use_tbl else z3.BoolVal(True)


def strict(context, left, right):
    return z3.And(context.Higher(left, right), z3.Not(context.Higher(right, left)))


def maximum(context, block, *, avoid=()):
    other = fresh_const(context.BoxSort, "maximum", avoid=(block, *avoid))
    return z3.ForAll(
        [other], z3.Implies(physical(context, other), context.Higher(block, other))
    )


def bottom(context, block, *, avoid=()):
    other = fresh_const(context.BoxSort, "bottom", avoid=(block, *avoid))
    return z3.ForAll(
        [other], z3.Implies(physical(context, other), context.Higher(other, block))
    )


def next_rep(context, block, representative, *, avoid=()):
    other = fresh_const(
        context.BoxSort, "next_height", avoid=(block, representative, *avoid)
    )
    return z3.And(
        strict(context, representative, block),
        z3.ForAll(
            [other],
            z3.Implies(
                z3.And(physical(context, other), strict(context, other, block)),
                context.Higher(other, representative),
            ),
        ),
    )


def add_height_representatives(context, solver):
    """H3, W1, W2, and optionally C1; retain the existing order/link axioms."""
    x, y, u = [
        fresh_const(context.BoxSort, name)
        for name in ("height_x", "height_y", "height_u")
    ]
    px, py = physical(context, x), physical(context, y)
    solver.assert_and_track(
        z3.ForAll(
            [x, y],
            z3.Implies(
                z3.And(px, py), z3.Or(context.Higher(x, y), context.Higher(y, x))
            ),
        ),
        "height_total",
    )
    solver.assert_and_track(
        z3.Implies(
            z3.Exists([x], px),
            z3.Exists([u], z3.And(physical(context, u), bottom(context, u))),
        ),
        "height_bottom_exists",
    )
    solver.assert_and_track(
        z3.ForAll(
            [x],
            z3.Implies(
                px,
                z3.Or(
                    maximum(context, x),
                    z3.Exists(
                        [u], z3.And(physical(context, u), next_rep(context, x, u))
                    ),
                ),
            ),
        ),
        "height_next_exists",
    )
    if context.wp_options.support_complete:
        solver.assert_and_track(
            z3.ForAll(
                [x, y],
                z3.Implies(
                    z3.And(px, py, context.Higher(x, y)),
                    z3.Exists(
                        [u],
                        z3.And(
                            physical(context, u),
                            context.ON_star(x, u),
                            context.Higher(u, y),
                            context.Higher(y, u),
                        ),
                    ),
                ),
            ),
            "height_support_complete",
        )
