"""Verification controls transcribed from POPL2027.pdf, page 46, Figs. 14–15.

The literal formulas are preserved. Explicitly named interpretations resolve
the undefined Unstack b or the printed Reverse tbk; they are not learned results.
None is assumed to be inductive for the current executable programs.
"""

import z3


def unstack_paper(context):
    u, v, b, b0, tbl = [context.get_consts(n) for n in ("u", "v", "b", "b0", "tbl")]
    on, higher, scattered = context.ON_star, context.Higher, context.Scattered
    return z3.ForAll(
        [u, v],
        z3.And(
            z3.Implies(higher(u, v), z3.Or(scattered(u, v), on(u, v))),
            z3.Implies(higher(b, u), z3.Or(higher(u, b), on(u, b0))),
            z3.Implies(higher(u, b), z3.Or(higher(b, u), on(u, b0))),
            z3.Implies(z3.And(higher(b, u), on(u, b0)), b0 == u),
            z3.Or(on(u, b0), higher(u, tbl), higher(b0, u)),
        ),
    )


def unstack_paper_base(context):
    """Explicit interpretation b=b0; the printed Unstack program has no b."""
    return z3.substitute(
        unstack_paper(context), (context.get_consts("b"), context.get_consts("b0"))
    )


def unstack_paper_physical_base(context):
    """Interpret b=b0 and retain b0's physical-block type in the loop invariant."""
    return z3.And(
        context.get_consts("b0") != context.get_consts("tbl"),
        unstack_paper_base(context),
    )


def reverse_paper(context, *, allow_final_base=False):
    u, v, b, b0, tbl, tbk = [
        context.get_consts(n) for n in ("u", "v", "b", "b0", "tbl", "tbk")
    ]
    on, frozen = context.ON_star, context.ON_star_zero
    return z3.ForAll(
        [u, v],
        z3.And(
            z3.Implies(frozen(u, v), z3.Or(frozen(u, b), on(u, v))),
            z3.Implies(z3.And(frozen(v, u), frozen(u, b)), on(u, v)),
            z3.Implies(on(u, v), z3.Or(on(b, u), frozen(u, v))),
            z3.Implies(z3.And(on(b, u), on(u, v)), frozen(v, u)),
            z3.Or(frozen(u, b), u == tbl, on(u, b0)),
            (
                z3.Implies(on(b, u), z3.Or(b == b0, z3.Not(on(u, b0))))
                if allow_final_base
                else z3.Implies(on(b, u), z3.Not(on(u, b0)))
            ),
            z3.Implies(on(u, b0), z3.Not(frozen(u, tbl))),
            z3.Or(on(tbk, b), frozen(b, b0)),
        ),
    )


def reverse_paper_table(context):
    """Explicit interpretation tbk=tbl; retain all eight printed clauses."""
    return z3.substitute(
        reverse_paper(context), (context.get_consts("tbk"), context.get_consts("tbl"))
    )


def reverse_paper_final(context):
    """Candidate repair: interpret tbk=tbl and admit the final b=b0 phase."""
    return z3.substitute(
        reverse_paper(context, allow_final_base=True),
        (context.get_consts("tbk"), context.get_consts("tbl")),
    )
