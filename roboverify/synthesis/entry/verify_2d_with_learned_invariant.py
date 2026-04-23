import argparse

from z3 import And, Const, Consts, ForAll, If, Implies, Not

import synthesis.verification_lib.highlevel_verification_lib as highlevel_verification_lib
from synthesis.api.instructions import GoalAssign, MarkGoal, MoveDown, MoveRight, While
from synthesis.api.program import Program


def _mk_mark_fn(context, mark_fn):
    return mark_fn if mark_fn is not None else (lambda t: context.Mark(t))


def invariant_outer(context, h0, i0, mark_fn=None):
    m = _mk_mark_fn(context, mark_fn)
    x = Const("x_I1", context.GoalSort)
    scanned = If(
        i0 == context.null,
        context._dr_reach(h0, x),
        And(context._flat_between(h0, x, i0), x != i0),
    )
    return And(
        h0 != context.null,
        context.l0(h0) == h0,
        Implies(i0 != context.null, context.d_star(h0, i0)),
        ForAll([x], Implies(scanned, m(x))),
    )


def invariant_inner(context, h0, i0, j0, mark_fn=None):
    m = _mk_mark_fn(context, mark_fn)
    x = Const("x_I2", context.GoalSort)
    inner_scanned = If(
        j0 == context.null,
        context.r_star(i0, x),
        And(context.r_star(i0, x), context._f_plus(context.r_star, x, j0)),
    )
    return And(
        h0 != context.null,
        context.l0(h0) == h0,
        context.d_star(h0, i0),
        Implies(j0 != context.null, context.r_star(i0, j0)),
        ForAll([x], Implies(And(context._flat_between(h0, x, i0), x != i0), m(x))),
        ForAll([x], Implies(inner_scanned, m(x))),
    )


def precondition(context, h0, mark_fn=None):
    m = _mk_mark_fn(context, mark_fn)
    x = Const("x_P", context.GoalSort)
    return And(
        h0 != context.null,
        context.l0(h0) == h0,
        ForAll([x], Implies(context._dr_reach(h0, x), Not(m(x)))),
    )


def postcondition(context, h0, mark_fn=None):
    m = _mk_mark_fn(context, mark_fn)
    x = Const("x_Q", context.GoalSort)
    return ForAll([x], Implies(context._dr_reach(h0, x), m(x)))


def verify_2d_program_with_learned_invariant():
    """Verify nested while-loop mark-all using program-level VC generation."""
    context = highlevel_verification_lib.HighLevelContext(
        mode="declare",
        verification_mode="goals",
    )

    h, i, j = Consts("h i j", context.GoalSort)

    instructions = [
        GoalAssign("i", "h"),
        While(
            instantiated_cond=i != context.null,
            guard_exists_vars=[],
            body=[
                GoalAssign("j", "i"),
                While(
                    instantiated_cond=j != context.null,
                    guard_exists_vars=[],
                    body=[MarkGoal("j"), MoveRight("j")],
                    invariant=invariant_inner(context, h, i, j),
                ),
                MoveDown("i"),
            ],
            invariant=invariant_outer(context, h, i),
        ),
    ]
    program = Program(2, instructions=instructions)

    pre = precondition(context, h)
    post = postcondition(context, h)
    return bool(program.highlevel_verification(pre, post, context=context))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.parse_args()
    success = verify_2d_program_with_learned_invariant()
    print(f"overall_ok: {success}")
