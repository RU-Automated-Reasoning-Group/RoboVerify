import argparse
from pathlib import Path
from typing import Optional, Tuple, Union

from z3 import And, Const, Consts, ExprRef, ForAll, If, Implies, Not

import synthesis.verification_lib.highlevel_verification_lib as highlevel_verification_lib
from synthesis.api.instructions import GoalAssign, MarkGoal, MoveDown, MoveRight, While
from synthesis.api.program import Program
from synthesis.inference_lib.inference import (
    run_2d_inner_loop_example,
    run_2d_outer_loop_example,
)


def _mk_mark_fn(context, mark_fn):
    return mark_fn if mark_fn is not None else (lambda t: context.Mark(t))


def _outer_template(context, head, i0, mark_fn=None):
    m = _mk_mark_fn(context, mark_fn)
    x = Const("x_I1", context.GoalSort)
    scanned = If(
        i0 == context.null,
        context._dr_reach(head, x),
        And(context._flat_between(head, x, i0), x != i0),
    )
    return And(
        head != context.null,
        context.l0(head) == head,
        Implies(i0 != context.null, context.d_star(head, i0)),
        ForAll([x], Implies(scanned, m(x))),
    )


def _inner_template(context, head, i0, j0, mark_fn=None):
    m = _mk_mark_fn(context, mark_fn)
    x = Const("x_I2", context.GoalSort)
    inner_scanned = If(
        j0 == context.null,
        context.r_star(i0, x),
        And(context.r_star(i0, x), context._f_plus(context.r_star, x, j0)),
    )
    return And(
        head != context.null,
        context.l0(head) == head,
        context.d_star(head, i0),
        Implies(j0 != context.null, context.r_star(i0, j0)),
        ForAll([x], Implies(And(context._flat_between(head, x, i0), x != i0), m(x))),
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


def _build_program(context, h, i, j, outer_inv, inner_inv):
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
                    invariant=inner_inv,
                ),
                MoveDown("i"),
            ],
            invariant=outer_inv,
        ),
    ]
    return Program(2, instructions=instructions)


def infer_loop_invariants(context) -> Tuple[ExprRef, ExprRef]:
    """
    Infer 2D nested-loop invariants through proposal+validation.

    Like stack, proposal is delegated to inference_lib. Unlike stack (constants only),
    this setup includes uniquely-defined function terms such as l0(constant).
    """
    learned_outer_invariant, learned_outer_invariant_lists = run_2d_outer_loop_example(
        context
    )
    learned_inner_invariant, learned_inner_invariant_lists = run_2d_inner_loop_example(
        context
    )
    return learned_outer_invariant, learned_inner_invariant


def verify_2d_program_with_learned_invariant(
    use_hardcoded_invariant: bool = False,
    counterexample_image_dir: Optional[Union[str, Path]] = None,
):
    """Verify nested while-loop mark-all using program-level VC generation."""
    context = highlevel_verification_lib.HighLevelContext(
        mode="declare",
        verification_mode="goals",
    )

    h, i, j = Consts("h i j", context.GoalSort)
    if use_hardcoded_invariant:
        # Original handwritten invariant choice.
        outer_inv = _outer_template(context, h, i)
        inner_inv = _inner_template(context, h, i, j)
    else:
        outer_inv, inner_inv = infer_loop_invariants(context)

    program = _build_program(context, h, i, j, outer_inv, inner_inv)

    pre = precondition(context, h)
    post = postcondition(context, h)
    return bool(
        program.highlevel_verification(
            pre,
            post,
            context=context,
            counterexample_image_dir=counterexample_image_dir,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use-hardcoded-invariant",
        action="store_true",
        help="Use the original handwritten outer/inner invariants instead of inferred ones.",
    )
    parser.add_argument(
        "--counterexample-dir",
        default="highlevel_counterexamples",
        metavar="DIR",
        help=(
            "Where to save Goal-mode counterexample PNGs when a VC check fails. "
            "Pass an empty string to disable."
        ),
    )
    args = parser.parse_args()
    ce = args.counterexample_dir.strip() or None
    success = verify_2d_program_with_learned_invariant(
        use_hardcoded_invariant=args.use_hardcoded_invariant,
        counterexample_image_dir=ce,
    )
    print(f"overall_ok: {success}")
