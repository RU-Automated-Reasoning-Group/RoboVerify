import argparse
import os
import time
from copy import deepcopy

import numpy as np
from z3 import And, Consts, ForAll, Implies, Not, Or

import synthesis.verification_lib.highlevel_verification_lib as highlevel_verification_lib
from synthesis.api.instructions import PickPlaceByName
from synthesis.api.program import Assign, Program, Put, While
from synthesis.entry.motion_options import add_motion_options, motion_noise_from_args
from synthesis.entry.run_rollouts import run_program_rollouts
from synthesis.inference_lib.demo_store import DemoStore, InvInference, tower_vocabulary
from synthesis.inference_lib.inference import (
    instantiate_invariant,
    serialize_invariant,
)
from synthesis.verification_lib.motion_verification import MotionContract


def build_stack_programs(
    context, learned_invariant=True, learned_invariant_lists=None, *, max_iters=10
):
    """Build the existing symbolic and physical Stack programs.

    Execution does not require an invariant; this also supplies the program used
    to collect loop-head demonstrations before invoking the learner.
    """
    b_prime, b, n, b0 = Consts("b_prime b n b0", context.BoxSort)
    instructions = [
        Assign("b", "b0"),
        While(
            instantiated_cond=And(
                ForAll(
                    [n],
                    Or(
                        b_prime == n,
                        Not(context.ON_star(n, b_prime)),
                    ),
                ),
                b_prime != b,
            ),
            guard_exists_vars=[b_prime],
            body=[Put("b_prime", "b"), Assign("b", "b_prime")],
            invariant=learned_invariant,
            max_iters=max_iters,
        ),
    ]
    program = Program(2, instructions=instructions)

    BOX_LENGTH = 0.05
    ll_instruction = deepcopy(instructions)
    ll_instruction[1].body = [
        PickPlaceByName(
            grab_box_name="b_prime",
            target_box_name_x="b_prime",
            target_box_name_y="b_prime",
            target_box_name_z="b",
            target_offset=[0.0, 0.0, 4 * BOX_LENGTH],
            release=False,
        ),
        PickPlaceByName(
            grab_box_name="b_prime",
            target_box_name_x="b",
            target_box_name_y="b",
            target_box_name_z="b",
            target_offset=[0.0, 0.0, 4 * BOX_LENGTH],
            release=False,
        ),
        PickPlaceByName(
            grab_box_name="b_prime",
            target_box_name_x="b",
            target_box_name_y="b",
            target_box_name_z="b",
            target_offset=[0.0, 0.0, 1.5 * BOX_LENGTH],
            release=True,
        ),
    ]
    ll_instruction[1].invariant = learned_invariant_lists
    ll_instruction[1].body.append(Assign("b", "b_prime"))
    ll_program = Program(2, instructions=ll_instruction)

    return program, ll_program


def verify_stack_program_with_learned_invariant(
    demo_store: DemoStore,
    loop_id: str = "1",
    verification_mode: str = "infinite",
    num_blocks: int = 4,
    visualize_finite_scene: bool = True,
    visualization_prefix: str = "verify_stack",
    noise=None,
    motion_timeout_ms: int = 5000,
):
    """Infer from recorded loop-head demonstrations and verify the stack program."""
    # Inference is always done in the infinite-block (DeclareSort) setting.
    inference_context = highlevel_verification_lib.HighLevelContext(mode="declare")
    learned_invariant, learned_invariant_lists = InvInference(
        demo_store, loop_id, tower_vocabulary("stack"), inference_context
    )

    if verification_mode == "finite":
        context = highlevel_verification_lib.HighLevelContext(
            mode="enum",
            num_blocks=num_blocks,
            visualize_enum_scene=visualize_finite_scene,
            visualization_prefix=visualization_prefix,
        )
        learned_spec = serialize_invariant(learned_invariant, inference_context)
        learned_invariant = instantiate_invariant(
            learned_spec, context, known_const_names=["b0", "b"]
        )
    else:
        context = highlevel_verification_lib.HighLevelContext(mode="declare")

    program, ll_program = build_stack_programs(
        context, learned_invariant, learned_invariant_lists
    )

    m, n = Consts("m n", context.BoxSort)
    precondition = And(
        ForAll(
            [m],
            ForAll([n], Or(m == n, Not(context.ON_star(n, m)))),
        ),
        # ForAll(
        #     [m],
        #     ForAll([n], context.Higher(n, m)),
        # ),
        ForAll([m, n], Implies(m != n, context.Scattered(m, n))),
    )

    m, b0 = Consts("m b0", context.BoxSort)
    postcondition = ForAll([m], context.ON_star(m, b0))

    hl_ok = program.highlevel_verification(precondition, postcondition, context=context)
    ll_ok = ll_program.lowlevel_verification(
        constants=["b0", "b", "b_prime"],
        contracts={"1": MotionContract("b_prime", "b")},
        noise=noise,
        timeout_ms=motion_timeout_ms,
    )
    print(f"hl_ok: {hl_ok}", f"ll_ok: {ll_ok}")
    return bool(hl_ok and ll_ok)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_motion_options(parser)
    parser.add_argument(
        "--demo-store",
        required=True,
        help="JSON loop-head traces saved by DemoStore.save().",
    )
    parser.add_argument(
        "--loop-id",
        default="1",
        help="Instruction path of the recorded loop (default: 1).",
    )
    parser.add_argument(
        "--verification-mode",
        choices=["infinite", "finite"],
        default="infinite",
        help="Use infinite (DeclareSort) or finite (EnumSort) verification.",
    )
    parser.add_argument(
        "--num-blocks",
        type=int,
        default=4,
        help="Number of blocks when verification mode is finite.",
    )
    parser.add_argument(
        "--disable-scene-viz",
        action="store_true",
        help="Disable image generation for finite-mode verification.",
    )
    parser.add_argument(
        "--viz-prefix",
        type=str,
        default="verify_stack",
        help="Output prefix for generated finite-mode scene images.",
    )
    args = parser.parse_args()
    noise = motion_noise_from_args(parser, args)

    verify_stack_program_with_learned_invariant(
        noise=noise,
        motion_timeout_ms=args.motion_timeout_ms,
        demo_store=DemoStore.load(args.demo_store),
        loop_id=args.loop_id,
        verification_mode=args.verification_mode,
        num_blocks=args.num_blocks,
        visualize_finite_scene=not args.disable_scene_viz,
        visualization_prefix=args.viz_prefix,
    )
