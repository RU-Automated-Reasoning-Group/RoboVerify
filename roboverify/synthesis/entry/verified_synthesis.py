"""Verify and refine the existing Stack program (Phase E, before CFG synthesis).

RunLogger retains every invariant and counterexample. Entry failures are saved
as NeedsResynthesis; Phase F is required to change the symbolic program.
"""

import argparse
import itertools

import numpy as np
import z3

from synthesis.entry.motion_options import add_motion_options, motion_noise_from_args
from synthesis.entry.verify_stack_with_learned_invariant import build_stack_programs
from synthesis.experiment.run_logger import RunLogger
from synthesis.inference_lib.demo_store import DemoStore, InvInference, tower_vocabulary
from synthesis.verification_lib.cegis import (
    MotionBlockSpec,
    NeedsResynthesis,
    optimize_motion_parameters,
    run_motion_cegis,
    run_symbolic_cegis,
)
from synthesis.verification_lib.highlevel_verification_lib import (
    HighLevelContext,
    InvariantSpec,
)
from synthesis.verification_lib.motion_verification import MotionContract

_CONTEXT_IDS = itertools.count()


def stack_problem_builder():
    ids = _CONTEXT_IDS

    def build(invariant, size):
        context = HighLevelContext(
            mode="declare" if size is None else "enum",
            num_blocks=size,
            sort_name="Box" if size is None else f"StackCEGIS{next(ids)}",
        )
        inv = context.spec_to_expr(
            InvariantSpec({"sexpr": invariant.sexpr()}), ["b0", "b"]
        )
        program, _ = build_stack_programs(context, inv)
        m, n, b0 = z3.Consts("m n b0", context.BoxSort)
        pre = z3.And(
            z3.ForAll([m, n], z3.Or(m == n, z3.Not(context.ON_star(n, m)))),
            z3.ForAll([m, n], z3.Implies(m != n, context.Scattered(m, n))),
        )
        post = z3.ForAll([m], context.ON_star(m, b0))
        return program, pre, post, context

    return build


def run(args, logger):
    noise = None
    if args.motion_noise is not None:
        from synthesis.verification_lib.bmc_lib import NoiseSpec

        noise = NoiseSpec(*args.motion_noise)
    store = DemoStore.load(args.demo_store)
    context = HighLevelContext()
    try:
        symbolic = run_symbolic_cegis(
            store,
            "1",
            tower_vocabulary("stack"),
            context,
            stack_problem_builder(),
            max_iterations=args.symbolic_iterations,
            max_blocks=args.max_blocks,
            timeout_ms=args.symbolic_timeout_ms,
            prove_unbounded=not args.bounded_only,
            learner=InvInference if args.learner == "legacy" else None,
            logger=logger,
        )
    except NeedsResynthesis as exc:
        logger.finish(
            "needs_resynthesis",
            phase_result="symbolic",
            failed_vc=exc.result.failed_vc_kind,
            num_blocks=exc.result.num_blocks,
            reason=str(exc),
            scene_reason=exc.result.reason,
        )
        return 2
    if not symbolic:
        logger.finish(
            symbolic.status,
            phase_result="symbolic",
            reason=symbolic.reason,
            iterations=symbolic.iterations,
        )
        return 2
    _, physical = build_stack_programs(
        context, symbolic.invariant, [symbolic.invariant]
    )
    loop = physical.instructions[1]
    spec = MotionBlockSpec(
        "1",
        (symbolic.invariant, loop.instantiated_cond),
        ("b0", "b", "b_prime"),
        MotionContract("b_prime", "b"),
        "Put(b_prime,b);Assign(b,b_prime)",
    )
    base_score = None
    if args.expert_states:
        from synthesis.mcmc.synthesis import Runner

        expert_states = np.load(args.expert_states, allow_pickle=False)

        def base_score(candidate):
            runner = Runner(
                candidate,
                expert_states,
                len(args.seeds),
                args.num_blocks,
                seeds=args.seeds,
                goal_feature_reward_weight=0.0,
            )
            return runner(candidate.register_trainable_parameter())

    def resynthesize(candidate, penalty, iteration):
        if base_score is None and not args.penalty_only:
            raise ValueError(
                "Motion repair needs --expert-states (NumPy observations), or explicit --penalty-only"
            )
        return optimize_motion_parameters(
            candidate,
            penalty,
            base_score=base_score,
            weight=args.penalty_weight,
            seed=args.seed + iteration,
            iterations=args.cem_iterations,
            samples=args.cem_samples,
            elites=args.cem_elites,
        )

    motion = run_motion_cegis(
        physical,
        [spec],
        resynthesize,
        noise=noise,
        timeout_ms=args.motion_timeout_ms,
        max_iterations=args.motion_iterations,
        logger=logger,
    )
    logger.write_artifact("program.txt", str(motion.program))
    logger.finish(
        motion.status,
        phase_result="motion",
        proof_scope=symbolic.verification.scope,
        motion_mode=motion.verification.mode,
        iterations=motion.iterations,
        objective="demonstration_mmd_plus_penalty" if base_score else "penalty_only",
    )
    return 0 if motion else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo-store", required=True)
    parser.add_argument("--run-root", default="runs")
    parser.add_argument("--slug", default="stack")
    parser.add_argument("--learner", choices=("monotone", "legacy"), default="monotone")
    parser.add_argument("--symbolic-iterations", type=int, default=20)
    parser.add_argument("--motion-iterations", type=int, default=3)
    parser.add_argument("--max-blocks", type=int, default=4)
    parser.add_argument("--symbolic-timeout-ms", type=int, default=5000)
    parser.add_argument(
        "--bounded-only",
        action="store_true",
        help="Check only 2..max-blocks; result explicitly reports finite scope",
    )
    parser.add_argument(
        "--expert-states",
        help="NumPy .npy array of expert observations for the MMD objective",
    )
    parser.add_argument(
        "--penalty-only",
        action="store_true",
        help="Explicitly permit motion repair without demonstration-distance scoring",
    )
    parser.add_argument("--penalty-weight", type=float, default=1.0)
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cem-iterations", type=int, default=2)
    parser.add_argument("--cem-samples", type=int, default=16)
    parser.add_argument("--cem-elites", type=int, default=4)
    add_motion_options(parser)
    args = parser.parse_args(argv)
    motion_noise_from_args(parser, args)
    if not (
        args.symbolic_iterations > 0
        and args.motion_iterations > 0
        and args.max_blocks >= 2
        and args.symbolic_timeout_ms > 0
        and 1 <= args.cem_elites <= args.cem_samples
        and args.cem_iterations >= 0
        and np.isfinite(args.penalty_weight)
        and args.penalty_weight >= 0
    ):
        parser.error("Invalid iteration, block, solver, CEM, or penalty budget")
    with RunLogger(args.run_root, "cegis", vars(args), slug=args.slug) as logger:
        logger.progress_line(f"CEGIS run: {logger.run_dir}")
        try:
            code = run(args, logger)
        except Exception as exc:
            logger.log_exception(exc)
            logger.finish("failed", reason=str(exc))
            raise
    print(
        f"CEGIS finished; read with: uv run python -m synthesis.experiment.report --run {logger.run_dir}"
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
