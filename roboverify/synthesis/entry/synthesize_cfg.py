"""Instrumented CFG synthesis; synthesized candidates still require verification."""

import argparse
import random
import signal
from functools import partial

from synthesis.api.instructions import (
    Move,
    MoveByName,
    Pick,
    PickByName,
    Release,
    ReleaseByName,
    Skip,
)
from synthesis.api.program import Program, generate_random_program
from synthesis.cfg.bindings import close_objects, mutate_scoped, require_closed
from synthesis.cfg.demo_sources import unstack_oracle
from synthesis.cfg.demos import DemoSegment, DemoTrace
from synthesis.cfg.execute import execute_cfg
from synthesis.cfg.graph import RelationalCFG
from synthesis.cfg.lower import lower, lower_region
from synthesis.cfg.recordings import load_traces, save_traces
from synthesis.cfg.region import BlockRegion, LoopRegion
from synthesis.cfg.reset import collect_recording
from synthesis.cfg.straightline import (
    SearchBudget,
    postcondition_reached,
    segment_rollout,
    straight_line_synthesize,
)
from synthesis.cfg.verified_synthesis import verified_synthesis
from synthesis.entry.motion_options import add_motion_options, motion_noise_from_args
from synthesis.experiment.run_logger import RunLogger
from synthesis.mcmc.synthesis import (
    make_roboverify_env,
    mutate_program,
    preserved_global_rng,
    set_np_seed,
)
from synthesis.predicates.language import Language
from synthesis.predicates.scene import evaluate
from synthesis.predicates.term import (
    atom,
    boolean,
    conjunction,
    disjunction,
    forall,
    implies,
    ref,
)
from synthesis.verification_lib.bmc_lib import NoiseSpec
from synthesis.verification_lib.highlevel_verification_lib import HighLevelContext


def task_spec(task):
    x, y, b0 = ref("x"), ref("y"), ref("b0")
    if task == "unstack":
        return forall(
            ["x"], disjunction(atom("eq", x, ref("tbl")), atom("ON_star", x, b0))
        ), forall(["x", "y"], implies(atom("ON_star", x, y), atom("eq", x, y)))
    return boolean(True), forall(["x"], atom("ON_star", x, b0))


def run(args, logger):
    context = HighLevelContext(use_tbl=args.task == "unstack")
    if args.demos:
        traces = load_traces(args.demos)
    else:
        if args.task == "unstack":
            oracle = unstack_oracle()
        else:
            from synthesis.entry.verify_stack_with_learned_invariant import (
                build_stack_programs,
            )

            _, oracle = build_stack_programs(context, max_iters=args.num_blocks)
        traces = []
        for seed in args.seeds:
            with preserved_global_rng():
                set_np_seed(seed)
                env = make_roboverify_env(args.task, num_blocks=args.num_blocks)
                try:
                    states, recording = collect_recording(oracle, env)
                finally:
                    env.close()
            traces.append(
                DemoTrace(
                    states,
                    tuple(recording.snapshots),
                    (tuple(recording.actions), tuple(recording.action_indices)),
                    seed,
                    args.task,
                    args.num_blocks,
                )
            )
        save_traces(logger.artifact_dir() / "demonstrations.npz", traces)
    if any(t.task != args.task or t.num_blocks != args.num_blocks for t in traces):
        raise ValueError(
            "Demonstration task/block count differs from requested synthesis"
        )
    segments = [
        DemoSegment(
            i,
            0,
            len(t.states) - 1,
            t,
            {
                "b0": 0,
                **{f"o{j}": j for j in range(t.num_blocks)},
                **({"tbl": "tbl"} if context.use_tbl else {}),
            },
        )
        for i, t in enumerate(traces)
    ]
    pre, post = task_spec(args.task)
    from synthesis.cfg.demo_validation import validate_demonstrations
    from synthesis.cfg.refine import scene_at

    validation = validate_demonstrations(segments, pre, post)
    demo_checks = validation.as_dict()
    logger.log_event(
        "demo_validation", f"Task predicate checks: {demo_checks}", force=True
    )
    if not validation:
        logger.finish(
            "invalid_demonstrations",
            demonstration_checks=demo_checks,
            formal_verification="not_run",
            reason="Provide recordings satisfying the initial and final task conditions",
        )
        return 2
    cfg = RelationalCFG.initial(
        segments, pre, post, ("b0", "tbl") if context.use_tbl else ("b0",)
    )
    budget = SearchBudget(
        iterations=args.iterations,
        cem_iterations=args.cem_iterations,
        cem_samples=args.cem_samples,
        cem_elites=args.cem_elites,
        theta=args.theta,
        distance=args.distance,
        seed=args.seed,
        inner_samples=args.inner_samples,
        final_samples=args.final_samples,
        pool_limit=args.pool_limit,
    )

    def env_factory(trace):
        return make_roboverify_env(trace.task, num_blocks=trace.num_blocks)

    rollout = partial(
        segment_rollout, env_factory=env_factory, reset_mode=args.reset_mode
    )

    def realize(node, demos, post, penalty=None):
        if isinstance(node.region, LoopRegion):
            physical = Program(
                len(lower_region(node.region, context, physical=True)),
                lower_region(node.region, context, physical=True),
            )
            return node.region, all(
                postcondition_reached(rollout(physical, s), s, post) for s in demos
            )
        initial = generate_random_program(
            args.slots, range(args.num_blocks), random.Random(args.seed)
        )
        if isinstance(node.region, BlockRegion) and node.region.physical:
            initial = Program(len(node.region.physical), list(node.region.physical))

        def propose(candidate, rng):
            with preserved_global_rng():
                set_np_seed(int(rng.integers(2**31)))
                return mutate_scoped(candidate, node.available_scope)

        initial = close_objects(
            initial,
            demos,
            node.available_scope,
            context,
            prefix="object_" + node.name.replace(".", "_"),
        )

        result = straight_line_synthesize(
            demos,
            post,
            initial,
            propose,
            rollout=rollout,
            budget=budget,
            logger=logger,
            block_id=node.name,
            penalty=penalty,
        )
        logger.log_event(
            "block_result",
            f"{node.name}: {result.status}",
            force=True,
            block_id=node.name,
            distance=result.distance,
            post_score=result.post_score,
            elapsed=result.elapsed,
            postscore_seconds=result.postscore_seconds,
        )
        exports = (
            require_closed(result.program.instructions, node.available_scope)
            - node.available_scope
        )
        # No unsupported relational summary is invented for arbitrary physical code.
        return (
            BlockRegion(
                node.region.symbolic if isinstance(node.region, BlockRegion) else None,
                tuple(result.program.instructions),
                exports,
            ),
            result.ok,
        )

    quotient_fn = None
    if args.quotient:
        from synthesis.cfg.invariants import infer_loop_invariant
        from synthesis.cfg.quotient import quotient

        infer_invariant = partial(
            infer_loop_invariant,
            context=context,
            learner=args.learner,
            relations=args.invariant_relations,
            variables=args.invariant_variables,
        )
        quotient_fn = partial(
            quotient,
            language=Language(timeout_seconds=args.predicate_seconds),
            infer_invariant=infer_invariant,
        )
    extra_paths = iter(args.additional_demos)

    def provide_demos(request):
        path = next(extra_paths, None)
        if path is None:
            return None
        extra = load_traces(path)
        if any(t.task != args.task for t in extra):
            raise ValueError("Additional demonstrations must use the same task")
        return [
            DemoSegment(
                i,
                0,
                len(t.states) - 1,
                t,
                {"b0": 0, **({"tbl": "tbl"} if context.use_tbl else {})},
            )
            for i, t in enumerate(extra, start=len(segments))
        ]

    def repair(node, demos, post, penalty):
        region, ok = realize(node, demos, post, penalty=penalty)
        return region if ok else None

    result = verified_synthesis(
        cfg,
        realize,
        partial(
            execute_cfg,
            context=context,
            env_factory=env_factory,
            reset_mode=args.reset_mode,
        ),
        context,
        quotient=quotient_fn,
        demo_provider=provide_demos,
        repair_motion=repair,
        symbolic_iterations=args.symbolic_iterations,
        motion_iterations=args.motion_iterations,
        timeout_ms=args.verification_timeout_ms,
        learner=args.learner,
        relations=args.invariant_relations,
        variables=args.invariant_variables,
        motion_options={
            "initial_arm": args.initial_arm,
            "noise": (
                NoiseSpec(*args.motion_noise) if args.motion_noise is not None else None
            ),
            "timeout_ms": args.motion_timeout_ms,
            "table_surface_height": args.table_surface_height,
        },
        max_refinements=args.refinements,
        language=Language(timeout_seconds=args.predicate_seconds),
        logger=logger,
    )
    import json

    logger.write_artifact(
        "cfg.json",
        json.dumps(
            {
                "order": cfg.order,
                "nodes": {name: str(node.region) for name, node in cfg.nodes.items()},
                "edges": [
                    {
                        "source": e.source,
                        "target": e.target,
                        "label": str(e.label),
                        "binds": sorted(e.binds),
                    }
                    for e in cfg.edges
                ],
                "segments": {
                    name: [
                        {
                            "demo": s.demo_idx,
                            "start": s.t_start,
                            "end": s.t_end,
                            "bindings": s.bindings,
                        }
                        for s in rows
                    ]
                    for name, rows in cfg.demos.segments.items()
                },
            },
            indent=2,
        ),
    )
    if result:
        program = lower(cfg, context, physical=True)
        logger.write_artifact("program.txt", str(program))
    logger.finish(
        result.status,
        stages=result.history,
        reason=result.reason,
        formal_verification=result.status,
        symbolic=str(result.symbolic),
        motion=str(result.motion),
        demonstration_checks=demo_checks,
        task=args.task,
    )
    return 0 if result else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("stack", "unstack"), default="unstack")
    parser.add_argument(
        "--demos",
        help="Lossless .npz recording; omit to collect the named historical oracle",
    )
    parser.add_argument("--run-root", default="runs")
    parser.add_argument("--slug", default="cfg")
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--slots", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--refinements", type=int, default=5)
    parser.add_argument("--cem-iterations", type=int, default=2)
    parser.add_argument("--cem-samples", type=int, default=8)
    parser.add_argument("--cem-elites", type=int, default=2)
    parser.add_argument("--inner-samples", type=int, default=128)
    parser.add_argument("--final-samples", type=int, default=2048)
    parser.add_argument("--pool-limit", type=int, default=10)
    parser.add_argument("--theta", type=float, default=1.0)
    parser.add_argument(
        "--distance", choices=("kl_kde", "kl_gmm", "mmd"), default="kl_kde"
    )
    parser.add_argument("--reset-mode", choices=("reset", "replay"), default="replay")
    parser.add_argument("--predicate-seconds", type=float, default=5.0)
    parser.add_argument(
        "--quotient", action="store_true", help="Enable conservative flat-loop folding"
    )
    parser.add_argument("--learner", choices=("legacy", "monotone"), default="legacy")
    parser.add_argument(
        "--invariant-relations",
        nargs="+",
        choices=("ON_star", "ON_star_zero", "Higher", "Scattered", "equality"),
    )
    parser.add_argument("--invariant-variables", type=int, default=2)
    add_motion_options(parser)
    parser.add_argument(
        "--table-surface-height",
        type=float,
        help="Physical table height required for table-placement contracts",
    )
    parser.add_argument(
        "--additional-demos",
        nargs="*",
        default=[],
        help="Recordings supplied for verification-requested resynthesis",
    )
    parser.add_argument("--symbolic-iterations", type=int, default=10)
    parser.add_argument("--motion-iterations", type=int, default=10)
    parser.add_argument("--verification-timeout-ms", type=int, default=5000)
    parser.add_argument(
        "--initial-arm",
        type=float,
        nargs=3,
        help="Explicit initial arm position for the geometric verification model",
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    motion_noise_from_args(parser, args)
    if args.smoke:
        args.iterations, args.cem_iterations, args.cem_samples, args.cem_elites = (
            2,
            1,
            4,
            2,
        )
        args.inner_samples, args.final_samples, args.refinements = 32, 128, 1
    if (
        args.num_blocks < 2
        or args.slots < 1
        or args.iterations < 0
        or args.refinements < 0
        or args.symbolic_iterations < 0
        or args.motion_iterations < 0
        or args.verification_timeout_ms <= 0
        or args.invariant_variables < 1
    ):
        parser.error("Invalid block, slot, or iteration budget")

    def deadline(signum, frame):
        raise TimeoutError("Unstack end-to-end 60-second budget exhausted")

    previous = signal.signal(signal.SIGALRM, deadline)
    with RunLogger(args.run_root, "cfg", vars(args), slug=args.slug) as logger:
        logger.progress_line(f"CFG run: {logger.run_dir}")
        if args.task == "unstack":
            signal.alarm(60)
        try:
            code = run(args, logger)
        except TimeoutError as exc:
            logger.finish("budget_exhausted", reason=str(exc))
            code = 2
        except Exception as exc:
            logger.log_exception(exc)
            logger.finish("failed", reason=str(exc))
            raise
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous)
    print(f"Read: uv run python -m synthesis.experiment.report --run {logger.run_dir}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
