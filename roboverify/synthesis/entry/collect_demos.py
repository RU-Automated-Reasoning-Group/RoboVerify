"""Collect tower, Grid or Pyramid DSL demonstrations, with optional videos.

Every task holds the gripper for 50 settling steps after a fresh reset before
recording. ``--post-settling-steps N`` (default 0) settles each completed
execution for N unrecorded steps and accepts it only if the settled
postcondition holds (``task-spec-v2``); 0 checks the last recorded state.
"""

import argparse
import json
from pathlib import Path

from synthesis.cfg.collection import VideoRecorder, record_execution, validate_trace
from synthesis.cfg.goal_collection import (
    GOAL_TASKS,
    PROVIDED_PROGRAMS,
    resolve_goal_configuration,
)
from synthesis.cfg.program_source import load_program
from synthesis.cfg.recordings import save_traces
from synthesis.cfg.task_success import criterion
from synthesis.cfg.tasks import task_identity
from synthesis.entry.predicate_options import add_predicate_options
from synthesis.util.on import using_higher_tolerance
from synthesis.verification_lib.highlevel_verification_lib import HighLevelContext


def resolve_seeds(args):
    if args.seeds is not None:
        if args.seed_start is not None:
            raise ValueError("Use --seeds or --seed-start, not both")
        if (
            args.num_trajectories is not None
            and len(args.seeds) != args.num_trajectories
        ):
            raise ValueError(
                "--num-trajectories must match the number of explicit seeds"
            )
        seeds = list(args.seeds)
    else:
        count = 5 if args.num_trajectories is None else args.num_trajectories
        first = 0 if args.seed_start is None else args.seed_start
        seeds = list(range(first, first + count))
    if (
        not seeds
        or len(set(seeds)) != len(seeds)
        or any(s < 0 or s >= 2**32 for s in seeds)
    ):
        raise ValueError(
            "Supply a positive trajectory count and distinct seeds in [0, 2**32)"
        )
    return seeds


def output_directory(args, count):
    if args.output_dir:
        path = Path(args.output_dir)
        path.mkdir(parents=True, exist_ok=False)
        return path
    layout = ""
    if args.task == "grid":
        layout = f"{args.grid_rows}x{args.grid_cols}-"
    elif args.task == "pyramid":
        layout = f"{args.bottom_row_blocks}-bottom-row-"
    base = (
        Path("demos")
        / args.task
        / f"{layout}{args.num_blocks}-blocks-{count}-trajectories"
    )
    index = 1
    while True:
        path = base if index == 1 else base.with_name(base.name + f"-{index}")
        try:
            path.mkdir(parents=True, exist_ok=False)
            return path
        except FileExistsError:
            index += 1


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--program",
        help="module:factory or path/to/program.py:factory; defaults to the task's provided groundtruth",
    )
    parser.add_argument(
        "--task",
        choices=["stack", "unstack", "reverse", "restack", *GOAL_TASKS],
        default="stack",
    )
    parser.add_argument(
        "--num-blocks",
        type=int,
        help="Defaults to 4 for Restack, 3 for other tower tasks; derived from goal layouts",
    )
    parser.add_argument(
        "--initial-tower-blocks",
        type=int,
        help="Restack source tower height; remaining blocks start loose (default max(2, N-1))",
    )
    parser.add_argument("--grid-rows", type=int, help="Grid rows (default 3)")
    parser.add_argument("--grid-cols", type=int, help="Grid columns (default 2)")
    parser.add_argument(
        "--bottom-row-blocks",
        type=int,
        help="Pyramid bottom-row width (default 3); creates k+(k-1)+...+1 blocks",
    )
    parser.add_argument(
        "--num-trajectories",
        type=int,
        help="Number of distinct seed runs (5 when omitted; explicit seeds set the count)",
    )
    parser.add_argument(
        "--seed-start", type=int, help="First consecutive seed (0 when omitted)"
    )
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument(
        "--output-dir",
        help="New collection directory; otherwise demos/<task>/<blocks>-blocks-<count>-trajectories",
    )
    parser.add_argument(
        "--save-video", action="store_true", help="Save one MP4 per seed at 20 FPS"
    )
    parser.add_argument(
        "--render", action="store_true", help="Show a live simulator window"
    )
    parser.add_argument("--max-loop-iterations", type=int, default=100)
    parser.add_argument("--trajectory-timeout-seconds", type=float, default=60)
    parser.add_argument(
        "--post-settling-steps",
        type=int,
        default=0,
        help="Unrecorded final settling steps before the postcondition check (0: last recorded state)",
    )
    add_predicate_options(parser)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    with using_higher_tolerance(args.higher_tolerance):
        return collect(args, parser)


def collect(args, parser):
    try:
        seeds = resolve_seeds(args)
        goal_task = args.task in GOAL_TASKS
        environment_config = None
        if args.initial_tower_blocks is not None and args.task != "restack":
            raise ValueError("--initial-tower-blocks requires --task restack")
        if goal_task:
            args.num_blocks, environment_config = resolve_goal_configuration(
                args.task,
                grid_rows=args.grid_rows,
                grid_cols=args.grid_cols,
                bottom_row_blocks=args.bottom_row_blocks,
                num_blocks=args.num_blocks,
            )
            for name, value in environment_config.items():
                setattr(args, name, value)
            args.program = args.program or PROVIDED_PROGRAMS[args.task]
        else:
            if any(
                value is not None
                for value in (args.grid_rows, args.grid_cols, args.bottom_row_blocks)
            ):
                raise ValueError("Goal layout options require --task grid or pyramid")
            if args.num_blocks is None:
                args.num_blocks = 4 if args.task == "restack" else 3
            if args.task == "restack":
                from synthesis.environment.restack_reset import resolve_tower_blocks

                args.initial_tower_blocks = resolve_tower_blocks(
                    args.num_blocks, args.initial_tower_blocks
                )
                environment_config = dict(
                    initial_tower_blocks=args.initial_tower_blocks
                )
                args.program = (
                    args.program or "synthesis.examples.restack:build_program"
                )
            args.program = (
                args.program or f"synthesis.examples.{args.task}:build_program"
            )
        if (
            args.num_blocks < (1 if goal_task else 2)
            or args.max_loop_iterations < 1
            or not 0 < args.trajectory_timeout_seconds < float("inf")
            or args.post_settling_steps < 0
        ):
            raise ValueError(
                "Invalid block count; execution budgets must be positive and finite"
            )
        if args.save_video:
            VideoRecorder.check_available()
        context = HighLevelContext(
            mode="declare", use_tbl=args.task in ("unstack", "reverse", "restack")
        )
        reference = load_program(
            args.program,
            context,
            args.num_blocks,
            allow_goal_program=goal_task,
            allow_observation_references=True,
        )
        output = output_directory(args, len(seeds))
    except (ValueError, FileExistsError, ImportError, AttributeError, TypeError) as exc:
        parser.error(str(exc))
    report = dict(
        task=task_identity(args.task),
        program=reference.metadata,
        seeds=seeds,
        num_blocks=args.num_blocks,
        higher_tolerance=args.higher_tolerance,
        **(
            dict(
                post_settling_steps=args.post_settling_steps,
                success_criterion=criterion(args.post_settling_steps),
            )
            if args.post_settling_steps
            else {}
        ),
        trajectories=[],
        status="collecting",
    )
    if environment_config is not None:
        report["environment"] = environment_config
    accepted = []
    for seed in seeds:
        # Rebuild for every seed; reject factories that change the executable.
        try:
            definition = load_program(
                args.program,
                context,
                args.num_blocks,
                allow_goal_program=goal_task,
                allow_observation_references=True,
            )
            if definition.metadata["fingerprint"] != reference.metadata["fingerprint"]:
                raise ValueError("Program factory changed between seeds")
            trace = record_execution(
                definition,
                seed=seed,
                task=args.task,
                num_blocks=args.num_blocks,
                max_loop_iterations=args.max_loop_iterations,
                timeout_seconds=args.trajectory_timeout_seconds,
                video_path=(
                    output / "videos" / f"seed_{seed:04d}.mp4"
                    if args.save_video
                    else None
                ),
                render=args.render,
                environment_config=environment_config,
                post_settling_steps=args.post_settling_steps,
            )
        except Exception as exc:
            from synthesis.cfg.demos import DemoTrace

            trace = DemoTrace(
                (),
                seed=seed,
                task=args.task,
                num_blocks=args.num_blocks,
                metadata=dict(
                    reference.metadata,
                    status="failed",
                    reason=f"{type(exc).__name__}: {exc}",
                ),
            )
        trace.metadata["task_spec"] = task_identity(args.task)
        valid = validate_trace(trace)
        if valid:
            accepted.append(trace)
        elif trace.states:
            diagnostics = output / "diagnostics"
            diagnostics.mkdir(exist_ok=True)
            save_traces(diagnostics / f"seed_{seed:04d}.npz", [trace])
        row = dict(seed=seed, observations=len(trace.states), **trace.metadata)
        if "video" in row:
            row["video"] = dict(
                row["video"], path=str(Path(row["video"]["path"]).relative_to(output))
            )
        report["trajectories"].append(row)
        (output / "collection.json").write_text(json.dumps(report, indent=2) + "\n")
        print(
            f"seed {seed}: {trace.metadata['status']}; {trace.metadata['reason']}",
            flush=True,
        )
    success = len(accepted) == len(seeds)
    if success:
        save_traces(output / "demonstrations.npz", accepted)
    else:
        # Preserve successful trajectories too when the requested batch is rejected.
        diagnostics = output / "diagnostics"
        diagnostics.mkdir(exist_ok=True)
        for trace in accepted:
            save_traces(diagnostics / f"seed_{trace.seed:04d}.npz", [trace])
    video_failed = any(
        r.get("video", {}).get("status") == "failed" or r.get("render_error")
        for r in report["trajectories"]
    )
    report["status"] = "valid" if success else "invalid_demonstrations"
    report["media_status"] = "failed" if video_failed else "complete"
    (output / "collection.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Collection: {output}")
    if success and args.task == "stack":
        import shlex

        archive = shlex.quote(str(output / "demonstrations.npz"))
        common = f"uv run python -m synthesis.entry.synthesize_cfg --task stack --num-blocks {args.num_blocks} --demos {archive} --higher-tolerance {args.higher_tolerance}"
        print(f"Full pipeline: {common} --mode full --quotient")
        print(
            f"Verify program: {common} --mode verify --program {shlex.quote(args.program)}"
        )
    return 0 if success and not video_failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
