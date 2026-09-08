"""CLI entry point for an instrumented MCMC search.

The MCMC path previously had no entry point: its only driver is
``synthesis/entry/main.py``, whose body is a single ``if __name__`` block mixing
commented-out experiments with an ``exit()`` partway through. This module gives the
search a real, reproducible invocation whose every argument is recorded in
``config.json``.

Examples::

    # Fast pathology check -- 20 iterations, no videos, finishes in minutes.
    uv run python -m synthesis.experiment.mcmc.run --smoke --demo-dir demos

    # A real search against a goal feature, with BMC pruning.
    uv run python -m synthesis.experiment.mcmc.run \
        --task stack --num-blocks 4 --iters 2000 \
        --demo-dir demos --goal-feature 'ON(1,0)' --slug stack-nb4

Then read it with::

    uv run python -m synthesis.experiment.report --run runs/mcmc/latest

Learning the goal feature is deliberately out of scope: this runs *one* search
against a feature you name. The decision-tree staging that discovers a feature and
splits demos around it lives in ``synthesis/mcmc/decision_tree.py`` and is driven by
``synthesis/entry/main.py``.
"""

import argparse
import re
import sys

from synthesis.api import program
from synthesis.experiment.config import add_cli_arguments, config_from_args
from synthesis.experiment.mcmc.search import MCMC
from synthesis.experiment.run_logger import RunLogger
from synthesis.mcmc import decision_tree, synthesis

GOAL_FEATURE_PATTERN = re.compile(r"^\s*ON\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*$", re.I)

AVAILABLE_INSTRUCTIONS = [program.Pick, program.Move, program.Release]


def parse_goal_feature(text: str):
    """Parse ``ON(b1, b2)`` into a :class:`decision_tree.ON_feature`."""
    match = GOAL_FEATURE_PATTERN.match(text)
    if not match:
        raise argparse.ArgumentTypeError(
            f"expected a goal feature of the form 'ON(b1, b2)', got {text!r}"
        )
    return decision_tree.ON_feature(int(match.group(1)), int(match.group(2)))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run an instrumented MCMC program search into a run directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_cli_arguments(parser)
    parser.add_argument(
        "--goal-feature",
        default=None,
        help=(
            "Dense goal reward feature, e.g. 'ON(1,0)'. Also supplies the BMC goal "
            "unless --no-bmc is given."
        ),
    )
    parser.add_argument(
        "--no-bmc",
        action="store_true",
        help="Skip BMC feasibility pruning even when a goal feature is given.",
    )
    parser.add_argument(
        "--no-refresh-best-metrics",
        action="store_true",
        help=(
            "Skip re-evaluating the objective at the winning CEM parameters. Saves "
            "one rollout batch per candidate but leaves mmd/success_rate unrecorded "
            "when no goal feature is configured."
        ),
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    config = config_from_args(args)

    goal_feature = parse_goal_feature(args.goal_feature) if args.goal_feature else None
    bmc_goal = None
    bmc_initial_constraints = None
    if goal_feature is not None and not args.no_bmc:
        bmc_goal = synthesis.bmc_goal_from_on_feature(goal_feature)
        bmc_initial_constraints = synthesis.roboverify_bmc_initial_constraints()

    trajectories, seeds, demo_num_blocks = synthesis.load_demo_trajectories(
        config.demo_dir
    )
    if not trajectories:
        print(f"no demo trajectories found in {config.demo_dir!r}", file=sys.stderr)
        return 1
    if demo_num_blocks is not None and demo_num_blocks != config.num_blocks:
        # Silently evaluating against demos from a different block count would
        # produce a meaningless MMD, so refuse rather than guess.
        print(
            f"demo block count ({demo_num_blocks}) does not match --num-blocks "
            f"({config.num_blocks})",
            file=sys.stderr,
        )
        return 1

    if config.num_seeds is None:
        config.num_seeds = len(trajectories)
    config.num_seeds = min(config.num_seeds, len(trajectories))
    active_seeds = list(seeds[: config.num_seeds])
    trajectories = trajectories[: config.num_seeds]
    expert_states = [state for traj in trajectories for state in traj]
    config.seeds = active_seeds

    available_operands = {"Box": list(range(config.num_blocks))}
    initial_program = program.Program(config.program_slots)

    synthesis.set_np_seed(config.rng_seed)

    config.describe_runtime(
        bmc_goal=bmc_goal,
        goal_feature=goal_feature,
        expert_states=expert_states,
        initial_program=initial_program,
        available_instructions=AVAILABLE_INSTRUCTIONS,
        available_operands=available_operands,
    )

    logger = RunLogger(
        config.run_root,
        config.run_name,
        config.to_json_dict(),
        slug=config.resolved_slug,
        capture_stdout=config.capture_stdout,
        event_min_step_gap=config.event_min_step_gap,
    )
    logger.progress_line(f"[run] {logger.run_dir}")
    logger.progress_line(
        f"[run] read it with: python -m synthesis.experiment.report "
        f"--run {config.run_root}/{config.run_name}/latest"
    )

    try:
        result = MCMC(
            initial_program,
            available_operands,
            AVAILABLE_INSTRUCTIONS,
            config,
            expert_states,
            logger=logger,
            seeds=active_seeds,
            bmc_goal=bmc_goal,
            bmc_initial_constraints=bmc_initial_constraints,
            goal_feature=goal_feature,
            refresh_best_metrics=not args.no_refresh_best_metrics,
        )
    except BaseException as exc:  # noqa: BLE001 - recorded, then re-raised
        logger.log_exception(exc)
        logger.finish("failed", exit_reason=f"{type(exc).__name__}: {exc}")
        raise

    logger.finish(
        "completed",
        best_cost=float(result.best_cost),
        best_iter=result.best_iter,
        iters_completed=result.iters_completed,
        best_program=str(result.best_program),
    )
    logger.progress_line(
        f"[run] done: best cost {result.best_cost:.6g} at iteration "
        f"{result.best_iter} over {result.iters_completed} iterations"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
