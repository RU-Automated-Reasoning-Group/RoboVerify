"""CLI entry point for MCMC or enumerative search with BMC and CEM.

Every argument, including the structural search strategy, is recorded in
``config.json``. Enumeration requires a BMC goal and screens each candidate
before invoking the same CEM optimizer used by MCMC.

Examples::

    # Fast pathology check -- 20 iterations, no videos, finishes in minutes.
    uv run python -m synthesis.experiment.mcmc.run --smoke --demos path/to/demonstrations.npz

    # A real search against a goal feature, with BMC pruning.
    uv run python -m synthesis.experiment.mcmc.run \
        --task stack --num-blocks 4 --iters 2000 \
        --demos path/to/demonstrations.npz --goal-feature 'ON(1,0)' --slug stack-nb4

Then read it with::

    uv run python -m synthesis.experiment.report --run runs/mcmc/latest

This runs one straight-line search against a feature you name. Structured search,
predicate discovery, refinement and quotienting use ``synthesis.entry.synthesize_cfg``.
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
        description="Run MCMC or enumerative program search into a run directory.",
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
    parser = build_parser()
    args = parser.parse_args(argv)
    config = config_from_args(args)

    if config.search_strategy == "enumeration" and (
        args.no_bmc or not args.goal_feature
    ):
        parser.error(
            "--search-strategy enumeration requires --goal-feature with BMC enabled"
        )

    goal_feature = parse_goal_feature(args.goal_feature) if args.goal_feature else None
    bmc_goal = None
    bmc_initial_constraints = None
    if goal_feature is not None and not args.no_bmc:
        bmc_goal = synthesis.bmc_goal_from_on_feature(goal_feature)
        bmc_initial_constraints = synthesis.roboverify_bmc_initial_constraints()

    from synthesis.cfg.recordings import load_traces

    if not config.demos:
        raise ValueError("--demos is required; first run synthesis.entry.collect_demos")
    traces = load_traces(config.demos, require_valid=True)
    if any(t.task != config.task or t.num_blocks != config.num_blocks for t in traces):
        raise ValueError("Demonstration task/block count does not match this search")
    trajectories = [list(t.states) for t in traces]
    seeds = [t.seed for t in traces]
    demo_num_blocks = traces[0].num_blocks
    if not trajectories:
        print(f"no demo trajectories found in {config.demos!r}", file=sys.stderr)
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
    if len(set(active_seeds)) != len(active_seeds):
        raise ValueError("MCMC demonstration seeds must be distinct")
    initial_snapshots = {t.seed: t.snapshots[0] for t in traces[: config.num_seeds]}
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

    config.runtime["initial_states"] = dict(
        source="archive_snapshots", state_index=0, seeds=active_seeds
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
            initial_snapshots=initial_snapshots,
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
        "no_feasible_candidate" if result.best_program is None else "completed",
        search_strategy=config.search_strategy,
        termination_reason=result.termination_reason,
        best_checks_passed=result.best_checks_passed,
        best_cost=float(result.best_cost) if result.best_program is not None else None,
        best_iter=result.best_iter,
        iters_completed=result.iters_completed,
        best_program=str(result.best_program),
    )
    if result.best_program is None:
        logger.progress_line(
            f"[run] done: no BMC-feasible candidate after {result.iters_completed} iterations"
        )
    else:
        logger.progress_line(
            f"[run] done: best cost {result.best_cost:.6g} at iteration "
            f"{result.best_iter} over {result.iters_completed} iterations"
        )
    return 0 if result.best_program is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
