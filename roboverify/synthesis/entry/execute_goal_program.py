"""Execute the supplied Grid/Pyramid program; no synthesis or verification."""

import argparse
import json
import random

import numpy as np

from synthesis.api.control import get_move_action
from synthesis.cfg.collection import VideoRecorder, execution_deadline
from synthesis.environment.goal_layout import occupancy
from synthesis.examples.goal_layout import build_program
from synthesis.experiment.run_logger import RunLogger
from synthesis.mcmc.synthesis import (
    make_roboverify_grid_env,
    make_roboverify_pyramid_env,
)


def execute(
    task,
    *,
    bottom_row_blocks=3,
    grid_rows=3,
    grid_cols=2,
    seed=0,
    video_path=None,
    on_event=None,
    max_loop_iterations=100,
):
    """Run the ordinary DSL executor and validate the actual final occupancy."""
    if task not in ("grid", "pyramid"):
        raise ValueError(f"Unsupported goal task: {task}")
    random.seed(seed)
    np.random.seed(seed)
    env = (
        make_roboverify_grid_env(grid_rows=grid_rows, grid_cols=grid_cols)
        if task == "grid"
        else make_roboverify_pyramid_env(bottom_row_blocks=bottom_row_blocks)
    )
    video, result = None, None
    try:
        inner = env.env
        obs, _ = env.reset()
        # Match the existing collection convention: start from a settled scene.
        target = obs[:3].copy()
        for _ in range(50):
            env.step(get_move_action(obs, target))
            obs = inner.flatten_observation(inner._get_obs())
        count = inner.num_blocks
        initial = np.array([obs[10 + 12 * i : 13 + 12 * i] for i in range(count)])
        if occupancy(initial, inner.goal_layout.positions).any():
            raise ValueError("Initial blocks must be outside the goal regions")
        video = VideoRecorder(video_path) if video_path is not None else None
        controls, placements, loop_exits = [], [], []

        def state(observation):
            if video is not None:
                video.append(env.render())

        def event(row):
            if on_event is not None:
                on_event(row)
            if row.get("control"):
                controls.append(row["control"])
                if not row["control"]["converged"]:
                    raise RuntimeError(
                        f"Primitive {row['path']} failed to converge: {row['control']}"
                    )
            if row["kind"] == "instruction_end" and row["path"] == "1.1.5":
                block, goal = row["bindings"]["b_prime"], row["goal_bindings"]["r"]
                xyz = inner.sim.data.get_site_xpos(f"object{block}")
                if not occupancy([xyz], [inner.goal_layout.goals[goal].position])[0, 0]:
                    raise RuntimeError(
                        f"Block {block} did not reach goal {goal} after Release"
                    )
                current = [
                    inner.sim.data.get_site_xpos(f"object{i}") for i in range(count)
                ]
                filled = occupancy(current, inner.goal_layout.positions).any(axis=0)
                if any(not filled[p["goal"]] for p in placements):
                    raise RuntimeError(
                        "Placement displaced a block from an earlier goal"
                    )
                placements.append({"block": block, "goal": goal})
            if row["kind"] == "loop_exit":
                loop_exits.append({"path": row["path"], "iterations": row["iteration"]})

        trajectory = build_program(num_blocks=count).eval(
            env,
            initial_observation=obs,
            on_event=event,
            on_state=state,
            max_loop_iterations=max_loop_iterations,
        )
        final = np.array(
            [trajectory[-1][10 + 12 * i : 13 + 12 * i] for i in range(count)]
        )
        filled = occupancy(final, inner.goal_layout.positions).any(axis=0)
        success = bool(filled.all() and len(placements) == count)
        result = dict(
            task=task,
            seed=seed,
            num_blocks=count,
            row_lengths=list(inner.goal_layout.row_lengths),
            success=success,
            filled_goals=int(filled.sum()),
            steps=len(trajectory) - 1,
            primitives=len(controls),
            placements=placements,
            loop_exits=loop_exits,
            initial_positions=initial.tolist(),
            final_positions=final.tolist(),
            goal_positions=inner.goal_layout.positions.tolist(),
        )
        return result
    finally:
        try:
            if video is not None:
                saved = video.close()
                if result is not None:
                    result["video"] = saved
                    if saved["status"] != "saved":
                        raise RuntimeError(f"Video recording failed: {saved['reason']}")
        finally:
            env.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("grid", "pyramid"), required=True)
    parser.add_argument("--bottom-row-blocks", type=int, default=3)
    parser.add_argument("--grid-rows", type=int, default=3)
    parser.add_argument("--grid-cols", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-loop-iterations", type=int, default=100)
    parser.add_argument("--timeout-seconds", type=float, default=120)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--output-dir", default="runs")
    args = parser.parse_args(argv)
    if min(
        args.bottom_row_blocks, args.grid_rows, args.grid_cols, args.max_loop_iterations
    ) < 1 or not 0 < args.timeout_seconds < float("inf"):
        parser.error("Dimensions and execution budgets must be positive and finite")
    config = vars(args).copy()
    with RunLogger(
        args.output_dir,
        "goal-execution",
        config,
        slug=f"{args.task}-seed{args.seed}",
        capture_stdout=False,
    ) as logger:
        video = logger.artifact_dir() / "execution.mp4" if args.save_video else None
        try:
            with execution_deadline(args.timeout_seconds):
                result = execute(
                    args.task,
                    bottom_row_blocks=args.bottom_row_blocks,
                    grid_rows=args.grid_rows,
                    grid_cols=args.grid_cols,
                    seed=args.seed,
                    video_path=video,
                    max_loop_iterations=args.max_loop_iterations,
                )
        except Exception as exc:
            logger.log_exception(exc)
            logger.finish("failed", exit_reason=str(exc))
            print(
                json.dumps(
                    {
                        "success": False,
                        "error": str(exc),
                        "run_dir": str(logger.run_dir),
                    }
                )
            )
            return 1
        logger.write_artifact("execution.json", json.dumps(result, indent=2) + "\n")
        logger.finish(
            "completed" if result["success"] else "failed",
            success=result["success"],
            num_blocks=result["num_blocks"],
            filled_goals=result["filled_goals"],
            steps=result["steps"],
        )
        print(
            json.dumps(
                {
                    "success": result["success"],
                    "blocks": result["num_blocks"],
                    "run_dir": str(logger.run_dir),
                }
            )
        )
        return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
