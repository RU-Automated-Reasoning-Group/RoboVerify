"""Goal-task collection configuration and numeric validation, without proof VCs."""

import numpy as np

from synthesis.cfg.demo_validation import DemoValidation
from synthesis.environment.goal_layout import occupancy, positive_size, pyramid_layers
from synthesis.util.on import scattered_implementation

GOAL_TASKS = ("grid", "pyramid")
PROVIDED_PROGRAMS = {
    "grid": "synthesis.examples.grid:build_program",
    "pyramid": "synthesis.examples.pyramid:build_program",
}


def resolve_goal_configuration(
    task, *, grid_rows=None, grid_cols=None, bottom_row_blocks=None, num_blocks=None
):
    if task == "grid":
        if bottom_row_blocks is not None:
            raise ValueError("--bottom-row-blocks applies only to Pyramid")
        rows = positive_size(3 if grid_rows is None else grid_rows, "grid_rows")
        cols = positive_size(2 if grid_cols is None else grid_cols, "grid_cols")
        count, config = rows * cols, dict(grid_rows=rows, grid_cols=cols)
    elif task == "pyramid":
        if grid_rows is not None or grid_cols is not None:
            raise ValueError("--grid-rows/--grid-cols apply only to Grid")
        bottom = 3 if bottom_row_blocks is None else bottom_row_blocks
        layers = pyramid_layers(bottom)
        count, config = sum(layers), dict(bottom_row_blocks=layers[0])
    else:
        raise ValueError(f"Unsupported goal task: {task}")
    if num_blocks is not None and num_blocks != count:
        raise ValueError(
            f"Layout requires {count} blocks; configure its dimensions instead of a conflicting --num-blocks"
        )
    return count, config


def make_goal_env(task, config):
    from synthesis.mcmc.synthesis import (
        make_roboverify_grid_env,
        make_roboverify_pyramid_env,
    )

    _, config = resolve_goal_configuration(task, **config)
    factory = (
        make_roboverify_grid_env if task == "grid" else make_roboverify_pyramid_env
    )
    return factory(**config)


def environment_configuration(env):
    if env.case == "RoboVerifyGrid":
        return dict(grid_rows=env.grid_rows, grid_cols=env.grid_cols)
    return dict(bottom_row_blocks=env.bottom_row_blocks)


def validate_goal_trace(trace, *, require_final_post=True):
    """Check real geometry, release effects and final coverage of fixed goals.

    ``require_final_post=False`` keeps final coverage as a diagnostic only, for
    the settled criterion; the Release-event checks always apply.
    """
    from synthesis.cfg.recordings import check_trace

    result = DemoValidation(1)

    def issue(reason):
        result.issues.append(dict(demo=0, reason=reason))

    try:
        check_trace(trace)
        layout = trace.snapshots[0].goal_layout
        count, config = resolve_goal_configuration(
            trace.task, **trace.metadata["environment"]
        )
        rows = (
            (config["grid_cols"],) * config["grid_rows"]
            if trace.task == "grid"
            else pyramid_layers(config["bottom_row_blocks"])
        )
        if count != trace.num_blocks or rows != layout.row_lengths:
            raise ValueError(
                "Environment dimensions disagree with the recorded goal layout"
            )
        states = np.asarray(trace.states)
        recorded_goals = states[:, -(3 * count + 3) : -3].reshape(-1, count, 3)
        if not np.all(recorded_goals == layout.positions):
            raise ValueError("Goal coordinates changed or disagree with the recording")
        positions = np.stack(
            [states[:, 10 + 12 * i : 13 + 12 * i] for i in range(count)], axis=1
        )
        occupied = occupancy(positions, layout.positions)
        filled = occupied.any(axis=1)
        initial = not filled[0].any() and all(
            scattered_implementation(positions[0, i], positions[0, j])
            for i in range(count)
            for j in range(i)
        )
        result.pre_holds = int(initial)
        result.post_at_end = int(filled[-1].all())
        result.post_reached = int(filled.all(axis=1).any())
        if not initial:
            issue("Initial blocks must be scattered and all goals empty")
        if not result.post_at_end and require_final_post:
            issue("Final state does not cover every goal")

        previous = np.zeros(count, dtype=bool)
        held = None
        for event in trace.events:
            if event["kind"] != "instruction_end":
                continue
            program = trace.metadata["program"]
            parts = list(map(int, event["path"].split(".")))
            for depth, index in enumerate(parts):
                instruction = program[index]
                if depth + 1 < len(parts):
                    program = instruction["body"]
            if instruction["instruction"] == "PickByName":
                held = event["bindings"][instruction["operands"][0]["val"]]
            elif instruction["instruction"] == "ReleaseByName":
                t = event["index"]
                if np.any(previous & ~filled[t]):
                    issue(
                        f"Placement at {event['path']} displaced an earlier occupied goal"
                    )
                # Release's operand supplies retreat height, not placement XY.
                # Validate the payload's actual goal occupancy independently.
                if held is None or not occupied[t, held].any():
                    issue("Released block did not reach a goal")
                previous |= filled[t]
                held = None
    except (ValueError, KeyError, TypeError, IndexError) as exc:
        issue(str(exc))
    return result
