"""Goal-task collection configuration and numeric validation, without proof VCs."""

import numpy as np
from synthesis.cfg.demo_validation import DemoValidation
from synthesis.environment.goal_layout import occupancy, positive_size, pyramid_layers
from synthesis.util.on import scattered_implementation

GOAL_TASKS = ("grid", "pyramid", "build3d")
PROVIDED_PROGRAMS = {
    "grid": "synthesis.examples.grid:build_program",
    "pyramid": "synthesis.examples.pyramid:build_program",
    "build3d": "synthesis.examples.build3d:build_program",
}


def resolve_goal_configuration(
    task,
    *,
    grid_rows=None,
    grid_cols=None,
    bottom_row_blocks=None,
    num_blocks=None,
    build_rows=None,
    build_cols=None,
    build_layers=None,
):
    if task != "build3d" and any(
        v is not None for v in (build_rows, build_cols, build_layers)
    ):
        raise ValueError(
            "--build-rows/--build-cols/--build-layers apply only to Build3D"
        )
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
    elif task == "build3d":
        if any(v is not None for v in (grid_rows, grid_cols, bottom_row_blocks)):
            raise ValueError(
                "Build3D dimensions use --build-rows/--build-cols/--build-layers"
            )
        config = {
            name: positive_size(2 if value is None else value, name)
            for name, value in (
                ("build_rows", build_rows),
                ("build_cols", build_cols),
                ("build_layers", build_layers),
            )
        }
        count = config["build_rows"] * config["build_cols"] * config["build_layers"]
    else:
        raise ValueError(f"Unsupported goal task: {task}")
    if num_blocks is not None and num_blocks != count:
        raise ValueError(
            f"Layout requires {count} blocks; configure its dimensions instead of a conflicting --num-blocks"
        )
    return count, config


def make_goal_env(task, config):
    from synthesis.mcmc.synthesis import (
        make_roboverify_build3d_env,
        make_roboverify_grid_env,
        make_roboverify_pyramid_env,
    )

    config = dict(config)
    relaxed = config.pop("relax_boundaries_for_generalization_test", False)
    _, config = resolve_goal_configuration(task, **config)
    factory = {
        "grid": make_roboverify_grid_env,
        "pyramid": make_roboverify_pyramid_env,
        "build3d": make_roboverify_build3d_env,
    }[task]
    if relaxed:
        if task == "build3d":
            raise ValueError("Build3D uses the nominal workspace bounds")
        config["relax_boundaries_for_generalization_test"] = True
    return factory(**config)


def environment_configuration(env):
    if env.case == "RoboVerifyGrid":
        return dict(grid_rows=env.grid_rows, grid_cols=env.grid_cols)
    if env.case == "RoboVerifyBuild3D":
        return dict(
            build_rows=env.build_rows,
            build_cols=env.build_cols,
            build_layers=env.build_layers,
        )
    return dict(bottom_row_blocks=env.bottom_row_blocks)


def goal_topology(task, config):
    """Return the expected rows and optional grouping into 3D layers."""
    if task == "build3d":
        return (
            (config["build_cols"],) * (config["build_rows"] * config["build_layers"]),
            (config["build_rows"],) * config["build_layers"],
        )
    if task == "grid":
        return (config["grid_cols"],) * config["grid_rows"], None
    return pyramid_layers(config["bottom_row_blocks"]), None


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
        rows, layers = goal_topology(trace.task, config)
        if (
            count != trace.num_blocks
            or rows != layout.row_lengths
            or layers != layout.layer_row_counts
        ):
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
