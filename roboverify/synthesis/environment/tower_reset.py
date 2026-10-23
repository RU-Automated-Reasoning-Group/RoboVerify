"""Lateral starting workspaces for the Unstack and Reverse tower tasks."""

import numpy as np

REVERSE_TOWER_Y_OFFSET = 0.15


def sample_tower_xy(num_blocks, robot_base_xy, *, task):
    if task == "unstack":
        # Center the eventual 15 cm-spaced row on the robot base.
        y_offset = 0.075 * (num_blocks - 1)
    elif task == "reverse":
        # Two towers, one on each side, irrespective of the block count.
        y_offset = REVERSE_TOWER_Y_OFFSET
    else:
        raise ValueError(f"Unsupported tower reset: {task}")
    return np.asarray(robot_base_xy) + np.array(
        [np.random.uniform(0.54, 0.58), y_offset + np.random.uniform(-0.01, 0.01)]
    )
