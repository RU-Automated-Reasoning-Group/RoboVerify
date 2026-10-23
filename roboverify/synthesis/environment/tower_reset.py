"""Lateral starting workspaces for the Unstack and Reverse tower tasks."""

import numpy as np

TOWER_X_RANGE = (0.54, 0.70)
TOWER_Y_RANGE = (0.15, 0.25)


def sample_tower_xy(num_blocks, robot_base_xy, *, task):
    """Sample the same side workspace for both tasks and every tower height."""
    if task not in ("unstack", "reverse"):
        raise ValueError(f"Unsupported tower reset: {task}")
    # Keep the source away from the initial arm while allowing farther starts.
    return np.asarray(robot_base_xy) + np.array(
        [np.random.uniform(*TOWER_X_RANGE), np.random.uniform(*TOWER_Y_RANGE)]
    )
