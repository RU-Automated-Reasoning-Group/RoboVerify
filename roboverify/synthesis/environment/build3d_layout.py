"""Shared rectangular Build3D goal geometry for simulator construction and reset."""

import numpy as np
from synthesis.environment.goal_layout import positive_size
from synthesis.util.on import BLOCK_LENGTH


def goal_positions(rows, cols, layers, *, gripper_position, robot_base_xy, height):
    """Return bottom-to-top, row-major goal centers with vertical supports.

    Ten-centimeter XY spacing matches Grid's finger clearance. Each upper
    goal sits one block height directly above its support in the previous layer.
    """
    rows = positive_size(rows, "build_rows")
    cols = positive_size(cols, "build_cols")
    layers = positive_size(layers, "build_layers")
    spacing = 2 * BLOCK_LENGTH
    origin_x = gripper_position[0] - 0.04 - 0.5 * (cols - 1) * spacing
    origin_y = robot_base_xy[1] + 0.10
    return np.asarray(
        [
            [
                origin_x + col * spacing,
                origin_y + row * spacing,
                height + layer * BLOCK_LENGTH,
            ]
            for layer in range(layers)
            for row in range(rows)
            for col in range(cols)
        ],
        dtype=np.float32,
    )
