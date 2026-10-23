"""Bounded lateral starts for Restack: one tower plus loose table blocks."""

import numpy as np

from synthesis.environment.tower_reset import TOWER_X_RANGE, TOWER_Y_RANGE
from synthesis.util.on import BLOCK_LENGTH

RESTACK_X_RANGE = (0.40, TOWER_X_RANGE[1])
RESTACK_SEPARATION = 2 * BLOCK_LENGTH + 0.005


def resolve_tower_blocks(num_blocks, initial_tower_blocks=None):
    if type(num_blocks) is not int or num_blocks < 2:
        raise ValueError("Restack requires at least two blocks")
    count = (
        max(2, num_blocks - 1) if initial_tower_blocks is None else initial_tower_blocks
    )
    if type(count) is not int or not 2 <= count <= num_blocks:
        raise ValueError("initial_tower_blocks must be between 2 and num_blocks")
    return count


def sample_restack_layout(num_blocks, tower_blocks, base_id, base_xy, gripper_xy):
    """Return block XY positions and integer heights, with b0 inside the tower.

    The tower shares Unstack/Reverse's workspace. Loose blocks may extend closer
    to the robot in X, but all source blocks remain on its higher-Y side.
    Bounded retries refuse configurations that do not fit; bounds never expand.
    """
    tower_blocks = resolve_tower_blocks(num_blocks, tower_blocks)
    base_xy, gripper_xy = np.asarray(base_xy), np.asarray(gripper_xy)
    columns = num_blocks - tower_blocks + 1
    # The Y interval is narrower than Scattered separation, so columns must fit in X.
    capacity = 1 + int((RESTACK_X_RANGE[1] - RESTACK_X_RANGE[0]) / RESTACK_SEPARATION)
    if columns > capacity:
        raise ValueError("Restack source workspace cannot fit this many loose blocks")
    order = np.random.permutation(
        [i for i in range(num_blocks) if i != base_id]
    ).tolist()
    level = int(np.random.randint(1, tower_blocks - 1)) if tower_blocks > 2 else 1
    tower = order[: tower_blocks - 1]
    tower.insert(level, base_id)
    loose = order[tower_blocks - 1 :]
    for _ in range(100):
        centers = []
        for column in range(columns):
            x_range = TOWER_X_RANGE if column == 0 else RESTACK_X_RANGE
            for _ in range(200):
                candidate = base_xy + np.array(
                    [np.random.uniform(*x_range), np.random.uniform(*TOWER_Y_RANGE)]
                )
                if np.linalg.norm(candidate - gripper_xy) < 0.10:
                    continue
                if all(
                    np.any(np.abs(candidate - p) >= RESTACK_SEPARATION) for p in centers
                ):
                    centers.append(candidate)
                    break
            else:
                break
        else:
            xy, heights = np.empty((num_blocks, 2)), np.zeros(num_blocks, dtype=int)
            for height, block in enumerate(tower):
                xy[block], heights[block] = centers[0], height
            for block, center in zip(loose, centers[1:]):
                xy[block] = center
            return xy, heights
    raise ValueError("Restack lateral reset exhausted its bounded layout retries")
