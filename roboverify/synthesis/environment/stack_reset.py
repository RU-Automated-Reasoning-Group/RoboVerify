"""Sample scattered Stack blocks inside a bounded region near the robot base."""

import numpy as np
from synthesis.util.on import BLOCK_LENGTH

# Offsets from the robot base, in metres. The forward lower bound avoids placing
# blocks underneath the robot; the radius excludes the former far table edge.
STACK_X_RANGE = (0.54, 0.70)
STACK_Y_RANGE = (-0.20, 0.20)
STACK_MAX_BASE_DISTANCE = 0.70
STACK_GRIPPER_CLEARANCE = 0.10


def sample_stack_xy(
    num_blocks,
    base_xy,
    gripper_xy,
    *,
    relax_boundaries_for_generalization_test=False,
    base_index=0,
):
    """Return a complete layout, or fail without relaxing the workspace bounds.

    Uses the environment's existing NumPy RNG so collection seeds remain
    reproducible. Restart the whole layout if earlier placements leave no room
    for a later block. The XY bound is not a robot reachability proof for arbitrary
    heights, orientations, or tower sizes.

    The opt-in relaxed mode (`relaxed_workspace`) instead places `b0`
    (`base_index`) at a high-reach tower site and the other blocks anywhere in
    the measured reachable workspace; the default draws are unchanged.
    """
    if relax_boundaries_for_generalization_test:
        from synthesis.environment.relaxed_workspace import sample_relaxed_stack_xy

        return sample_relaxed_stack_xy(num_blocks, base_xy, gripper_xy, base_index)
    base_xy = np.asarray(base_xy, dtype=float)
    gripper_xy = np.asarray(gripper_xy, dtype=float)
    separation = 2 * BLOCK_LENGTH
    low = [STACK_X_RANGE[0], STACK_Y_RANGE[0]]
    high = [STACK_X_RANGE[1], STACK_Y_RANGE[1]]
    for _ in range(100):
        positions = []
        for _ in range(num_blocks):
            for _ in range(200):
                candidate = base_xy + np.random.uniform(low, high)
                if np.linalg.norm(candidate - base_xy) > STACK_MAX_BASE_DISTANCE:
                    continue
                if np.linalg.norm(candidate - gripper_xy) < STACK_GRIPPER_CLEARANCE:
                    continue
                if any(
                    np.all(np.abs(candidate - other) < separation)
                    for other in positions
                ):
                    continue
                positions.append(candidate)
                break
            else:
                break
        else:
            return positions
    raise ValueError(
        f"Could not sample {num_blocks} scattered Stack blocks within "
        f"{STACK_MAX_BASE_DISTANCE:.2f} m of the robot base; "
        "bounded retries exhausted; reduce the block count."
    )
