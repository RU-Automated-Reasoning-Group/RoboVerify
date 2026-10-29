"""Opt-in relaxed reset and goal bounds for block-count generalization tests.

`relax_boundaries_for_generalization_test` raises MuJoCo's contact buffers
(RELAXED_CONTACT_BUFFER) and replaces each task's reset and goal
workspace with one shared region, measured as reachable by the arm:
base-relative X >= 0.35 m and XY radius <= 0.80 m. Every 5 cm cell of that
region was reached by the empty gripper at Z >= 0.64 m on a straight or
rise-traverse-descend path (path-aware reach maps of the 2026-10-28 max-block
probe and its relaxed run). Closer to the base, X <= 0.30 m near Y = 0 lies
under the robot.

Tower sites (Stack's b0, the Unstack/Reverse/Restack source tower) are sampled
in X [0.40, 0.60] m. Every tower top of the relaxed run, up to 19 blocks
(1.32 m), was at least 0.14 m below the reach measured above its column. Separation rules, gripper clearance and
the side on which each task keeps its sources and goals are unchanged, and
exhausted retries still fail explicitly.

The default bounds of each task are untouched; nothing here runs unless the
option is enabled. Relaxed scenes differ from the trained layout distribution
even at the trained block count.
"""

import numpy as np

from synthesis.util.on import BLOCK_LENGTH

RELAXED_MIN_X = 0.35
RELAXED_MAX_RADIUS = 0.80
RELAXED_TOWER_X_RANGE = (0.40, 0.60)
RELAXED_STACK_BASE_Y_RANGE = (-0.20, 0.20)
# Unstack/Reverse/Restack keep their source tower on the higher-Y side.
RELAXED_LATERAL_TOWER_Y_RANGE = (0.15, 0.25)
RELAXED_LATERAL_MIN_Y = 0.15
# Goal layouts are centred here in X, then shifted forward to RELAXED_MIN_X.
RELAXED_GOAL_CENTER_X = 0.50
# Grid/Pyramid sources stay on the lower-Y side, as in the default reset.
RELAXED_SOURCE_MAX_Y = -0.08
GRIPPER_CLEARANCE = 0.10
# MuJoCo contact and constraint-row capacity (nconmax, njmax). The defaults
# (100, 500) overflow at about 20 blocks. A toppling 40-block tower peaked at
# 306 contacts and 1,230 rows (about 8 and 31 per block), so this leaves a
# 3x margin. MuJoCo sizes its stack roughly with njmax squared (about 640 MB
# here, against 2.5 GB at njmax 8000); step time is unchanged.
RELAXED_CONTACT_BUFFER = (1000, 4000)
_EPS = 1e-6  # goal layouts are float32


def in_relaxed_workspace(relative_xy):
    relative_xy = np.asarray(relative_xy, dtype=float)
    return bool(
        relative_xy[0] >= RELAXED_MIN_X - _EPS
        and np.linalg.norm(relative_xy) <= RELAXED_MAX_RADIUS + _EPS
    )


def _scattered(candidate, others, separation):
    return all(np.any(np.abs(candidate - other) >= separation) for other in others)


def _sample_region(
    count,
    base_xy,
    gripper_xy,
    *,
    y_range,
    separation,
    placed=(),
    layouts=100,
    candidates=400,
):
    """Scatter `count` centres in the relaxed workspace within `y_range`.

    Random sequential placement with whole-layout restarts. If every restart
    jams, fall back to a random subset of the region's separation lattice,
    which is Scattered by construction. Raises when even the lattice is short.
    """
    base_xy, gripper_xy = np.asarray(base_xy, float), np.asarray(gripper_xy, float)
    low = np.array([RELAXED_MIN_X, max(y_range[0], -RELAXED_MAX_RADIUS)])
    high = np.array([RELAXED_MAX_RADIUS, min(y_range[1], RELAXED_MAX_RADIUS)])

    def accept(relative, positions):
        return (
            in_relaxed_workspace(relative)
            and y_range[0] - _EPS <= relative[1] <= y_range[1] + _EPS
            and np.linalg.norm(base_xy + relative - gripper_xy) >= GRIPPER_CLEARANCE
            and _scattered(base_xy + relative, positions, separation)
        )

    placed = [np.asarray(p, float) for p in placed]
    for _ in range(layouts):
        positions = list(placed)
        for _ in range(count):
            for _ in range(candidates):
                relative = np.random.uniform(low, high)
                if accept(relative, positions):
                    positions.append(base_xy + relative)
                    break
            else:
                break
        else:
            return positions[len(placed) :]
    step = separation + 0.001
    lattice = [
        base_xy + np.array([x, y])
        for x in np.arange(low[0], high[0] + _EPS, step)
        for y in np.arange(low[1], high[1] + _EPS, step)
    ]
    lattice = [p for p in lattice if accept(p - base_xy, placed)]
    if len(lattice) < count:
        raise ValueError(
            f"Cannot fit {count} scattered blocks in the relaxed workspace "
            f"(lattice capacity {len(lattice)})"
        )
    order = np.random.permutation(len(lattice))[:count]
    return [lattice[int(i)] for i in order]


def sample_relaxed_stack_xy(num_blocks, base_xy, gripper_xy, base_index=0):
    """b0 first at a high-reach tower site, then the rest anywhere reachable."""
    base_xy = np.asarray(base_xy, float)
    separation = 2 * BLOCK_LENGTH
    for _ in range(100):
        b0 = base_xy + np.random.uniform(
            [RELAXED_TOWER_X_RANGE[0], RELAXED_STACK_BASE_Y_RANGE[0]],
            [RELAXED_TOWER_X_RANGE[1], RELAXED_STACK_BASE_Y_RANGE[1]],
        )
        if np.linalg.norm(b0 - np.asarray(gripper_xy)) >= GRIPPER_CLEARANCE:
            break
    else:
        raise ValueError("Could not place the relaxed Stack tower base")
    others = _sample_region(
        num_blocks - 1,
        base_xy,
        gripper_xy,
        y_range=(-RELAXED_MAX_RADIUS, RELAXED_MAX_RADIUS),
        separation=separation,
        placed=[b0],
    )
    others.insert(base_index, b0)
    return others


def sample_relaxed_tower_xy(robot_base_xy):
    return np.asarray(robot_base_xy) + np.array(
        [
            np.random.uniform(*RELAXED_TOWER_X_RANGE),
            np.random.uniform(*RELAXED_LATERAL_TOWER_Y_RANGE),
        ]
    )


def sample_relaxed_restack_centers(columns, base_xy, gripper_xy, separation):
    """Tower column first, then loose blocks anywhere on the higher-Y side."""
    tower = sample_relaxed_tower_xy(base_xy)
    loose = _sample_region(
        columns - 1,
        base_xy,
        gripper_xy,
        y_range=(RELAXED_LATERAL_MIN_Y, RELAXED_MAX_RADIUS),
        separation=separation,
        placed=[tower],
    )
    return [tower] + loose


def sample_relaxed_goal_sources(count, base_xy, gripper_xy):
    """Grid/Pyramid source blocks on the lower-Y side."""
    return _sample_region(
        count,
        base_xy,
        gripper_xy,
        y_range=(-RELAXED_MAX_RADIUS, RELAXED_SOURCE_MAX_Y),
        separation=2 * BLOCK_LENGTH + 0.005,
    )


def relaxed_goal_origin_x(base_x, width):
    """Left goal X of a layout `width` wide, centred at 0.50 m when possible."""
    return base_x + max(RELAXED_MIN_X, RELAXED_GOAL_CENTER_X - 0.5 * width)


def check_relaxed_goals(positions, base_xy):
    relative = np.asarray(positions)[:, :2] - np.asarray(base_xy)
    if not all(in_relaxed_workspace(p) for p in relative):
        raise ValueError("Goal layout exceeds the relaxed reachable workspace")
