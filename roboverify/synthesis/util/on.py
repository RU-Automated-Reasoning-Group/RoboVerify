from __future__ import annotations

import numpy as np
import z3

BLOCK_LENGTH = (
    0.025 * 2
)  # (0.025, 0.025, 0.025) in the xml file of the gym env is half length


def on_star_eval(block1, block2) -> bool:
    """define the numerical interpretation of the on(block1, block2) between two blocks"""
    x1, y1, z1 = block1
    x2, y2, z2 = block2
    return (
        abs(x1 - x2) < BLOCK_LENGTH / 2
        and abs(y1 - y2) < BLOCK_LENGTH / 2
        and 0 <= z1 - z2
    )


def on(block1, block2) -> bool:
    """define the numerical interpretation of the on(block1, block2) between two blocks"""
    x1, y1, z1 = block1
    x2, y2, z2 = block2
    return (
        abs(x1 - x2) < BLOCK_LENGTH / 2
        and abs(y1 - y2) < BLOCK_LENGTH / 2
        and 0 <= z1 - z2 < 1.5 * BLOCK_LENGTH
    )


def _clamped_linear_reward(error: float, tolerance: float) -> float:
    """Return 1 at zero error and linearly decay to 0 beyond ``tolerance``."""
    if tolerance <= 0.0:
        return 1.0 if error <= 0.0 else 0.0
    return float(max(0.0, 1.0 - error / tolerance))


def on_reward(
    block1,
    block2,
    block_length: float = BLOCK_LENGTH,
) -> float:
    """Dense [0, 1] reward aligned with :func:`on` geometry.

    Returns 1.0 when ``on(block1, block2)`` holds; decays smoothly as xy
    misalignment or vertical gap move outside the valid ON band.
    """
    x1, y1, z1 = np.asarray(block1, dtype=float)
    x2, y2, z2 = np.asarray(block2, dtype=float)
    half_xy = block_length / 2.0
    max_dz = 1.5 * block_length

    score_x = _clamped_linear_reward(abs(x1 - x2), half_xy)
    score_y = _clamped_linear_reward(abs(y1 - y2), half_xy)
    score_xy = score_x * score_y

    dz = z1 - z2
    if 0.0 <= dz < max_dz:
        score_z = 1.0
    elif dz < 0.0:
        score_z = _clamped_linear_reward(-dz, half_xy)
    else:
        score_z = _clamped_linear_reward(dz - max_dz, half_xy)

    return float(score_xy * score_z)


def z3_on(
    x1: z3.ArithRef,
    y1: z3.ArithRef,
    z1: z3.ArithRef,
    x2: z3.ArithRef,
    y2: z3.ArithRef,
    z2: z3.ArithRef,
    block_length: float = BLOCK_LENGTH,
) -> z3.BoolRef:
    """Z3 encoding aligned with :func:`on` (same xy tolerance and vertical gap)."""
    half_xy = z3.RealVal(block_length / 2.0)
    max_dz = z3.RealVal(1.5 * block_length)
    return z3.And(
        z3.Abs(x1 - x2) < half_xy,
        z3.Abs(y1 - y2) < half_xy,
        z1 - z2 >= 0,
        z1 - z2 < max_dz,
    )


def on_star_implementation(block1, block2) -> bool:
    """define the numerical interpretation of the on(block1, block2) between two blocks"""
    x1, y1, z1 = block1
    x2, y2, z2 = block2
    return (
        abs(x1 - x2) < BLOCK_LENGTH / 2
        and abs(y1 - y2) < BLOCK_LENGTH / 2
        and 0 <= z1 - z2
    )


def d_star_implementation(block1, block2) -> bool:
    """define the numerical interpretation of the d_star(block1, block2) between two blocks"""
    x1, y1, z1 = block1
    x2, y2, z2 = block2
    return x1 == x2 and z1 == z2 and y1 <= y2


def r_star_implementation(block1, block2) -> bool:
    """define the numerical interpretation of the r_star(block1, block2) between two blocks"""
    x1, y1, z1 = block1
    x2, y2, z2 = block2
    return y1 == y2 and z1 == z2 and x1 <= x2


def higher_implementation(block1, block2) -> bool:
    """define the numerical interpretation of the on(block1, block2) between two blocks"""
    x1, y1, z1 = block1
    x2, y2, z2 = block2
    return (0 <= z1 - z2 and z1 >= 0.0 and z2 >= 0.0) or (
        x1 == x2 and y1 == y2 and z1 == z2
    )


def scattered_implementation(block1, block2) -> bool:
    """define the numerical interpretation of the scattered(block1, block2) between two blocks"""
    x1, y1, z1 = block1
    x2, y2, z2 = block2
    return (
        (abs(x1 - x2) >= 2 * BLOCK_LENGTH or abs(y1 - y2) >= 2 * BLOCK_LENGTH)
        and z1 >= 0.0
        and z2 >= 0.0
    )


def top_implementation(block, all_blocks) -> bool:
    """Check if a block is on top"""
    top_flag = True
    for other_block in all_blocks:
        if other_block != block and on_star_implementation(other_block, block):
            top_flag = False
            break
    return top_flag


def get_block_pos(obs, block_id):
    start_idx = 10 + 12 * block_id
    end_idx = start_idx + 3
    return np.array(obs[start_idx:end_idx])


def state_comparison_indices(num_blocks: int) -> list[int]:
    """Indices into flattened RoboVerifyStack obs for policy vs expert comparison.

    Includes gripper position (0-2), finger opening (3-4), and every block's xyz.
    """
    agent_dim = 10
    object_dyn_dim = 12
    indices = [0, 1, 2, 3, 4]
    for block_id in range(num_blocks):
        start = agent_dim + block_id * object_dyn_dim
        indices.extend(range(start, start + 3))
    return indices


def print_block_layout(obs, num_block):
    for i in range(0, num_block):
        for j in range(0, num_block):
            print(f"on({i}, {j})", on(get_block_pos(obs, i), get_block_pos(obs, j)))
