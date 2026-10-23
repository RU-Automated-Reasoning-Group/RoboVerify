"""Reverse a tower from right to left using the standard physical DSL.

The original top becomes the destination base. Each remaining source top is
placed on the growing destination tower, including the original base last.
"""

import z3

from synthesis.api.instructions import (
    Assign,
    Get,
    MoveByName,
    PickByName,
    ReleaseByName,
    While,
)
from synthesis.api.program import Program
from synthesis.environment.tower_reset import REVERSE_TOWER_Y_OFFSET
from synthesis.util.on import BLOCK_LENGTH


def build_program(context, *, num_blocks):
    if num_blocks not in (2, 3, 4):
        raise ValueError("The Reverse groundtruth program supports 2--4 blocks")
    selected, other, base, destination = z3.Consts(
        "b_prime n b0 dest_base", context.BoxSort
    )
    top = z3.ForAll(
        [other], z3.Or(other == selected, z3.Not(context.ON_star(other, selected)))
    )
    source_top = z3.And(context.ON_star(selected, base), top)
    # Both towers fit below this height, measured from a stationary ground block.
    clearance = (num_blocks - 1) * BLOCK_LENGTH + 0.10
    leftward = -2 * REVERSE_TOWER_Y_OFFSET
    bootstrap = [
        Get(selected, source_top),
        MoveByName("b_prime", "b_prime", "b0", target_offset=[0, 0, clearance]),
        PickByName("b_prime"),
        MoveByName("b_prime", "b_prime", "b0", target_offset=[0, 0, clearance]),
        MoveByName("b0", "b0", "b0", target_offset=[0, leftward, clearance]),
        MoveByName("b0", "b0", "b0", target_offset=[0, leftward, 0]),
        ReleaseByName("b0", target_z=clearance),
        Assign("dest_base", "b_prime"),
        Assign("b", "b_prime"),
    ]
    body = [
        MoveByName("b_prime", "b_prime", "dest_base", target_offset=[0, 0, clearance]),
        PickByName("b_prime"),
        MoveByName("b_prime", "b_prime", "dest_base", target_offset=[0, 0, clearance]),
        MoveByName(
            "dest_base", "dest_base", "dest_base", target_offset=[0, 0, clearance]
        ),
        MoveByName("dest_base", "dest_base", "b", target_offset=[0, 0, BLOCK_LENGTH]),
        ReleaseByName("dest_base", target_z=clearance),
        Assign("b", "b_prime"),
    ]
    instructions = [
        *bootstrap,
        While(
            z3.And(source_top, z3.Not(context.ON_star(selected, destination))),
            [selected],
            body,
            invariant=None,
        ),
    ]
    return Program(len(instructions), instructions)
