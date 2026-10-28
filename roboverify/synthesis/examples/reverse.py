"""Reverse a tower from right to left using the standard physical DSL.

The loop starts with b = tbl and moves every source top onto b. Table-relative
XY offsets position the new tower; the original base moves last.
"""

import z3

from synthesis.api.instructions import (
    Assign,
    MoveByName,
    PickByName,
    ReleaseByName,
    While,
)
from synthesis.api.program import Program
from synthesis.util.on import BLOCK_LENGTH

DESTINATION_OFFSET = (0.45, -0.15)


def build_program(context, *, num_blocks):
    if num_blocks not in (2, 3, 4, 5):
        raise ValueError("The Reverse groundtruth program supports 2--5 blocks")
    selected, other, base, current, table = z3.Consts(
        "b_prime n b0 b tbl", context.BoxSort
    )
    top = z3.ForAll(
        [other], z3.Or(other == selected, z3.Not(context.ON_star(other, selected)))
    )
    guard = z3.And(
        selected != table, selected != current, context.ON_star(selected, base), top
    )
    dx, dy = DESTINATION_OFFSET
    # Keep the lifted gripper height during lateral transport; re-evaluating
    # Z_m while carrying the block would command an unnecessary extra rise.
    body = [
        MoveByName("b_prime", "b_prime", "Z_m", target_offset=[0, 0, 0.10]),
        PickByName("b_prime"),
        MoveByName("b_prime", "b_prime", "Z_m", target_offset=[0, 0, 0.10]),
        MoveByName("tbl", "tbl", "gripper", target_offset=[dx, dy, 0]),
        MoveByName("tbl", "tbl", "b", target_offset=[dx, dy, BLOCK_LENGTH]),
        ReleaseByName("Z_m", target_z=0.10),
        Assign("b", "b_prime"),
    ]
    instructions = [
        Assign("b", "tbl"),
        While(guard, [selected], body, invariant=None),
    ]
    return Program(len(instructions), instructions)
