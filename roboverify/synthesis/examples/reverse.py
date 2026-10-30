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
    selected, other, base, current, table = z3.Consts(
        "b_prime n b0 b tbl", context.BoxSort
    )
    top = z3.ForAll(
        [other], z3.Or(other == selected, z3.Not(context.ON_star(other, selected)))
    )
    guard = z3.And(
        selected != table, selected != current, context.ON_star(selected, base), top
    )
    # tbl's support reference is half a block below the table surface.
    # This clearance is 10 cm above the highest nominal block center.
    clearance = num_blocks * BLOCK_LENGTH + 0.10
    dx, dy = DESTINATION_OFFSET
    body = [
        MoveByName("b_prime", "b_prime", "tbl", target_offset=[0, 0, clearance]),
        PickByName("b_prime"),
        MoveByName("b_prime", "b_prime", "tbl", target_offset=[0, 0, clearance]),
        MoveByName("tbl", "tbl", "tbl", target_offset=[dx, dy, clearance]),
        MoveByName("tbl", "tbl", "b", target_offset=[dx, dy, BLOCK_LENGTH]),
        ReleaseByName("tbl", target_z=clearance),
        Assign("b", "b_prime"),
    ]
    instructions = [
        Assign("b", "tbl"),
        While(guard, [selected], body, invariant=None),
    ]
    return Program(len(instructions), instructions)
