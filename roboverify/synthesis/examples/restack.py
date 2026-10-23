"""Executable adaptation of Fig. 16's two-phase Restack program.

The printed unstack guard repeats b0 forever after freeing it. Tracking the last
parked block p excludes it once p = b0. Table-relative waypoints replace the
paper's moving extrema and place the temporary row on the robot's lower-Y side.
Only existing Pick/Move/Release/Assign/While instructions are used.
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
from synthesis.environment.restack_reset import resolve_tower_blocks
from synthesis.util.on import BLOCK_LENGTH

PARKING_X_OFFSET = 0.45
PARKING_Y_STEP = -0.12


def build_program(context, *, num_blocks):
    resolve_tower_blocks(num_blocks)
    selected, other, base, current, parked, table = z3.Consts(
        "b_prime n b0 b p tbl", context.BoxSort
    )
    top = z3.ForAll(
        [other], z3.Or(other == selected, z3.Not(context.ON_star(other, selected)))
    )
    clearance = num_blocks * BLOCK_LENGTH + 0.10
    unstack_body = [
        MoveByName("b_prime", "b_prime", "tbl", target_offset=[0, 0, clearance]),
        PickByName("b_prime"),
        MoveByName("b_prime", "b_prime", "tbl", target_offset=[0, 0, clearance]),
        MoveByName(
            "tbl",
            "p",
            "tbl",
            target_offset=[PARKING_X_OFFSET, PARKING_Y_STEP, clearance],
        ),
        MoveByName(
            "tbl",
            "p",
            "tbl",
            target_offset=[PARKING_X_OFFSET, PARKING_Y_STEP, BLOCK_LENGTH],
        ),
        ReleaseByName("tbl", target_z=clearance),
        Assign("p", "b_prime"),
    ]
    stack_body = [
        MoveByName("b_prime", "b_prime", "tbl", target_offset=[0, 0, clearance]),
        PickByName("b_prime"),
        MoveByName("b_prime", "b_prime", "tbl", target_offset=[0, 0, clearance]),
        MoveByName("b0", "b0", "tbl", target_offset=[0, 0, clearance]),
        MoveByName("b0", "b0", "b", target_offset=[0, 0, BLOCK_LENGTH]),
        ReleaseByName("tbl", target_z=clearance),
        Assign("b", "b_prime"),
    ]
    instructions = [
        Assign("p", "tbl"),
        While(
            z3.And(
                selected != table,
                selected != parked,
                context.ON_star(selected, base),
                top,
            ),
            [selected],
            unstack_body,
            invariant=None,
        ),
        Assign("b", "b0"),
        While(
            z3.And(selected != table, selected != current, top),
            [selected],
            stack_body,
            invariant=None,
        ),
    ]
    return Program(len(instructions), instructions)
