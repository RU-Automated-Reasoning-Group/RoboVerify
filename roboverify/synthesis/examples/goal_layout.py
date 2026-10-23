"""The paper's shared Grid/Pyramid program, using goal-only pointer traversal.

This executable does not supply invariants or claim synthesis/verification support.
"""

import z3

from synthesis.api.instructions import (
    GoalAssign,
    MoveByName,
    MoveDown,
    MoveRight,
    PickByName,
    ReleaseByName,
    While,
)
from synthesis.api.program import Program


def build_program(context=None, *, num_blocks=None):
    box_sort = context.BoxSort if context is not None else z3.DeclareSort("Box")
    goal_sort = z3.DeclareSort("Goal")
    row, goal, null = z3.Consts("d r null", goal_sort)
    block = z3.Const("b_prime", box_sort)
    at_goal = z3.Function("AtGoal", box_sort, z3.BoolSort())
    body = [
        MoveByName("b_prime", "b_prime", "r", target_offset=[0, 0, 0.15]),
        PickByName("b_prime"),
        MoveByName("b_prime", "b_prime", "r", target_offset=[0, 0, 0.15]),
        MoveByName("r", "r", "r", target_offset=[0, 0, 0.15]),
        MoveByName("r", "r", "r", target_offset=[0, 0, 0]),
        ReleaseByName("r", target_z=0.15),
        MoveRight("r"),
    ]
    inner = While(
        z3.And(goal != null, z3.Not(at_goal(block))),
        [block],
        body,
        invariant=None,
        max_iters=num_blocks,
    )
    outer = While(
        row != null,
        [],
        [GoalAssign("r", "d"), inner, MoveDown("d")],
        invariant=None,
        max_iters=num_blocks,
    )
    return Program(2, [GoalAssign("d", "h0"), outer])
