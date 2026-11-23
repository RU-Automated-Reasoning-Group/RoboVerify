"""Build a rectangular m-by-n-by-k goal layout, one complete layer at a time."""

import z3
from synthesis.api.instructions import GoalAssign, MoveDown, MoveUp, While
from synthesis.api.program import Program
from synthesis.examples.goal_layout import build_program as build_rows


def build_program(context=None, *, num_blocks=None):
    # Reuse the 2D row traversal and unchanged six-primitive placement body.
    row_program = build_rows(context, num_blocks=num_blocks)
    goal_sort = z3.DeclareSort("Goal")
    layer, null = z3.Consts("u null", goal_sort)
    row_loop = row_program.instructions[1]
    assert isinstance(row_loop.body[-1], MoveDown)
    outer = While(
        layer != null,
        [],
        [GoalAssign("d", "u"), row_loop, MoveUp("u")],
        invariant=None,
        max_iters=num_blocks,
    )
    return Program(2, [GoalAssign("u", "h0"), outer])
