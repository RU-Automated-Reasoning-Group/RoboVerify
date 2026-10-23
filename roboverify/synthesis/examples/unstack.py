"""Provided Unstack groundtruth: lateral transfers that never move the base.

The guard strengthens the original condition with b_prime != b0 so a scene
containing only the base exits immediately. Collection uses this factory by
default; supplied-program verification loads the same executable.
"""

import z3

from synthesis.api.instructions import MoveByName, PickByName, ReleaseByName, While
from synthesis.api.program import Program


def build_program(context, *, num_blocks):
    selected, other, base, table = z3.Consts("b_prime n b0 tbl", context.BoxSort)
    # Keep the base fixed, including when it is the only physical block.
    guard = z3.And(
        selected != base,
        z3.ForAll(
            [other],
            z3.Implies(
                z3.And(
                    other != table,
                    z3.Or(
                        context.ON_star(base, selected), context.ON_star(other, base)
                    ),
                ),
                context.ON_star(selected, other),
            ),
        ),
    )
    body = [
        MoveByName("b_prime", "b_prime", "b_prime", target_offset=[0, 0, 0.15]),
        PickByName("b_prime"),
        MoveByName("b_prime", "b_prime", "b_prime", target_offset=[0, 0, 0.15]),
        MoveByName("b0", "Y_min", "gripper", target_offset=[0, -0.15, 0]),
        MoveByName("b0", "Y_min", "b0", target_offset=[0, 0, 0]),
        ReleaseByName("Z_m", target_z=0.15),
    ]
    return Program(1, [While(guard, [selected], body, invariant=None)])
