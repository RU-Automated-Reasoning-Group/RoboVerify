"""Mixed block/goal assertions for future goal-program verification.

The historical goal-only Mark model does not represent physical occupancy.
These symbols instead share the runtime AtGoal/HasBlock meanings. Layout shape
and traversal belong to the environment/program, not the task conditions.
No motion or WP rules are provided.
"""

import z3

from synthesis.util.symbols import fresh_const
from synthesis.verification_lib.highlevel_verification_lib import HighLevelContext


class GoalTaskContext:
    """Block algebra plus a separate Goal sort with an excluded null sentinel.

    The physical interpretation fixes the goal set and coordinates throughout
    execution. No row/layer shape or traversal assumptions are imposed here.
    """

    def __init__(self, blocks=None, *, goal_sort_name="Goal"):
        self.blocks = blocks if blocks is not None else HighLevelContext(use_tbl=True)
        if not self.blocks.use_tbl or self.blocks.verification_mode != "box":
            raise ValueError("Goal contracts require a block context with use_tbl=True")
        self.BoxSort = self.blocks.BoxSort
        self.GoalSort = z3.DeclareSort(goal_sort_name)
        self.tbl = self.blocks.get_consts("tbl")
        self.null = z3.Const("null", self.GoalSort)
        self.Scattered = self.blocks.Scattered
        self.Occupies = z3.Function(
            "Occupies", self.BoxSort, self.GoalSort, z3.BoolSort()
        )
        self.AtGoal = z3.Function("AtGoal", self.BoxSort, z3.BoolSort())
        self.HasBlock = z3.Function("HasBlock", self.GoalSort, z3.BoolSort())
        # Ghost resource bijection, unrelated to which block fills which goal.
        self.supply = z3.Function("goal_supply", self.GoalSort, self.BoxSort)
        self.slot = z3.Function("block_slot", self.BoxSort, self.GoalSort)

    def occupancy_axioms(self):
        """Interpret unary runtime predicates and disjoint goal regions.

        A block cannot occupy two goals. Goal occupancy is not assumed injective
        in the other direction: final coverage plus equal finite counts suffices.
        These are state-domain constraints, to reestablish after physical actions.
        """
        b = fresh_const(self.BoxSort, "block", avoid=(self.tbl,))
        g, h = [
            fresh_const(self.GoalSort, name, avoid=(self.null,))
            for name in ("goal", "other_goal")
        ]
        return z3.And(
            z3.ForAll(
                [b, g],
                z3.Implies(self.Occupies(b, g), z3.And(b != self.tbl, g != self.null)),
            ),
            z3.ForAll(
                [b, g, h],
                z3.Implies(z3.And(self.Occupies(b, g), self.Occupies(b, h)), g == h),
            ),
            z3.ForAll([b], self.AtGoal(b) == z3.Exists([g], self.Occupies(b, g))),
            z3.ForAll([g], self.HasBlock(g) == z3.Exists([b], self.Occupies(b, g))),
        )

    def task_spec(self, task):
        if task not in ("grid", "pyramid", "build3d"):
            raise ValueError(f"Unsupported goal task: {task}")
        b, c = [
            fresh_const(self.BoxSort, name, avoid=(self.tbl,))
            for name in ("block", "other_block")
        ]
        g = fresh_const(self.GoalSort, "goal", avoid=(self.null,))
        pre = z3.And(
            z3.Exists([g], g != self.null),
            z3.ForAll(
                [g],
                z3.Implies(
                    g != self.null,
                    z3.Not(self.HasBlock(g)),
                ),
            ),
            z3.ForAll(
                [b, c],
                z3.Implies(
                    z3.And(b != self.tbl, c != self.tbl, b != c),
                    self.Scattered(b, c),
                ),
            ),
            z3.ForAll(
                [g],
                z3.Implies(
                    g != self.null,
                    z3.And(self.supply(g) != self.tbl, self.slot(self.supply(g)) == g),
                ),
            ),
            z3.ForAll(
                [b],
                z3.Implies(
                    b != self.tbl,
                    z3.And(self.slot(b) != self.null, self.supply(self.slot(b)) == b),
                ),
            ),
        )
        post = z3.ForAll([g], z3.Implies(g != self.null, self.HasBlock(g)))
        return pre, post

    def new_solver(self, timeout_ms=5000):
        """Block and occupancy domain only; callers add task/state assertions."""
        solver = self.blocks.new_solver(timeout_ms)
        solver.add(self.occupancy_axioms())
        return solver
