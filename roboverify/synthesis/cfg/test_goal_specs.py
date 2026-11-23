"""Nonvacuous goal contracts, resource counts and concrete occupancy semantics."""

import itertools
import unittest

import numpy as np
import z3

from synthesis.cfg.goal_specs import GoalTaskContext
from synthesis.cfg.tasks import symbolic_task_spec
from synthesis.environment.goal_layout import occupancy
from synthesis.util.on import scattered_implementation
from synthesis.verification_lib.highlevel_verification_lib import HighLevelContext

_contexts = itertools.count()


def finite_scene(num_blocks, num_goals):
    """Explicit finite universes for tests only; table/null are distinct members."""
    suffix = next(_contexts)
    context = GoalTaskContext(
        HighLevelContext(sort_name=f"SpecBlocks{suffix}", use_tbl=True),
        goal_sort_name=f"SpecGoals{suffix}",
    )
    blocks = [z3.Const(f"block_{i}", context.BoxSort) for i in range(num_blocks)]
    goals = [z3.Const(f"goal_{i}", context.GoalSort) for i in range(num_goals)]
    solver = context.new_solver()
    for members, sentinel in ((blocks, context.tbl), (goals, context.null)):
        probe = z3.FreshConst(sentinel.sort())
        solver.add(
            z3.Distinct(*members, sentinel),
            z3.ForAll([probe], z3.Or(*(probe == v for v in (*members, sentinel)))),
        )
    return context, solver, blocks, goals


def bind_scene(context, blocks, goals, positions, targets):
    occupied = occupancy(positions, targets)
    return [
        *(
            context.Occupies(b, g) == bool(occupied[i, j])
            for i, b in enumerate(blocks)
            for j, g in enumerate(goals)
        ),
        *(
            context.Scattered(b, c)
            == bool(scattered_implementation(positions[i], positions[j]))
            for i, b in enumerate(blocks)
            for j, c in enumerate(blocks)
        ),
    ]


class GoalSpecTests(unittest.TestCase):
    def assert_truth(self, solver, formula, expected):
        # Check premises separately so an inconsistent scene cannot pass.
        self.assertEqual(solver.check(), z3.sat, solver.reason_unknown())
        solver.push()
        solver.add(z3.Not(formula) if expected else formula)
        self.assertEqual(solver.check(), z3.unsat, solver.reason_unknown())
        solver.pop()

    def test_all_goal_tasks_accept_empty_scattered_starts_and_permuted_coverage(self):
        for task in ("grid", "pyramid", "build3d"):
            with self.subTest(task=task):
                context, solver, blocks, goals = finite_scene(3, 3)
                pre, post = symbolic_task_spec(task, context)
                targets = np.array([[0.1 * i, 0.3, 0.425] for i in range(3)])
                initial = [[0.12 * i, -0.3, 0.425] for i in range(3)]
                solver.push()
                solver.add(*bind_scene(context, blocks, goals, initial, targets), pre)
                self.assertEqual(solver.check(), z3.sat, solver.reason_unknown())
                self.assert_truth(solver, post, False)
                solver.pop()
                solver.push()
                solver.add(*bind_scene(context, blocks, goals, targets[::-1], targets))
                self.assert_truth(solver, post, True)
                self.assert_truth(solver, pre, False)
                solver.pop()
                missing = targets[::-1].copy()
                missing[0, 1] = -0.3
                solver.add(*bind_scene(context, blocks, goals, missing, targets))
                self.assert_truth(solver, post, False)

    def test_precondition_rejects_unseparated_sources_and_occupied_goals(self):
        targets = [[0, 0.3, 0.425], [0.1, 0.3, 0.425]]
        for positions in (
            [[0, -0.3, 0.425], [0.075, -0.3, 0.425]],
            [[0, 0.3, 0.425], [0.3, -0.3, 0.425]],
        ):
            context, solver, blocks, goals = finite_scene(2, 2)
            pre, _ = symbolic_task_spec("grid", context)
            solver.add(*bind_scene(context, blocks, goals, positions, targets))
            self.assert_truth(solver, pre, False)

    def test_resource_bijection_requires_equal_nonzero_counts(self):
        for blocks, goals, expected in (
            (0, 0, z3.unsat),
            (0, 1, z3.unsat),
            (1, 1, z3.sat),
            (1, 2, z3.unsat),
            (2, 2, z3.sat),
            (3, 2, z3.unsat),
        ):
            with self.subTest(blocks=blocks, goals=goals):
                context, solver, _, _ = finite_scene(blocks, goals)
                pre, _ = symbolic_task_spec("grid", context)
                solver.add(pre)
                self.assertEqual(solver.check(), expected, solver.reason_unknown())

    def test_runtime_predicates_have_physical_occupancy_witnesses(self):
        context, solver, blocks, goals = finite_scene(2, 2)
        b, c = blocks
        g, h = goals
        for impossible in (
            context.HasBlock(context.null),
            context.AtGoal(context.tbl),
            context.Occupies(context.tbl, g),
            context.Occupies(b, context.null),
            z3.And(context.Occupies(b, g), context.Occupies(b, h)),
            z3.And(context.Occupies(b, g), z3.Not(context.HasBlock(g))),
            z3.And(context.Occupies(c, g), z3.Not(context.AtGoal(c))),
        ):
            solver.push()
            solver.add(impossible)
            self.assertEqual(solver.check(), z3.unsat)
            solver.pop()

    def test_untyped_context_and_unknown_task_fail_explicitly(self):
        with self.assertRaisesRegex(TypeError, "GoalTaskContext"):
            symbolic_task_spec("grid", HighLevelContext())
        with self.assertRaisesRegex(ValueError, "use_tbl"):
            GoalTaskContext(HighLevelContext())
        with self.assertRaisesRegex(ValueError, "Unsupported goal"):
            GoalTaskContext().task_spec("unknown")

    def test_parametric_contract_has_no_fixed_domain_size(self):
        context = GoalTaskContext()
        pre, post = symbolic_task_spec("build3d", context)
        solver = context.new_solver()
        solver.add(pre)
        self.assertEqual(solver.check(), z3.sat, solver.reason_unknown())
        b = z3.Const("probe", context.BoxSort)
        self.assert_truth(solver, z3.Exists([b], b != context.tbl), True)
        self.assert_truth(solver, post, False)


if __name__ == "__main__":
    unittest.main()
