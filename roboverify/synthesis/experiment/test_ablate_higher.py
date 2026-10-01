"""Concrete diagnostic witnesses for the Higher ablation, without saved demos."""

import unittest
from unittest.mock import patch

import z3
from synthesis.api.instructions import Assign, Put
from synthesis.api.program import to_seq, wp
from synthesis.experiment.ablate_higher import structural_invariant, uses_higher
from synthesis.experiment.motion_premise_ablation import (
    quantifier_count,
    without_extra_motion_premises,
)
from synthesis.inference_lib.demo_store import LoopHeadState
from synthesis.util.on import using_higher_tolerance
from synthesis.verification_lib.counterexamples import stacks_to_positions, state_holds
from synthesis.verification_lib.highlevel_verification_lib import HighLevelContext
from synthesis.verification_lib.lowlevel_verification_lib import LowLevelContext
from synthesis.verification_lib.motion_verification import MotionContract
from synthesis.verification_lib.primitive_motion import PrimitiveMotionProblem


class HigherAblationTests(unittest.TestCase):
    def setUp(self):
        self.context = HighLevelContext()
        self.invariant = structural_invariant(self.context)

    def test_structural_control_covers_stack_heads_and_exit_without_higher(self):
        blocks = [f"x{i}" for i in range(4)]
        entry = stacks_to_positions([[b] for b in blocks])
        self.assertFalse(uses_higher(self.invariant))
        for top in range(4):
            state = LoopHeadState(
                "1",
                stacks_to_positions(
                    [blocks[: top + 1]] + [[b] for b in blocks[top + 1 :]]
                ),
                entry,
                {"b0": blocks[0], "b": blocks[top]},
            )
            self.assertTrue(state_holds(self.invariant, state))

    def test_second_tower_exposes_attachment_preservation_mismatch(self):
        # This is the three-clause invariant observed in the no-Higher run:
        # clear top, root, and separation. It omits the single-tower clause.
        weak = z3.And(
            self.invariant.arg(0), self.invariant.arg(1), self.invariant.arg(3)
        )
        positions = stacks_to_positions([["a"], ["d", "c"]])
        state = LoopHeadState(
            "1", positions, positions, {"b": "a", "b0": "a", "b_prime": "c"}
        )
        selected, b = self.context.get_consts("b_prime"), self.context.get_consts("b")
        n = z3.Const("guard_object", self.context.BoxSort)
        guard = z3.And(
            selected != b,
            z3.ForAll(
                [n], z3.Implies(self.context.ON_star(n, selected), n == selected)
            ),
        )
        body = to_seq([Put("b_prime", "b"), Assign("b", "b_prime")])
        self.assertTrue(state_holds(weak, state))
        self.assertTrue(state_holds(guard, state))
        self.assertFalse(state_holds(wp(body, weak, self.context), state))
        self.assertFalse(state_holds(self.invariant, state))
        # Real detachment loses ON*(c,d); the abstract attachment WP retains it.
        successor = LoopHeadState(
            "1",
            stacks_to_positions([["a", "c"], ["d"]]),
            positions,
            {"b": "c", "b0": "a", "b_prime": "c"},
        )
        self.assertTrue(state_holds(weak, successor))

    def test_relational_structure_alone_does_not_bound_singleton_height(self):
        positions = stacks_to_positions([["a"], ["c"]])
        before = LoopHeadState(
            "1", positions, positions, {"b": "a", "b0": "a", "c": "c"}
        )
        raised = {name: list(point) for name, point in positions.items()}
        raised["c"][2] += 0.5
        after = LoopHeadState("1", raised, positions, before.constants)
        self.assertTrue(state_holds(self.invariant, before))
        self.assertTrue(state_holds(self.invariant, after))
        for name in ("ON_star", "Scattered"):
            for a in ("b", "c"):
                for b in ("b", "c"):
                    predicate = getattr(self.context, name)(
                        self.context.get_consts(a), self.context.get_consts(b)
                    )
                    self.assertEqual(
                        state_holds(predicate, before), state_holds(predicate, after)
                    )
        higher = self.context.Higher(
            self.context.get_consts("b"), self.context.get_consts("c")
        )
        with using_higher_tolerance(0.001):
            self.assertTrue(state_holds(higher, before))
            self.assertFalse(state_holds(higher, after))

    def test_removed_premises_admit_bad_geometry_but_checks_still_refute_it(self):
        from synthesis.cfg import verification
        from synthesis.verification_lib import tower_geometry

        original_height = tower_geometry.assume_tower_geometry
        original_clearance = tower_geometry.arm_clearance
        problem = PrimitiveMotionProblem(
            LowLevelContext(default_L=0.05),
            [],
            ["a", "b"],
            MotionContract("a", "b"),
            None,
            "test",
            3000,
            initial_positions={"a": (0, 0, 0.3), "b": (0.02, 0, 0.5)},
            initial_arm=(0, 0, 0.1),
        )
        with without_extra_motion_premises():
            tower_geometry.assume_tower_geometry(problem, 0.4)
            problem.solver.add(
                tower_geometry.arm_clearance(problem),
                tower_geometry.column_alignment(problem),
            )
            verification.assume_input_alignment(problem, ["a"])
            self.assertEqual(quantifier_count(problem.solver.assertions()), 0)
            self.assertEqual(problem.check("consistency").status, "valid")
            # The low arm, below-table block and offset column remain failures
            # of the actual geometric conclusions, never trivial True checks.
            for name in ("tower_height_invariant", "arm_clearance", "column_alignment"):
                violation = z3.Not(
                    getattr(tower_geometry, name)(problem, fields=problem.fields)
                )
                self.assertEqual(problem.check(name, violation).status, "refuted")
        self.assertIs(tower_geometry.assume_tower_geometry, original_height)
        self.assertIs(tower_geometry.arm_clearance, original_clearance)

    def test_effect_goals_remain_exact_without_quantified_premise_copies(self):
        from synthesis.cfg import verification

        x = z3.Const("effect_test_x", self.context.BoxSort)
        b = self.context.get_consts("b")
        condition = z3.ForAll([x], self.context.Higher(b, x))
        problem = PrimitiveMotionProblem(
            LowLevelContext(default_L=0.05),
            [condition],
            ["a", "b"],
            MotionContract("a", "b"),
            None,
            "test",
            3000,
        )
        original_conditions = problem.initial_condition
        goals = {}

        def inspect_query(obligation, violation=None):
            self.assertEqual(quantifier_count(problem.solver.assertions()), 0)
            goals[obligation] = violation

        with without_extra_motion_premises(), patch.object(
            problem, "check", inspect_query
        ):
            verification.check_abstract_effects(problem)
        self.assertIs(problem.initial_condition, original_conditions)
        self.assertEqual(
            set(goals), {"effect_ON_star", "effect_Higher", "effect_Scattered"}
        )
        self.assertGreater(quantifier_count([goals["effect_Higher"]]), 0)


if __name__ == "__main__":
    unittest.main()
