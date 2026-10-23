"""Soundness boundaries and an unbounded lateral-descent regression."""

import unittest

import z3

from synthesis.api.instructions import MoveByName
from synthesis.util.symbols import fresh_const
from synthesis.verification_lib.lowlevel_verification_lib import LowLevelContext
from synthesis.verification_lib.motion_verification import MotionContract, MotionProblem
from synthesis.verification_lib.primitive_motion import PrimitiveMotionProblem
from synthesis.verification_lib.universal_instances import (
    is_quantifier_free,
    universal_instance_relaxation,
)


class UniversalInstanceTests(unittest.TestCase):
    def setUp(self):
        self.low = LowLevelContext(default_L=0.05)
        self.a, self.x, self.hidden = [
            fresh_const(self.low.BoxSort, name) for name in ("a", "x", "hidden")
        ]

    def test_relaxation_is_implied_without_closing_the_domain(self):
        premise = z3.ForAll([self.x], self.low.Y(self.x) >= 0)
        relaxed = universal_instance_relaxation([premise], [self.a])
        solver = z3.Solver()
        solver.add(premise, z3.Not(z3.And(*relaxed)))
        self.assertEqual(solver.check(), z3.unsat)
        # A relaxed SAT model can have another object violating the universal.
        solver = z3.Solver()
        solver.add(*relaxed, self.hidden != self.a, self.low.Y(self.hidden) < 0)
        self.assertEqual(solver.check(), z3.sat)

    def test_negative_universal_does_not_constrain_the_named_object(self):
        premise = z3.Not(z3.ForAll([self.x], self.low.Y(self.x) >= 0))
        relaxed = universal_instance_relaxation([premise], [self.a])
        solver = z3.Solver()
        solver.add(*relaxed, self.low.Y(self.a) >= 0, self.low.Y(self.hidden) < 0)
        self.assertEqual(solver.check(), z3.sat)

    def test_nested_negative_universal_is_not_finitely_replaced(self):
        y = fresh_const(self.low.BoxSort, "y")
        # Every global minimum is at zero. A named object at one is allowed
        # because another object can be at zero. Replacing the inner forall
        # with just y=a would incorrectly require Y(a)=0.
        premise = z3.ForAll(
            [self.x],
            z3.Implies(
                z3.ForAll([y], self.low.Y(y) >= self.low.Y(self.x)),
                self.low.Y(self.x) == 0,
            ),
        )
        relaxed = universal_instance_relaxation([premise], [self.a])
        solver = z3.Solver()
        solver.add(*relaxed, self.low.Y(self.a) == 1, self.low.Y(self.hidden) == 0)
        self.assertEqual(solver.check(), z3.sat)

    def test_instance_budget_only_weakens_the_premises(self):
        premise = z3.ForAll([self.x], self.low.Y(self.x) >= 0)
        relaxed = universal_instance_relaxation([premise], [self.a], max_instances=0)
        solver = z3.Solver()
        solver.add(*relaxed, self.low.Y(self.a) < 0)
        self.assertEqual(solver.check(), z3.sat)
        self.assertIsNone(
            universal_instance_relaxation([self.low.Y(self.a) >= 0], [self.a])
        )
        self.assertTrue(is_quantifier_free(z3.And(*relaxed)))

    def problem(self):
        return MotionProblem(
            self.low, [], ["a", "b"], MotionContract("a", "b"), None, "test", 3000
        )

    def test_relaxed_sat_falls_back_to_the_original_universal(self):
        problem = self.problem()
        problem.solver.add(z3.ForAll([self.x], self.low.Y(self.x) >= 0))
        # hidden is deliberately absent from the instantiation candidates.
        result = problem.check("hidden_nonnegative", self.low.Y(self.hidden) < 0)
        self.assertEqual(result.status, "valid")
        self.assertEqual(result.reason, "")
        self.assertIsNone(result.counterexample)

    def test_relaxed_sat_is_not_a_spurious_success(self):
        problem = self.problem()
        problem.solver.add(z3.ForAll([self.x], self.low.Y(self.x) >= 0))
        result = problem.check("not_forced_above_one", self.low.Y(self.hidden) < 1)
        self.assertEqual(result.status, "refuted")
        self.assertIsNotNone(result.counterexample)

    def test_lateral_descent_is_proved_with_an_unbounded_obstacle(self):
        p = PrimitiveMotionProblem(
            self.low,
            [],
            ["a", "b"],
            MotionContract("a", "b"),
            None,
            "descent",
            5000,
            initial_positions={"a": [0, 0, 0.7], "b": [0, 0, 0.425]},
            extrema_witness_count=2,
        )
        p.solver.add(
            z3.ForAll(
                [self.x],
                z3.Implies(
                    z3.Not(self.low.lowlevel_box_equal(self.x, p.constants["a"])),
                    self.low.Z(self.x) == z3.RealVal("0.425"),
                ),
            )
        )
        p.held_name = "a"
        p.arm = p.current["a"]
        p.execute(
            [
                MoveByName("a", "Y_min", "gripper", target_offset=[0, -0.15, 0]),
                MoveByName("a", "Y_min", "b", target_offset=[0, 0, 0]),
            ]
        )
        # A finite SAT witness checks nonvacuity only; the safety proof above
        # does not assert a finite domain or pin the arbitrary obstacle sym.
        p.consistency_candidates = tuple(p.constants.values())
        self.assertEqual(p.check("consistency").status, "valid")
        descent = next(c for c in p.checks if c.obligation == "collision_2_sym")
        self.assertEqual(descent.status, "valid")
        self.assertIn("universal instances", descent.reason)
        self.assertTrue(all(c.status == "valid" for c in p.checks), str(p.checks))


if __name__ == "__main__":
    unittest.main()
