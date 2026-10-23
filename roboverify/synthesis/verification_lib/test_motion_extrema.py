"""Independent extrema/obstacle witnesses and instruction-entry reference semantics."""

import contextlib
import io
import unittest

import z3

from synthesis.api.instructions import MoveByName, ReleaseByName
from synthesis.examples.unstack import build_program
from synthesis.verification_lib.highlevel_verification_lib import HighLevelContext
from synthesis.verification_lib.lowlevel_verification_lib import LowLevelContext
from synthesis.verification_lib.motion_verification import (
    MotionContract,
    verify_motion_block,
)
from synthesis.verification_lib.primitive_motion import (
    PrimitiveMotionProblem,
    count_extrema_references,
)


class MotionExtremaTests(unittest.TestCase):
    def problem(self, count=1, *, positions=None, conditions=(), use_tbl=False):
        return PrimitiveMotionProblem(
            LowLevelContext(default_L=0.05, use_tbl=use_tbl),
            conditions,
            ["a", "b"] + (["tbl"] if use_tbl else []),
            MotionContract("a", "b", table_surface_height=0.4),
            None,
            "test",
            1000,
            initial_positions=positions,
            initial_arm=[1, 0.5, 0.5],
            extrema_witness_count=count,
        )

    def assert_implied(self, problem, condition):
        problem.solver.push()
        problem.solver.add(z3.Not(condition))
        self.assertEqual(problem.solver.check(), z3.unsat)
        problem.solver.pop()

    def test_minimizing_block_and_collision_obstacle_are_independent(self):
        p = self.problem(
            positions={
                "a": [1, 0.2, 0.5],
                "b": [2, 0.3, 0.5],
                "sym2": [3, -0.05, 0.5],
            }
        )
        # Keep sym arbitrary in the VC machinery, selecting an admissible scene
        # where C collides with the target determined by a different block D.
        p.solver.add(*(v == n for v, n in zip(p.initial["sym"], [1, 0.1, 0.5])))
        p.execute([MoveByName("a", "Y_min", "a", target_offset=[0, 0.15, 0])])
        self.assert_implied(p, p.arm[1] == z3.RealVal("0.1"))
        collision = next(c for c in p.checks if c.obligation == "collision_1_sym")
        self.assertEqual(collision.status, "refuted")
        self.assertNotEqual(
            collision.counterexample.bindings["sym"],
            collision.counterexample.bindings["sym2"],
        )

    def test_extrema_witness_can_alias_the_obstacle_and_a_named_block(self):
        p = self.problem(use_tbl=True)
        p.solver.add(p.same("sym2", "sym"), p.same("sym2", "a"))
        self.assertEqual(p.solver.check(), z3.sat)

    def test_new_witness_receives_entry_invariant_instances(self):
        high = HighLevelContext()
        q = z3.FreshConst(high.BoxSort)
        premise = z3.ForAll([q], high.Higher(q, high.get_consts("a")))
        with contextlib.redirect_stdout(io.StringIO()):
            p = self.problem(conditions=[premise])
        self.assertEqual(p.solver.check(), z3.sat)
        p.solver.add(p.initial["sym2"][2] == 0, p.initial["a"][2] == 0.5)
        self.assertEqual(p.solver.check(), z3.unsat)

    def test_held_block_participates_and_minimum_is_recomputed_next_instruction(self):
        p = self.problem(
            2,
            positions={
                "a": [1, 0.2, 0.5],
                "b": [2, 0.3, 0.4],
                "sym": [3, 1, 0.4],
                "sym2": [4, 1, 0.4],
                "sym3": [5, 1, 0.4],
            },
        )
        p.held_name = "a"
        p.arm = p.current["a"]
        p.execute([MoveByName("a", "Y_min", "gripper", target_offset=[0, -0.15, 0])])
        self.assert_implied(p, p.current["a"][1] == z3.RealVal("0.05"))
        # The same instruction reference is frozen, despite the payload update.
        self.assert_implied(p, p.reference_position("Y_min")[1] == z3.RealVal("0.2"))
        p.execute([MoveByName("a", "Y_min", "a", target_offset=[0, 0, 0])])
        self.assert_implied(p, p.current["a"][1] == z3.RealVal("0.05"))
        self.assert_implied(p, p.arm[2] == z3.RealVal("0.5"))

    def test_table_alias_is_not_an_extremum_candidate(self):
        p = self.problem(
            use_tbl=True,
            positions={
                "a": [1, 0.2, 0.5],
                "sym": [2, 0.3, 0.5],
                "sym2": [3, 0.4, 0.5],
            },
        )
        p.solver.add(
            p.constants["b"] == p.context.table_const(),
            p.context.Y(p.constants["b"]) == -100,
        )
        self.assert_implied(p, p.reference_position("Y_min")[1] == z3.RealVal("0.2"))

    def test_ties_allow_either_extremizing_block_without_mixing_coordinates(self):
        p = self.problem(
            positions={
                "a": [1, 0.2, 0.5],
                "b": [2, 0.2, 0.6],
                "sym": [3, 0.3, 0.4],
                "sym2": [4, 0.4, 0.4],
            }
        )
        position = p.reference_position("Y_min")
        for x, z in ((1, "0.5"), (2, "0.6")):
            p.solver.push()
            p.solver.add(position[0] == x, position[2] == z3.RealVal(z))
            self.assertEqual(p.solver.check(), z3.sat)
            p.solver.pop()
        self.assert_implied(
            p,
            z3.Or(
                z3.And(position[0] == 1, position[2] == z3.RealVal("0.5")),
                z3.And(position[0] == 2, position[2] == z3.RealVal("0.6")),
            ),
        )

    def test_release_freezes_gripper_xy_and_maximum_z_before_opening(self):
        p = self.problem(
            positions={
                "a": [1, 0.2, 0.7],
                "b": [2, 0.3, 0.4],
                "sym": [3, 1, 0.4],
                "sym2": [4, 1, 0.4],
            }
        )
        p.held_name = "a"
        p.arm = p.current["a"]
        p.execute([ReleaseByName("Z_m", target_z=0.15)])
        self.assertIsNone(p.held_name)
        self.assert_implied(
            p,
            z3.And(
                p.arm[0] == 1,
                p.arm[1] == z3.RealVal("0.2"),
                p.arm[2] == z3.RealVal("0.85"),
            ),
        )
        # Its unsupported drop is still rejected; a frozen retreat target must
        # not depend on the unconstrained fallen payload position.
        self.assertEqual(
            next(c.status for c in p.checks if c.obligation == "release_support_1"),
            "refuted",
        )

    def test_no_extrema_preserves_the_original_symbol_universe(self):
        p = self.problem(0)
        self.assertEqual(set(p.constants), {"a", "b", "sym"})
        with self.assertRaisesRegex(NotImplementedError, "independent"):
            p.reference_position("Y_min")

    def test_current_unstack_body_is_fully_encoded_on_a_supported_scene(self):
        high = HighLevelContext(use_tbl=True)
        body = build_program(high, num_blocks=2).instructions[0].body
        self.assertEqual(count_extrema_references(body), 3)
        q = z3.FreshConst(high.BoxSort)
        source, base, table = [high.get_consts(n) for n in ("b_prime", "b0", "tbl")]
        conditions = [z3.ForAll([q], z3.Or(q == source, q == base, q == table))]
        positions = {"b_prime": [0.3, 0.2, 0.475], "b0": [0.3, 0.2, 0.425]}
        for name in ("sym", "sym2", "sym3", "sym4"):
            positions[name] = positions["b0"]
        with contextlib.redirect_stdout(io.StringIO()):
            result = verify_motion_block(
                conditions,
                body,
                ["b_prime", "b0", "tbl"],
                MotionContract("b_prime", "tbl", table_surface_height=0.4),
                initial_positions=positions,
                initial_arm=[0.3, 0.5, 0.6],
                timeout_ms=3000,
            )
        self.assertTrue(
            result,
            [
                (c.obligation, c.status, c.reason)
                for c in result.checks
                if c.status != "valid"
            ],
        )


if __name__ == "__main__":
    unittest.main()
