"""Pick source clearance agrees in nominal proposals and the BMC screen."""

import unittest

import numpy as np

from synthesis.api.control import DEFAULT_PICK_CONTROL
from synthesis.api.instructions import MoveByName, PickByName
from synthesis.api.program import Program
from synthesis.cfg.search_feasibility import check_scene_feasibility
from synthesis.predicates.scene import Scene
from synthesis.search.contract import (
    Action,
    ContractConfig,
    ContractProblem,
    SearchExample,
)
from synthesis.util.primitive_geometry import PICK_APPROACH_XY_TOLERANCE


class PickApproachTests(unittest.TestCase):
    def problem(self, gripper, *, envelope=True):
        positions = np.array([[0.0, 0.0, 0.575], [0.0, 0.0, 0.525]])
        example = SearchExample(
            positions,
            np.array(gripper),
            {"source": 0, "support": 1},
            np.array([gripper]),
            positions[None],
        )
        return ContractProblem(
            [example],
            lambda state, index: False,
            allowed_operands=("source", "support"),
            actions=(Action("Pick", ("source",)),),
            config=ContractConfig(gripper_envelope=envelope),
        )

    def screen(self, program, gripper):
        return check_scene_feasibility(
            program,
            Scene(
                {0: np.array([0.0, 0.0, 0.575]), 1: np.array([0.0, 0.0, 0.525])},
                {"source": 0, "support": 1},
            ),
            gripper=gripper,
            initial_holding=None,
            fixed_offsets=True,
        )

    def test_source_height_lateral_approach_rejected_in_both_models(self):
        for height in (0.575, 0.595, 0.622):
            with self.subTest(height=height):
                gripper = [0.0, -0.2, height]
                problem = self.problem(gripper)
                self.assertIsNone(
                    problem.transition(problem.initial, Action("Pick", ("source",)))
                )
                result = self.screen(Program(1, [PickByName("source")]), gripper)
                self.assertFalse(result)
                self.assertEqual(result.status, "unsat")

    def test_clear_lateral_approach_and_aligned_grasp_remain_enabled(self):
        for gripper in ([0.0, -0.2, 0.625], [0.0, 0.0, 0.575], [0.001, 0, 0.575]):
            with self.subTest(gripper=gripper):
                problem = self.problem(gripper)
                self.assertIsNotNone(
                    problem.transition(problem.initial, Action("Pick", ("source",)))
                )
                self.assertTrue(
                    self.screen(Program(1, [PickByName("source")]), gripper)
                )

    def test_generated_height_reference_can_enable_pick(self):
        gripper = [0.0, -0.2, 0.575]
        for offset, expected in ((0.0, False), (0.05, True)):
            with self.subTest(offset=offset):
                program = Program(
                    2,
                    [
                        MoveByName(
                            "gripper", "gripper", "Z_m", target_offset=[0, 0, offset]
                        ),
                        PickByName("source"),
                    ],
                )
                self.assertEqual(bool(self.screen(program, gripper)), expected)

    def test_envelope_ablation_is_explicit(self):
        problem = self.problem([0.0, -0.2, 0.575], envelope=False)
        self.assertIsNotNone(
            problem.transition(problem.initial, Action("Pick", ("source",)))
        )

    def test_xy_deadband_matches_default_pick_controller(self):
        self.assertEqual(
            PICK_APPROACH_XY_TOLERANCE, DEFAULT_PICK_CONTROL.position_tolerance
        )


if __name__ == "__main__":
    unittest.main()
