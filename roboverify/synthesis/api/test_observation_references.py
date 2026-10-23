"""Instruction-entry extrema/gripper references retain bounded servo behavior."""

import unittest

import numpy as np

from synthesis.api.instructions import MoveByName, ReleaseByName
from synthesis.api.program import Program
from synthesis.api.test_control import ServoEnvironment
from synthesis.cfg.program_source import describe_program, load_program, prepare_program
from synthesis.environment.motion_reference import OBSERVATION_REFERENCE_MODE
from synthesis.verification_lib.highlevel_verification_lib import HighLevelContext


class ObservationReferenceTests(unittest.TestCase):
    def test_move_captures_gripper_height_and_recomputes_extrema_each_instruction(self):
        env = ServoEnvironment()
        env.obs[10:13] = [0, 0.10, 0.425]
        env.obs[22:25] = [0.20, -0.10, 0.475]
        z = float(env.obs[2])
        move = MoveByName("base", "Y_min", "gripper", target_offset=[0, -0.15, 0])
        move.eval(env, [])
        self.assertTrue(move.last_control_result.converged)
        np.testing.assert_allclose(env.obs[:3], [0, -0.25, z], atol=0.002, rtol=0)
        env.obs[10:13] = [0.03, -0.30, 0.525]
        move.eval(env, [])
        self.assertTrue(move.last_control_result.converged)
        np.testing.assert_allclose(env.obs[:3], [0.03, -0.45, z], atol=0.002, rtol=0)

    def test_release_uses_highest_physical_block_and_keeps_xy(self):
        env = ServoEnvironment()
        env.obs[10:13] = [0, 0, 0.525]
        env.obs[22:25] = [0.2, 0, 0.425]
        xy = env.obs[:2].copy()
        release = ReleaseByName("Z_m", target_z=0.15)
        release.eval(env, [])
        self.assertTrue(release.last_control_result.converged)
        np.testing.assert_allclose(env.obs[:3], [*xy, 0.675], atol=0.002, rtol=0)

    def test_fingerprint_records_semantics_and_normal_loader_rejects_them(self):
        program = Program(1, [MoveByName("base", "Y_min", "gripper")])
        self.assertEqual(
            describe_program(program)[0]["observation_reference"],
            OBSERVATION_REFERENCE_MODE,
        )
        with self.assertRaisesRegex(ValueError, "execution/collection only"):
            prepare_program(program, 2)
        definition = load_program(
            "synthesis.examples.unstack:build_program",
            HighLevelContext(use_tbl=True),
            4,
            allow_observation_references=True,
        )
        self.assertEqual(definition.initial_bindings, {"b0": 0, "tbl": "tbl"})


if __name__ == "__main__":
    unittest.main()
