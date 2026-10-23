"""Paired physical diagnostics use exact starts and remain scoped to experiments."""

import unittest
from types import SimpleNamespace

import numpy as np

from synthesis.api.control import PrimitiveController
from synthesis.experiment.tower_workspace import reset_probe, run_trial


class TowerWorkspaceTests(unittest.TestCase):
    def test_reset_probe_detects_centerline_overlap_and_clear_side(self):
        args = SimpleNamespace(task="reverse", num_blocks=4)
        center = reset_probe(args, (0.66, 0.0), 0)
        side = reset_probe(args, (0.66, 0.15), 0)
        self.assertTrue(center["geometry"]["robot_block_contacts"])
        self.assertEqual(side["geometry"]["robot_block_contacts"], [])

    def test_paired_feedback_uses_identical_physics_and_distinct_identity(self):
        args = SimpleNamespace(task="reverse", num_blocks=4)
        # Keep the known off-center grasp independent of reset sampling policy.
        baseline, trace = run_trial(args, seed=0, point=(0.56, 0.15))
        replay, replay_trace = run_trial(args, seed=0, snapshot=trace.snapshots[0])
        self.assertTrue(baseline["valid"], baseline["reason"])
        self.assertTrue(replay["valid"], replay["reason"])
        np.testing.assert_allclose(trace.states, replay_trace.states, atol=1e-8, rtol=0)
        np.testing.assert_allclose(
            trace.actions[0], replay_trace.actions[0], atol=1e-8, rtol=0
        )
        original = PrimitiveController.move_relative
        row, intervention = run_trial(
            args, seed=0, snapshot=trace.snapshots[0], variant="payload_xy"
        )
        self.assertIs(PrimitiveController.move_relative, original)
        self.assertTrue(row["valid"], row["reason"])
        np.testing.assert_array_equal(
            trace.snapshots[0].gt_state, intervention.snapshots[0].gt_state
        )
        for name, value in trace.snapshots[0].arrays.items():
            np.testing.assert_array_equal(value, intervention.snapshots[0].arrays[name])
        self.assertNotEqual(
            trace.metadata["fingerprint"], intervention.metadata["fingerprint"]
        )
        self.assertIn("experimental_controller", intervention.metadata)
        self.assertGreater(baseline["adjacent_xy_mm"][0], 8)
        self.assertLess(row["adjacent_xy_mm"][0], 3)


if __name__ == "__main__":
    unittest.main()
