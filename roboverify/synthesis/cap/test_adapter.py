"""Shared predicate/controller parity on a deterministic non-MuJoCo fixture."""

import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
from synthesis.cfg.reset import capture, restore
from synthesis.predicates.scene import evaluate, scene_from_obs
from synthesis.predicates.term import atom, block_id

from .adapter import RoboVerifyAdapter
from .budget import Budget
from .contracts import CapError, ObjectHandle, PrimitiveFailure
from .logger import CapLogger


class RobotFixture:
    """Minimal action/state fixture, not a physics model or benchmark evaluator."""

    def __init__(self, count=2, *, success=False, stuck=False):
        self.env = self.unwrapped = self
        self.num_blocks, self.roboverify_base_block_id = count, 0
        self.symbolic_name_to_box_id = {"b0": 0}
        self.obs = np.zeros(13 + 15 * count)
        self.obs[:3] = [0, 0, 0.2]
        self.obs[3:5] = [0.04, 0.04]
        for i in range(count):
            self.obs[10 + 12 * i : 13 + 12 * i] = [0.2 * i, 0, 0.025]
        self.sim = SimpleNamespace(
            data=SimpleNamespace(
                ctrl=np.zeros(2),
                mocap_pos=np.zeros((1, 3)),
                mocap_quat=np.zeros((1, 4)),
                qacc_warmstart=np.zeros(2),
            ),
            forward=lambda: None,
        )
        self.success, self.stuck, self.closed = success, stuck, False
        self.actions = []

    def _get_obs(self):
        split = 10 + 12 * self.num_blocks
        return {
            "observation": self.obs[:split].copy(),
            "desired_goal": self.obs[split:].copy(),
        }

    def flatten_observation(self, obs):
        return np.r_[obs["observation"], obs["desired_goal"]]

    def step(self, action):
        self.actions.append(np.array(action).copy())
        if not self.stuck:
            self.obs[:3] += np.asarray(action[:3]) * 0.05
            self.obs[3:5] = np.clip(self.obs[3:5] + action[3] * 0.05, 0, 0.04)
        return self.obs.copy(), 0.0, False, False, {"is_success": self.success}

    def get_GT_state(self):
        return self.obs.copy()

    def set_GT_state(self, state):
        self.obs = np.array(state).copy()

    def _is_success(self, obs):
        return self.success

    def close(self):
        self.closed = True

    def render(self, mode="rgb_array"):
        return np.full((32, 32, 3), len(self.actions) % 256, dtype=np.uint8)


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.logger = CapLogger(self.tmp.name, console=False)
        self.env = RobotFixture()
        self.adapter = RoboVerifyAdapter(
            self.env, self.logger, Budget(), task="reverse"
        )

    def tearDown(self):
        self.logger.close()
        self.tmp.cleanup()

    def test_predicates_match_shared_evaluator_including_frozen_history(self):
        self.env.obs[22:25] = [0, 0, 0.075]
        scene = scene_from_obs(
            self.env.obs, 2, entry_obs=self.adapter.entry_observation
        )
        for predicate in ["ON", "ON_star", "Higher", "Scattered", "ON_star_zero", "eq"]:
            for a in range(2):
                for b in range(2):
                    with self.subTest(predicate=predicate, a=a, b=b):
                        expected = evaluate(
                            atom(predicate, block_id(a), block_id(b)), scene
                        )
                        actual = self.adapter.call(
                            predicate,
                            [self.adapter.handles[a], self.adapter.handles[b]],
                        )
                        self.assertEqual(actual, expected)
        self.assertTrue(
            self.adapter.call("ON_star", [self.adapter.handles[1], self.adapter.b0])
        )
        self.assertFalse(
            self.adapter.call(
                "ON_star_zero", [self.adapter.handles[1], self.adapter.b0]
            )
        )

    def test_query_order_and_no_witness(self):
        self.assertEqual(self.adapter.call("get", [lambda x: True]), self.adapter.b0)
        with self.assertRaises(CapError):
            self.adapter.call("get", [lambda x: False])
        with self.assertRaises(CapError):
            self.adapter.call("exists", [lambda x: self.adapter.call("Pick", [x])])
        self.assertEqual(self.env.actions, [])

    def test_handles_do_not_cross_trials(self):
        with self.assertRaises(CapError):
            self.adapter.call(
                "Higher", [ObjectHandle("b0", "another"), self.adapter.b0]
            )

    def test_primitive_wrapper_matches_existing_controller_actions(self):
        from synthesis.api.instructions import MoveByName, PickByName, ReleaseByName

        start = capture(self.env)
        for name, args in [
            ("Pick", [self.adapter.b0]),
            ("Move", [self.adapter.b0] * 3 + [[0, 0, 0.1]]),
            ("Release", [self.adapter.b0, 0.2]),
        ]:
            self.adapter.call(name, args)
        actual = np.array(self.env.actions)
        restore(self.env, start)
        self.env.actions.clear()
        trajectory = [self.env.obs.copy()]
        for instruction in [
            PickByName("b0"),
            MoveByName("b0", "b0", "b0", target_offset=[0, 0, 0.1]),
            ReleaseByName("b0", target_z=0.2),
        ]:
            instruction.eval(self.env, trajectory)
            self.assertTrue(instruction.last_control_result.converged)
        np.testing.assert_allclose(actual, self.env.actions)

    def test_primitive_failure_is_not_boolean_feedback(self):
        self.env.stuck = True
        with self.assertRaises(PrimitiveFailure):
            self.adapter.call("Pick", [self.adapter.b0])


if __name__ == "__main__":
    unittest.main()
