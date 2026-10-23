"""The production Unstack reset uses the lateral workspace for every caller."""

import unittest

import numpy as np

from synthesis.cfg.reset import inner_env
from synthesis.environment.cee_us_env.fpp_construction_env import (
    FetchPickAndPlaceConstruction,
)
from synthesis.experiment.unstack_lateral import lateral_env
from synthesis.mcmc.synthesis import (
    ROBOVERIFY_UNSTACK_ENV_KWARGS,
    make_roboverify_env,
    make_roboverify_unstack_env,
)
from synthesis.util.on import BLOCK_LENGTH, get_block_pos


class UnstackResetTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(np.random.set_state, np.random.get_state())

    def assert_lateral_tower(self, env, obs, count):
        inner = inner_env(env)
        xyz = np.array([get_block_pos(obs, i) for i in range(count)])
        order = inner._roboverify_canonical_tower_order()
        offsets = xyz[:, :2] - inner.robot_base_xy
        self.assertTrue(np.all(offsets[:, 0] >= 0.54))
        self.assertTrue(np.all(offsets[:, 0] <= 0.70))
        self.assertTrue(np.all(offsets[:, 1] >= 0.15))
        self.assertTrue(np.all(offsets[:, 1] <= 0.25))
        np.testing.assert_array_equal(xyz[:, :2], np.tile(xyz[0, :2], (count, 1)))
        np.testing.assert_array_equal(
            xyz[order, 2],
            (inner.height_offset + BLOCK_LENGTH * np.arange(count)).astype(xyz.dtype),
        )
        self.assertEqual(order[0], inner.symbolic_name_to_box_id["b0"])
        self.assertFalse(inner._roboverify_task_complete(xyz))
        row = xyz.copy()
        row[:, 1] -= 0.15 * np.arange(count)
        row[:, 2] = inner.height_offset
        self.assertTrue(inner._roboverify_task_complete(row))

    def test_all_factories_use_the_same_seeded_lateral_layout(self):
        self.assertIs(lateral_env, make_roboverify_unstack_env)
        for count in (2, 3, 4):
            expected = {}
            for factory in (
                make_roboverify_unstack_env,
                lambda num_blocks: make_roboverify_env("unstack", num_blocks),
            ):
                env = factory(num_blocks=count)
                try:
                    for seed in (0, 4, 73):
                        with self.subTest(count=count, factory=factory, seed=seed):
                            np.random.seed(seed)
                            obs, _ = env.reset()
                            if seed in expected:
                                np.testing.assert_array_equal(obs, expected[seed])
                            else:
                                expected[seed] = obs.copy()
                            self.assert_lateral_tower(env, obs, count)
                    self.assertFalse(np.array_equal(expected[0], expected[4]))
                    sampled = np.array(
                        [get_block_pos(o, 0)[:2] for o in expected.values()]
                    )
                    sampled -= inner_env(env).robot_base_xy
                    self.assertGreater(sampled[:, 0].max(), 0.65)
                    self.assertGreater(sampled[:, 1].max(), 0.20)
                finally:
                    env.close()

    def test_direct_environment_preserves_nonzero_base_binding(self):
        env = FetchPickAndPlaceConstruction(
            name="unstack_nonzero_base",
            num_blocks=4,
            **dict(ROBOVERIFY_UNSTACK_ENV_KWARGS, base_block_id=2),
        )
        self.addCleanup(env.close)
        np.random.seed(0)
        obs = env.reset()
        self.assert_lateral_tower(env, obs, 4)
        self.assertEqual(env.symbolic_name_to_box_id, {"b0": 2})


if __name__ == "__main__":
    unittest.main()
