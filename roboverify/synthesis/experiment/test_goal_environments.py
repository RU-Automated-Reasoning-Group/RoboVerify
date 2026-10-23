"""Simulator regression checks for goal layouts and the supplied nested program."""

import unittest

import numpy as np

from synthesis.entry.execute_goal_program import execute
from synthesis.environment.goal_layout import GoalRef
from synthesis.mcmc.synthesis import (
    make_roboverify_grid_env,
    make_roboverify_pyramid_env,
)
from synthesis.util.on import scattered_implementation


class GoalEnvironmentTests(unittest.TestCase):
    def setUp(self):
        state = np.random.get_state()
        self.addCleanup(np.random.set_state, state)

    def test_configurable_layers_reset_graph_and_markers_survive_actions(self):
        for bottom in (1, 3, 4):
            env = make_roboverify_pyramid_env(bottom_row_blocks=bottom)
            try:
                inner = env.env
                np.random.seed(0)
                obs, _ = env.reset()
                self.assertEqual(inner.num_blocks, bottom * (bottom + 1) // 2)
                self.assertEqual(
                    inner.goal_layout.row_lengths, tuple(range(bottom, 0, -1))
                )
                self.assertEqual(inner.sim.model.nmocap, inner.num_blocks + 1)
                markers = inner.goal_layout.positions.copy()
                for i in range(inner.num_blocks):
                    for j in range(i):
                        self.assertTrue(
                            scattered_implementation(
                                obs[10 + 12 * i : 13 + 12 * i],
                                obs[10 + 12 * j : 13 + 12 * j],
                            )
                        )
                env.step(np.array([0.1, -0.1, 0, 0]))
                for i in range(inner.num_blocks):
                    mocap = inner.sim.model.body_mocapid[
                        inner.sim.model.body_name2id(f"grid_marker{i}")
                    ]
                    np.testing.assert_array_equal(
                        inner.sim.data.mocap_pos[mocap], markers[i]
                    )
                inner.symbolic_name_to_goal["r"] = GoalRef(0)
                inner.symbolic_name_to_box_id["b_prime"] = 0
                env.reset()
                self.assertEqual(
                    inner.symbolic_name_to_goal, {"h0": GoalRef(0), "null": None}
                )
                self.assertEqual(inner.symbolic_name_to_box_id, {})
            finally:
                env.close()

    def test_success_accepts_permutation_and_restore_reinstalls_goal_coordinates(self):
        env = make_roboverify_grid_env(grid_rows=2, grid_cols=3)
        try:
            inner = env.env
            env.reset()
            target = np.r_[inner.goal_layout.positions.ravel(), [0, 0, 0]]
            achieved = np.r_[inner.goal_layout.positions[::-1].ravel(), [0, 0, 0]]
            self.assertTrue(inner._roboverify_goal_layout_complete(achieved, target))
            batch = np.stack([achieved, achieved + 0.03])
            np.testing.assert_array_equal(
                inner._roboverify_goal_layout_complete(
                    batch, np.stack([target, target])
                ),
                [True, False],
            )
            saved = inner.get_GT_state()
            shifted = saved.copy()
            shifted[-inner.goal_space_size : -3 : 3] += 0.01
            inner.set_GT_state(shifted)
            np.testing.assert_allclose(
                inner.goal_layout.positions[:, 0],
                target[:-3].reshape(-1, 3)[:, 0] + 0.01,
            )
            inner.set_GT_state(saved)
            np.testing.assert_array_equal(
                inner.goal_layout.positions, target[:-3].reshape(-1, 3)
            )
        finally:
            env.close()

    def test_grid_and_pyramids_execute_each_transfer_and_exit_both_loops(self):
        cases = [
            ("grid", {}),
            ("pyramid", {"bottom_row_blocks": 3}),
            ("pyramid", {"bottom_row_blocks": 4}),
        ]
        for task, kwargs in cases:
            for seed in (0, 1):
                with self.subTest(task=task, seed=seed, **kwargs):
                    result = execute(task, seed=seed, **kwargs)
                    self.assertTrue(result["success"])
                    count = result["num_blocks"]
                    self.assertEqual(result["primitives"], 6 * count)
                    self.assertEqual(
                        [p["goal"] for p in result["placements"]], list(range(count))
                    )
                    self.assertEqual(
                        len({p["block"] for p in result["placements"]}), count
                    )
                    self.assertEqual(
                        [
                            e["iterations"]
                            for e in result["loop_exits"]
                            if e["path"] == "1.1"
                        ],
                        result["row_lengths"],
                    )


if __name__ == "__main__":
    unittest.main()
