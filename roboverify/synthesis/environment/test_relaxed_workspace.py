"""The opt-in relaxed reset keeps task geometry inside the reachable workspace."""

import unittest
from unittest.mock import patch

import numpy as np

from synthesis.environment.relaxed_workspace import (
    RELAXED_LATERAL_MIN_Y,
    RELAXED_LATERAL_TOWER_Y_RANGE,
    RELAXED_SOURCE_SEPARATION,
    RELAXED_STACK_BASE_Y_RANGE,
    RELAXED_TOWER_X_RANGE,
    check_relaxed_goals,
    in_relaxed_workspace,
    relaxed_goal_origin_x,
    relaxed_goal_origin_y,
    sample_relaxed_goal_sources,
)
from synthesis.environment.restack_reset import (
    RESTACK_SEPARATION,
    sample_restack_layout,
)
from synthesis.environment.stack_reset import sample_stack_xy
from synthesis.environment.tower_reset import sample_tower_xy
from synthesis.util.on import scattered_implementation

BASE = np.array([0.69189994, 0.74409998])
GRIPPER = BASE + [0.652, 0.005]


def within(value, bounds):
    return bounds[0] - 1e-9 <= value <= bounds[1] + 1e-9


class RelaxedWorkspaceTests(unittest.TestCase):
    def setUp(self):
        state = np.random.get_state()
        self.addCleanup(np.random.set_state, state)

    def assert_scattered(self, xy, separation):
        xy = np.asarray(xy)
        for i in range(len(xy)):
            for j in range(i):
                self.assertTrue(np.any(np.abs(xy[i] - xy[j]) >= separation - 1e-9))

    def assert_reachable(self, xy):
        for p in np.asarray(xy):
            self.assertTrue(in_relaxed_workspace(p - BASE), p - BASE)
            self.assertGreaterEqual(np.linalg.norm(p - GRIPPER), 0.10 - 1e-9)

    def test_default_stack_draws_are_unchanged(self):
        # Positions of the sampler before the option existed, for seed 73.
        np.random.seed(73)
        positions = sample_stack_xy(4, BASE, GRIPPER)
        np.testing.assert_allclose(
            positions,
            [
                [1.310778, 0.629885],
                [1.27053, 0.906388],
                [1.237347, 0.757132],
                [1.378809, 0.857166],
            ],
            atol=1e-6,
        )

    def test_relaxed_stack_fits_eighteen_blocks_with_b0_at_a_tower_site(self):
        for seed in range(20):
            for base_index in (0, 5):
                with self.subTest(seed=seed, base_index=base_index):
                    np.random.seed(seed)
                    xy = sample_stack_xy(
                        18,
                        BASE,
                        GRIPPER,
                        relax_boundaries_for_generalization_test=True,
                        base_index=base_index,
                    )
                    self.assertEqual(len(xy), 18)
                    self.assert_reachable(xy)
                    for i, p in enumerate(xy):
                        for q in xy[:i]:
                            self.assertTrue(
                                scattered_implementation(
                                    np.r_[p, 0.425], np.r_[q, 0.425]
                                )
                            )
                    b0 = np.asarray(xy[base_index]) - BASE
                    self.assertTrue(within(b0[0], RELAXED_TOWER_X_RANGE))
                    self.assertTrue(within(b0[1], RELAXED_STACK_BASE_Y_RANGE))

    def test_relaxed_stack_is_seeded(self):
        np.random.seed(11)
        first = sample_stack_xy(
            8, BASE, GRIPPER, relax_boundaries_for_generalization_test=True
        )
        np.random.seed(11)
        again = sample_stack_xy(
            8, BASE, GRIPPER, relax_boundaries_for_generalization_test=True
        )
        np.testing.assert_array_equal(first, again)

    def test_relaxed_tower_and_restack_sources_stay_on_the_higher_y_side(self):
        for seed in range(20):
            np.random.seed(seed)
            tower = (
                sample_tower_xy(
                    18,
                    BASE,
                    task="reverse",
                    relax_boundaries_for_generalization_test=True,
                )
                - BASE
            )
            self.assertTrue(within(tower[0], RELAXED_TOWER_X_RANGE))
            self.assertTrue(within(tower[1], RELAXED_LATERAL_TOWER_Y_RANGE))
            xy, heights = sample_restack_layout(
                16, 8, 0, BASE, GRIPPER, relax_boundaries_for_generalization_test=True
            )
            self.assertEqual(sorted(heights.tolist()), [0] * 8 + list(range(8)))
            self.assertIn(heights[0], range(1, 7))
            columns = np.unique(xy.round(9), axis=0)
            self.assertEqual(len(columns), 9)
            self.assert_reachable(columns)
            self.assert_scattered(columns, RESTACK_SEPARATION)
            self.assertTrue(np.all(columns[:, 1] - BASE[1] >= RELAXED_LATERAL_MIN_Y))

    def test_default_restack_still_refuses_three_loose_blocks(self):
        with self.assertRaisesRegex(ValueError, "cannot fit this many loose blocks"):
            sample_restack_layout(6, 3, 0, BASE, GRIPPER)

    def assert_clear_of_goals(self, sources, goals):
        for p in np.asarray(sources)[:, :2]:
            for g in np.asarray(goals)[:, :2]:
                self.assertGreaterEqual(
                    np.abs(p - g).max(), RELAXED_SOURCE_SEPARATION - 1e-6
                )

    def test_goal_sources_and_layouts(self):
        goals = BASE + np.array(
            [[0.35 + 0.1 * c, -0.15 + 0.1 * r] for r in range(4) for c in range(4)]
        )
        np.random.seed(3)
        sources = sample_relaxed_goal_sources(16, BASE, GRIPPER, goals)
        self.assert_reachable(sources)
        self.assert_scattered(sources, RELAXED_SOURCE_SEPARATION)
        self.assert_clear_of_goals(sources, goals)
        # Sources may now lie on both sides of the goals.
        ys = np.asarray(sources)[:, 1] - BASE[1]
        self.assertTrue(ys.min() < -0.15 and ys.max() > 0.15)
        self.assertAlmostEqual(relaxed_goal_origin_y(BASE[1], 0.3) - BASE[1], -0.15)
        self.assertAlmostEqual(relaxed_goal_origin_x(BASE[0], 0.2) - BASE[0], 0.40)
        self.assertAlmostEqual(relaxed_goal_origin_x(BASE[0], 0.4) - BASE[0], 0.35)
        check_relaxed_goals(BASE + np.array([[0.35, 0.1], [0.65, 0.4]]), BASE)
        with self.assertRaisesRegex(ValueError, "relaxed reachable workspace"):
            check_relaxed_goals(BASE + np.array([[0.75, 0.3]]), BASE)

    def test_jammed_sampling_falls_back_to_a_scattered_lattice(self):
        with patch(
            "synthesis.environment.relaxed_workspace.np.random.uniform",
            return_value=np.array([0.5, -0.3]),
        ):
            sources = sample_relaxed_goal_sources(6, BASE, GRIPPER, [BASE + [0.5, 0.0]])
        self.assertEqual(len(sources), 6)
        self.assert_reachable(sources)
        self.assert_scattered(sources, RELAXED_SOURCE_SEPARATION)
        self.assert_clear_of_goals(sources, [BASE + [0.5, 0.0]])
        with self.assertRaisesRegex(ValueError, "lattice capacity"):
            sample_relaxed_goal_sources(200, BASE, GRIPPER, [])

    def test_simulator_resets(self):
        from synthesis.cfg.goal_collection import make_goal_env
        from synthesis.mcmc.synthesis import make_roboverify_env

        env = make_roboverify_env(
            "stack", 12, relax_boundaries_for_generalization_test=True
        )
        self.addCleanup(env.close)
        np.random.seed(5)
        obs, _ = env.reset()
        xyz = np.array([obs[10 + 12 * i : 13 + 12 * i] for i in range(12)])
        np.testing.assert_allclose(xyz[:, 2], env.env.height_offset, atol=1e-7)
        self.assert_reachable(xyz[:, :2])
        self.assertEqual(env.env.symbolic_name_to_box_id, {"b0": 0})
        self.assertTrue(within(xyz[0, 0] - BASE[0], RELAXED_TOWER_X_RANGE))

        grid = make_goal_env(
            "grid",
            dict(
                grid_rows=4, grid_cols=4, relax_boundaries_for_generalization_test=True
            ),
        )
        self.addCleanup(grid.close)
        np.random.seed(5)
        obs, _ = grid.reset()
        goals = grid.env.goal_layout.positions[:, :2]
        self.assertTrue(all(in_relaxed_workspace(g - BASE) for g in goals))
        # Rows are centred on the robot's midline.
        self.assertAlmostEqual(float(goals[:, 1].mean() - BASE[1]), 0.0, places=5)
        blocks = np.array([obs[10 + 12 * i : 12 + 12 * i] for i in range(16)])
        self.assert_reachable(blocks)
        self.assert_clear_of_goals(blocks, goals)

    def test_contact_buffers_are_raised_only_when_relaxed(self):
        from synthesis.environment.relaxed_workspace import RELAXED_CONTACT_BUFFER
        from synthesis.mcmc.synthesis import make_roboverify_env

        for relaxed, sizes in ((False, (100, 500)), (True, RELAXED_CONTACT_BUFFER)):
            env = make_roboverify_env(
                "unstack", 4, relax_boundaries_for_generalization_test=relaxed
            )
            self.addCleanup(env.close)
            model = env.env.sim.model
            self.assertEqual((model.nconmax, model.njmax), sizes)

    def test_default_goal_layout_unchanged_and_relaxed_rejects_far_goals(self):
        from synthesis.mcmc.synthesis import make_roboverify_grid_env

        default = make_roboverify_grid_env(3, 2)
        self.addCleanup(default.close)
        np.random.seed(0)
        default.reset()
        np.testing.assert_allclose(
            default.env.goal_layout.positions[0, :2] - BASE,
            [GRIPPER[0] - BASE[0] - 0.05, 0.10],
            atol=2e-3,
        )
        # Seven columns span 0.60 m, beyond the region's 0.35-0.80 m in X.
        far = make_roboverify_grid_env(
            1, 7, relax_boundaries_for_generalization_test=True
        )
        self.addCleanup(far.close)
        with self.assertRaisesRegex(ValueError, "relaxed reachable workspace"):
            far.reset()


if __name__ == "__main__":
    unittest.main()
