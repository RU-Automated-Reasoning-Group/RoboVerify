"""Rectangular 3D goal geometry and goal-only layer/row/column traversal."""

import copy
import unittest

import numpy as np
from synthesis.cfg.goal_collection import resolve_goal_configuration
from synthesis.environment.goal_layout import GoalLayout, GoalRef
from synthesis.mcmc.synthesis import make_roboverify_build3d_env


class Build3DLayoutTests(unittest.TestCase):
    def test_dimensions_are_independent_and_invalid_shapes_are_rejected(self):
        self.assertEqual(
            resolve_goal_configuration("build3d"),
            (8, dict(build_rows=2, build_cols=2, build_layers=2)),
        )
        for rows, cols, layers in [(1, 3, 2), (2, 3, 2), (2, 2, 3)]:
            count, config = resolve_goal_configuration(
                "build3d", build_rows=rows, build_cols=cols, build_layers=layers
            )
            self.assertEqual(count, rows * cols * layers)
            self.assertEqual(list(config.values()), [rows, cols, layers])
        for task, kwargs in [
            ("build3d", dict(build_layers=0)),
            ("build3d", dict(build_rows=True)),
            ("build3d", dict(build_cols=2.5)),
            ("build3d", dict(num_blocks=7)),
            ("build3d", dict(grid_cols=3)),
            ("grid", dict(build_layers=2)),
        ]:
            with self.subTest(task=task, kwargs=kwargs), self.assertRaises(ValueError):
                resolve_goal_configuration(task, **kwargs)

    def test_layer_links_visit_every_goal_once_and_do_not_cross_row_boundaries(self):
        positions = [
            [0.1 * col, 0.1 * row, 0.05 * layer]
            for layer in range(3)
            for row in range(2)
            for col in range(3)
        ]
        layout = GoalLayout.from_positions(positions, [3] * 6, layer_row_counts=[2] * 3)
        visited = []
        layer = layout.head
        while layer is not None:
            row = layer
            while row is not None:
                column = row
                while column is not None:
                    visited.append(column.index)
                    column = layout.get(column).r
                row = layout.get(row).d
            layer = layout.get(layer).u
        self.assertEqual(visited, list(range(18)))
        self.assertEqual(layout.get(GoalRef(0)).u, GoalRef(6))
        self.assertIsNone(layout.get(GoalRef(3)).d)
        self.assertIsNone(layout.get(GoalRef(1)).u)
        self.assertEqual(GoalLayout.from_dict(layout.as_dict()), layout)
        bad = copy.deepcopy(layout.as_dict())
        bad["u"][0] = 3
        with self.assertRaisesRegex(ValueError, "topology"):
            GoalLayout.from_dict(bad)
        with self.assertRaisesRegex(ValueError, "partition"):
            GoalLayout.from_positions(positions, [3] * 6, layer_row_counts=[2, 3])

    def test_2d_archive_topology_is_unchanged(self):
        layout = GoalLayout.from_positions([[0, 0, 0], [0.1, 0, 0]], [1, 1])
        value = layout.as_dict()
        self.assertEqual(set(value), {"positions", "row_lengths", "head", "r", "d"})
        self.assertEqual(value["d"], [1, None])
        self.assertEqual(GoalLayout.from_dict(value), layout)

    def test_single_row_column_and_layer_keep_the_same_traversal_contract(self):
        for rows, cols, layers in [(1, 3, 2), (3, 1, 2), (2, 3, 1), (1, 1, 1)]:
            with self.subTest(rows=rows, cols=cols, layers=layers):
                positions = [
                    [0.1 * col, 0.1 * row, 0.05 * layer]
                    for layer in range(layers)
                    for row in range(rows)
                    for col in range(cols)
                ]
                layout = GoalLayout.from_positions(
                    positions,
                    [cols] * (rows * layers),
                    layer_row_counts=[rows] * layers,
                )
                visited, layer = [], layout.head
                while layer is not None:
                    row = layer
                    while row is not None:
                        column = row
                        while column is not None:
                            visited.append(column.index)
                            column = layout.get(column).r
                        row = layout.get(row).d
                    layer = layout.get(layer).u
                self.assertEqual(visited, list(range(rows * cols * layers)))
                self.assertEqual(GoalLayout.from_dict(layout.as_dict()), layout)

    def test_simulator_has_rectangular_layers_and_fixed_goal_markers(self):
        state = np.random.get_state()
        self.addCleanup(np.random.set_state, state)
        env = make_roboverify_build3d_env(2, 3, 2)
        self.addCleanup(env.close)
        obs, _ = env.reset()
        inner = env.env
        self.assertEqual(inner.num_blocks, 12)
        self.assertEqual(inner.goal_layout.row_lengths, (3, 3, 3, 3))
        self.assertEqual(inner.goal_layout.layer_row_counts, (2, 2))
        goals = inner.goal_layout.positions.reshape(2, 2, 3, 3)
        np.testing.assert_allclose(goals[1, :, :, :2], goals[0, :, :, :2])
        np.testing.assert_allclose(
            goals[1, :, :, 2] - goals[0, :, :, 2], 0.05, atol=1e-7
        )
        np.testing.assert_allclose(np.diff(goals[..., 0], axis=2), 0.1, atol=1e-7)
        np.testing.assert_allclose(np.diff(goals[..., 1], axis=1), 0.1, atol=1e-7)
        self.assertEqual(inner.symbolic_name_to_box_id, {})
        self.assertEqual(inner.symbolic_name_to_goal, {"h0": GoalRef(0), "null": None})
        self.assertFalse(inner._is_success(obs))
        env.step(np.array([0.1, -0.1, 0, 0]))
        for index, goal in enumerate(inner.goal_layout.goals):
            body = inner.sim.model.body_name2id(f"grid_marker{index}")
            mocap = inner.sim.model.body_mocapid[body]
            np.testing.assert_array_equal(
                inner.sim.data.mocap_pos[mocap], goal.position
            )


if __name__ == "__main__":
    unittest.main()
