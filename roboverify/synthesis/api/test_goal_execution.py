"""Goal typing, geometric guards and nested traversal through Program.eval."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import z3

from synthesis.api.guard_eval import evaluate_z3
from synthesis.api.instructions import (
    GoalAssign,
    MoveByName,
    MoveDown,
    MoveRight,
    PickByName,
    ReleaseByName,
)
from synthesis.api.test_control import ServoEnvironment
from synthesis.environment.goal_layout import (
    GoalLayout,
    GoalRef,
    occupancy,
    pyramid_layers,
)
from synthesis.examples.goal_layout import build_program
from synthesis.predicates.scene import Scene


class GoalExecutionTests(unittest.TestCase):
    def test_graph_visits_every_goal_once_and_only_heads_have_down_links(self):
        for rows in ((1,), (2, 2, 2), pyramid_layers(3), pyramid_layers(4)):
            xyz = [[i * 0.1, 0, 0.425] for i in range(sum(rows))]
            layout = GoalLayout.from_positions(xyz, rows)
            visited, lengths, row = [], [], layout.head
            while row is not None:
                goal, length = row, 0
                while goal is not None:
                    visited.append(goal.index)
                    length += 1
                    if goal != row:
                        self.assertIsNone(layout.get(goal).d)
                    goal = layout.get(goal).r
                lengths.append(length)
                row = layout.get(row).d
            self.assertEqual(visited, list(range(sum(rows))))
            self.assertEqual(lengths, list(rows))
            with self.assertRaises(ValueError):
                layout.get(None)
        for invalid in (0, -1, 3.5, True, "4"):
            with self.assertRaises(ValueError):
                pyramid_layers(invalid)

    def test_occupancy_is_identity_independent_and_regions_are_disjoint(self):
        goals = np.array([[0, 0, 0.425], [0.1, 0, 0.425]])
        self.assertTrue(occupancy(goals[::-1], goals).any(axis=0).all())
        self.assertFalse(occupancy(goals + [0, 0.03, 0], goals).any())
        with self.assertRaisesRegex(ValueError, "disjoint"):
            GoalLayout.from_positions([[0, 0, 0], [0.03, 0, 0]], [2])

    def test_goal_and_block_with_index_zero_are_distinct_coordinate_references(self):
        env = ServoEnvironment()
        env.goal_layout = GoalLayout.from_positions([[0.3, 0.2, 0.5]], [1])
        env.symbolic_name_to_goal = {"h0": GoalRef(0), "null": None}
        GoalAssign("r", "h0").eval(env, [])
        move = MoveByName("other", "r", "r", target_offset=[0, 0, 0.15])
        move.eval(env, [])
        self.assertTrue(move.last_control_result.converged)
        np.testing.assert_allclose(env.obs[:3], [0.2, 0.2, 0.65], atol=0.002, rtol=0)
        with self.assertRaises(KeyError):
            PickByName("r").eval(env, [])
        MoveRight("r").eval(env, [])
        with self.assertRaisesRegex(ValueError, "null"):
            move.eval(env, [])
        with self.assertRaises(ValueError):
            MoveDown("h0").eval(env, [])

    def test_mixed_goal_block_guards_use_geometry_and_separate_quantifier_domains(self):
        box, goal = z3.DeclareSort("Box"), z3.DeclareSort("Goal")
        b, g, null = z3.Const("b", box), z3.Const("g", goal), z3.Const("null", goal)
        at = z3.Function("AtGoal", box, z3.BoolSort())
        has = z3.Function("HasBlock", goal, z3.BoolSort())
        layout = GoalLayout.from_positions([[0, 0, 0.425]], [1])
        scene = Scene({0: [0, 0, 0.425], 1: [0.2, 0, 0.425]})
        kwargs = dict(goal_layout=layout, goal_bindings={"g": GoalRef(0), "null": None})
        self.assertTrue(evaluate_z3(z3.And(has(g), at(b)), scene, {"b": 0}, **kwargs))
        self.assertFalse(evaluate_z3(at(b), scene, {"b": 1}, **kwargs))
        self.assertTrue(
            evaluate_z3(z3.ForAll([g], z3.Or(g == null, has(g))), scene, **kwargs)
        )
        self.assertFalse(evaluate_z3(z3.ForAll([b], at(b)), scene, **kwargs))

    def test_same_nested_program_traverses_rectangular_and_triangular_layouts(self):
        for rows in ((2, 2, 2), (4, 3, 2, 1)):
            count = sum(rows)
            obs = np.zeros(13 + 15 * count)
            for i in range(count):
                obs[10 + 12 * i : 13 + 12 * i] = [0.1 * i, -0.3, 0.425]
            layout = GoalLayout.from_positions(
                [[0.1 * i, 0.3, 0.425] for i in range(count)], rows
            )
            env = SimpleNamespace(
                num_blocks=count,
                symbolic_name_to_box_id={},
                symbolic_name_to_goal={"h0": layout.head, "null": None},
                goal_layout=layout,
            )
            env.env = env
            selected, events = [], []

            def release(instruction, current, trajectory, return_image=False):
                block = current.symbolic_name_to_box_id["b_prime"]
                goal = current.symbolic_name_to_goal["r"]
                selected.append((block, goal.index))
                obs[10 + 12 * block : 13 + 12 * block] = layout.get(goal).position
                trajectory.append(obs.copy())
                return []

            with patch.object(MoveByName, "eval", return_value=[]), patch.object(
                PickByName, "eval", return_value=[]
            ), patch.object(ReleaseByName, "eval", release):
                build_program(num_blocks=count).eval(
                    env, initial_observation=obs.copy(), on_event=events.append
                )
            self.assertEqual(selected, list(zip(range(count), range(count))))
            self.assertIsNone(env.symbolic_name_to_goal["d"])
            self.assertEqual(
                [
                    e["iteration"]
                    for e in events
                    if e["kind"] == "loop_exit" and e["path"] == "1.1"
                ],
                list(rows),
            )
            self.assertEqual(env.symbolic_name_to_box_id, {"b_prime": count - 1})


if __name__ == "__main__":
    unittest.main()
