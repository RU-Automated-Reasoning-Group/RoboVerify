"""partial-reward-v1 on synthetic block layouts (no simulator)."""

import unittest

import numpy as np

from synthesis.util.on import BLOCK_LENGTH

from ..reports.partial_reward import block_positions, rollout_reward

L = BLOCK_LENGTH
REST = 0.425


def column(x, y, blocks):
    """{block: (x, y, z)} for ``blocks`` stacked bottom-up at (x, y)."""
    return {b: (x, y, REST + i * L) for i, b in enumerate(blocks)}


def layout(*columns, carried=None):
    placed = {}
    for c in columns:
        placed |= c
    if carried is not None:
        placed[carried[0]] = carried[1]
    return np.array([placed[b] for b in sorted(placed)])


def reward(task, entry, final, b0=0, states=()):
    return rollout_reward(task, entry, final, b0, states)


class TowerTests(unittest.TestCase):
    def test_stack_counts_blocks_above_b0(self):
        entry = layout(*(column(0.1 * i, 0, [i]) for i in range(4)))
        self.assertEqual(reward("stack", entry, entry), 0.0)
        two = layout(column(0, 0, [0, 2, 3]), column(0.1, 0, [1]))
        self.assertAlmostEqual(reward("stack", entry, two), 2 / 3)
        done = layout(column(0, 0, [0, 3, 1, 2]))
        self.assertEqual(reward("stack", entry, done), 1.0)

    def test_unstack_counts_resting_blocks_other_than_b0(self):
        entry = layout(column(0, 0, [0, 1, 2, 3]))
        self.assertEqual(reward("unstack", entry, entry), 0.0)
        one = layout(column(0, 0, [0, 1, 2]), column(0.1, 0, [3]))
        self.assertAlmostEqual(reward("unstack", entry, one), 1 / 3)
        held = layout(
            column(0, 0, [0, 1, 2]), carried=(3, (0.2, 0, REST + 0.1))
        )  # in the gripper, above nothing
        self.assertEqual(reward("unstack", entry, held), 0.0)
        done = layout(*(column(0.1 * i, 0, [i]) for i in range(4)))
        self.assertEqual(reward("unstack", entry, done), 1.0)

    def test_reverse_counts_the_correct_prefix_of_the_new_tower(self):
        entry = layout(column(0, 0, [0, 1, 2, 3]))  # top t1 = 3
        self.assertEqual(reward("reverse", entry, entry), 0.0)
        first = layout(column(0, 0, [0, 1, 2]), column(0.2, 0, [3]))
        self.assertEqual(reward("reverse", entry, first), 0.25)
        three = layout(column(0, 0, [0]), column(0.2, 0, [3, 2, 1]))
        self.assertEqual(reward("reverse", entry, three), 0.75)
        wrong = layout(column(0, 0, [0, 1]), column(0.2, 0, [3]), column(0.3, 0, [2]))
        self.assertEqual(reward("reverse", entry, wrong), 0.25)
        done = layout(column(0.2, 0, [3, 2, 1, 0]))
        self.assertEqual(reward("reverse", entry, done), 1.0)

    def test_restack_scores_unstacking_over_the_rollout_and_stacking_at_the_end(self):
        # Source tower 1, b0, 2 (bottom-up) plus loose 3 and 4.
        loose = [column(0.2, 0, [3]), column(0.3, 0, [4])]
        entry = layout(column(0, 0, [1, 0, 2]), *loose)
        self.assertEqual(reward("restack", entry, entry), 0.0)
        parked = layout(
            column(0, 0, [1]), column(0.1, 0, [0]), column(0.4, 0, [2]), *loose
        )
        self.assertEqual(reward("restack", entry, parked), 0.5)
        done = layout(column(0.1, 0, [0, 2, 1, 3, 4]))
        # Block 2 ends on b0 again: its unstacking is seen only in the rollout.
        self.assertEqual(reward("restack", entry, done, states=[parked]), 1.0)
        self.assertEqual(reward("restack", entry, done), 0.75)


class GoalTests(unittest.TestCase):
    def test_goal_tasks_use_the_occupied_share(self):
        self.assertEqual(
            rollout_reward("pyramid", None, None, goal_occupied=[1, 1, 0, 0, 0, 1]),
            0.5,
        )


class ObservationTests(unittest.TestCase):
    def test_block_positions_follow_the_observation_layout(self):
        observation = np.arange(10 + 12 * 2, dtype=float)
        self.assertEqual(
            block_positions(observation, 2).tolist(), [[10, 11, 12], [22, 23, 24]]
        )
        self.assertEqual(
            block_positions(np.stack([observation] * 3), 2).shape, (3, 2, 3)
        )


if __name__ == "__main__":
    unittest.main()
