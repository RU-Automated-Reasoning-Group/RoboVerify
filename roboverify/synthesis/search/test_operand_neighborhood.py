"""Reference-policy proposals preserve complete nominal contracts and budgets."""

import time
import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np

from synthesis.search.contract import (
    Action,
    ContractConfig,
    ContractInputError,
    ContractProblem,
    Node,
    SearchExample,
)
from synthesis.search.operand_neighborhood import (
    OperandNeighborhoodConfig,
    expand_operand_neighborhood,
    select_reference_diverse,
)


def problem_and_seed():
    positions = np.array([[0.2, 0.1, 0.425], [0, 0, 0.475], [0, 0, 0.425]])
    gripper = np.array([0.2, -0.2, 0.7])
    path = np.array(
        [
            gripper,
            positions[0],
            [0.2, 0.1, 0.625],
            [0, 0, 0.625],
            [0, 0, 0.525],
            [0, 0, 0.625],
        ]
    )
    object_path = np.repeat(positions[None], len(path), axis=0)
    object_path[2:, 0] = path[2:]
    object_path[-1, 0] = path[-2]
    example = SearchExample(
        positions, gripper, {"item": 0, "support": 1, "anchor": 2}, path, object_path
    )
    problem = ContractProblem(
        [
            example,
            replace(
                example,
                positions=positions + [0.3, 0.1, 0],
                gripper=gripper + [0.3, 0.1, 0],
                gripper_path=path + [0.3, 0.1, 0],
                positions_path=object_path + [0.3, 0.1, 0],
            ),
        ],
        lambda state, i: bool(
            np.allclose(state.positions[i, 0], state.positions[i, 1] + [0, 0, 0.05])
        ),
        config=ContractConfig(
            observation_references=False, observation_proposals=False
        ),
    )
    actions = (
        Action("Pick", ("item",)),
        Action("Move", ("item",) * 3, (0, 0, 0.2)),
        Action("Move", ("support", "support", "item"), (0, 0, 0)),
        Action("Move", ("support",) * 3, (0, 0, 0.05)),
        Action("Release", ("support",), (0.15,)),
    )
    seed = problem.replay(actions)
    assert seed is not None and problem.goal(seed.state)
    return problem, seed


class OperandNeighborhoodTests(unittest.TestCase):
    def test_all_candidates_replay_complete_goal_and_keep_offsets_and_kinds(self):
        problem, seed = problem_and_seed()
        result = expand_operand_neighborhood(
            problem, [seed], OperandNeighborhoodConfig(relative_slack=1, pool_limit=20)
        )
        self.assertGreater(result.stats["goal_candidates"], 1)
        self.assertGreater(result.stats["nominal_rejected"], 0)
        self.assertTrue(result.candidates)
        for node in result.candidates:
            self.assertTrue(problem.goal(problem.replay(node.actions).state))
            self.assertAlmostEqual(node.score, problem.rank(node, complete=True))
            self.assertEqual(
                [(a.kind, a.offsets) for a in node.actions],
                [(a.kind, a.offsets) for a in seed.actions],
            )
            self.assertLessEqual(
                sum(
                    a.operands != b.operands for a, b in zip(node.actions, seed.actions)
                ),
                1,
            )
        self.assertTrue(
            any(
                node.actions[3].operands == ("anchor", "anchor", "support")
                for node in result.candidates
            )
        )

    def test_numeric_operands_need_no_role_names_or_observation_references(self):
        named, seed = problem_and_seed()
        problem = ContractProblem(
            named.examples,
            named.goal_callback,
            config=named.config,
            allowed_operands=(0, 1, 2),
        )
        bindings = named.examples[0].bindings
        numeric = problem.replay(
            tuple(
                Action(
                    action.kind,
                    tuple(bindings[ref] for ref in action.operands),
                    action.offsets,
                )
                for action in seed.actions
            )
        )
        result = expand_operand_neighborhood(problem, [numeric])
        self.assertTrue(result.candidates)
        for node in result.candidates:
            self.assertTrue(problem.goal(node.state))
            self.assertTrue(
                all(
                    type(ref) is int
                    for action in node.actions
                    for ref in action.operands
                )
            )

    def test_only_original_full_goals_can_seed_neighborhood(self):
        problem, seed = problem_and_seed()
        partial = problem.replay(seed.actions[:2])
        result = expand_operand_neighborhood(problem, [partial])
        self.assertEqual(result.candidates, ())
        self.assertEqual(result.stats["goal_rejected"], 1)

    def test_candidate_budget_deduplicates_and_returns_checked_seed(self):
        problem, seed = problem_and_seed()
        result = expand_operand_neighborhood(
            problem, [seed, seed], OperandNeighborhoodConfig(max_candidates=1)
        )
        self.assertEqual(result.status, "candidate_budget_exhausted")
        self.assertEqual(result.stats["candidates_checked"], 1)
        self.assertEqual(result.candidates[0].actions, seed.actions)

    def test_original_deadline_is_never_extended_and_is_restored(self):
        problem, seed = problem_and_seed()
        deadline = time.perf_counter() - 1
        problem.deadline = deadline
        result = expand_operand_neighborhood(
            problem, [seed], OperandNeighborhoodConfig(seconds=100)
        )
        self.assertEqual(result.status, "deadline_exhausted")
        self.assertEqual(result.candidates, ())
        self.assertEqual(problem.deadline, deadline)

    def test_local_transition_limit_restores_existing_limit(self):
        problem, seed = problem_and_seed()
        original = problem.counts["transition_trials"] + 100
        problem.transition_limit = original
        result = expand_operand_neighborhood(
            problem, [seed], OperandNeighborhoodConfig(max_transition_trials=2)
        )
        self.assertEqual(result.status, "transition_budget_exhausted")
        self.assertEqual(result.candidates, ())
        self.assertEqual(result.stats["transition_trials"], 2)
        self.assertEqual(problem.transition_limit, original)

    def test_external_timeout_is_not_swallowed(self):
        problem, seed = problem_and_seed()
        with patch.object(problem, "replay", side_effect=TimeoutError("outer alarm")):
            with self.assertRaisesRegex(TimeoutError, "outer alarm"):
                expand_operand_neighborhood(problem, [seed])
        self.assertIsNone(problem.deadline)
        self.assertIsNone(problem.transition_limit)

    def test_scalar_references_remain_on_their_axes(self):
        problem, _ = problem_and_seed()
        problem.config = replace(problem.config, observation_references=True)
        x, y, z = problem.operand_references("Move")
        self.assertNotIn("Y_min", x)
        self.assertNotIn("Z_m", y)
        self.assertIn("Y_min", y)
        self.assertIn("Z_m", z)
        self.assertNotIn("gripper", problem.operand_references("Pick")[0])

    def test_coherent_reference_choices_survive_release_only_variants(self):
        # Equal endpoint arrays intentionally demonstrate why nominal equality
        # cannot remove different controller-reference policies.
        state = np.zeros((1, 3))

        def node(xy="target", release="target", score=1.0):
            return Node(
                state,
                (
                    Action("Move", (xy, xy, "target"), (0, 0, 0.05)),
                    Action("Release", (release,), (0.1,)),
                ),
                state,
                score,
            )

        baseline = node()
        choices = [
            baseline,
            node(release="payload", score=1.001),
            node(release="gripper", score=1.002),
            node("anchor", score=1.003),
            node("payload", score=1.004),
            node("far", score=2),
        ]
        selected = select_reference_diverse(choices, limit=3, relative_slack=0.01)
        self.assertIs(selected[0], baseline)
        self.assertEqual(
            {n.actions[0].operands[0] for n in selected},
            {"target", "anchor", "payload"},
        )
        self.assertTrue(all(n.score <= 1.01 for n in selected))

    def test_invalid_budgets_are_rejected(self):
        for options in (
            {"max_candidates": 0},
            {"seconds": float("nan")},
            {"relative_slack": -1},
            {"pool_limit": False},
        ):
            with self.subTest(options=options), self.assertRaises(ContractInputError):
                OperandNeighborhoodConfig(**options)


if __name__ == "__main__":
    unittest.main()
