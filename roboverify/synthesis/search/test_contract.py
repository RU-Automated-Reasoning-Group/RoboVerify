"""Behavioral checks for the nominal proposal system; no saved demos required."""

import time
import unittest
from dataclasses import replace

import numpy as np

from synthesis.search.contract import (
    Action,
    ContractConfig,
    ContractInputError,
    ContractProblem,
    DeadlineExceeded,
    NominalState,
    SearchConfig,
    SearchExample,
    TransitionBudgetExceeded,
    beam_search,
    resample_batch,
    select_candidates,
    state_key,
)


def example(held=None, elevated=False):
    positions = np.array([[0.0, 0.0, 0.425], [0.2, 0.0, 0.425]])
    if elevated:
        positions[0] = [0.2, 0, 0.475]
    ee = positions[0].copy() if held is not None else np.array([0.0, -0.2, 0.65])
    gripper_path = np.array(
        [
            ee,
            positions[0],
            [0, 0, 0.575],
            [0.2, 0, 0.575],
            [0.2, 0, 0.475],
            [0.2, 0, 0.625],
        ]
    )
    path = np.repeat(positions[None], len(gripper_path), axis=0)
    path[2:, 0] = gripper_path[2:]
    path[-1, 0] = [0.2, 0, 0.475]
    return SearchExample(
        positions,
        ee,
        {"source": 0, "source_alias": 0, "target": 1, "tbl": "tbl"},
        gripper_path,
        path,
        held,
    )


def stack_goal(state, index):
    return bool(
        np.allclose(
            state.positions[index, 0],
            state.positions[index, 1] + [0, 0, 0.05],
            atol=1e-7,
        )
    )


def transfer_actions():
    return (
        Action("Pick", ("source",)),
        Action("Move", ("source",) * 3, (0, 0, 0.15)),
        Action("Move", ("target", "target", "source"), (0, 0, 0)),
        Action("Move", ("target",) * 3, (0, 0, 0.05)),
        Action("Release", ("target",), (0.2,)),
    )


class ContractTests(unittest.TestCase):
    def problem(self, **kwargs):
        return ContractProblem(
            [example()],
            stack_goal,
            allowed_operands=("source", "source_alias", "target"),
            **kwargs,
        )

    def test_partial_prefixes_are_opt_in_and_never_goal_candidates(self):
        action = Action("Pick", ("source",))
        called = []
        ordinary = beam_search(
            self.problem(actions=(action,)),
            SearchConfig(max_length=1, seconds=2),
            on_candidate=called.append,
        )
        self.assertEqual(ordinary.candidates, ())
        self.assertEqual(ordinary.partial_candidates, ())
        problem = self.problem(actions=(action,))
        partial = beam_search(
            problem,
            SearchConfig(max_length=1, seconds=2, partial_pool_limit=2),
            on_candidate=called.append,
        )
        self.assertEqual(partial.candidates, ())
        self.assertEqual(len(partial.partial_candidates), 1)
        self.assertEqual(partial.partial_candidates[0].actions, (action,))
        self.assertFalse(problem.goal(partial.partial_candidates[0].state))
        self.assertEqual(called, [])

    def test_budget_preserves_fully_scored_partial_prefix(self):
        problem = self.problem(actions=(Action("Pick", ("source",)),))
        result = beam_search(
            problem,
            SearchConfig(
                max_length=3, max_expansions=1, seconds=2, partial_pool_limit=1
            ),
        )
        self.assertEqual(result.status, "expansion_budget_exhausted")
        self.assertEqual(result.candidates, ())
        self.assertEqual(len(result.partial_candidates), 1)
        self.assertTrue(np.isfinite(result.partial_candidates[0].score))

    def test_actual_transfer_and_instruction_entry_references(self):
        problem = self.problem(actions=transfer_actions())
        entry = problem.initial.positions.copy()
        node = problem.replay(transfer_actions())
        self.assertIsNotNone(node)
        self.assertTrue(problem.goal(node.state))
        self.assertEqual(node.state.holding, (-1,))
        np.testing.assert_array_equal(problem.initial.positions, entry)
        np.testing.assert_array_equal(node.state.positions[:, 1], entry[:, 1])
        self.assertFalse(node.state.positions.flags.writeable)
        numeric = node.to_program()
        self.assertEqual(len(numeric.instructions), 5)
        self.assertEqual(len(numeric.register_trainable_parameter()), 10)

    def test_batch_rank_preserves_history_and_scalar_scores(self):
        problem = self.problem(actions=transfer_actions())
        nodes = [problem.replay(transfer_actions()[:i]) for i in range(6)]
        # Two different histories may reach one endpoint; both must be scored.
        repeated = replace(
            nodes[-1],
            path=np.concatenate(
                [nodes[-1].path[:, :1], nodes[-1].path[:, :1] + 0.03, nodes[-1].path],
                axis=1,
            ),
        )
        nodes += [repeated, nodes[2], nodes[0]]
        flags = [problem.goal(node.state) for node in nodes]
        expected = np.asarray([problem.rank(n, f) for n, f in zip(nodes, flags)])
        for chunk in (1, 2, 64):
            np.testing.assert_allclose(
                problem.rank_many(nodes, flags, chunk_size=chunk),
                expected,
                rtol=1e-14,
                atol=1e-16,
            )
        self.assertNotEqual(expected[5], expected[6])
        self.assertEqual(problem.rank_many([]).shape, (0,))

    def test_batched_successors_equal_independent_scalar_contracts(self):
        for envelope in (False, True):
            problem = self.problem(config=ContractConfig(gripper_envelope=envelope))
            for prefix in range(5):
                state = problem.replay(transfer_actions()[:prefix]).state
                expected = [
                    (action, successor)
                    for action in problem.actions
                    if (successor := problem.transition(state, action)) is not None
                ]
                actual = problem.successors(state)
                self.assertEqual([a for a, _ in expected], [a for a, _ in actual])
                self.assertEqual(
                    [state_key(s) for _, s in expected],
                    [state_key(s) for _, s in actual],
                )

    def test_move_preserves_observed_grasp_offset_in_scalar_and_batch(self):
        ex = example(held=0)
        positions = ex.positions.copy()
        positions[0, 2] += 0.2
        gripper = positions[0] + [-0.01, 0.005, -0.007]
        ex = replace(
            ex,
            positions=positions,
            gripper=gripper,
            positions_path=np.array([positions]),
            gripper_path=np.array([gripper]),
        )
        moves = (
            Action("Move", ("gripper",) * 3, (0.1, 0, 0)),
            Action("Move", ("gripper",) * 3, (0, 0, 0.1)),
        )
        problem = ContractProblem(
            [ex],
            lambda state, index: False,
            actions=moves,
            config=ContractConfig(gripper_envelope=False),
        )
        expected_offset = positions[0] - gripper
        batched = dict(problem.successors(problem.initial))
        for move in moves:
            state = problem.transition(problem.initial, move)
            self.assertIsNotNone(state)
            np.testing.assert_allclose(
                state.positions[0, 0] - state.gripper[0], expected_offset
            )
            np.testing.assert_array_equal(state.positions[0, 1], positions[1])
            self.assertEqual(state_key(state), state_key(batched[move]))

    def test_held_clearance_and_sweep_use_payload_centre(self):
        cases = (
            # The gripper is above the support-clearance band, but its payload
            # remains on the support and must be lifted before lateral motion.
            (np.array([[0, 0, 0.475], [0, 0, 0.425]]), 0.028, True),
            # The gripper passes below an obstacle that intersects the payload.
            (np.array([[0, 0, 0.5], [0.1, 0, 0.545]]), -0.028, False),
        )
        move = Action("Move", ("gripper",) * 3, (0.2, 0, 0))
        for positions, height_offset, clearance in cases:
            with self.subTest(clearance=clearance):
                gripper = positions[0] + [0, 0, height_offset]
                ex = replace(
                    example(held=0),
                    positions=positions,
                    gripper=gripper,
                    positions_path=np.array([positions]),
                    gripper_path=np.array([gripper]),
                )
                problem = ContractProblem(
                    [ex],
                    lambda state, index: False,
                    actions=(move,),
                    config=ContractConfig(
                        gripper_envelope=False, support_clearance=clearance
                    ),
                )
                self.assertIsNone(problem.transition(problem.initial, move))
                self.assertEqual(problem.successors(problem.initial), ())

    def test_shared_named_program_respects_each_examples_distinct_binding(self):
        first = example()
        second = SearchExample(
            first.positions[::-1],
            first.gripper,
            {"source": 1, "source_alias": 1, "target": 0},
            first.gripper_path,
            first.positions_path[:, ::-1],
        )
        examples = [first, second]

        def goal(state, i):
            source, target = (
                examples[i].bindings["source"],
                examples[i].bindings["target"],
            )
            return bool(
                np.allclose(
                    state.positions[i, source],
                    state.positions[i, target] + [0, 0, 0.05],
                )
            )

        problem = ContractProblem(examples, goal, actions=transfer_actions())
        picked = problem.transition(problem.initial, transfer_actions()[0])
        self.assertEqual(picked.holding, (0, 1))
        node = problem.replay(transfer_actions())
        self.assertTrue(problem.goal(node.state))
        np.testing.assert_array_equal(
            node.state.positions[0], node.state.positions[1, ::-1]
        )
        # One failing example is enough to reject the nominal goal.
        wrong = node.state.positions.copy()
        wrong[1, 1, 0] += 0.02
        invalid = NominalState(wrong, node.state.gripper.copy(), node.state.holding)
        self.assertFalse(problem.goal(invalid))

    def test_pick_cannot_escape_allowed_operand_scope(self):
        problem = ContractProblem([example()], stack_goal, allowed_operands=("source",))
        with self.assertRaises(ContractInputError):
            problem.transition(problem.initial, Action("Pick", ("target",)))
        with self.assertRaises(ContractInputError):
            replace(example(), bindings={"gripper": 0})

    def test_support_clearance_and_airborne_release_are_rejected(self):
        ex = example(held=0, elevated=True)
        problem = ContractProblem(
            [ex], stack_goal, config=ContractConfig(gripper_envelope=False)
        )
        lateral = Action("Move", ("gripper",) * 3, (0.15, 0, 0))
        self.assertIsNone(problem.transition(problem.initial, lateral))
        lift = problem.transition(
            problem.initial, Action("Move", ("gripper",) * 3, (0, 0, 0.15))
        )
        self.assertIsNotNone(lift)
        self.assertIsNone(
            problem.transition(lift, Action("Release", ("gripper",), (0.05,)))
        )
        self.assertIsNotNone(problem.transition(lift, lateral))
        # At the original aligned support, opening and retreating is admissible.
        self.assertIsNotNone(
            problem.transition(
                problem.initial, Action("Release", ("gripper",), (0.15,))
            )
        )

    def test_scalar_extrema_are_axis_scoped_and_use_current_entry(self):
        problem = self.problem(
            config=ContractConfig(gripper_envelope=False, support_clearance=False)
        )
        picked = problem.transition(problem.initial, Action("Pick", ("source",)))
        first = problem.transition(
            picked, Action("Move", ("target", "gripper", "Z_m"), (0, 0, 0.15))
        )
        self.assertIsNotNone(first)
        np.testing.assert_allclose(first.gripper[0], [0.2, 0, 0.575])
        second = problem.transition(
            first, Action("Move", ("gripper", "Y_min", "Z_m"), (0, -0.15, 0.1))
        )
        np.testing.assert_allclose(second.gripper[0], [0.2, -0.15, 0.675])
        with self.assertRaises(ContractInputError):
            problem.transition(
                first, Action("Move", ("Z_m", "target", "target"), (0, 0, 0))
            )
        with self.assertRaises(ContractInputError):
            problem.transition(first, Action("Pick", ("Y_min",)))

    def test_no_table_coordinates_are_invented(self):
        problem = self.problem()
        self.assertNotIn("tbl", problem.operands)
        with self.assertRaises(ContractInputError):
            ContractProblem([example()], stack_goal, allowed_operands=("tbl", "source"))
        ex = replace(example(), table_reference=np.array([0.1, -0.2, 0.375]))
        problem = ContractProblem([ex], stack_goal)
        np.testing.assert_allclose(
            problem.reference(problem.initial, "tbl"), [[0.1, -0.2, 0.375]]
        )
        self.assertTrue(any("tbl" in a.operands for a in problem.actions))
        self.assertFalse(
            any(a.kind == "Pick" and "tbl" in a.operands for a in problem.actions)
        )

    def test_unknown_holding_and_malformed_paths_fail_explicitly(self):
        with self.assertRaises(ContractInputError):
            replace(example(), initial_holding="unknown")
        with self.assertRaises(ContractInputError):
            replace(example(), positions_path=np.zeros((2, 2, 3)))
        with self.assertRaises(ContractInputError):
            replace(example(), initial_holding=0)  # Recorded gripper is far away.
        with self.assertRaises(ContractInputError):
            self.problem(batch_goal=lambda s: np.ones((2,))).goal_mask(
                self.problem().initial
            )

    def test_numeric_scope_and_fresh_program_parameters(self):
        problem = ContractProblem(
            [example()],
            stack_goal,
            allowed_operands=(0, 1),
            config=ContractConfig(observation_references=False),
        )
        self.assertTrue(
            all(all(type(r) is int for r in a.operands) for a in problem.actions)
        )
        nodes = []
        for source in ("source", "source_alias"):
            acts = (Action("Pick", (source,)),) + transfer_actions()[1:]
            nodes.append(self.problem(actions=acts).replay(acts))
        a, b = nodes[0].to_program(), nodes[0].to_program()
        p = a.register_trainable_parameter()
        b.register_trainable_parameter()
        a.update_trainable_parameter([v + 0.01 for v in p])
        self.assertNotEqual(
            a.register_trainable_parameter(), b.register_trainable_parameter()
        )
        with self.assertRaises(ContractInputError):
            nodes[0].to_program(named=False)

    def test_deadline_and_transition_budget_fail_closed(self):
        with self.assertRaises(DeadlineExceeded):
            self.problem(deadline=time.perf_counter() - 1)
        problem = self.problem(actions=transfer_actions())
        result = beam_search(problem, SearchConfig(max_transition_trials=1))
        self.assertEqual(result.status, "transition_budget_exhausted")
        self.assertEqual(result.stats["transition_trials"], 1)
        self.assertFalse(result.candidates)
        self.assertIsNone(problem.transition_limit)
        problem.transition_limit = problem.counts["transition_trials"]
        with self.assertRaises(TransitionBudgetExceeded):
            problem.transition(problem.initial, transfer_actions()[0])
        problem.transition_limit = None
        result = beam_search(problem, SearchConfig(deadline=time.perf_counter() - 1))
        self.assertEqual(result.status, "deadline_exhausted")
        self.assertFalse(result.candidates)

    def test_successor_cache_preserves_goal_pool_and_history_rank(self):
        acts = transfer_actions() + (Action("Pick", ("source_alias",)),)
        results = []
        for memo in (False, True):
            result = beam_search(
                self.problem(actions=acts),
                SearchConfig(
                    max_length=5,
                    beam_width=32,
                    max_expansions=500,
                    seconds=10,
                    memoize=memo,
                    diversity=False,
                ),
            )
            results.append(result)
        self.assertTrue(results[0].candidates)
        self.assertEqual(
            [[a.row() for a in n.actions] for n in results[0].candidates],
            [[a.row() for a in n.actions] for n in results[1].candidates],
        )
        self.assertEqual(
            [n.score for n in results[0].candidates],
            [n.score for n in results[1].candidates],
        )
        self.assertGreater(results[1].stats.get("memo_hits", 0), 0)
        self.assertLess(
            results[1].stats["transition_trials"], results[0].stats["transition_trials"]
        )

    def test_callback_sees_complete_goal_only(self):
        problem = self.problem(actions=transfer_actions())
        observed = []

        def callback(node):
            observed.append(node)
            return False

        result = beam_search(
            problem,
            SearchConfig(max_length=5, beam_width=32, seconds=10),
            on_candidate=callback,
        )
        self.assertEqual(result.status, "callback_stopped")
        self.assertEqual(len(observed), 1)
        self.assertTrue(problem.goal(observed[0].state))
        self.assertTrue(result.candidates)
        self.assertIsNotNone(result.stats["first_goal_seconds"])

    def test_goal_holding_policy_and_empty_body_are_explicit(self):
        ex = example(held=0, elevated=True)
        held_problem = ContractProblem([ex], stack_goal)
        self.assertTrue(held_problem.goal_mask(held_problem.initial)[0])
        self.assertFalse(held_problem.goal(held_problem.initial))
        permissive = ContractProblem(
            [ex], stack_goal, config=ContractConfig(require_empty_hand=False)
        )
        result = beam_search(permissive, SearchConfig(max_length=0))
        self.assertEqual(result.candidates[0].actions, ())
        disallow = ContractProblem(
            [ex],
            stack_goal,
            config=ContractConfig(require_empty_hand=False, allow_empty_goal=False),
        )
        self.assertFalse(beam_search(disallow, SearchConfig(max_length=0)).candidates)

    def test_observation_proposals_and_state_key(self):
        problem = self.problem()
        self.assertTrue(
            any(a.proposal_id.startswith("observation") for a in problem.actions)
        )
        self.assertTrue(
            any(
                a.operands[0] != a.operands[1]
                for a in problem.actions
                if a.kind == "Move"
            )
        )
        a = problem.initial
        b = NominalState(a.positions.copy(), a.gripper.copy(), (0,))
        self.assertNotEqual(state_key(a), state_key(b))
        changed = a.gripper.copy()
        changed[0, 0] += 1e-12
        c = NominalState(a.positions.copy(), changed, a.holding)
        self.assertNotEqual(state_key(a), state_key(c))
        with self.assertRaises(ValueError):
            a.positions[0, 0, 0] = 100

    def test_arc_resampling_handles_dwells_without_nonfinite_values(self):
        p = np.array(
            [[[0, 0], [0, 0], [1, 0], [1, 0]], [[2, 2], [2, 2], [2, 2], [2, 2]]],
            dtype=float,
        )
        value = resample_batch(p, 5)
        np.testing.assert_allclose(value[0, :, 0], np.linspace(0, 1, 5))
        np.testing.assert_allclose(value[1], np.full((5, 2), 2))
        self.assertTrue(np.isfinite(value).all())


if __name__ == "__main__":
    unittest.main()
