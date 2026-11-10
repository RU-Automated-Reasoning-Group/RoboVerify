"""Critical fairness, input-boundary and interrupted-evaluation regressions."""

import json
import tempfile
import unittest
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np

from .data import load_training
from .matched import MatchedConfig, mutate_actions, search
from .metrics import arc_resample, objective
from .numeric import EvaluationBudgetExceeded, SharedEvaluator, bmc_screen
from .protocol import TrialConfig, candidate_seed


class InputAndMetricTests(unittest.TestCase):
    def test_training_loader_never_loads_heldout_or_exposes_teacher_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "public").mkdir()
            observations = np.zeros((52, 34))
            meta = {"bindings": {"b0": 0, "b_prime": 1}, "snapshot_arrays": []}
            np.savez(
                root / "public/train.npz",
                observations=observations,
                initial_gt_state=np.zeros(5),
                metadata=json.dumps(meta),
                teacher_commands=np.ones((51, 4)),
            )
            rows = [
                dict(
                    id="train",
                    task="unstack",
                    split="train",
                    seed=0,
                    num_blocks=2,
                    transfer_index=0,
                    success=True,
                    public_file="public/train.npz",
                    teacher_target="secret",
                ),
                dict(
                    id="test",
                    task="unstack",
                    split="heldout",
                    seed=60,
                    num_blocks=2,
                    transfer_index=0,
                    success=True,
                    public_file="public/does-not-exist.npz",
                ),
            ]
            (root / "manifest.json").write_text(json.dumps({"attempts": rows}))
            examples = load_training(root, "unstack", seeds=(0,))
            self.assertEqual([e["id"] for e in examples], ["train"])
            self.assertFalse(any("teacher" in k for k in examples[0]))
            rows[0]["public_file"] = "../private_eval/commands.npz"
            (root / "manifest.json").write_text(json.dumps({"attempts": rows}))
            with self.assertRaisesRegex(ValueError, "public NPZ"):
                load_training(root, "unstack", seeds=(0,))

    def test_arc_alignment_removes_dwell_but_retains_order(self):
        a = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
        np.testing.assert_allclose(
            arc_resample(a), arc_resample(np.repeat(a, [3, 7, 2], axis=0))
        )
        self.assertGreater(np.linalg.norm(arc_resample(a) - arc_resample(a[::-1])), 1.0)
        self.assertGreater(objective(100.0, 0), objective(0.0, 1))

    def test_numeric_noise_is_candidate_keyed_and_arm_independent(self):
        rows = [dict(kind="Move", operands=["a", "a", "b"], offsets=[0.0, 0.0, 0.1])]
        seed = candidate_seed(3, rows)
        self.assertEqual(seed, candidate_seed(3, json.loads(json.dumps(rows))))
        self.assertNotEqual(seed, candidate_seed(4, rows))
        first = np.random.default_rng(seed).normal(size=(2, 3))
        np.random.default_rng(321).normal(size=1000)
        np.testing.assert_array_equal(
            first, np.random.default_rng(candidate_seed(3, rows)).normal(size=(2, 3))
        )


class MatchedTests(unittest.TestCase):
    def test_primitive_families_are_not_diluted_by_offset_grid_size(self):
        from synthesis.search.contract import Action

        rng = np.random.default_rng(3)
        alphabet = (Action("Pick", ("a",)), Action("Release", ("a",), (0.1,))) + tuple(
            Action("Move", ("a",) * 3, (0.0, 0.0, float(i))) for i in range(100)
        )
        counts = Counter(
            mutate_actions((), alphabet, rng, 2)[0][0].kind for _ in range(900)
        )
        self.assertTrue(
            all(200 < counts[kind] < 400 for kind in ("Pick", "Move", "Release")),
            counts,
        )

    def test_initial_deadline_restores_callers_transition_limit(self):
        class Problem:
            transition_limit = 7
            counts = Counter()

            def replay(self, actions):
                raise TimeoutError("expired")

        problem = Problem()
        candidates, stats, status = search(problem)
        self.assertEqual(status, "deadline")
        self.assertFalse(candidates)
        self.assertEqual(problem.transition_limit, 7)

    def test_mutations_cover_all_edits_and_legal_lengths(self):
        rng = np.random.default_rng(1)
        operations = set()
        actions = (1, 2)
        for _ in range(500):
            result, operation = mutate_actions(actions, (1, 2, 3), rng, 4)
            operations.add(operation)
            self.assertLessEqual(len(result), 4)
            self.assertTrue(set(result) <= {1, 2, 3})
        self.assertEqual(operations, {"insert", "delete", "replace", "swap"})
        self.assertEqual(mutate_actions((), (1,), rng, 2)[0], (1,))

    def test_partial_programs_can_reach_a_terminal_goal(self):
        from synthesis.search.contract import Action

        alphabet = tuple(Action("Pick", (r,)) for r in ("a", "b"))

        @dataclass
        class Node:
            actions: tuple
            state: tuple
            score: float = 0.0

        class Problem:
            actions = alphabet
            transition_limit = None
            counts = Counter()

            def replay(self, actions):
                self.counts["transition_trials"] += len(actions)
                return Node(actions, actions)

            def goal(self, state):
                return len(state) == 2 and state[0] != state[1]

            def rank(self, node, complete=None):
                return -0.001 * min(len(node.actions), 2)

        candidates, stats, status = search(
            Problem(),
            MatchedConfig(max_length=3, max_iterations=500, seconds=2, seed=5),
        )
        self.assertTrue(candidates)
        self.assertGreater(stats["accepted"], 0)
        self.assertTrue(all(len(n.actions) == 2 for n in candidates))


class NumericBudgetTests(unittest.TestCase):
    def test_bmc_preserves_relational_table_without_inventing_coordinates(self):
        rows = [dict(kind="Pick", operands=["b_prime"], offsets=[])]
        example = dict(
            initial_positions={0: [0.0, 0.0, 0.425], 1: [0.2, 0.0, 0.425]},
            bindings={"b0": 0, "b_prime": 1, "tbl": "tbl"},
            observations=np.zeros((1, 34)),
        )
        result = bmc_screen(rows, [example], deadline=perf_counter() + 10)
        self.assertTrue(result["feasible"], result)
        rows = [dict(kind="Move", operands=["tbl"] * 3, offsets=[0.0, 0.0, 0.05])]
        result = bmc_screen(rows, [example], deadline=perf_counter() + 10)
        self.assertFalse(result["feasible"])
        self.assertIn("explicit runtime frame", result["scenes"][0]["reason"])

    def examples(self):
        obs = np.zeros((2, 34))
        return [
            dict(
                id=str(i),
                task="stack",
                num_blocks=2,
                bindings={"b0": 0, "b_prime": 1},
                observations=obs.copy(),
                execution_end_index=1,
            )
            for i in range(2)
        ]

    def test_partial_shared_vector_cannot_replace_completed_best(self):
        class Problem:
            def replay(self, actions):
                return object()

        class Env:
            def close(self):
                pass

        calls = []

        def rollout(env, rows, example, **kw):
            calls.append(example["id"])
            if rows[0]["offsets"][0] > 0 and example["id"] == "1":
                raise EvaluationBudgetExceeded("test interruption")
            return example["observations"].copy(), dict(
                eligible=True, strict_1mm_eligible=True, execution_end_index=1
            )

        rows = [dict(kind="Move", operands=["b0"] * 3, offsets=[0.0, 0.0, 0.0])]
        with tempfile.TemporaryDirectory() as tmp:
            start = perf_counter()
            ev = SharedEvaluator(
                Problem(),
                self.examples(),
                TrialConfig("stack", "beam", max_rollouts=10),
                tmp,
                deadline=start + 20,
                start=start,
                rollout_fn=rollout,
                env_factory=Env,
                metric=lambda trajectories: 0.01,
            )
            ev.evaluate(rows, [0.0, 0.0, 0.0])
            key = ev.best["program_sha256"]
            with self.assertRaises(EvaluationBudgetExceeded):
                ev.evaluate(rows, [0.001, 0.0, 0.0])
            self.assertEqual(ev.best["program_sha256"], key)
            self.assertEqual(ev.counts["completed_vector_batches"], 1)
            self.assertEqual(ev.counts["interrupted_vector_batches"], 1)
            self.assertEqual(ev.counts["simulator_rollouts"], 4)
            self.assertEqual(ev.counts["completed_simulator_rollouts"], 3)
            self.assertFalse(ev.records[-1]["complete"])
            ev.close()

    def test_cached_vector_still_respects_each_proposals_parameter_radius(self):
        class Problem:
            def replay(self, actions):
                return object()

        class Env:
            def close(self):
                pass

        def rollout(env, rows, example, **kw):
            return example["observations"].copy(), dict(
                eligible=True, strict_1mm_eligible=True, execution_end_index=1
            )

        rows = [dict(kind="Move", operands=["b0"] * 3, offsets=[0.0, 0.0, 0.0])]
        with tempfile.TemporaryDirectory() as tmp:
            start = perf_counter()
            ev = SharedEvaluator(
                Problem(),
                self.examples(),
                TrialConfig("stack", "beam"),
                tmp,
                deadline=start + 20,
                start=start,
                rollout_fn=rollout,
                env_factory=Env,
                metric=lambda trajectories: 0.01,
            )
            score = ev.evaluate(rows, [0.03, 0.0, 0.0])
            far_base = [dict(rows[0], offsets=[0.1, 0.0, 0.0])]
            self.assertLess(ev.evaluate(far_base, [0.03, 0.0, 0.0]), score)
            self.assertEqual(ev.counts["cache_hits"], 0)
            self.assertEqual(ev.counts["simulator_rollouts"], 2)
            self.assertEqual(ev.counts["radius_rejected"], 1)
            ev.close()

    def test_bmc_rejection_never_constructs_environment_or_calls_cem(self):
        class Problem:
            pass

        with tempfile.TemporaryDirectory() as tmp:
            start = perf_counter()
            ev = SharedEvaluator(
                Problem(),
                self.examples(),
                TrialConfig("stack", "beam"),
                tmp,
                deadline=start + 20,
                start=start,
                env_factory=lambda: self.fail(
                    "BMC rejected candidate reached simulator"
                ),
                metric=lambda trajectories: 0.0,
            )
            with patch(
                "synthesis.experiment.search_comparison.numeric.bmc_screen",
                return_value={"feasible": False},
            ):
                ev.fit_candidate([dict(kind="Pick", operands=["b0"], offsets=[])])
            self.assertEqual(ev.counts["bmc_rejected"], 1)
            self.assertIsNone(ev.best)
            self.assertEqual(ev.counts["objective_calls"], 0)


if __name__ == "__main__":
    unittest.main()
