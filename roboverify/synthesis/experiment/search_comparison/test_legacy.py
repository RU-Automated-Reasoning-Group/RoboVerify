import json
import tempfile
import unittest
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np

from synthesis.api.instructions import MoveByName, Skip
from synthesis.api.program import Program
from synthesis.cfg.bindings import mutate_scoped
from synthesis.cfg.straightline import straight_line_synthesize
from synthesis.experiment.search_comparison.legacy import (
    LegacyConfig,
    LegacyRollouts,
    adapt_examples,
    encode,
    local_post,
    run_trial,
)
from synthesis.experiment.search_comparison.numeric import EvaluationBudgetExceeded


def examples():
    result = []
    for seed in range(2):
        observation = np.zeros(34)
        observation[:3] = (0.2, 0.1, 0.7)
        observation[3:5] = 0.04
        observation[10:13] = (0, 0, 0.425)
        observation[22:25] = (0.2, 0, 0.425)
        result.append(
            dict(
                id=f"stack-{seed}",
                seed=seed,
                task="stack",
                num_blocks=2,
                bindings={"b0": 0, "b": 0, "b_prime": 1},
                observations=np.stack([observation, observation]),
                initial_snapshot=object(),
                execution_end_index=1,
                public_sha256=f"hash-{seed}",
            )
        )
    return result


class Environment:
    def close(self):
        pass


def rollout(env, rows, example, **kwargs):
    return example["observations"], dict(
        execution_end_index=1, eligible=False, strict_1mm_eligible=False
    )


class LegacyReferenceTests(unittest.TestCase):
    def test_unstack_symbolic_post_does_not_assign_geometry_to_table(self):
        from synthesis.predicates.scene import Scene, evaluate
        from synthesis.util.on import TABLE

        positions = {
            0: (0, 0, 0.425),
            1: (0, 0, 0.475),
            2: (0.2, 0, 0.425),
            "tbl": TABLE,
        }
        bindings = {"b0": 0, "b_prime": 2, "tbl": "tbl"}
        self.assertTrue(evaluate(local_post("unstack"), Scene(positions, bindings)))
        positions[2] = (0, 0, 0.525)
        self.assertFalse(evaluate(local_post("unstack"), Scene(positions, bindings)))
        positions[2] = (0.2, 0, 0.8)
        self.assertTrue(
            evaluate(local_post("unstack"), Scene(positions, bindings)),
            "Physical table height belongs to the separate simulator acceptance audit",
        )

    def test_skip_encoding_retains_observation_multiplicity(self):
        from synthesis.experiment.search_comparison.safety import decode

        program = Program(
            2, [Skip(7), MoveByName("b", "b", "b", target_offset=[0, 0, 0.1])]
        )
        rows = encode(program)
        self.assertEqual(rows[0]["skip_steps"], 7)
        restored = decode(rows)
        self.assertEqual(restored[0].skip_steps, 7)
        self.assertEqual(encode(Program(2, restored)), rows)

    def test_complete_batch_required_and_repeated_calls_are_not_cached(self):
        data = examples()
        segments = adapt_examples(data)
        config = LegacyConfig("stack", max_rollouts=4)
        program = Program(1, [Skip()])
        with tempfile.TemporaryDirectory() as folder:
            audit = LegacyRollouts(
                data,
                config,
                folder,
                deadline=perf_counter() + 10,
                start=perf_counter(),
                rollout_fn=rollout,
                env_factory=Environment,
            )
            audit(program, segments[0])
            self.assertIsNone(audit.best)
            audit(program, segments[1])
            self.assertEqual(audit.best["accepted_examples"], 0)
            for segment in segments:
                audit(program, segment)
            self.assertEqual(audit.counts["simulator_rollouts"], 4)
            self.assertEqual(audit.counts["completed_vector_batches"], 2)
            with self.assertRaisesRegex(
                EvaluationBudgetExceeded, "complete vector batch"
            ):
                audit(program, segments[0])
            audit.close()

    def test_interrupted_vector_cannot_be_exported(self):
        data = examples()
        with tempfile.TemporaryDirectory() as folder:
            audit = LegacyRollouts(
                data,
                LegacyConfig("stack"),
                folder,
                deadline=perf_counter() + 10,
                start=perf_counter(),
                rollout_fn=rollout,
                env_factory=Environment,
            )
            audit(Program(1, [Skip()]), adapt_examples(data)[0])
            audit.close()
            self.assertIsNone(audit.best)
            self.assertEqual(audit.counts["interrupted_vector_batches"], 1)
            record = json.loads(Path(folder, "evaluations.jsonl").read_text())
            self.assertFalse(record["complete"])

    def test_runner_calls_actual_production_mutation_and_search(self):
        data = examples()
        with tempfile.TemporaryDirectory() as folder, patch(
            "synthesis.experiment.search_comparison.data.load_training",
            return_value=data,
        ), patch(
            "synthesis.experiment.search_comparison.safety.run_rows",
            side_effect=rollout,
        ), patch.object(
            LegacyRollouts, "_environment", return_value=Environment()
        ), patch(
            "synthesis.cfg.bindings.mutate_scoped", wraps=mutate_scoped
        ) as mutation, patch(
            "synthesis.cfg.straightline.straight_line_synthesize",
            wraps=straight_line_synthesize,
        ) as search:
            report = run_trial(
                LegacyConfig("stack", max_rollouts=6), "unused", folder, pilot=True
            )
            self.assertEqual(
                report["status"], "budget_exhausted", report.get("traceback")
            )
            self.assertFalse(report["production_returned"])
            self.assertFalse(report["training_accepted"])
            self.assertEqual(report["evaluation_counts"]["simulator_rollouts"], 6)
            self.assertEqual(report["evaluation_counts"]["completed_vector_batches"], 3)
            self.assertEqual(search.call_count, 1)
            self.assertGreater(mutation.call_count, 0)
            budget = search.call_args.kwargs["budget"]
            self.assertEqual(budget.search_strategy, "mcmc")
            self.assertEqual(budget.distance, "kl_kde")
            self.assertEqual(
                (budget.cem_iterations, budget.cem_samples, budget.cem_elites),
                (2, 8, 2),
            )
            self.assertTrue(
                all(isinstance(i, Skip) for i in search.call_args.args[2].instructions)
            )


if __name__ == "__main__":
    unittest.main()
