"""Combined report over several CaP batches, and the Revision 6 driver's configs."""

import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from ..experiments import revision6_experiment as experiment
from ..reports.batch_set_report import find_batches, write


def make_batch(directory, task, model, outcomes):
    directory.mkdir(parents=True)
    jobs = [
        {
            "id": f"job-{i:03d}",
            "run_config": {
                "task": task,
                "num_blocks": 4,
                "seed": i,
                "method": "cap_method_2",
                "condition_mode": "dsl_only",
                "prompt_version": "typed_v5",
                "timeout": 3600,
                "max_model_calls": 12,
                "max_tokens": 100000,
                "max_sim_steps": 10000,
                "recovery": {"enabled": False},
                "demonstrations": {
                    "enabled": True,
                    "token_counter": "server_tokenize",
                    "waypoint_tolerance": 0.01,
                },
            },
            "model_config": {
                "model": model,
                "provider": "vllm",
                "extra_body": {},
                "max_output_tokens": 32768,
            },
        }
        for i in range(len(outcomes))
    ]
    (directory / "manifest.json").write_text(json.dumps({"jobs": jobs}))
    (directory / "schedule.json").write_text("{}")
    rows = [
        {
            "trial": i + 1,
            "job_id": jobs[i]["id"],
            "status": "completed" if ok else "goal_failed",
            "task_spec_success": ok,
            "first_attempt_task_spec_success": ok,
            "success": ok,
            "settled_environment_success": ok,
            "dsl_compliance": "pass",
            "known_tokens": 100,
            "demonstration_tokens": 900,
            "demonstration_block_tokens": 900,
            "provider_total_tokens": 1000,
            "model_calls": 1,
            "simulation_steps": 1200 + 50 * i,
            "attempts": [
                {
                    "status": "completed" if ok else "goal_failed",
                    "task_spec_success": ok,
                    "simulation_steps": 1200 + 50 * i,
                }
            ],
            "run_dir": "x",
        }
        for i, ok in enumerate(outcomes)
    ]
    (directory / "summary.json").write_text(json.dumps({"trials": rows}))
    (directory / "comparison_plan.json").write_text(
        json.dumps(
            {
                "frozen_config": "cap-frozen-v3",
                "frozen_config_source": "v3.py",
                "git_sha": "abcdef0123",
            }
        )
    )


class BatchSetReportTests(unittest.TestCase):
    def test_batches_are_found_by_model_then_task_order_and_combined(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_batch(root / "m" / "unstack", "unstack", "m", [True, False])
            make_batch(root / "m" / "stack", "stack", "m", [True])
            # A trial directory with its own manifest is not a batch.
            (root / "m" / "stack" / "trials" / "t").mkdir(parents=True)
            (root / "m" / "stack" / "trials" / "t" / "manifest.json").write_text("{}")
            self.assertEqual([d.name for d in find_batches(root)], ["stack", "unstack"])
            text = write(root, title="T", notes=["n1"]).read_text()
        self.assertIn("# CaP batch report: T", text)
        self.assertIn("(3/3 trials finished in 2 batches)", text)
        self.assertIn(
            "| [m/unstack](m/unstack/report.md) | unstack | m | 4 | 0–1 |", text
        )
        self.assertIn("| 1/2 | 1/2 |", text)
        self.assertIn("100 (+900 demo)", text)
        self.assertIn("[trial 2](m/unstack/trials/job-001/report.txt)", text)
        self.assertIn("- n1", text)
        # Env steps: batch total and mean, per trial, per round.
        self.assertIn("| 2/2 | 2,450 | 1,225.0 | 2/2 |", text)
        self.assertIn("| 1 | 0 | 1,250 | 0.00 | 0.00 |", text)
        self.assertIn("| 2 | 0 | goal_failed | False | None | 1,250 |", text)

    def test_a_single_batch_keeps_its_own_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            make_batch(Path(tmp) / "b", "stack", "m", [True])
            with self.assertRaisesRegex(ValueError, "single batch"):
                write(Path(tmp) / "b")


class DriverConfigurationTests(unittest.TestCase):
    def test_local_model_is_config_frozen_v3_unchanged(self):
        config, model, _, overrides = experiment.configuration(
            "stack", "local-qwen3.8-27b-fp8"
        )
        frozen, frozen_model, _ = experiment.FROZEN["v3"].configuration("stack", seed=0)
        self.assertEqual(config, frozen)
        self.assertEqual(model, frozen_model)
        self.assertEqual(overrides, {})

    def test_hosted_models_change_only_endpoint_reasoning_and_token_counter(self):
        for label in ("qwen3.8-27b", "claude-haiku-5.5"):
            config, model, _, overrides = experiment.configuration("grid", label)
            frozen, frozen_model, _ = experiment.FROZEN["v3"].configuration(
                "grid", seed=0
            )
            self.assertEqual(model.provider, "openrouter")
            self.assertEqual(model.extra_body, {"reasoning": {"effort": "medium"}})
            self.assertEqual(model.max_output_tokens, frozen_model.max_output_tokens)
            self.assertEqual(config.demonstrations.token_counter, "provider_usage")
            self.assertEqual(
                config.demonstrations.archive_sha256,
                frozen.demonstrations.archive_sha256,
            )
            self.assertEqual(
                set(overrides), {"model_config", "demonstrations.token_counter"}
            )
        off, _, _, overrides = experiment.configuration(
            "grid", "claude-haiku-5.5", demonstrations=False
        )
        self.assertFalse(off.demonstrations.enabled)
        self.assertEqual(set(overrides), {"model_config"})

    def test_frozen_v6_supplies_its_run_config_and_instruction(self):
        v6 = experiment.FROZEN["v6"]
        for demonstrations in (True, False):
            config, _, instruction, _ = experiment.configuration(
                "pyramid",
                "claude-haiku-5.5",
                demonstrations=demonstrations,
                frozen="v6",
            )
            frozen, _, frozen_instruction = v6.configuration(
                "pyramid", demonstrations, seed=0
            )
            self.assertEqual(instruction, frozen_instruction)
            changed = {
                key
                for key, value in asdict(config)["demonstrations"].items()
                if asdict(frozen)["demonstrations"][key] != value
            }
            self.assertEqual(changed, {"token_counter"} if demonstrations else set())
            self.assertEqual(
                asdict(config) | {"demonstrations": None},
                asdict(frozen) | {"demonstrations": None},
            )

    def test_analysis_budget_is_an_explicit_recorded_override(self):
        frozen, _, _ = experiment.FROZEN["final_v2_wo_demo"].configuration(
            "restack", seed=0
        )
        self.assertIsNone(frozen.max_analysis_steps)
        config, _, _, overrides = experiment.configuration(
            "restack",
            "claude-opus-5.5",
            demonstrations=False,
            frozen="final_v2_wo_demo",
            max_analysis_steps=1_000_000,
        )
        self.assertEqual(config.max_analysis_steps, 1_000_000)
        self.assertEqual(overrides["max_analysis_steps"], 1_000_000)
        self.assertEqual(asdict(config) | {"max_analysis_steps": None}, asdict(frozen))


if __name__ == "__main__":
    unittest.main()
