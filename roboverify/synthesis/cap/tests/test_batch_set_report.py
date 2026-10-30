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

    def test_any_openrouter_id_is_a_hosted_model_override(self):
        frozen, _, _ = experiment.FROZEN["final_v2_wo_demo"].configuration(
            "stack", seed=0
        )
        config, model, _, overrides = experiment.configuration(
            "stack",
            "openai/gpt-6.1-sol",
            demonstrations=False,
            frozen="final_v2_wo_demo",
        )
        self.assertEqual(config, frozen)
        self.assertEqual(
            (model.provider, model.model), ("openrouter", "openai/gpt-6.1-sol")
        )
        self.assertEqual(model.extra_body, {"reasoning": {"effort": "medium"}})
        self.assertEqual(overrides["model_config"]["model"], "openai/gpt-6.1-sol")
        with self.assertRaisesRegex(ValueError, "OpenRouter id"):
            experiment.model_settings("gpt")

    def test_a_batch_without_gpus_has_no_gpu_scheduler(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "batch"
            experiment.create(
                directory,
                "stack",
                "claude-opus-5.5",
                trials=1,
                seed_start=0,
                workers=1,
                demonstrations=False,
                frozen="final_v2_wo_demo",
                gpus=[],
            )
            manifest = json.loads((directory / "manifest.json").read_text())
            plan = json.loads((directory / "comparison_plan.json").read_text())
        self.assertIsNone(manifest["gpu_config"])
        self.assertEqual(plan["worker_gpus"], [])


class FinalReportUsageTests(unittest.TestCase):
    def test_a_replayed_reply_is_counted_once_with_its_rerun(self):
        from ..reports import final_report

        with tempfile.TemporaryDirectory() as tmp:
            usage = {"prompt_tokens": 3, "completion_tokens": 2, "cost": 0.5}
            original = Path(tmp, "replaced/job-1/call_0001/response.json")
            original.parent.mkdir(parents=True)
            original.write_text(json.dumps({"usage": usage}))
            other = Path(tmp, "replaced/job-2/call_0001/response.json")
            other.parent.mkdir(parents=True)
            other.write_text(json.dumps({"usage": usage}))
            replayed = Path(tmp, "trials/job/call_0001/response.json")
            replayed.parent.mkdir(parents=True)
            replayed.write_text(
                json.dumps({"usage": usage, "replayed_from": str(original)})
            )
            sources = final_report.replayed_sources(tmp)
            self.assertEqual(sources, {str(original)})
            scored, _ = final_report.provider_usage(tmp)
            self.assertEqual((scored["calls"], scored["replayed"]), (1, 1))
            replaced, _ = final_report.provider_usage(tmp, "replaced", sources)
            self.assertEqual((replaced["calls"], replaced["cost"]), (1, 0.5))


if __name__ == "__main__":
    unittest.main()
