"""Recovery preserves initial states, method boundaries, history and total costs."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from synthesis.cfg.reset import restore

from .budget import Budget
from .config import ModelConfig, RecoveryConfig, RunConfig
from .contracts import BudgetExceeded
from .llm import ScriptedModel
from .recovery import RoundBudget, select_relevant_evidence, token_bound
from .runner import run
from .test_adapter import RobotFixture


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.envs = []

    def tearDown(self):
        self.tmp.cleanup()

    def factory(self, config, snapshot):
        env = RobotFixture(config.num_blocks, success=True)
        if snapshot:
            restore(env, snapshot)
        self.envs.append(env)
        return env

    def execute(self, method, responses, **options):
        config = RunConfig(
            method=method,
            num_blocks=1,
            save_video=False,
            output_root=self.tmp.name,
            prompt_version="typed_v3",
            recovery=RecoveryConfig(**options),
        )
        backend = ScriptedModel(responses)
        with contextlib.redirect_stdout(io.StringIO()):
            result = run(
                config,
                ModelConfig(max_output_tokens=16384),
                backend=backend,
                environment_factory=self.factory,
                capture_stdout=False,
            )
        return result, backend

    def test_all_methods_recover_from_no_witness_with_exact_initial_state(self):
        for method in ("cap_google_baseline", "cap_method_1", "cap_method_2"):
            with self.subTest(method=method):
                result, backend = self.execute(
                    method,
                    ["Pick(b0)\nx=get(lambda x: not ON_star(x,x))", "pass"],
                    enabled=True,
                )
                self.assertTrue(result["success"], result)
                self.assertFalse(result["first_attempt_success"])
                self.assertEqual(result["recovery_rounds"], 1)
                self.assertEqual(result["rollout_count"], 2)
                self.assertEqual(result["model_calls"], 2)
                self.assertNotIn("RECOVERY REQUEST", backend.prompts[0])
                self.assertIn("query_failure", backend.prompts[1])
                self.assertIn("saved initial state", backend.prompts[1])
                root = Path(result["attempts"][1]["run_dir"]) / "artifacts/trial_001"
                events = [
                    json.loads(s)
                    for s in (root / "timeline.jsonl").read_text().splitlines()
                ]
                restoration = next(
                    e for e in events if e["kind"] == "initial_state_restoration"
                )
                self.assertTrue(restoration["exact_match"])
                self.assertTrue((root / "recovery_feedback.json").exists())
                feedback = json.loads((root / "recovery_feedback.json").read_text())
                self.assertEqual(feedback["evidence"]["earlier_failures"], [])
                self.assertIn('"earlier_failures":[]', backend.prompts[1])

    def test_disabled_is_one_attempt_and_unchanged_prompt(self):
        failed = "x=get(lambda x: not ON_star(x,x))"
        off, off_model = self.execute("cap_method_2", [failed, "pass"], enabled=False)
        on, on_model = self.execute("cap_method_2", [failed, "pass"], enabled=True)
        self.assertFalse(off["success"])
        self.assertEqual(len(off_model.prompts), 1)
        self.assertEqual(off_model.prompts[0], on_model.prompts[0])
        self.assertTrue(on["success"])

    def test_four_recoveries_accumulate_history_and_allow_duplicates(self):
        source = "x=get(lambda x: not ON_star(x,x))"
        result, model = self.execute("cap_method_1", [source] * 5, enabled=True)
        self.assertFalse(result["success"])
        self.assertEqual(result["recovery_rounds"], 4)
        self.assertEqual(result["rollout_count"], 5)
        self.assertEqual(len(model.prompts), 5)
        root = Path(result["run_dir"]) / "artifacts/trial_001"
        feedback = json.loads((root / "rounds/003/feedback.json").read_text())
        self.assertEqual(len(feedback["evidence"]["earlier_failures"]), 3)
        identity = json.loads((root / "rounds/004/identity.json").read_text())
        self.assertEqual(identity["duplicate_of"], 0)
        self.assertLessEqual(feedback["feedback_token_bound"], 8192)
        for index, attempt in enumerate(result["attempts"][1:], 1):
            delivered = json.loads(
                (
                    Path(attempt["run_dir"])
                    / "artifacts/trial_001/recovery_feedback.json"
                ).read_text()
            )
            self.assertEqual(len(delivered["evidence"]["earlier_failures"]), index - 1)
            prepared = json.loads(
                (root / f"rounds/{index-1:03d}/feedback.json").read_text()
            )
            self.assertEqual(prepared, delivered)

    def test_generation_failure_has_no_invented_rollout(self):
        result, model = self.execute("cap_method_2", [""], enabled=True)
        self.assertFalse(result["success"])
        self.assertEqual(result["rollout_count"], 0)
        self.assertEqual(result["stop_reason"], "preparation_failure_no_rollout")
        self.assertEqual(len(model.prompts), 1)

    def test_baseline_keeps_posthoc_audit_and_optional_dsl_trigger(self):
        source = "if 1 < 2:\n    pass"
        result, model = self.execute("cap_google_baseline", [source], enabled=True)
        self.assertTrue(result["success"])
        self.assertEqual(result["dsl_compliance"], "fail")
        result, model = self.execute(
            "cap_google_baseline",
            [source, "pass"],
            enabled=True,
            on_dsl_noncompliance=True,
        )
        self.assertTrue(result["success"])
        self.assertEqual(result["recovery_rounds"], 1)
        self.assertEqual(result["dsl_compliance"], "pass")

    def test_all_failure_categories_use_outcome_not_an_allowlist(self):
        result, model = self.execute(
            "cap_google_baseline", ["missing_value", "pass"], enabled=True
        )
        self.assertTrue(result["success"], result)
        self.assertEqual(result["recovery_rounds"], 1)

    def test_goal_primitive_tick_and_unknown_evaluation_failures_can_recover(self):
        from unittest.mock import patch
        from .tasks import evaluate_terminal

        for scenario, source in (
            ("goal", "pass"),
            ("primitive", "Pick(b0)"),
            ("ticks", "while ON_star(b0,b0):\n    pass"),
            ("unknown_goal", "pass"),
        ):
            with self.subTest(scenario=scenario):
                envs = []

                def factory(config, snapshot):
                    env = RobotFixture(
                        1,
                        success=scenario != "goal" or bool(envs),
                        stuck=scenario == "primitive" and not envs,
                    )
                    if snapshot:
                        restore(env, snapshot)
                    envs.append(env)
                    return env

                calls = [0]

                def evaluate(*args):
                    calls[0] += 1
                    if calls[0] <= 2:
                        raise ValueError("Synthetic unavailable task evaluation")
                    return evaluate_terminal(*args)

                context = (
                    patch(
                        "synthesis.cap.runner.evaluate_terminal", side_effect=evaluate
                    )
                    if scenario == "unknown_goal"
                    else contextlib.nullcontext()
                )
                config = RunConfig(
                    method="cap_method_2",
                    num_blocks=1,
                    save_video=False,
                    output_root=self.tmp.name,
                    prompt_version="typed_v3",
                    max_steps=100,
                    recovery=RecoveryConfig(enabled=True),
                )
                with context, contextlib.redirect_stdout(io.StringIO()):
                    result = run(
                        config,
                        ModelConfig(),
                        backend=ScriptedModel([source, "pass"]),
                        environment_factory=factory,
                        capture_stdout=False,
                    )
                self.assertTrue(result["success"])
                self.assertFalse(result["first_attempt_success"])
                self.assertEqual(result["recovery_rounds"], 1)
                if scenario == "unknown_goal":
                    self.assertIsNone(result["attempts"][0]["environment_success"])

    def test_insufficient_feedback_records_stop_without_request(self):
        result, model = self.execute(
            "cap_method_2",
            ["x=get(lambda x: not ON_star(x,x))"],
            enabled=True,
            feedback_tokens=1,
        )
        self.assertEqual(result["stop_reason"], "required_feedback_exceeds_token_limit")
        self.assertEqual(len(model.prompts), 1)


class RecoveryBudgetTests(unittest.TestCase):
    def budget(self):
        parent = Budget(max_tokens=100, max_model_calls=2, max_steps=2, max_sim_steps=2)
        parent.charged_tokens = 0
        child = RoundBudget(max_tokens=100)
        child.parent = parent
        child.output_tokens = 10
        return child, parent

    def test_reservations_reconcile_once_and_survive_unknown_usage(self):
        child, parent = self.budget()
        child.reserve_request("abc")
        child.account_tokens(7)
        self.assertEqual(parent.tokens, 7)
        self.assertEqual(parent.charged_tokens, 7)
        child.reserve_request("abc")
        child.account_tokens(None)
        self.assertEqual(parent.charged_tokens, 20)
        self.assertEqual(parent.unknown_usage_calls, 1)

    def test_cumulative_limit_cannot_reset_with_new_round(self):
        child, parent = self.budget()
        child.tick()
        child.tick()
        other = RoundBudget()
        other.parent = parent
        with self.assertRaises(BudgetExceeded):
            other.tick()
        child.model_call()
        child.model_call()
        with self.assertRaises(BudgetExceeded):
            other.model_call()

    def test_dispatch_checks_input_and_output_reservation(self):
        child, parent = self.budget()
        with self.assertRaises(BudgetExceeded):
            child.reserve_request("a" * 91)
        self.assertEqual(parent.calls, 0)
        self.assertEqual(parent.charged_tokens, 0)


if __name__ == "__main__":
    unittest.main()
