"""Revision 4.3: configurable recovery feedback items and executed-program history."""

import contextlib
import io
import json
import re
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest import mock

from synthesis.cfg.reset import restore

from .batch import variant_config
from .config import (
    FEEDBACK_ITEMS,
    ModelConfig,
    PrimitiveFeedbackConfig,
    RecoveryConfig,
    RunConfig,
)
from .llm import ScriptedModel, local_model_config, server_provenance
from .recovery import build_history, freeze_attempt
from .runner import run
from .test_adapter import RobotFixture

# Recorded from the pre-Revision 4.3 implementation; never regenerate from new code.
GOLDEN = Path(__file__).parent / "testdata/revision43_v1_golden.json"
NO_WITNESS = "Pick(b0)\nx=get(lambda x: not ON_star(x,x))"

# name -> (method, responses, fixture options, primitive feedback, run options)
SCENARIOS = {
    "baseline_no_witness_history": (
        "cap_google_baseline",
        [NO_WITNESS] * 5,
        {},
        {},
        {},
    ),
    "method_1_no_witness_history": ("cap_method_1", [NO_WITNESS] * 5, {}, {}, {}),
    "method_2_no_witness_history": ("cap_method_2", [NO_WITNESS] * 5, {}, {}, {}),
    "method_2_primitive_enriched": (
        "cap_method_2",
        ["Pick(b0)", "Pick(b0)", "pass"],
        {"stuck": 2},
        {"target_derivation": True, "controller_phases": True, "action_context": True},
        {},
    ),
    "method_1_goal_failure": ("cap_method_1", ["pass", "pass"], {"goal": 1}, {}, {}),
    "baseline_name_error": (
        "cap_google_baseline",
        ["missing_value", "pass"],
        {},
        {},
        {},
    ),
    "method_2_tick_budget": (
        "cap_method_2",
        ["while ON_star(b0,b0):\n    pass", "pass"],
        {},
        {},
        {"max_steps": 100},
    ),
}


def run_scenario(name, root, recovery=None, responses=None, **kwargs):
    """Run a scripted recovery and return prepared feedback and recovery prompts."""
    method, scripted, fixture, primitive, options = SCENARIOS[name]
    responses = scripted if responses is None else responses
    envs = []

    def factory(config, snapshot):
        failing = len(envs) < fixture.get("stuck", 0)
        env = RobotFixture(
            1,
            success=len(envs) >= fixture.get("goal", 0),
            stuck=failing,
        )
        if snapshot:
            restore(env, snapshot)
        envs.append(env)
        return env

    config = RunConfig(
        method=method,
        num_blocks=1,
        save_video=False,
        output_root=str(root),
        prompt_version="typed_v3",
        recovery=RecoveryConfig(**({"enabled": True} | (recovery or {}))),
        primitive_feedback=PrimitiveFeedbackConfig(**primitive),
        **options,
    )
    backend = ScriptedModel(responses)
    with contextlib.redirect_stdout(io.StringIO()):
        result = run(
            config,
            ModelConfig(max_output_tokens=16384),
            backend=backend,
            environment_factory=factory,
            capture_stdout=False,
            **kwargs,
        )
    rounds = Path(result["run_dir"]) / "artifacts/trial_001/rounds"
    feedback = [
        json.loads(path.read_text()) for path in sorted(rounds.glob("*/feedback.json"))
    ]
    prompts = [
        p[p.index("\n\nRECOVERY REQUEST") :]
        for p in backend.prompts
        if "\n\nRECOVERY REQUEST" in p
    ]
    return result, feedback, prompts, backend


EVIDENCE_KEYS = {
    "failure_location": ["last_source_location"],
    "failure_events": ["failure_context"],
    "query_candidates": ["query_candidates"],
    "keyframes": ["keyframes"],
    "goal_clauses": ["derived_goal_evidence"],
    "dsl_diagnostics": ["dsl_diagnostics"],
    "supporting_events": ["supporting_context"],
    "event_counts": ["event_counts"],
    "selection_metadata": ["selected_event_ids", "omitted_event_count"],
}
DSL_FAILURE = "if 1 < 2:\n    pass"
DISTINCT_FAILURES = [
    "x=get(lambda x: not ON_star(x,x))",
    "y=get(lambda y: not ON_star(y,y))",
    "Pick(b0)\nz=get(lambda z: not ON_star(z,z))",
    "x=get(lambda x: not ON_star(x,x))",
    "w=get(lambda w: not ON_star(w,w))",
]


def delivered_evidence(prompt):
    """The evidence object exactly as serialized into a recovery prompt."""
    lines = prompt.splitlines()
    index = next(
        i for i, line in enumerate(lines) if line.startswith("OBSERVED FAILURE")
    )
    return json.loads(lines[index + 1])


class Revision43Case(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def scenario(self, name, **kwargs):
        return run_scenario(name, Path(self.tmp.name) / str(id(kwargs)), **kwargs)


class DefaultProfileTests(Revision43Case):
    def test_default_profile_reproduces_v1_feedback_and_prompts(self):
        golden = json.loads(GOLDEN.read_text())
        for name in SCENARIOS:
            with self.subTest(scenario=name):
                _, feedback, prompts, _ = self.scenario(name)
                self.assertEqual(feedback, golden[name]["feedback"])
                self.assertEqual(prompts, golden[name]["prompts"])
                self.assertTrue(all("item_delivery" not in f for f in feedback))


class ConfigurationTests(unittest.TestCase):
    def test_defaults_and_round_trip(self):
        config = RecoveryConfig()
        self.assertTrue(config.default_profile)
        self.assertEqual(config.evidence_cap, 8192)
        nested = RunConfig(recovery={"history": {"mode": "executed_programs"}})
        again = RunConfig(**asdict(nested))
        self.assertEqual(again.recovery.history.mode, "executed_programs")
        self.assertFalse(again.recovery.default_profile)
        self.assertEqual(
            RecoveryConfig(feedback={"evidence_bytes": 32768}).evidence_cap, 32768
        )

    def test_invalid_settings_are_rejected(self):
        for settings in (
            {"feedback": {"bogus": True}},
            {"feedback": {"keyframes": "yes"}},
            {"feedback": {"evidence_layout": "packed"}},
            {"feedback": {"supporting_event_window": 513}},
            {"history": {"mode": "full"}},
            {"history": {"source_diff": True}},
            {
                "feedback": {"current_source": False},
                "history": {"mode": "executed_programs"},
            },
        ):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                RecoveryConfig(**settings)

    def test_batch_variants_merge_recovery_blocks_per_field(self):
        base = asdict(RunConfig(recovery={"enabled": True, "round_tokens": 200000}))
        merged = variant_config(
            base,
            {
                "recovery": {
                    "feedback": {"keyframes": False},
                    "history": {"mode": "none"},
                }
            },
        )
        recovery = RunConfig(**merged).recovery
        self.assertFalse(recovery.feedback.keyframes)
        self.assertTrue(recovery.feedback.goal_clauses)
        self.assertEqual(recovery.history.mode, "none")
        self.assertEqual(recovery.round_tokens, 200000)

    def test_local_model_configuration_and_provenance(self):
        config = local_model_config("qwen", "http://127.0.0.1:8106/v1")
        self.assertEqual(config.provider, "vllm")
        self.assertEqual(
            config.extra_body, {"chat_template_kwargs": {"enable_thinking": True}}
        )
        with self.assertRaises(ValueError):
            local_model_config("qwen", "http://example.com/v1")
        with self.assertRaises(ValueError):
            ModelConfig(provider="vllm", extra_body={"reasoning": {"effort": "medium"}})
        seen = []

        class Reply(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        def opener(request, timeout):
            seen.append((request.full_url, request.get_header("Authorization")))
            body = (
                {"version": "0.32.0"}
                if request.full_url.endswith("/version")
                else {
                    "data": [{"id": "qwen", "root": "Qwen/x", "max_model_len": 65536}]
                }
            )
            return Reply(json.dumps(body).encode())

        with mock.patch.dict("os.environ", {"VLLM_API_KEY": "secret-key"}):
            provenance = server_provenance(config, opener=opener)
        self.assertEqual(provenance["server_version"], "0.32.0")
        self.assertEqual(provenance["served_models"][0]["max_model_len"], 65536)
        self.assertEqual([u for u, _ in seen][0], "http://127.0.0.1:8106/v1/models")
        self.assertEqual(seen[1][0], "http://127.0.0.1:8106/version")
        self.assertNotIn("secret-key", json.dumps(provenance))


class ItemGatingTests(Revision43Case):
    def test_each_switch_removes_only_its_item_and_declares_it(self):
        for item in FEEDBACK_ITEMS:
            name, extra = "method_1_no_witness_history", {}
            if item == "dsl_diagnostics":
                name, extra = "baseline_name_error", {
                    "responses": [DSL_FAILURE, "pass"],
                }
            recovery = {"feedback": {item: False}}
            if item == "dsl_diagnostics":
                recovery["on_dsl_noncompliance"] = True
            with self.subTest(item=item):
                _, feedback, prompts, _ = self.scenario(
                    name, recovery=recovery, **extra
                )
                first = feedback[0]
                evidence = first["evidence"]
                self.assertEqual(evidence["schema"], "cap-recovery-v2")
                self.assertIn(item, evidence["omitted_by_configuration"])
                self.assertEqual(first["item_delivery"][item], "disabled")
                self.assertEqual(delivered_evidence(prompts[0]), evidence)
                for key in EVIDENCE_KEYS.get(item, []):
                    self.assertNotIn(key, evidence)
                if item == "current_source":
                    self.assertIsNone(first["current_sources"])
                    self.assertNotIn("CURRENT EXECUTED SOURCE", prompts[0])
                else:
                    self.assertIn("CURRENT EXECUTED SOURCE", prompts[0])
                others = set(FEEDBACK_ITEMS) - {item}
                self.assertFalse(
                    any(first["item_delivery"][o] == "disabled" for o in others)
                )

    def test_query_candidates_are_independent_of_failure_events(self):
        _, feedback, _, _ = self.scenario(
            "method_2_no_witness_history",
            recovery={"feedback": {"failure_events": False}},
        )
        evidence = feedback[0]["evidence"]
        self.assertNotIn("failure_context", evidence)
        self.assertEqual(len(evidence["query_candidates"]), 1)
        self.assertIn("candidates", evidence["query_candidates"][0])
        _, feedback, _, _ = self.scenario(
            "method_2_no_witness_history",
            recovery={"feedback": {"query_candidates": False}},
        )
        evidence = feedback[0]["evidence"]
        failure = [
            e for e in evidence["failure_context"] if e["kind"] == "query_failure"
        ]
        self.assertTrue(failure)
        self.assertTrue(all("candidates" not in e for e in failure))
        self.assertNotIn("query_candidates", evidence)

    def test_fixed_items_delivers_every_enabled_item_by_count(self):
        recovery = {
            "feedback": {
                "evidence_layout": "fixed_items",
                "evidence_bytes": 65536,
                "supporting_event_window": 2,
            }
        }
        _, feedback, _, _ = self.scenario(
            "method_1_no_witness_history", recovery=recovery
        )
        self.assertEqual(len(feedback[0]["evidence"]["supporting_context"]), 2)
        _, feedback, _, _ = self.scenario(
            "method_2_primitive_enriched", recovery=recovery
        )
        first = feedback[0]
        self.assertEqual(first["evidence_cap"], 65536)
        for item, status in first["item_delivery"].items():
            self.assertNotIsInstance(status, dict, item)
        self.assertEqual(first["item_delivery"]["action_context"], "complete")
        tight = {"feedback": recovery["feedback"] | {"evidence_bytes": 2000}}
        result, feedback, _, _ = self.scenario(
            "method_2_primitive_enriched", recovery=tight
        )
        self.assertEqual(result["stop_reason"], "required_feedback_exceeds_token_limit")

    def test_history_none_omits_earlier_attempts(self):
        _, feedback, prompts, _ = self.scenario(
            "method_1_no_witness_history", recovery={"history": {"mode": "none"}}
        )
        evidence = feedback[-1]["evidence"]
        self.assertNotIn("earlier_failures", evidence)
        self.assertIn("earlier_attempts", evidence["omitted_by_configuration"])
        self.assertIn("OBSERVED FAILURE (latest attempt)", prompts[-1])


def attempt(round_index, sources, bindings=None, status="primitive_failed"):
    return {
        "round": round_index,
        "candidate": f"c{round_index}",
        "duplicate_of": None,
        "changed_from_parent": bool(round_index),
        "rolled_out": True,
        "sources": sources,
        "bindings": bindings,
        "outcome": {"status": status, "reason": "x"},
        "failure_point": {"source_line": 1},
    }


class HistoryTests(Revision43Case):
    def settings(self, **values):
        return RecoveryConfig(history={"mode": "executed_programs"} | values).history

    def test_round_four_request_contains_all_earlier_programs(self):
        recovery = {"history": {"mode": "executed_programs", "source_diff": True}}
        result, feedback, prompts, _ = self.scenario(
            "method_1_no_witness_history",
            recovery=recovery,
            responses=DISTINCT_FAILURES,
        )
        self.assertEqual(result["recovery_rounds"], 4)
        self.assertNotIn("SOURCE REGISTRY", prompts[0])
        self.assertIsNone(feedback[0]["history"])
        last = feedback[-1]["history"]
        self.assertEqual(last["included_rounds"], [0, 1, 2])
        self.assertEqual(last["current"]["round"], 3)
        registry = last["registry"]
        bodies = list(registry.values())
        self.assertEqual(len(bodies), len(set(bodies)))
        for source in DISTINCT_FAILURES[:4]:
            self.assertTrue(any(source in body for body in bodies), source)
        references = set(re.findall(r"src_[0-9a-f]{12}", prompts[-1]))
        self.assertTrue(references)
        self.assertLessEqual(references, set(registry))
        self.assertIn("SOURCE REGISTRY", prompts[-1])
        self.assertIn("EXECUTED ATTEMPT HISTORY", prompts[-1])
        self.assertNotIn("earlier_failures", feedback[-1]["evidence"])
        # Round 3 repeats round 0's program: one registry body serves both.
        self.assertEqual(last["attempts"][0]["sources"], last["current"]["sources"])
        for index, attempt_result in enumerate(result["attempts"][1:]):
            delivered = json.loads(
                (
                    Path(attempt_result["run_dir"])
                    / "artifacts/trial_001/recovery_feedback.json"
                ).read_text()
            )
            self.assertEqual(delivered, feedback[index])

    def test_frozen_attempts_ignore_later_mutation(self):
        record = {
            "result": {"status": "goal_failed", "reason": "", "run_dir": "x"},
            "candidate_sha256": "c0",
            "rolled_out": True,
            "sources": {"artifacts/trial_001/checkpoints/cp_0001/policy.py": "a=1\n"},
            "bindings": {"policy": "p", "functions": {}},
            "events": [],
        }
        frozen = freeze_attempt(record, 0)
        record["sources"]["artifacts/trial_001/checkpoints/cp_0001/policy.py"] = "b"
        record["bindings"]["policy"] = "q"
        self.assertEqual(
            frozen["sources"],
            {"artifacts/trial_001/checkpoints/cp_0001/policy.py": "a=1\n"},
        )
        self.assertEqual(frozen["bindings"]["policy"], "p")

    def test_bindings_edit_summary_and_limits(self):
        helper_v1 = "def helper():\n    return 1\n"
        helper_v2 = "def helper():\n    return 2\n"
        bindings = lambda body: {
            "policy": "policy()",
            "functions": {"function_0": {"name": "helper", "source": body}},
        }
        earlier = [
            attempt(
                0, {"a/checkpoints/cp_0001/executed.py": "x=1\n"}, bindings(helper_v1)
            )
        ]
        latest = attempt(
            1, {"a/checkpoints/cp_0002/executed.py": "x=2\ny=3\n"}, bindings(helper_v2)
        )
        history = build_history(earlier, latest, self.settings(source_diff=True))
        registry = history["registry"]
        entry = history["attempts"][0]
        self.assertEqual(
            registry[entry["bindings"]["functions"]["function_0"]["source"]], helper_v1
        )
        self.assertEqual(
            registry[
                history["current"]["bindings"]["functions"]["function_0"]["source"]
            ],
            helper_v2,
        )
        summary = history["current"]["edit_summary"]
        self.assertEqual(
            summary["units_changed"],
            [{"unit": "executed.py", "lines_added": 2, "lines_removed": 1}],
        )
        self.assertIn("executed.py", history["current"]["diff"])
        tight = build_history(earlier, latest, self.settings(source_bytes=10))
        self.assertEqual(tight["reason"], "history_exceeds_source_limit")
        size = history["history_bytes"]
        no_diff = build_history(
            earlier, latest, self.settings(source_diff=True, source_bytes=size - 1)
        )
        self.assertEqual(no_diff.get("diffs_omitted"), "exceeds_source_bytes")
        self.assertNotIn("diff", no_diff["current"])
        self.assertIsNone(build_history([], latest, self.settings()))


class SeedTests(Revision43Case):
    def test_branch_starts_from_archived_failed_attempt(self):
        failed, _, _, _ = self.scenario(
            "method_2_no_witness_history", responses=[NO_WITNESS, "pass"]
        )
        seed = Path(failed["run_dir"]) / "artifacts/trial_001/rounds/000/result.json"
        branch, feedback, prompts, backend = self.scenario(
            "method_2_no_witness_history", responses=["pass"], seed_attempt=seed
        )
        self.assertTrue(branch["success"], branch)
        self.assertEqual(branch["recovery_rounds"], 1)
        self.assertEqual(len(backend.prompts), 1)
        self.assertIn("RECOVERY REQUEST", backend.prompts[0])
        self.assertEqual(branch["attempts"][0]["seeded_from"], str(seed.resolve()))
        self.assertEqual(
            branch["model_calls"], failed["attempts"][0]["model_calls"] + 1
        )
        identity = json.loads(
            (Path(branch["run_dir"]) / "artifacts/trial_001/seed.json").read_text()
        )
        self.assertEqual(identity["seed_run_dir"], failed["attempts"][0]["run_dir"])
        self.assertEqual(
            feedback[0]["current_sources"],
            json.loads(
                (
                    Path(failed["run_dir"])
                    / "artifacts/trial_001/rounds/000/feedback.json"
                ).read_text()
            )["current_sources"],
        )
        succeeded = (
            Path(failed["run_dir"]) / "artifacts/trial_001/rounds/001/result.json"
        )
        with self.assertRaises(ValueError):
            self.scenario(
                "method_2_no_witness_history",
                responses=["pass"],
                seed_attempt=succeeded,
            )
