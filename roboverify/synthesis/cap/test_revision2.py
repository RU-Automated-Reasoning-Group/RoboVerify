"""Shared-checker, final-output audit, and whole-candidate repair regressions."""

import contextlib
import io
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from .audit import audit_run
from .checker import AnalysisFailure, DSLContract, check_bundle
from .config import ModelConfig, RunConfig
from .contracts import Bundle
from .llm import ScriptedModel
from .runner import run
from .test_adapter import RobotFixture


class SharedCheckerTests(unittest.TestCase):
    def test_task_mode_local_and_complete_scopes(self):
        for mode in ("dsl_only", "coordinate_predicates"):
            contract = DSLContract.for_task("stack", mode)
            result = check_bundle(Bundle("if 1 < 2:\n    pass"), contract)
            self.assertEqual(result.status, "fail" if mode == "dsl_only" else "pass")
            unavailable = check_bundle(Bundle("ON_star_zero(b0,b0)"), contract)
            self.assertEqual(unavailable.diagnostics[0]["rule"], "API_UNAVAILABLE")
            self.assertEqual(
                check_bundle(Bundle("missing(b0)"), contract, scope="local").status,
                "pending",
            )
            self.assertEqual(
                check_bundle(Bundle("missing(b0)"), contract).status, "fail"
            )
        self.assertEqual(
            check_bundle(
                Bundle("ON_star_zero(b0,b0)"), DSLContract.for_task("reverse")
            ).status,
            "pass",
        )

    def test_analysis_errors_and_deadlines_are_not_code_failures(self):
        with patch(
            "synthesis.cap.checker.validate", side_effect=RuntimeError("internal")
        ):
            self.assertEqual(
                check_bundle(Bundle("pass"), DSLContract()).status, "error"
            )
        with patch(
            "synthesis.cap.checker.validate",
            side_effect=lambda *a, **k: time.sleep(0.2),
        ):
            report = check_bundle(Bundle("pass"), DSLContract(), timeout=0.01)
            self.assertEqual(report.status, "error")
            self.assertIn("deadline", report.reason)


class Revision2RunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def execute(self, method, responses, mode="dsl_only", audit=True, attempts=3):
        self.env = RobotFixture(1, success=True)
        self.backend = ScriptedModel(responses)
        with contextlib.redirect_stdout(io.StringIO()):
            return run(
                RunConfig(
                    method=method,
                    condition_mode=mode,
                    prompt_version="typed_v3",
                    num_blocks=1,
                    save_video=False,
                    output_root=self.tmp.name,
                    baseline_audit=audit,
                    max_attempts=attempts,
                ),
                ModelConfig(),
                backend=self.backend,
                environment_factory=lambda c, s: self.env,
                capture_stdout=False,
            )

    def test_method2_repairs_whole_candidate_before_any_action(self):
        result = self.execute(
            "cap_method_2", ["perform(b0)", "def perform(b):\n    Pick(b)\nperform(b0)"]
        )
        self.assertTrue(result["success"])
        self.assertEqual(len(self.backend.prompts), 2)
        self.assertIn("entire replacement source", self.backend.prompts[1])
        root = Path(result["run_dir"]) / "artifacts/trial_001"
        events = [
            json.loads(line)
            for line in (root / "timeline.jsonl").read_text().splitlines()
        ]
        self.assertEqual(
            [e["purpose"] for e in events if e["kind"] == "generation_start"],
            ["policy", "repair"],
        )
        self.assertEqual(sum(e["kind"] == "primitive_start" for e in events), 1)

    def test_method2_invalid_candidates_never_execute_and_no_rollout_repair(self):
        result = self.execute(
            "cap_method_2", ["if Pick(b0):\n    pass"] * 2, attempts=2
        )
        self.assertEqual(result["status"], "constraint_failed")
        self.assertEqual(len(self.env.actions), 0)
        result = self.execute("cap_method_2", ["Move(b0,b0,b0,[99,99,99])"])
        self.assertEqual(result["status"], "primitive_failed")
        self.assertEqual(len(self.backend.prompts), 1)

    def test_baseline_audit_is_posthoc_and_mode_aware(self):
        import numpy as np

        actions = []
        for mode in ("dsl_only", "coordinate_predicates"):
            result = self.execute(
                "cap_google_baseline", ["if 1 < 2:\n    Pick(b0)"], mode
            )
            self.assertTrue(result["success"])
            self.assertEqual(
                result["dsl_compliance"], "fail" if mode == "dsl_only" else "pass"
            )
            actions.append(self.env.actions.copy())
        result = self.execute(
            "cap_google_baseline", ["if 1 < 2:\n    Pick(b0)"], audit=False
        )
        actions.append(self.env.actions)
        for action in actions[1:]:
            np.testing.assert_array_equal(actions[0], action)

    def test_final_inline_helper_supersedes_unused_generated_version(self):
        result = self.execute(
            "cap_google_baseline",
            [
                "def ready(b):\n    return Higher(b,b)\nif ready(b0):\n    Pick(b0)",
                "def ready(b):\n    return get_block_position(b)[2] > 0",
            ],
        )
        self.assertTrue(result["success"])
        self.assertEqual(result["dsl_compliance"], "pass")

    def test_final_dependency_binding_retains_a_referenced_version(self):
        result = self.execute(
            "cap_google_baseline",
            [
                "def ready(b):\n    return Higher(b,b)\ndef check(b):\n    return ready(b)\nif check(b0):\n    Pick(b0)",
                "def ready(b):\n    return get_block_position(b)[2] > 0",
                "def check(b):\n    return ready(b)",
            ],
        )
        self.assertTrue(result["success"])
        self.assertEqual(result["dsl_compliance"], "fail")

    def test_query_lambda_captures_actual_global_helper_version(self):
        responses = [
            "def ready(b):\n    return Higher(b,b)\ndef check(b):\n    return exists(lambda x: ready(x))\nif check(b0):\n    Pick(b0)",
            "def ready(b):\n    return get_block_position(b)[2] > 0",
            "def check(b):\n    return exists(lambda x: ready(x))",
        ]
        for mode, expected in (("dsl_only", "fail"), ("coordinate_predicates", "pass")):
            with self.subTest(mode=mode):
                result = self.execute("cap_google_baseline", responses, mode=mode)
                self.assertTrue(result["success"])
                self.assertEqual(result["dsl_compliance"], expected)
                root = Path(result["run_dir"])
                report = json.loads(
                    (root / "artifacts/trial_001/dsl_compliance.json").read_text()
                )
                if expected == "fail":
                    self.assertEqual(
                        report["diagnostics"][0]["rule"], "NUMERIC_CONDITION"
                    )
                pointer = root / "artifacts/trial_001/final_output.json"
                target = root / json.loads(pointer.read_text())["artifact"]
                output = json.loads(target.read_text())
                self.assertEqual(output["schema"], "cap-final-output-v2")
                parent = output["functions"][output["bindings"]["check"]]
                child = output["functions"][parent["bindings"]["ready"]]
                self.assertEqual(child["source_kind"], "helper")

                # Simulate the earlier incomplete capture without inventing bindings.
                output["schema"] = "cap-final-output-v1"
                parent["bindings"].pop("ready")
                legacy = pointer.parent / "legacy_output.json"
                legacy.write_text(json.dumps(output))
                pointer.write_text(
                    json.dumps({"artifact": str(legacy.relative_to(root))})
                )
                self.assertEqual(audit_run(root)["status"], "unsupported")

    def test_incomplete_no_source_and_physical_failure_are_distinct(self):
        result = self.execute("cap_google_baseline", ["missing(b0)"])
        self.assertEqual(result["dsl_compliance"], "incomplete")
        result = self.execute("cap_google_baseline", [])
        self.assertEqual(result["dsl_compliance"], "not_available")
        result = self.execute("cap_google_baseline", ["Move(b0,b0,b0,[99,99,99])"])
        self.assertEqual(result["status"], "primitive_failed")
        self.assertEqual(result["dsl_compliance"], "pass")
        again = audit_run(result["run_dir"])
        self.assertEqual(again["status"], "pass")

    def test_three_methods_share_valid_action_execution(self):
        import numpy as np

        actions = []
        for mode in ("dsl_only", "coordinate_predicates"):
            for method in ("cap_google_baseline", "cap_method_1", "cap_method_2"):
                result = self.execute(
                    method, ["Pick(b0)\nMove(b0,b0,b0,[0,0,.1])"], mode
                )
                self.assertTrue(result["success"])
                self.assertEqual(result["dsl_compliance"], "pass")
                actions.append(self.env.actions.copy())
        for action in actions[1:]:
            np.testing.assert_array_equal(actions[0], action)


if __name__ == "__main__":
    unittest.main()
