"""Artifact analysis must not confuse code edits with verified repair causes."""

import ast
import unittest
from pathlib import Path

from .recovery_analysis import (
    features,
    outcome_class,
    parse_recovery_prompt,
    source_changes,
    source_units,
)


class RecoveryAnalysisTests(unittest.TestCase):
    def test_missing_response_is_not_an_action_deletion(self):
        result = source_changes({"policy.py": "Move(b0,b0,b0,[0,0,0.1])"}, {})
        self.assertEqual(result["tags"], ["no_replacement_source"])

    def test_numeric_change_separated_from_condition_change(self):
        old = {"policy.py": "while ON_star(x, b0):\n    Move(x,x,x,[0,0,0.05])"}
        new = {"policy.py": "while ON_star(x, b0):\n    Move(x,x,x,[0,0,0.15])"}
        self.assertIn("numeric_literals_only", source_changes(old, new)["tags"])
        self.assertNotIn(
            "condition_or_selection_changed", source_changes(old, new)["tags"]
        )
        new["policy.py"] = new["policy.py"].replace(
            "while ON_star", "while not ON_star"
        )
        self.assertNotIn("numeric_literals_only", source_changes(old, new)["tags"])
        self.assertIn(
            "condition_or_selection_changed", source_changes(old, new)["tags"]
        )

    def test_boolean_constants_are_not_numeric_parameters(self):
        self.assertNotIn(
            "numeric_literals_only",
            source_changes({"a.py": "flag=True"}, {"a.py": "flag=False"})["tags"],
        )

    def test_duplicate_paths_do_not_multiply_same_source(self):
        sources = {"cp1/policy.py": "Pick(b0)\n", "cp2/executed.py": "Pick(b0)"}
        self.assertEqual(len(source_units(sources)), 1)
        self.assertEqual(features(sources)["primitive_structure"], ["Pick"])

    def test_formatting_differences_do_not_duplicate_ast_units(self):
        a = {"cp1/policy.py": "Pick( b0 )", "cp2/executed.py": "Pick(b0)"}
        b = {"policy.py": "Pick(b0)"}
        self.assertIn("same_ast", source_changes(a, b)["tags"])
        self.assertEqual(features(a)["primitive_structure"], ["Pick"])

    def test_helper_versions_are_not_collapsed(self):
        self.assertEqual(
            len(
                source_units(
                    {
                        "cp1/helper.py": "def f(): return True",
                        "cp2/helper.py": "def f(): return False",
                    }
                )
            ),
            2,
        )

    def test_preparation_is_not_a_failed_rollout(self):
        result = {"status": "failed", "reason": "SyntaxError: invalid syntax"}
        self.assertEqual(outcome_class(result, [], False), "preparation_no_rollout")
        self.assertEqual(outcome_class(result, [], True), "SyntaxError")

    def test_primitive_class_uses_failed_controller(self):
        result = {"status": "primitive_failed"}
        events = [
            {"kind": "primitive_end", "primitive": "Pick", "converged": True},
            {"kind": "primitive_end", "primitive": "Move", "converged": False},
        ]
        self.assertEqual(outcome_class(result, events, True), "primitive_Move")

    def test_pinned_baseline_lambda_cannot_read_top_level_local(self):
        core = Path(__file__).parent / "vendor/google_cap/core.py"
        namespace = {"ast": ast}
        exec(compile(core.read_text(), str(core), "exec"), namespace)
        exec_safe = namespace["exec_safe"]

        values = {"b0": object(), "exists": lambda predicate: predicate(None)}
        local_values = {}
        with self.assertRaisesRegex(NameError, "base"):
            exec_safe(
                "base = b0\nanswer = exists(lambda x: x is not base)",
                values,
                local_values,
            )
        self.assertIs(local_values["base"], values["b0"])
        # A policy-local variable is a lexical closure, not an exec top-level local.
        local_values = {}
        exec_safe(
            "def policy():\n    base = b0\n    return exists(lambda x: x is not base)\nanswer = policy()",
            values,
            local_values,
        )
        self.assertTrue(local_values["answer"])

    def test_parse_actual_payload_not_keyword_presence(self):
        prompt = 'prefix\nCURRENT EXECUTED SOURCE AND BINDINGS\n{"sources":{},"bindings":null}\nOBSERVED FAILURE AND ACCUMULATED HISTORY\n{"latest":{"round":2}}\nOutput complete replacement policy'
        sources, evidence = parse_recovery_prompt(prompt)
        self.assertEqual(evidence["latest"]["round"], 2)
        self.assertIsNone(sources["bindings"])


if __name__ == "__main__":
    unittest.main()
