"""Offline integration tests for both actual method implementations."""

import ast
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from ..adapter import GenerationOnlyAdapter
from ..budget import Budget
from ..cap_google_baseline import GoogleBaseline
from ..cap_method_1 import Method1
from ..checkpoints import Checkpoints
from ..contracts import BudgetExceeded, ConstraintFailure, ObjectHandle
from ..llm import LoggedModel, ScriptedModel
from ..logger import CapLogger


class FakeAdapter(GenerationOnlyAdapter):
    def __init__(self):
        super().__init__()
        self.actions = []

    def call(self, name, args):
        if name in {"Higher", "ON_star", "eq"}:
            return args[0] == args[1]
        if name == "Scattered":
            return False
        if name in {"exists", "forall", "get"}:
            result = args[0](self.b0)
            return self.b0 if name == "get" and result else result
        self.actions.append((name, args))
        return None


class MethodsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.logger = CapLogger(self.tmp.name, console=False)
        self.budget = Budget(timeout=20)
        self.adapter = FakeAdapter()

    def tearDown(self):
        self.logger.close()
        self.tmp.cleanup()

    def method(self, responses, baseline=False, attempts=3):
        self.backend = ScriptedModel(responses)
        model = LoggedModel(self.backend, self.logger, self.budget)
        name = "cap_google_baseline" if baseline else "cap_method_1"
        cp = Checkpoints(self.logger, method=name)
        return (
            GoogleBaseline(model, self.adapter, self.logger, cp, self.budget)
            if baseline
            else Method1(model, self.adapter, self.logger, cp, self.budget, attempts)
        )

    def events(self):
        return [
            json.loads(line)
            for line in (self.logger.root / "timeline.jsonl").read_text().splitlines()
        ]

    def test_recursive_helper_and_local_repair(self):
        method = self.method(
            [
                "if ready(b0):\n    perform(b0)",
                "def ready(b):\n    return get_obj_pos(b)[2] > 0.1",
                "def ready(b):\n    return Higher(b, b)",
                "def perform(b):\n    grab(b)",
                "def grab(b):\n    Pick(b)",
            ]
        )
        method("Manipulate the input block")
        self.assertEqual([a[0] for a in self.adapter.actions], ["Pick"])
        self.assertEqual(len(self.backend.prompts), 5)
        self.assertTrue(any(e["kind"] == "validation_failed" for e in self.events()))
        self.assertIn("FORBIDDEN_ACCESS", self.backend.prompts[2])
        manifests = list((self.logger.root / "checkpoints").glob("*/manifest.json"))
        self.assertGreater(len(manifests), 5)
        child_checkpoints = [
            json.loads(p.read_text())
            for p in manifests
            if json.loads(p.read_text()).get("unit") == "grab"
        ]
        self.assertTrue(child_checkpoints)
        self.assertTrue(
            all("helpers/perform.py" in cp["sources"] for cp in child_checkpoints)
        )

    def test_initial_policy_prompt_matches_original_lmp_assembly(self):
        from ..cap_method_1.generation import prompt

        method = self.method([], baseline=True)
        original, _ = method.policy.build_prompt(
            "Manipulate", context="b0 is the input block."
        )
        constrained = prompt(
            "policy",
            api_names=self.adapter.api_names,
            instruction="Manipulate",
            context="b0 is the input block.",
        )
        self.assertEqual(original, constrained)

    def test_vendored_core_matches_pinned_notebook_definitions(self):
        root = Path(__file__).parents[1] / "vendor/google_cap"
        raw = (root / "Interactive_Demo.ipynb").read_bytes()
        provenance = json.loads((root / "PROVENANCE.json").read_text())
        self.assertEqual(hashlib.sha256(raw).hexdigest(), provenance["sha256"])
        notebook = json.loads(raw)
        core = {
            node.name: node
            for node in ast.parse((root / "core.py").read_text()).body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef))
        }
        for item in provenance["extraction"]:
            source = "".join(notebook["cells"][item["cell_index"]]["source"])
            original = next(
                n
                for n in ast.parse(source).body
                if getattr(n, "name", None) == item["name"]
            )
            self.assertEqual(ast.dump(original), ast.dump(core[item["name"]]))

    def test_top_level_repair_before_any_action(self):
        method = self.method(
            [
                "Pick(b0)\nif get_obj_pos(b0)[2]:\n    Pick(b0)",
                "if Higher(b0, b0):\n    Pick(b0)",
            ]
        )
        method("Pick the input")
        self.assertEqual(len(self.adapter.actions), 1)

    def test_helper_condition_actions_rejected_and_repaired(self):
        method = self.method(
            [
                "if ready(b0):\n    Release(b0, 0.1)",
                "def ready(b):\n    Pick(b)\n    return True",
                "def ready(b):\n    return eq(b, b)",
            ]
        )
        method("Release")
        self.assertEqual([a[0] for a in self.adapter.actions], ["Release"])

    def test_failed_attempt_never_runs(self):
        method = self.method(["if Pick(b0):\n    Pick(b0)"] * 2, attempts=2)
        with self.assertRaises(ConstraintFailure):
            method("Pick")
        self.assertEqual(self.adapter.actions, [])

    def test_baseline_retains_original_recursive_generation(self):
        method = self.method(
            [
                "perform(b0)",
                "def perform(b):\n    grab(b)",
                "def grab(b):\n    Pick(b)",
            ],
            baseline=True,
        )
        method("Pick")
        self.assertEqual([a[0] for a in self.adapter.actions], ["Pick"])
        self.assertFalse(any(e["kind"] == "validation_failed" for e in self.events()))
        self.assertTrue(any(e["kind"] == "python_line" for e in self.events()))
        for path in (self.logger.root / "checkpoints").glob("*/manifest.json"):
            self.assertEqual(json.loads(path.read_text())["validation"], "not_run")

    def test_baseline_has_no_dsl_repair_or_condition_gate(self):
        method = self.method(["if Pick(b0):\n    Release(b0, 0.1)"], baseline=True)
        method("Pick")
        self.assertEqual([a[0] for a in self.adapter.actions], ["Pick"])
        self.assertEqual(len(self.backend.prompts), 1)

    def test_baseline_keeps_initial_policy_when_helper_binding_fails(self):
        method = self.method(
            ["perform(b0)", "def perform(b):\n    import os"], baseline=True
        )
        with self.assertRaises(AssertionError):
            method("Pick")
        sources = list((self.logger.root / "checkpoints").glob("*/policy.py"))
        self.assertTrue(sources)
        self.assertTrue(all(p.read_text().strip() == "perform(b0)" for p in sources))
        self.assertTrue(
            list((self.logger.root / "checkpoints").glob("*/helpers/perform.py"))
        )

    def test_baseline_infinite_loop_is_bounded(self):
        self.budget.max_steps = 30
        method = self.method(["while True:\n    pass"], baseline=True)
        with self.assertRaises(BudgetExceeded):
            method("Wait")

    def test_method_loop_is_bounded(self):
        self.budget.max_steps = 30
        method = self.method(["while True:\n    pass"])
        with self.assertRaises(BudgetExceeded):
            method("Wait")

    def test_model_budget_is_shared_across_helpers(self):
        self.budget.max_model_calls = 2
        method = self.method(["first(b0)", "def first(b):\n    second(b)"])
        with self.assertRaises(BudgetExceeded):
            method("Pick")

    def test_inline_helpers_queries_and_nested_loops(self):
        method = self.method(
            [
                "def ready(b):\n    return Higher(b,b)\nb = get(lambda x: eq(x,b0))\nwhile ready(b):\n    while exists(lambda x: eq(x,b)):\n        Pick(b)\n        break\n    break"
            ]
        )
        method("Pick")
        self.assertEqual(len(self.adapter.actions), 1)


if __name__ == "__main__":
    unittest.main()
