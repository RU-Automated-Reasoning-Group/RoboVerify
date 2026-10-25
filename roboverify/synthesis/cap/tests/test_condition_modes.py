"""Mode boundaries across source checking, generation, and actual execution."""

import inspect
import json
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from ..budget import Budget
from ..cap_google_baseline import GoogleBaseline
from ..cap_method_1 import Method1
from ..cap_method_1.generation import prompt
from ..checkpoints import Checkpoints
from ..config import RunConfig
from ..contracts import CONDITION_MODES, Bundle, ConstraintFailure, Position3
from ..llm import LoggedModel, ScriptedModel
from ..logger import CapLogger
from ..runtime import Runtime
from ..validation import validate
from .test_methods import FakeAdapter


class PositionAdapter(FakeAdapter):
    def __init__(self):
        super().__init__()
        self.reads = []

    def call(self, name, args):
        if name in {"get_gripper_position", "get_block_position"}:
            self.reads.append(name)
            return Position3((0.123456789, 0.2, 0.3))
        if name in {"exists", "forall", "get"}:
            query = args[0]
            arity = getattr(query, "arity", None) or len(
                inspect.signature(query).parameters
            )
            result = query(*([self.b0] * arity))
            return self.b0 if name == "get" and result else result
        return super().call(name, args)


class ConditionModeTests(unittest.TestCase):
    @contextmanager
    def execution(self, mode="dsl_only", responses=(), baseline=False):
        with tempfile.TemporaryDirectory() as tmp:
            logger = CapLogger(tmp, console=False)
            try:
                adapter, budget = PositionAdapter(), Budget(timeout=20)
                backend = ScriptedModel(responses)
                model = LoggedModel(backend, logger, budget)
                name = "cap_google_baseline" if baseline else "cap_method_1"
                checkpoints = Checkpoints(logger, method=name, condition_mode=mode)
                method = (GoogleBaseline if baseline else Method1)(
                    model, adapter, logger, checkpoints, budget, condition_mode=mode
                )
                yield method, adapter, backend, logger, budget
            finally:
                logger.close()

    def test_modes_validate_configuration_and_match_prompt_text(self):
        self.assertEqual(RunConfig().condition_mode, "dsl_only")
        with self.assertRaises(ValueError):
            RunConfig(condition_mode="unknown")
        for mode in CONDITION_MODES:
            with self.subTest(mode=mode), self.execution(mode, baseline=True) as data:
                method, adapter, _, _, _ = data
                actual, _ = method.policy.build_prompt(
                    "Task", context="b0 is the input block."
                )
                self.assertEqual(
                    actual,
                    prompt(
                        "policy",
                        api_names=adapter.api_names,
                        condition_mode=mode,
                        instruction="Task",
                        context="b0 is the input block.",
                    ),
                )
                self.assertIn(f"Condition mode: {mode}", actual)
                self.assertIn("get_gripper_position", actual)

    def test_numeric_conditions_direct_indirect_and_query_modes(self):
        sources = [
            "if get_block_position(b0)[2] > 0.1:\n    pass",
            "p = get_gripper_position()\nx = p[0]\nready = x <= 2\nif ready:\n    pass",
            "def inner(x):\n    return x >= 0\ndef outer(b):\n    return inner(get_block_position(b)[2])\nready = outer(b0)\nwhile ready:\n    break",
            "get(lambda b: get_block_position(b)[0] != 3)",
            "forall(lambda a, b: get_block_position(a)[2] == get_block_position(b)[2])",
            "counter = 0\nwhile counter < 2:\n    counter = counter + 1",
            "def below(a,b):\n    return a < b\nif below(1,2):\n    pass",
            "if 0 < 1 <= 2 != 3:\n    pass",
        ]
        for source in sources:
            with self.subTest(source=source):
                with self.assertRaises(ConstraintFailure):
                    validate(Bundle(source))
                validate(Bundle(source), condition_mode="coordinate_predicates")
                with self.execution("coordinate_predicates", [source]) as data:
                    method, adapter, backend, _, _ = data
                    method("Exercise numeric conditions")
                    self.assertEqual(len(backend.prompts), 1)
                    self.assertEqual(adapter.actions, [])

    def test_numeric_comparisons_preserve_integer_precision(self):
        source = "n=9007199254740992 + 1\nif n > 9007199254740992:\n    Pick(b0)"
        for baseline in (False, True):
            with (
                self.subTest(baseline=baseline),
                self.execution(
                    "coordinate_predicates", [source], baseline=baseline
                ) as data,
            ):
                method, adapter, _, _, _ = data
                method("Compare integers exactly")
                self.assertEqual(len(adapter.actions), 1)

    def test_position_parameters_work_in_both_modes(self):
        source = "def height(p):\n    return p[2]\ng = get_gripper_position()\nMove(b0,b0,b0,[0,0,height(g)])"
        for mode in CONDITION_MODES:
            with self.subTest(mode=mode), self.execution(mode, [source]) as data:
                method, adapter, backend, _, _ = data
                method("Move")
                self.assertEqual(adapter.actions[0][1][-1], [0, 0, 0.3])
                self.assertEqual(adapter.reads, ["get_gripper_position"])
                self.assertNotIn("0.123456789", "\n".join(backend.prompts))

    def test_illegal_index_types_effects_and_handles_in_both_modes(self):
        sources = [
            "p = get_gripper_position()\ni=0\nx=p[i]",
            "x=get_block_position(b0)[3]",
            "x=get_gripper_position()[-1]",
            "x=get_gripper_position()[True]",
            "x=[1,2,3][0]",
            "x=get_gripper_position()[:2]",
            "x=get_block_position(1)",
            "get_gripper_position=b0",
            "if b0 > b0:\n    pass",
            "if b0 == 1:\n    pass",
            "if get_block_position(b0):\n    pass",
            "if True == 1:\n    pass",
            "def dirty(b):\n    Pick(b)\n    return get_block_position(b)\np=dirty(b0)\nif p[0] > 0:\n    pass",
            "exists(lambda b: Pick(b))",
            "x=1e999",
        ]
        for mode in CONDITION_MODES:
            for source in sources:
                with (
                    self.subTest(mode=mode, source=source),
                    self.assertRaises(ConstraintFailure),
                ):
                    validate(Bundle(source), condition_mode=mode)

    def test_runtime_finite_values_and_mode_mismatch(self):
        with self.execution("coordinate_predicates") as data:
            _, adapter, _, logger, budget = data
            runtime = Runtime(adapter, budget, logger, "coordinate_predicates")
            with self.assertRaises(ConstraintFailure):
                runtime.execute(Bundle("Pick(b0)", condition_mode="dsl_only"))
            self.assertEqual(adapter.actions, [])
            from ..contracts import CapError

            with self.assertRaises(CapError):
                runtime.execute(Bundle("x=1e308 * 1e308"))

    def test_recursive_generation_and_repairs_carry_mode_without_observations(self):
        responses = [
            "if ready(b0):\n    Pick(b0)",
            "def ready(b):\n    return child(b)",
            "def child(b):\n    Pick(b)\n    return True",
            "def child(b):\n    return get_block_position(b)[0] > 0",
        ]
        with self.execution("coordinate_predicates", responses) as data:
            method, adapter, backend, logger, _ = data
            method("Test recursive helper roles")
            self.assertEqual(len(backend.prompts), 4)
            self.assertEqual(len(adapter.actions), 1)
            for text in backend.prompts:
                self.assertIn("Condition mode: coordinate_predicates", text)
                self.assertNotIn("0.123456789", text)
            self.assertIn("Checker diagnostics:", backend.prompts[-1])
            for path in logger.root.glob("checkpoints/*/manifest.json"):
                manifest = json.loads(path.read_text())
                self.assertEqual(manifest["condition_mode"], "coordinate_predicates")
                self.assertIsNotNone(manifest["checker_version"])
                self.assertEqual(manifest["observation_api_version"], "positions-v1")

    def test_strict_mode_repairs_numeric_predicate_before_actions(self):
        responses = [
            "if get_block_position(b0)[0] > 0:\n    Pick(b0)",
            "if Higher(b0,b0):\n    Pick(b0)",
        ]
        with self.execution(responses=responses) as data:
            method, adapter, backend, _, _ = data
            method("Pick")
            self.assertEqual(adapter.reads, [])
            self.assertEqual(len(adapter.actions), 1)
            self.assertIn("NUMERIC_CONDITION", backend.prompts[1])
            self.assertIn("Condition mode: dsl_only", backend.prompts[1])

    def test_baseline_modes_never_invoke_checker_or_repair(self):
        responses = [
            "if ready(b0):\n    Pick(b0)",
            "def ready(b):\n    return get_block_position(b)[0] > 0",
        ]
        for mode in CONDITION_MODES:
            with (
                self.subTest(mode=mode),
                self.execution(mode, responses, baseline=True) as data,
            ):
                method, adapter, backend, logger, _ = data
                with (
                    patch(
                        "synthesis.cap.validation.validate",
                        side_effect=AssertionError("baseline invoked checker"),
                    ),
                    patch(
                        "synthesis.cap.cap_method_1.generation.Generator.prepare",
                        side_effect=AssertionError("baseline invoked repair"),
                    ),
                ):
                    method("Pick")
                self.assertEqual(len(adapter.actions), 1)
                self.assertEqual(len(backend.prompts), 2)
                for text in backend.prompts:
                    self.assertIn(f"Condition mode: {mode}", text)
                    self.assertNotIn("0.123456789", text)
                for path in logger.root.glob("checkpoints/*/manifest.json"):
                    manifest = json.loads(path.read_text())
                    self.assertEqual(manifest["condition_mode"], mode)
                    self.assertEqual(manifest["validation"], "not_run")
                    self.assertFalse(manifest["refinement_enabled"])
                    self.assertIsNone(manifest["checker_version"])


if __name__ == "__main__":
    unittest.main()
