"""Regression boundaries for generic types and loop-carried effects."""

import tempfile
import unittest
from unittest.mock import patch

from .audit import normalize_output, read_final_output
from .budget import Budget
from .checker import AnalysisFailure, DSLContract, check_bundle
from .checkpoints import Checkpoints
from .contracts import Bundle, ConstraintFailure
from .llm import LoggedModel, ScriptedModel
from .logger import CapLogger
from .runner import make_method
from .test_condition_modes import PositionAdapter

MODES = ("dsl_only", "coordinate_predicates")
DIRTY = "def dirty(x):\n    Pick(x)\n    return True\n"
OBJECT_LOOP = """def policy(base):
    top = base
    while exists(lambda x: not ON_star(x, base)):
        if eq(top, base):
            pass
        top = get(lambda x: not ON_star(x, base))
policy(b0)
"""
DELAYED = DIRTY + """a = True
b = True
done = False
stage1 = False
stage2 = False
while not done:
    if a:
        pass
    a = b
    b = dirty(b0)
    done = stage2
    stage2 = stage1
    stage1 = True
"""


class Revision3CheckerTests(unittest.TestCase):
    def assert_status(self, source, status, mode, rule=None, helpers=None):
        result = check_bundle(
            Bundle(source, helpers or {}), DSLContract.for_task("stack", mode)
        )
        self.assertEqual(result.status, status, result.to_dict())
        if rule:
            self.assertIn(rule, [d["rule"] for d in result.diagnostics])
        return result

    def test_generic_object_loop_accepts_real_object_call(self):
        for mode in MODES:
            with self.subTest(mode=mode):
                self.assert_status(OBJECT_LOOP, "pass", mode)
                definition, call = OBJECT_LOOP.rsplit("policy(b0)", 1)
                self.assert_status(
                    "policy(b0)", "pass", mode, helpers={"policy": definition}
                )

    def test_generic_alias_return_and_position_loops(self):
        sources = [
            "def identity(x):\n    return x\n"
            + OBJECT_LOOP.replace("top = base", "top = identity(base)"),
            "def use(p):\n    q=p\n    while Higher(b0,b0):\n        Move(b0,b0,b0,[0,0,q[2]])\n        q=get_gripper_position()\n        break\nuse(get_gripper_position())",
            "def offset(p):\n    return p[2]\nMove(b0,b0,b0,[0,0,offset(get_gripper_position())])",
        ]
        for mode in MODES:
            for source in sources:
                with self.subTest(mode=mode, source=source):
                    self.assert_status(source, "pass", mode)

    def test_unused_generic_constraints_reject_concrete_conflicts(self):
        sources = [
            "def bad(x):\n    Pick(x)\n    return x + 1",
            "def identity(x):\n    return x\ndef bad(x):\n    y=identity(x)\n    Pick(y)\n    return x + 1",
            OBJECT_LOOP.replace("policy(b0)", "policy(1)"),
            OBJECT_LOOP.replace("policy(b0)", "policy(get_gripper_position())"),
            OBJECT_LOOP + "policy(1)",
            "def bad(x):\n    if False:\n        Move(x,x,x,0,0,1)",
        ]
        for mode in MODES:
            for source in sources:
                with self.subTest(mode=mode, source=source):
                    self.assert_status(source, "fail", mode)

    def test_generic_equality_constraints_remain_linked(self):
        self.assert_status(
            "def bad(a,b):\n    if a == b:\n        pass\n    Pick(a)\n    return b + 1",
            "fail",
            "coordinate_predicates",
        )
        self.assert_status(
            "def same(a,b):\n    return a == b\nif same(b0,b0):\n    pass\nif same(1,2):\n    pass",
            "pass",
            "coordinate_predicates",
        )

    def test_unused_definitions_keep_mode_and_purity_checks(self):
        self.assert_status(
            "def numeric(x):\n    return x < 2", "fail", "dsl_only", "NUMERIC_CONDITION"
        )
        for mode in MODES:
            self.assert_status(
                DIRTY + "def bad(x):\n    if dirty(x):\n        pass",
                "fail",
                mode,
                "ACTION_IN_CONDITION",
            )

    def test_focused_generic_helper_is_only_local_pass(self):
        source = OBJECT_LOOP[: OBJECT_LOOP.rfind("policy(b0)")]
        for mode in MODES:
            report = check_bundle(
                Bundle("", {"policy": source}),
                DSLContract.for_task("stack", mode),
                scope="local",
                focus="policy",
            )
            self.assertEqual(report.status, "local_pass", report.to_dict())
            self.assertEqual(report.coverage, "local_generic")

    def test_delayed_effect_reaches_condition(self):
        for mode in MODES:
            self.assert_status(DELAYED, "fail", mode, "ACTION_IN_CONDITION")
            for length in (3, 7):
                initial = "\n".join(f"v{i}=True" for i in range(length))
                shifts = "\n".join(f"    v{i}=v{i+1}" for i in range(length - 1))
                source = (
                    DIRTY
                    + initial
                    + "\nwhile True:\n    if v0:\n        pass\n"
                    + shifts
                    + f"\n    v{length-1}=dirty(b0)"
                )
                with self.subTest(mode=mode, length=length):
                    self.assert_status(source, "fail", mode, "ACTION_IN_CONDITION")

    def test_delayed_non_boolean_type_is_rejected_before_execution(self):
        source = "a=True\nb=True\nc=True\nwhile True:\n    if a:\n        pass\n    a=b\n    b=c\n    c=1"
        for mode in MODES:
            self.assert_status(source, "fail", mode, "TYPE")

    def test_effects_of_evaluated_helper_arguments_are_retained(self):
        source = (
            "def ignore(x):\n    return True\nflag=ignore(Pick(b0))\nif flag:\n    pass"
        )
        for mode in MODES:
            self.assert_status(source, "fail", mode, "ACTION_IN_CONDITION")

    def test_control_transfers_do_not_erase_effects(self):
        sources = [
            DIRTY
            + "flag=True\nwhile True:\n    flag=dirty(b0)\n    break\n    flag=False\nif flag:\n    pass",
            DIRTY
            + "flag=True\ndone=False\nwhile not done:\n    flag=dirty(b0)\n    done=True\n    continue\n    flag=False\nif flag:\n    pass",
            DIRTY
            + "a=True\nb=True\nwhile True:\n    if a:\n        pass\n    while True:\n        a=b\n        b=dirty(b0)\n        break",
            DIRTY
            + "flag=True\nwhile True:\n    if Higher(b0,b0):\n        flag=dirty(b0)\n        break\n    else:\n        continue\nif flag:\n    pass",
        ]
        for mode in MODES:
            for source in sources:
                with self.subTest(mode=mode, source=source):
                    self.assert_status(source, "fail", mode, "ACTION_IN_CONDITION")

    def test_break_state_is_not_a_loop_back_edge(self):
        source = "top=b0\nwhile Higher(top,b0):\n    Pick(top)\n    top=1\n    break"
        for mode in MODES:
            self.assert_status(source, "pass", mode)

    def test_break_exit_bypasses_while_else_assignments(self):
        source = (
            DIRTY
            + "flag=True\nwhile Higher(b0,b0):\n    flag=dirty(b0)\n    break\nelse:\n    flag=True\nif flag:\n    pass"
        )
        for mode in MODES:
            self.assert_status(source, "fail", mode, "ACTION_IN_CONDITION")

    def test_pure_loops_and_return_control_flow(self):
        sources = [
            "flag=True\nwhile flag:\n    if Higher(b0,b0):\n        flag=False\n        continue\n    else:\n        break",
            "def value(x):\n    while Higher(x,x):\n        if eq(x,x):\n            return x\n        else:\n            break\n    return x\nPick(value(b0))",
            OBJECT_LOOP,
        ]
        for mode in MODES:
            for source in sources:
                self.assert_status(source, "pass", mode)
        self.assert_status(
            "n=0\nwhile n<3:\n    n=n+1", "pass", "coordinate_predicates"
        )
        self.assert_status("n=0\nwhile n<3:\n    n=n+1", "fail", "dsl_only")

    def test_analysis_work_exhaustion_is_error(self):
        with patch("synthesis.cap.validation.MAX_ANALYSIS_STEPS", 2):
            report = check_bundle(Bundle("x=b0\ny=b0\nPick(x)"), DSLContract())
        self.assertEqual(report.status, "error", report.to_dict())
        self.assertIn("analysis", report.reason.lower())


class Revision3MethodTests(unittest.TestCase):
    def test_checked_methods_reject_delayed_effect_before_actions(self):
        for name in ("cap_method_1", "cap_method_2"):
            for mode in MODES:
                with (
                    self.subTest(method=name, mode=mode),
                    tempfile.TemporaryDirectory() as tmp,
                ):
                    logger = CapLogger(tmp, console=False)
                    try:
                        adapter = PositionAdapter()
                        budget = Budget(timeout=10)
                        backend = ScriptedModel([DELAYED])
                        model = LoggedModel(backend, logger, budget)
                        checkpoints = Checkpoints(
                            logger,
                            method=name,
                            condition_mode=mode,
                            prompt_version="typed_v3",
                        )
                        method = make_method(
                            name,
                            model,
                            adapter,
                            logger,
                            checkpoints,
                            budget,
                            1,
                            condition_mode=mode,
                            prompt_version="typed_v3",
                        )
                        with self.assertRaises(ConstraintFailure):
                            method("Check before actions")
                        self.assertEqual(adapter.actions, [])
                        self.assertEqual(len(backend.prompts), 1)
                    finally:
                        logger.close()

    def test_analysis_limit_does_not_trigger_model_repair(self):
        for name in ("cap_method_1", "cap_method_2"):
            with self.subTest(method=name), tempfile.TemporaryDirectory() as tmp:
                logger = CapLogger(tmp, console=False)
                try:
                    adapter = PositionAdapter()
                    budget = Budget(timeout=10)
                    backend = ScriptedModel(["x=b0\ny=b0\nPick(x)", "pass"])
                    model = LoggedModel(backend, logger, budget)
                    checkpoints = Checkpoints(
                        logger, method=name, prompt_version="typed_v3"
                    )
                    method = make_method(
                        name,
                        model,
                        adapter,
                        logger,
                        checkpoints,
                        budget,
                        3,
                        prompt_version="typed_v3",
                    )
                    with patch("synthesis.cap.validation.MAX_ANALYSIS_STEPS", 2):
                        with self.assertRaises(AnalysisFailure):
                            method("Bound analysis")
                    self.assertEqual(adapter.actions, [])
                    self.assertEqual(len(backend.prompts), 1)
                finally:
                    logger.close()

    def test_baseline_still_executes_and_posthoc_checker_rejects(self):
        for mode in MODES:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                logger = CapLogger(tmp, console=False)
                try:
                    adapter = PositionAdapter()
                    budget = Budget(timeout=10)
                    backend = ScriptedModel([DELAYED, DIRTY])
                    model = LoggedModel(backend, logger, budget)
                    checkpoints = Checkpoints(
                        logger,
                        method="cap_google_baseline",
                        condition_mode=mode,
                        prompt_version="typed_v3",
                    )
                    method = make_method(
                        "cap_google_baseline",
                        model,
                        adapter,
                        logger,
                        checkpoints,
                        budget,
                        1,
                        condition_mode=mode,
                        prompt_version="typed_v3",
                    )
                    method("Keep baseline execution unchanged")
                    before = list(adapter.actions)
                    output = read_final_output(tmp)
                    self.assertEqual(output["status"], "complete", output)
                    bundle, _ = normalize_output(output)
                    report = check_bundle(bundle, DSLContract.for_task("stack", mode))
                    self.assertEqual(report.status, "fail", report.to_dict())
                    self.assertIn(
                        "ACTION_IN_CONDITION", [d["rule"] for d in report.diagnostics]
                    )
                    self.assertEqual(len(before), 3)
                    self.assertEqual(adapter.actions, before)
                    self.assertEqual(len(backend.prompts), 2)
                finally:
                    logger.close()

    def test_checked_methods_accept_valid_object_loop(self):
        for name in ("cap_method_1", "cap_method_2"):
            for mode in MODES:
                with (
                    self.subTest(method=name, mode=mode),
                    tempfile.TemporaryDirectory() as tmp,
                ):
                    logger = CapLogger(tmp, console=False)
                    try:
                        adapter = PositionAdapter()
                        budget = Budget(timeout=10)
                        backend = ScriptedModel([OBJECT_LOOP + "Pick(b0)"])
                        model = LoggedModel(backend, logger, budget)
                        checkpoints = Checkpoints(
                            logger,
                            method=name,
                            condition_mode=mode,
                            prompt_version="typed_v3",
                        )
                        method = make_method(
                            name,
                            model,
                            adapter,
                            logger,
                            checkpoints,
                            budget,
                            1,
                            condition_mode=mode,
                            prompt_version="typed_v3",
                        )
                        method("Accept the object-valued loop")
                        self.assertEqual(
                            [name for name, args in adapter.actions], ["Pick"]
                        )
                        self.assertEqual(len(backend.prompts), 1)
                    finally:
                        logger.close()
