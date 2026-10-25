"""Goal interface for Grid/Pyramid (Revision 5), without MuJoCo."""

import ast
import contextlib
import io
import tempfile
import unittest

import numpy as np

from ..adapter import RoboVerifyAdapter
from ..budget import Budget
from ..checker import DSLContract, check_bundle
from ..contracts import (
    API_NAMES,
    CONDITION_MODES,
    Bundle,
    CapError,
    ConstraintFailure,
    GoalHandle,
    Position3,
    PrimitiveFailure,
    task_api_names,
    task_input_types,
)
from ..logger import CapLogger
from ..prompt_package import package, render_api
from ..runtime import Runtime
from ..validation import GOAL, OBJECT, validate
from .test_adapter import RobotFixture

# The paper's Grid/Pyramid program (Figs. 17-18) in CaP form.
REFERENCE = """d = h0
while d != null:
    r = d
    while r != null and exists(lambda x: not AtGoal(x)):
        b = get(lambda x: not AtGoal(x))
        Move(b, b, r, [0.0, 0.0, 0.15])
        Pick(b)
        Move(b, b, r, [0.0, 0.0, 0.15])
        Move(r, r, r, [0.0, 0.0, 0.15])
        Move(r, r, r, [0.0, 0.0, 0.0])
        Release(r, 0.15)
        MoveRight(r)
    MoveDown(d)
"""


def check(source, task="grid", mode="dsl_only", helpers=None):
    contract = DSLContract.for_task(task, mode)
    return validate(
        Bundle(source, helpers or {}),
        inputs={
            name: GOAL if kind == "goal" else OBJECT
            for name, kind in contract.input_types
        },
        condition_mode=mode,
        api_names=contract.api_names,
    )


class CheckerTests(unittest.TestCase):
    def rejects(self, rule, source, **options):
        with self.assertRaises(ConstraintFailure) as caught:
            check(source, **options)
        self.assertEqual(caught.exception.diagnostics[0].rule, rule, source)

    def test_reference_program_passes_in_both_modes(self):
        for mode in CONDITION_MODES:
            with self.subTest(mode=mode):
                self.assertEqual(check(REFERENCE, mode=mode), [])

    def test_contract_inputs_and_apis_per_task(self):
        self.assertEqual(task_input_types("grid"), (("h0", "goal"), ("null", "goal")))
        self.assertEqual(task_input_types("stack"), (("b0", "object"),))
        self.assertIn("MoveRight", task_api_names("pyramid"))
        self.assertNotIn("ON_star_zero", task_api_names("grid"))
        for task in ("stack", "unstack", "reverse", "restack"):
            self.assertFalse(task_api_names(task) - API_NAMES)

    def test_traversal_statement_rules(self):
        self.rejects("TRAVERSAL_TARGET", "g = h0\ng = MoveRight(g)")
        self.rejects("TRAVERSAL_TARGET", "MoveRight(h0)")
        self.rejects("TRAVERSAL_TARGET", "MoveDown(null)")
        self.rejects("TRAVERSAL_TARGET", "g = h0\nMoveRight(g, g)")
        self.rejects("TYPE", "b = get(lambda x: True)\nMoveRight(b)")
        self.rejects("UNBOUND_NAME", "MoveRight(g)")
        # A cursor update in a condition helper is not pure.
        self.rejects(
            "ACTION_IN_CONDITION",
            "def nxt(g):\n    MoveRight(g)\n    return HasBlock(g)\n"
            "g = h0\nif nxt(g):\n    pass",
        )
        self.assertEqual(check("g = h0\nMoveRight(g)\nMoveDown(g)"), [])

    def test_inputs_are_protected(self):
        self.rejects("PROTECTED_BINDING", "h0 = null")
        self.rejects("PROTECTED_BINDING", "def f(null):\n    return True\nf(h0)")
        self.rejects("PROTECTED_BINDING", "def AtGoal(x):\n    return True")

    def test_goal_and_block_kinds_do_not_mix(self):
        block = "b = get(lambda x: True)\n"
        for source in (
            "Pick(h0)",
            block + "HasBlock(b)",
            "AtGoal(h0)",
            block + "ON(h0, b)",
            "get_block_position(h0)",
            block + "get_goal_position(b)",
            block + "if b == h0:\n    pass",
        ):
            with self.subTest(source=source):
                self.rejects("TYPE", source)
        self.assertEqual(
            check(block + "Move(b, b, h0, [0, 0, 0.1])\nRelease(h0, 0.1)"), []
        )

    def test_comparisons_and_positions_in_conditions(self):
        self.rejects(
            "IDENTITY_COMPARISON",
            "if h0 < null:\n    pass",
            mode="coordinate_predicates",
        )
        self.rejects(
            "NUMERIC_CONDITION", "if get_goal_position(h0)[2] > 0.5:\n    pass"
        )
        self.assertEqual(
            check(
                "if get_goal_position(h0)[2] > 0.5:\n    pass",
                mode="coordinate_predicates",
            ),
            [],
        )

    def test_queries_bind_blocks_only(self):
        self.rejects("TYPE", "if exists(lambda x: HasBlock(x)):\n    pass")


class TowerParityTests(unittest.TestCase):
    """Tower tasks keep the language they had before the goal interface."""

    def test_goal_names_are_ordinary_helpers_in_tower_tasks(self):
        # Not an API there: a call is an unresolved helper, a definition is allowed.
        with self.assertRaises(ConstraintFailure) as caught:
            check("if AtGoal(b0):\n    pass", task="stack")
        self.assertEqual(caught.exception.diagnostics[0].rule, "UNRESOLVED_HELPER")
        source = "def MoveRight(x):\n    return x\nMoveRight(b0)\nh0 = b0\nnull = h0"
        self.assertEqual(check(source, task="stack"), [])

    def test_tower_messages_are_unchanged(self):
        with self.assertRaises(ConstraintFailure) as caught:
            check("if b0 < b0:\n    pass", task="stack", mode="coordinate_predicates")
        self.assertEqual(
            caught.exception.diagnostics[0].message, "Objects support only == and !="
        )
        with self.assertRaises(ConstraintFailure) as caught:
            check("Move(1, b0, b0, [0, 0, 0])", task="stack")
        self.assertEqual(
            caught.exception.diagnostics[0].message, "Expected object, got number"
        )

    def test_recorded_v3_contract_is_still_auditable(self):
        contract = DSLContract(
            "dsl_only",
            tuple(sorted(task_api_names("stack"))),
            (("b0", "object"),),
            "shared-contract-v3",
            "positions-v1",
        )
        report = check_bundle(Bundle("Pick(b0)"), contract)
        self.assertEqual(report.status, "pass")

    def test_tower_prompts_are_byte_identical_to_typed_v3(self):
        v3, v4 = package("typed_v3"), package("typed_v4")
        for key in (
            "method1_templates",
            "method2_templates",
            "baseline_prefixes",
            "condition_rules",
            "output_rules",
        ):
            self.assertEqual(v3[key], v4[key], key)
        for task in ("stack", "unstack", "reverse", "restack"):
            for mode in CONDITION_MODES:
                with self.subTest(task=task, mode=mode):
                    names = task_api_names(task)
                    self.assertEqual(
                        render_api(names, mode, "typed_v3", 0.001),
                        render_api(names, mode, "typed_v4", 0.001),
                    )

    def test_goal_prompt_describes_the_goal_api(self):
        text = render_api(task_api_names("grid"), "dsl_only", "typed_v4", 0.001)
        for name in (
            "GoalHandle",
            "MoveRight(goal_var)",
            "HasBlock(goal",
            "get_goal_position",
        ):
            self.assertIn(name, text)
        self.assertNotIn("The input b0 is a block handle", text)
        self.assertNotIn("ON_star_zero", text)


class GoalFixture(RobotFixture):
    """A RobotFixture with a 2x2 goal layout and goal bindings."""

    def __init__(self, count=4):
        from synthesis.environment.goal_layout import GoalLayout, GoalRef

        super().__init__(count)
        self.roboverify_base_block_id = None
        self.symbolic_name_to_box_id = {}
        self.goal_layout = GoalLayout.from_positions(
            [
                [0.5, 0.0, 0.025],
                [0.6, 0.0, 0.025],
                [0.5, 0.1, 0.025],
                [0.6, 0.1, 0.025],
            ],
            (2, 2),
        )
        self.symbolic_name_to_goal = {"h0": GoalRef(0), "null": None}
        for i in range(count):
            self.obs[10 + 12 * i : 13 + 12 * i] = [0.2 * i, 0.5, 0.025]


class GoalAdapterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.logger = CapLogger(self.tmp.name, console=False)
        self.env = GoalFixture()
        self.adapter = RoboVerifyAdapter(self.env, self.logger, Budget(), task="grid")
        self.h0, self.null = self.adapter.inputs["h0"], self.adapter.inputs["null"]

    def tearDown(self):
        self.logger.close()
        self.tmp.cleanup()

    def test_traversal_follows_the_layout(self):
        right = self.adapter.call("MoveRight", [self.h0])
        self.assertIsInstance(right, GoalHandle)
        self.assertEqual(self.adapter.call("MoveRight", [right]), self.null)
        down = self.adapter.call("MoveDown", [self.h0])
        self.assertEqual(self.adapter.call("MoveDown", [down]), self.null)
        # Only row heads have a d link.
        self.assertEqual(self.adapter.call("MoveDown", [right]), self.null)
        with self.assertRaisesRegex(CapError, "cannot advance null"):
            self.adapter.call("MoveRight", [self.null])
        self.assertEqual(self.env.actions, [])

    def test_occupancy_predicates_and_goal_positions(self):
        block = self.adapter.handles[0]
        self.assertFalse(self.adapter.call("AtGoal", [block]))
        self.assertFalse(self.adapter.call("HasBlock", [self.h0]))
        self.env.obs[10:13] = [0.51, 0.0, 0.03]  # within 20 mm of goal 0
        self.assertTrue(self.adapter.call("AtGoal", [block]))
        self.assertTrue(self.adapter.call("HasBlock", [self.h0]))
        self.assertFalse(self.adapter.call("HasBlock", [self.null]))
        self.assertEqual(
            self.adapter.call("get_goal_position", [self.h0]),
            Position3((0.5, 0.0, 0.025)),
        )
        with self.assertRaisesRegex(CapError, "null has none"):
            self.adapter.call("get_goal_position", [self.null])

    def test_goals_are_references_not_blocks(self):
        with self.assertRaisesRegex(CapError, "goals cannot be picked"):
            self.adapter.call("Pick", [self.h0])
        with self.assertRaisesRegex(CapError, "not a block"):
            self.adapter.call("get_block_position", [self.h0])
        with self.assertRaisesRegex(CapError, "null has none"):
            self.adapter.call("Move", [self.null, self.h0, self.h0, [0, 0, 0]])
        targets, emit = [], self.logger.emit

        def record(kind, **fields):
            if kind == "controller_phase" and fields.get("target") is not None:
                targets.append(fields["target"])
            return emit(kind, **fields)

        self.logger.emit = record
        right = self.adapter.call("MoveRight", [self.h0])
        try:
            self.adapter.call("Move", [right, self.h0, right, [0, 0, 0.1]])
        except PrimitiveFailure:
            pass  # The fixture is not a convergent controller.
        np.testing.assert_allclose(targets[0], [0.6, 0.0, 0.125])

    def test_runtime_rebinds_cursor_variables(self):
        runtime = Runtime(self.adapter, Budget(), self.logger)
        source = (
            "g = h0\nMoveRight(g)\nfirst = g\nMoveRight(g)\n"
            "def advance(c):\n    MoveDown(c)\n    return c\n"
            "row = advance(h0)\nsame = h0 == h0\nend = g == null\n"
        )
        _, values = runtime.execute(Bundle(source))
        self.assertEqual(values["g"], self.null)
        self.assertNotEqual(values["first"], self.null)
        self.assertEqual(values["row"], self.adapter.call("MoveDown", [self.h0]))
        self.assertTrue(values["same"] and values["end"])
        self.assertEqual(values["h0"], self.h0)


class BaselineRewriteTests(unittest.TestCase):
    def test_statements_become_assignments_only_for_goal_tasks(self):
        from ..cap_google_baseline.compatibility import _TraversalStatements

        tree = _TraversalStatements({"MoveRight"}).visit(
            ast.parse("MoveRight(g)\nx = MoveRight(g)\nMoveDown(g)")
        )
        self.assertEqual(
            ast.unparse(tree), "g = MoveRight(g)\nx = MoveRight(g)\nMoveDown(g)"
        )


class FrozenV2Tests(unittest.TestCase):
    def test_v2_is_v1_plus_the_goal_interface(self):
        from ..configs import config_frozen_v1 as v1
        from ..configs import config_frozen_v2 as v2

        for task in ("stack", "unstack", "reverse", "restack"):
            a, b = v1.run_settings(task), v2.run_settings(task)
            self.assertEqual(b.pop("prompt_version"), "typed_v4")
            a.pop("prompt_version")
            self.assertEqual(a, b, task)
            self.assertEqual(v1.instruction(task), v2.instruction(task))
        for task in ("grid", "pyramid"):
            config, model, instruction = v2.configuration(task, seed=2)
            self.assertEqual((config.num_blocks, config.seed), (6, 2))
            self.assertIn("HasBlock(g)", instruction)
            self.assertEqual(model.model, v1.MODEL["model"])
        self.assertEqual(v2.run_config("grid").grid_rows, 3)


if __name__ == "__main__":
    unittest.main()
