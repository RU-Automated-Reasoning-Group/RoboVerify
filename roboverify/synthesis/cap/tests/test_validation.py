"""Semantic boundary checks, including cross-line and transitive violations."""

import unittest

from ..contracts import Bundle, ConstraintFailure
from ..validation import missing_calls, validate


class ValidationTests(unittest.TestCase):
    def test_rejects_unchecked_python_and_perception(self):
        bad = [
            "import os",
            "x = b0.token",
            "x = get_obj_pos(b0)",
            "x = getattr(b0, 'token')",
            "x = [0,1,2][0]",
            "f = Pick",
            "Pick = b0",
            "if 1 < 2:\n    Pick(b0)",
            "if 1 == 1:\n    Pick(b0)",
            "x = Pick(b0)\nif x:\n    Pick(b0)",
            "while b0:\n    pass",
            "def Higher(a,b):\n    return True",
            "get(lambda Pick: True)",
            "def f(x=Pick(b0)):\n    pass",
            "get(lambda x: Pick(x))",
        ]
        for source in bad:
            with self.subTest(source=source), self.assertRaises(ConstraintFailure):
                validate(Bundle(source))

    def test_action_provenance_through_assignment_and_helper(self):
        source = "def ready(b):\n    Pick(b)\n    return Higher(b,b)\ncondition = ready(b0)\nif condition:\n    Release(b0,0.1)"
        with self.assertRaises(ConstraintFailure) as caught:
            validate(Bundle(source))
        self.assertIn("ACTION_IN_CONDITION", str(caught.exception))

    def test_transitive_helper_effect(self):
        source = "def query(b):\n    return child(b)\ndef child(b):\n    Pick(b)\n    return True\nwhile query(b0):\n    break"
        with self.assertRaises(ConstraintFailure):
            validate(Bundle(source))

    def test_empty_missing_calls_still_requires_type_check(self):
        source = "if Pick(b0):\n    pass"
        self.assertEqual(missing_calls(source), [])
        with self.assertRaises(ConstraintFailure):
            validate(Bundle(source))

    def test_missing_helpers_are_pending_in_local_check(self):
        pending = validate(
            Bundle("ready = query(b0)\nif ready:\n    Pick(b0)"), complete=False
        )
        self.assertEqual(pending[0].name, "query")
        self.assertEqual(
            missing_calls("ready = query(b0)\nif ready:\n    Pick(b0)")[0].role,
            "condition",
        )

    def test_wrong_arity_and_return_type(self):
        for source in [
            "Pick(b0,b0)",
            "def ready(b):\n    return 3\nif ready(b0):\n    pass",
            "def ready(b):\n    if eq(b,b):\n        return True\nif ready(b0):\n    pass",
        ]:
            with self.subTest(source=source), self.assertRaises(ConstraintFailure):
                validate(Bundle(source))

    def test_all_branches_and_loop_carried_types(self):
        for source in [
            "if False:\n    if Pick(b0):\n        pass",
            "b=b0\nwhile Higher(b,b0):\n    b=0",
        ]:
            with self.subTest(source=source), self.assertRaises(ConstraintFailure):
                validate(Bundle(source))

    def test_recursive_cycle(self):
        with self.assertRaises(ConstraintFailure):
            validate(Bundle("def again(b):\n    return again(b)\nagain(b0)"))

    def test_valid_parameter_helpers(self):
        validate(
            Bundle(
                "def offset(x):\n    return x + 0.05\nMove(b0,b0,b0,[0,0,offset(0.1)])"
            )
        )

    def test_helper_wrong_name_and_extra_statements(self):
        with self.assertRaises(ConstraintFailure):
            validate(
                Bundle("pick_it(b0)", {"pick_it": "def different(b):\n    Pick(b)"})
            )

    def test_uncalled_definition_still_checks_action_arity(self):
        with self.assertRaises(ConstraintFailure) as caught:
            validate(Bundle("def policy(b):\n    Move(b,b,b,0.0,0.0,0.15)"))
        self.assertIn("ARITY", str(caught.exception))

    def test_operator_diagnostic_has_real_source_location(self):
        with self.assertRaises(ConstraintFailure) as caught:
            validate(Bundle("if b0 is b0:\n    pass"))
        self.assertEqual(caught.exception.diagnostics[0].line, 1)


if __name__ == "__main__":
    unittest.main()
