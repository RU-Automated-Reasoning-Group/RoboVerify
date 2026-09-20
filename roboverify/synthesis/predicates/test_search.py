import unittest

from synthesis.predicates.classifier import learn_classifier
from synthesis.predicates.enumerate import enumerate_separator
from synthesis.predicates.guard import loop_guard_synthesis
from synthesis.predicates.language import Language
from synthesis.predicates.scene import Scene, evaluate
from synthesis.predicates.term import free_names


class SearchTests(unittest.TestCase):
    def test_classifier_finds_exact_existential_separator(self):
        positive = Scene({0: (0, 0, 0.425), 1: (0, 0, 0.475)}, {"b": 0})
        negative = Scene({0: (0, 0, 0.425), 1: (0.2, 0, 0.425)}, {"b": 0})
        result = learn_classifier(
            [positive],
            [negative],
            {"b"},
            language=Language(
                relations=("ON_star", "eq"), max_depth=4, max_variables=1
            ),
        )
        self.assertTrue(result, result.status)
        self.assertEqual(result.term.op, "exists")
        self.assertTrue(evaluate(result.term, positive))
        self.assertFalse(evaluate(result.term, negative))

    def test_search_is_invariant_to_program_variable_names(self):
        positions = {0: (0, 0, 0.425), 1: (0, 0, 0.475)}
        for mode in ("classifier", "guard"):
            for name in ("anchor", "v0", "v1"):
                with self.subTest(mode=mode, name=name):
                    examples = [
                        (Scene(positions, {name: 0}), True),
                        (Scene(positions, {name: 1}), False),
                    ]
                    result = enumerate_separator(
                        examples,
                        (name,),
                        mode=mode,
                        language=Language(
                            relations=("Higher",), max_depth=3, max_variables=1
                        ),
                    )
                    self.assertTrue(result, result.status)
                    self.assertEqual(free_names(result.term), {name})
                    for scene, expected in examples:
                        self.assertEqual(evaluate(result.term, scene), expected)

    def test_guard_rejects_ambiguous_witnesses(self):
        scene = Scene({0: (0, 0, 0.425), 1: (0.2, 0, 0.425)}, {"b": 0})
        result = loop_guard_synthesis(
            [(scene, {"x": 0})],
            [scene],
            ("x",),
            {"b"},
            language=Language(max_candidates=100),
        )
        self.assertFalse(result)

    def test_budget_exhaustion_is_explicit(self):
        scene = Scene({0: (0, 0, 0)}, {"b": 0})
        result = learn_classifier(
            [scene], [scene], {"b"}, language=Language(max_candidates=1)
        )
        self.assertEqual(result.status, "budget_exhausted")


if __name__ == "__main__":
    unittest.main()
