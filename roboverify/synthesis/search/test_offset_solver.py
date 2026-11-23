"""Shared-offset proposals are goals across examples, not independent solves."""

import unittest
from dataclasses import replace
from unittest.mock import Mock, patch

import numpy as np
import z3

from synthesis.predicates.scene import Scene, evaluate
from synthesis.predicates.term import (
    atom,
    boolean,
    conjunction,
    exists,
    forall,
    implies,
    negate,
    ref,
)
from synthesis.search.contract import (
    Action,
    ContractConfig,
    ContractProblem,
    SearchConfig,
    SearchExample,
    beam_search,
)
from synthesis.search.offset_solver import (
    compile_endpoint,
    solve_terminal_offsets,
    strengthen,
)
from synthesis.util import on

MOVE = Action("Move", ("payload", "payload", "base"), (0, 0, 0))
RELEASE = Action("Release", ("base",), (0.15,))


def held_examples(errors=(-0.00044, 0.001803, 0.00025, 0.001, -0.0001)):
    examples = []
    for error in errors:
        positions = np.array([[0, 0, 0.425], [0.2, 0, 0.6]])
        gripper = np.array([0.2, 0, 0.6 - error])
        path = np.array([gripper, [0.2, 0, 0.425], [0.2, 0, 0.575]])
        frames = np.repeat(positions[None], 3, axis=0)
        frames[1:, 1] = [0.2, 0, 0.425]
        examples.append(
            SearchExample(
                positions,
                gripper,
                {"base": 0, "payload": 1},
                path,
                frames,
                initial_holding=1,
            )
        )
    return examples


def problem(*, errors=None, solve=True, term=None):
    examples = held_examples() if errors is None else held_examples(errors)
    term = atom("Higher", ref("base"), ref("payload")) if term is None else term

    def goal(state, index):
        coordinates = dict(enumerate(state.positions[index]))
        return evaluate(term, Scene(coordinates, examples[index].bindings))

    def symbolic(coordinates, index):
        return compile_endpoint(
            term,
            coordinates,
            examples[index].bindings,
            dict(enumerate(examples[index].positions)),
        )

    return ContractProblem(
        examples,
        goal,
        allowed_operands=("base", "payload"),
        actions=(MOVE, RELEASE),
        symbolic_goal=symbolic,
        config=ContractConfig(solve_offsets=solve),
    )


class EndpointCompilerTests(unittest.TestCase):
    def test_geometric_language_matches_production_including_frozen_and_table(self):
        positions = {
            0: (0, 0, 0.425),
            1: (0, 0, 0.475),
            2: (0.1, 0, 0.425),
            "tbl": on.TABLE,
        }
        entry = dict(positions, **{})
        entry[1] = (0.2, 0, 0.425)
        coordinates = {
            key: tuple(z3.RealVal(str(value)) for value in point)
            for key, point in positions.items()
            if key != "tbl"
        }
        for relation in (
            "ON",
            "ON_star",
            "ON_star_zero",
            "Higher",
            "EqualHeight",
            "Scattered",
            "eq",
        ):
            for left in positions:
                for right in positions:
                    with self.subTest(relation=relation, left=left, right=right):
                        term = atom(relation, ref("left"), ref("right"))
                        bindings = {"left": left, "right": right}
                        compiled = z3.simplify(
                            compile_endpoint(term, coordinates, bindings, entry)
                        )
                        expected = evaluate(term, Scene(positions, bindings, entry))
                        self.assertEqual(z3.is_true(compiled), expected)
                        self.assertTrue(z3.is_true(compiled) or z3.is_false(compiled))
        term = forall(
            ("x",),
            exists(
                ("y",),
                conjunction(
                    implies(
                        atom("eq", ref("x"), ref("y")),
                        atom("ON_star_zero", ref("x"), ref("y")),
                    ),
                    negate(atom("Scattered", ref("x"), ref("y"))),
                ),
            ),
        )
        self.assertEqual(
            z3.is_true(z3.simplify(compile_endpoint(term, coordinates, {}, entry))),
            evaluate(term, Scene(positions, {}, entry)),
        )

    def test_margin_strengthening_implies_original_with_negation_and_disjunction(self):
        x = z3.Real("x")
        for expression in (
            x < 1,
            x >= 0,
            z3.Not(x <= 2),
            z3.Or(x < -1, x >= 1),
            z3.Implies(x >= 0, x > 1),
        ):
            solver = z3.Solver()
            solver.add(strengthen(expression, z3.RealVal("0.001")), z3.Not(expression))
            self.assertEqual(solver.check(), z3.unsat)
        self.assertTrue(z3.is_true(z3.simplify(strengthen(x >= x, z3.RealVal("0.1")))))


class SharedOffsetTests(unittest.TestCase):
    def test_one_float_vector_satisfies_five_different_grasp_offsets(self):
        p = problem()
        original = p.transition(p.initial, MOVE)
        self.assertIsNotNone(original)
        self.assertFalse(p.goal(p.transition(original, RELEASE)))
        result = solve_terminal_offsets(
            p, p.initial, MOVE, RELEASE, timeout_ms=1000, radius=0.02
        )
        self.assertEqual(result.status, "sat", result.reason)
        self.assertGreater(result.margin, 0.0003)
        self.assertAlmostEqual(result.actions[0].offsets[2], -0.0011815, places=8)
        self.assertEqual(result.actions[0].offsets[:2], (0, 0))
        self.assertTrue(p.goal(result.states[-1]))
        self.assertEqual(result.states[-1].holding, (-1,) * 5)

    def test_individually_satisfiable_examples_can_have_no_shared_offsets(self):
        p = problem(errors=(-0.003, 0.002))
        combined = solve_terminal_offsets(
            p, p.initial, MOVE, RELEASE, timeout_ms=1000, radius=0.02
        )
        self.assertEqual(combined.status, "unsat")
        self.assertEqual(combined.actions, ())
        for error in (-0.003, 0.002):
            local = problem(errors=(error,))
            solved = solve_terminal_offsets(
                local, local.initial, MOVE, RELEASE, timeout_ms=1000, radius=0.02
            )
            self.assertEqual(solved.status, "sat", solved.reason)

    def test_unknown_is_not_a_candidate(self):
        p = problem()
        solver = Mock()
        solver.check.return_value = z3.unknown
        solver.reason_unknown.return_value = "timeout"
        with patch("synthesis.search.offset_solver.z3.Optimize", return_value=solver):
            result = solve_terminal_offsets(
                p, p.initial, MOVE, RELEASE, timeout_ms=10, radius=0.02
            )
        self.assertEqual(result.status, "unknown")
        self.assertEqual(result.reason, "timeout")
        self.assertEqual(result.actions, ())

    def test_exact_nominal_contract_rejection_cannot_promote_solver_model(self):
        p = problem()
        with patch.object(p, "transition", return_value=None):
            result = solve_terminal_offsets(
                p, p.initial, MOVE, RELEASE, timeout_ms=1000, radius=0.02
            )
        self.assertEqual(result.status, "rejected")
        self.assertEqual(result.actions, ())

    def test_beam_uses_solver_without_changing_length_or_default_search(self):
        control = beam_search(
            problem(solve=False), SearchConfig(max_length=2, seconds=3)
        )
        self.assertEqual(control.candidates, ())
        solved = beam_search(problem(), SearchConfig(max_length=2, seconds=3))
        self.assertTrue(solved.candidates)
        self.assertEqual(len(solved.candidates[0].actions), 2)
        self.assertGreaterEqual(solved.stats["offset_solver_sat"], 1)
        short = beam_search(problem(), SearchConfig(max_length=1, seconds=3))
        self.assertEqual(short.candidates, ())
        self.assertEqual(short.stats.get("offset_solver_attempts", 0), 0)

    def test_attempt_budget_and_missing_symbolic_goal_are_explicit(self):
        p = problem()
        p.config = replace(p.config, solver_attempts=1)
        result = beam_search(p, SearchConfig(max_length=3, seconds=3))
        self.assertLessEqual(result.stats.get("offset_solver_attempts", 0), 1)
        p = problem()
        p.symbolic_goal = None
        result = solve_terminal_offsets(
            p, p.initial, MOVE, RELEASE, timeout_ms=1000, radius=0.02
        )
        self.assertEqual(result.status, "unsupported")


if __name__ == "__main__":
    unittest.main()
