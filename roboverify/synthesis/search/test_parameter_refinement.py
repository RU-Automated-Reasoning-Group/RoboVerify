"""Numeric repair proposes exact bounded samples; evaluator owns acceptance."""

import unittest
from copy import deepcopy
from unittest.mock import Mock, patch

import numpy as np
import z3

from synthesis.api.instructions import Assign, Move, Release, Skip, While
from synthesis.api.program import Program
from synthesis.cfg.program_source import program_fingerprint
from synthesis.search.parameter_refinement import (
    ParameterRefinementBudget,
    refine_parameters,
)


def candidate():
    return Program(
        3,
        [
            Assign("pointer", "anchor"),
            Move(0, 1, 2, target_offset=[0.01, 0.02, 0.03]),
            Release(1, target_z=0.1),
        ],
    )


def values(program):
    return deepcopy(program).register_trainable_parameter()


class ParameterRefinementTests(unittest.TestCase):
    def test_every_coordinate_both_signs_and_multipliers_around_original(self):
        original = candidate()
        fingerprint = program_fingerprint(original)
        centre = np.array(values(original))
        seen = []

        def evaluate(trial, index, deadline):
            seen.append(np.array(values(trial)))
            self.assertEqual(
                trial.instructions[1].get_operand(),
                original.instructions[1].get_operand(),
            )
            self.assertEqual(
                trial.instructions[1].control, original.instructions[1].control
            )
            return {"accepted": False, "complete": True}

        result = refine_parameters(original, evaluate)
        self.assertFalse(result)
        self.assertIsNone(result.program)
        self.assertIsNone(result.report)
        self.assertEqual(result.status, "space_exhausted")
        expected = []
        for multiplier in (1, 2):
            for coordinate in range(4):
                for sign in (1, -1):
                    trial = centre.copy()
                    trial[coordinate] += sign * multiplier * 0.001
                    expected.append(trial)
        np.testing.assert_allclose(seen, expected)
        self.assertEqual(len(result.history), 16)
        self.assertEqual(result.original_fingerprint, fingerprint)
        self.assertEqual(program_fingerprint(original), fingerprint)
        self.assertIsNone(original.instructions[1].target_offset[0].pos)
        self.assertIsNone(original.instructions[2].target_z_offset.pos)

    def test_freeze_precedes_evaluation_and_first_complete_acceptance_stops(self):
        original = candidate()
        events = []

        def freeze(trial, index, deadline):
            events.append(("freeze", index, program_fingerprint(trial), deadline))

        def evaluate(trial, index, deadline):
            self.assertEqual(
                events[-1], ("freeze", index, program_fingerprint(trial), deadline)
            )
            events.append(("evaluate", index, program_fingerprint(trial), deadline))
            return {"accepted": index == 1, "complete": True, "status": "finished"}

        result = refine_parameters(original, evaluate, on_trial=freeze)
        self.assertTrue(result.accepted)
        self.assertEqual(len(result.history), 2)
        self.assertTrue(result.report["accepted"])
        self.assertAlmostEqual(values(result.program)[0], 0.009)
        self.assertEqual(
            [row[:2] for row in events],
            [("freeze", 0), ("evaluate", 0), ("freeze", 1), ("evaluate", 1)],
        )
        self.assertEqual(len({row[3] for row in events}), 1)
        self.assertEqual(
            result.history[-1]["fingerprint"], program_fingerprint(result.program)
        )

    def test_nested_loop_registers_all_move_xyz_and_release_values(self):
        loop = While(z3.BoolVal(False), [], candidate().instructions, None)
        program = Program(1, [loop])
        seen = []
        result = refine_parameters(
            program,
            lambda trial, index, deadline: seen.append(values(trial))
            or {"accepted": False, "complete": True},
            budget=ParameterRefinementBudget(coordinate_multipliers=(1,)),
        )
        self.assertEqual(len(seen), 8)
        self.assertEqual({row["coordinate"] for row in result.history}, {0, 1, 2, 3})

    def test_missing_or_false_completeness_cannot_promote(self):
        reports = [
            {"accepted": True},
            {"accepted": True, "complete": False},
            {"accepted": "yes", "complete": True},
        ]
        result = refine_parameters(
            candidate(),
            lambda trial, index, deadline: reports[index],
            budget=ParameterRefinementBudget(max_trials=3),
        )
        self.assertFalse(result)
        self.assertEqual(result.status, "trial_budget_exhausted")
        self.assertTrue(all(not row["accepted"] for row in result.history))

    def test_shared_action_budget_report_stops_without_promotion(self):
        evaluate = Mock(
            return_value={
                "accepted": False,
                "complete": False,
                "budget_exhausted": True,
            }
        )
        result = refine_parameters(candidate(), evaluate)
        self.assertFalse(result)
        self.assertEqual(result.status, "budget_exhausted")
        self.assertEqual(evaluate.call_count, 1)

    def test_past_absolute_deadline_prevents_freeze_or_evaluation(self):
        freeze, evaluate = Mock(), Mock()
        result = refine_parameters(
            candidate(),
            evaluate,
            on_trial=freeze,
            budget=ParameterRefinementBudget(deadline=0),
        )
        self.assertEqual(result.status, "deadline_exhausted")
        self.assertEqual(result.history, ())
        freeze.assert_not_called()
        evaluate.assert_not_called()

    def test_expired_freeze_prevents_execution(self):
        now = [10.0]

        def freeze(*args):
            now[0] = 12.0

        evaluate = Mock()
        with patch(
            "synthesis.search.parameter_refinement.perf_counter",
            side_effect=lambda: now[0],
        ):
            result = refine_parameters(
                candidate(),
                evaluate,
                on_trial=freeze,
                budget=ParameterRefinementBudget(seconds=1),
            )
        self.assertEqual(result.status, "deadline_exhausted")
        self.assertFalse(result.history[0]["evaluated"])
        evaluate.assert_not_called()

    def test_late_success_is_rejected(self):
        now = [10.0]

        def evaluate(*args):
            now[0] = 12.0
            return {"accepted": True, "complete": True}

        with patch(
            "synthesis.search.parameter_refinement.perf_counter",
            side_effect=lambda: now[0],
        ):
            result = refine_parameters(
                candidate(), evaluate, budget=ParameterRefinementBudget(seconds=1)
            )
        self.assertFalse(result)
        self.assertEqual(result.status, "deadline_exhausted")
        self.assertTrue(result.history[0]["evaluated"])
        self.assertFalse(result.history[0]["accepted"])

    def test_external_timeout_and_callback_errors_propagate(self):
        for callback in ("evaluate", "on_trial"):
            with self.subTest(callback=callback):
                options = {
                    "evaluate": lambda *args: {"accepted": False, "complete": True},
                    "on_trial": None,
                }
                options[callback] = Mock(side_effect=TimeoutError("outer alarm"))
                with self.assertRaisesRegex(TimeoutError, "outer alarm"):
                    refine_parameters(candidate(), **options)

    def test_callbacks_cannot_change_frozen_parameters(self):
        def mutate(trial, *args):
            trial.instructions[1].target_offset[0].val = 0.9
            return {"accepted": True, "complete": True}

        with self.assertRaisesRegex(ValueError, "on_trial changed"):
            refine_parameters(candidate(), Mock(), on_trial=mutate)
        with self.assertRaisesRegex(ValueError, "evaluate changed"):
            refine_parameters(candidate(), mutate)

    def test_parameterless_program_has_no_baseline_evaluation(self):
        evaluate = Mock()
        result = refine_parameters(Program(1, [Skip(0)]), evaluate)
        self.assertEqual(result.status, "no_parameters")
        self.assertFalse(result)
        evaluate.assert_not_called()

    def test_budget_validation(self):
        for options in (
            {"step": 0},
            {"seconds": float("inf")},
            {"max_trials": True},
            {"coordinate_multipliers": ()},
            {"coordinate_multipliers": (1, 1)},
            {"deadline": float("nan")},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                ParameterRefinementBudget(**options)


if __name__ == "__main__":
    unittest.main()
