import contextlib
import io
import itertools
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import z3

from synthesis.api.instructions import Assign, While
from synthesis.api.program import Program
from synthesis.experiment.invariant_learning.runner import (
    ExperimentConfig,
    run_experiment,
)
from synthesis.experiment.invariant_learning.witness import WitnessQuery
from synthesis.inference_lib.demo_store import (
    InferenceVocabulary,
    InvInference,
    LoopHeadState,
)
from synthesis.verification_lib.highlevel_verification_lib import HighLevelContext
from synthesis.verification_lib.symbolic_verify import (
    SymbolicVerificationResult,
    discharge_vc,
)


class BindingTask:
    """Small independent adapter: bind a to b, without any Stack names or geometry."""

    loop_id = "0"
    vocabulary = InferenceVocabulary(0, ("equality",), ("a", "b"))

    def __init__(self):
        self.context = HighLevelContext()
        self.invariant = z3.BoolVal(False)
        self.verify_motion = Mock(return_value=SimpleNamespace(checks=[]))
        self.describe = Mock(return_value={"task": "binding"})
        self.executions = 0
        self.valid = True

    def set_invariant(self, value):
        self.invariant = value

    def verify_symbolic(self, timeout_ms):
        a, b = self.context.get_consts("a"), self.context.get_consts("b")
        program = Program(1, [While(a != b, [], [Assign("a", "b")], self.invariant)])
        return SymbolicVerificationResult(
            [
                discharge_vc(vc, self.context, timeout_ms)
                for vc in program.VC_gen(a != b, a == b, self.context)
            ],
            scope="unbounded",
        )

    def search_query(self, size, timeout_ms):
        solver = z3.Solver()
        return WitnessQuery(
            solver, z3.BoolVal(size >= 2), lambda model: {"size": size}, 1
        )

    def execute(self, search, **kwargs):
        self.executions += 1
        return SimpleNamespace(states=(), metadata={"status": "completed"})

    def validate(self, trace):
        return self.valid

    def learning_states(self, trace):
        positions = {"x1": [0, 0, 0.425], "x2": [0.15, 0, 0.425]}
        return [
            LoopHeadState("0", positions, positions, {"a": "x1", "b": "x2"}),
            LoopHeadState("0", positions, positions, {"a": "x2", "b": "x2"}),
        ]


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.output = contextlib.redirect_stdout(io.StringIO())
        self.output.__enter__()
        self.addCleanup(self.output.__exit__, None, None, None)

    def test_empty_start_uses_intended_learner_and_skips_motion(self):
        task = BindingTask()
        with patch(
            "synthesis.experiment.invariant_learning.runner.InvInference",
            wraps=InvInference,
        ) as learner:
            result = run_experiment(task, ExperimentConfig())
        self.assertEqual(result.status, "verified_symbolic", result.reason)
        self.assertEqual(result.verification_attempts, 2)
        self.assertEqual(result.history[0]["n_states"], 0)
        self.assertEqual(result.counterexample_executions, 1)
        learner.assert_called_once()
        task.verify_motion.assert_not_called()

    def test_motion_runs_after_symbolic_success(self):
        task = BindingTask()
        result = run_experiment(task, ExperimentConfig(verification_level="both"))
        self.assertEqual(result.status, "verified_model")
        task.verify_motion.assert_called_once()
        self.assertEqual(result.symbolic_status, "verified")

    def test_invalid_execution_is_not_training_data(self):
        task = BindingTask()
        task.valid = False
        with patch(
            "synthesis.experiment.invariant_learning.runner.InvInference"
        ) as learner:
            result = run_experiment(task, ExperimentConfig(verification_level="both"))
        self.assertEqual(result.status, "execution_failed")
        learner.assert_not_called()
        task.verify_motion.assert_not_called()

    def test_search_exhaustion_is_not_success(self):
        result = run_experiment(
            BindingTask(), ExperimentConfig(max_counterexample_blocks=1)
        )
        self.assertEqual(result.status, "no_reachable_counterexample")
        self.assertFalse(result)

    def test_motion_failure_retains_symbolic_success(self):
        class MotionFailure:
            checks = [
                SimpleNamespace(
                    obligation="clearance", status="invalid", reason="collision"
                )
            ]

            def __bool__(self):
                return False

        task = BindingTask()
        task.verify_motion.return_value = MotionFailure()
        result = run_experiment(task, ExperimentConfig(verification_level="both"))
        self.assertEqual(result.status, "motion_failed")
        self.assertEqual(result.symbolic_status, "verified")
        self.assertEqual(result.motion_status, "failed")

    def test_learner_must_cover_every_accumulated_state(self):
        with patch(
            "synthesis.experiment.invariant_learning.runner.InvInference",
            return_value=(z3.BoolVal(False), None),
        ):
            result = run_experiment(BindingTask(), ExperimentConfig())
        self.assertEqual(result.status, "learning_failed")


if __name__ == "__main__":
    unittest.main()
