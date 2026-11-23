"""CEM must return its best evaluated point, including disconnected optima."""

import contextlib
import io
import unittest
from unittest.mock import patch

import numpy as np

from synthesis.experiment.mcmc.cem import cem_optimize as instrumented
from synthesis.mcmc.cem import cem_optimize


def disconnected_objective(parameters):
    """Two feasible regions separated by an infeasible elite mean."""
    return -abs(abs(float(parameters[0])) - 1.0)


class CEMTests(unittest.TestCase):
    def test_retains_feasible_sample_when_elite_mean_is_infeasible(self):
        for optimize in (cem_optimize, instrumented):
            for workers in (0, 1):
                with self.subTest(optimizer=optimize.__module__, workers=workers):
                    with patch(
                        "numpy.random.randn", return_value=np.array([[-1.0], [1.0]])
                    ), contextlib.redirect_stdout(io.StringIO()):
                        score, parameters, *_ = optimize(
                            disconnected_objective,
                            1,
                            iterations=1,
                            N=2,
                            K=2,
                            init_std=1.0,
                            num_workers=workers,
                        )
                    self.assertEqual(score, 0.0)
                    self.assertEqual(abs(parameters[0]), 1.0)
                    self.assertEqual(disconnected_objective(parameters), score)

    def test_returns_best_evaluation_and_preserves_instrumentation_parity(self):
        rng_state = np.random.get_state()
        runs = []
        try:
            for optimize in (cem_optimize, instrumented):
                evaluated = []

                def objective(parameters):
                    score = -float(np.square(parameters - [0.3, -0.2]).sum())
                    evaluated.append((parameters.copy(), score))
                    return score

                np.random.seed(17)
                with contextlib.redirect_stdout(io.StringIO()):
                    result = optimize(
                        objective, 2, iterations=4, N=8, K=3, num_workers=0
                    )
                score, parameters = result[:2]
                self.assertEqual(len(evaluated), 1 + 4 * (8 + 1))
                self.assertEqual(score, max(s for _, s in evaluated))
                self.assertTrue(
                    any(
                        np.array_equal(parameters, p) and score == s
                        for p, s in evaluated
                    )
                )
                if len(result) == 3:
                    self.assertEqual(result[2].evals, len(evaluated))
                    self.assertEqual(result[2].best_score, score)
                runs.append((result[:2], evaluated, np.random.rand(3)))
        finally:
            np.random.set_state(rng_state)
        self.assertEqual(runs[0][0][0], runs[1][0][0])
        np.testing.assert_array_equal(runs[0][0][1], runs[1][0][1])
        for left, right in zip(runs[0][1], runs[1][1]):
            np.testing.assert_array_equal(left[0], right[0])
            self.assertEqual(left[1], right[1])
        np.testing.assert_array_equal(runs[0][2], runs[1][2])

    def test_zero_iterations_returns_initial_parameters(self):
        for optimize in (cem_optimize, instrumented):
            with self.subTest(optimizer=optimize.__module__):
                with contextlib.redirect_stdout(io.StringIO()):
                    score, parameters, *_ = optimize(
                        disconnected_objective,
                        1,
                        iterations=0,
                        init_mu=[0.7],
                        num_workers=0,
                    )
                self.assertAlmostEqual(score, -0.3)
                np.testing.assert_array_equal(parameters, [0.7])


if __name__ == "__main__":
    unittest.main()
