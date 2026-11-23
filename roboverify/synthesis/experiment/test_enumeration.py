import contextlib
import io
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from synthesis.api.instructions import Move, Pick, Release
from synthesis.api.program import Program
from synthesis.experiment.config import MCMCConfig, config_from_args
from synthesis.experiment.mcmc import run, search
from synthesis.experiment.run_logger import RunLogger
from synthesis.mcmc.synthesis import (
    preserved_global_rng,
    roboverify_bmc_initial_constraints,
)


class QuadraticRunner:
    """Synthetic rollout objective; the BMC solver and CEM remain real."""

    last_goal_feature_report = "synthetic objective"
    last_mmd = None
    last_success_rate = None

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, parameters):
        return -float(np.square(np.asarray(parameters) - 0.1).sum())


class StandaloneEnumerationTests(unittest.TestCase):
    def test_shared_bmc_and_cem_scorer_exhausts_space_and_tunes_floats(self):
        events = []
        real_check = search.check_bmc_candidate
        real_optimize = search.optimize_program

        def check(p, *args, **kwargs):
            result = real_check(p, *args, **kwargs)
            events.append(("bmc", result[0]))
            return result

        def optimize(*args, **kwargs):
            self.assertEqual(events[-1], ("bmc", True))
            events.append(("cem", True))
            return real_optimize(*args, **kwargs)

        with tempfile.TemporaryDirectory() as root, preserved_global_rng():
            np.random.seed(7)
            config = MCMCConfig(
                search_strategy="enumeration",
                iters=20,
                program_slots=2,
                num_blocks=1,
                cem_N=16,
                cem_K=4,
                cem_iterations=3,
                video_policy="none",
                checkpoint_every=0,
                progress_every=0,
            )
            with RunLogger(
                root, "enumeration", config.to_json_dict(), capture_stdout=False
            ) as logger, patch.object(
                search, "check_bmc_candidate", side_effect=check
            ), patch.object(
                search, "optimize_program", side_effect=optimize
            ), patch.object(
                search, "InstrumentedRunner", QuadraticRunner
            ), patch.object(
                search, "mutate_program", side_effect=AssertionError("No mutation")
            ), patch.object(
                search.random,
                "random",
                side_effect=AssertionError("No Metropolis draw"),
            ):
                result = search.MCMC(
                    Program(0),
                    {"Box": [0]},
                    [Pick, Move, Release],
                    config,
                    [],
                    logger=logger,
                    bmc_goal=lambda sym: sym.bz["0"][sym.T] > sym.bz["0"][0] + 0.05,
                    bmc_initial_constraints=roboverify_bmc_initial_constraints(),
                    motion_penalty=lambda p: 0,  # Run real CEM in-process.
                )
                metrics = [
                    json.loads(s)
                    for s in (logger.run_dir / "metrics.jsonl").read_text().splitlines()
                ]
            self.assertEqual(result.termination_reason, "space_exhausted")
            self.assertEqual(result.iters_completed, 13)
            self.assertEqual(len(result.costs), 14)  # warm seed plus 13 structures
            self.assertTrue(result.best_checks_passed)
            self.assertTrue(result.best_program.register_trainable_parameter())
            self.assertGreater(
                result.best_program.register_trainable_parameter()[0], 0.05
            )
            self.assertEqual(
                sum(e[0] == "cem" for e in events),
                sum(e == ("bmc", True) for e in events),
            )
            self.assertTrue(any(m.get("cem_delta", 0) > 0 for m in metrics))
            self.assertTrue(all(m["acceptance_ratio"] is None for m in metrics[1:]))

    def test_failed_bmc_never_optimizes_or_returns_best_candidate(self):
        with tempfile.TemporaryDirectory() as root:
            config = MCMCConfig(
                search_strategy="enumeration",
                iters=3,
                program_slots=1,
                video_policy="none",
                checkpoint_every=0,
                progress_every=0,
            )
            with RunLogger(
                root, "empty", {}, capture_stdout=False
            ) as logger, patch.object(
                search, "check_bmc_candidate", return_value=(False, "BMC: INFEASIBLE")
            ), patch.object(
                search,
                "optimize_program",
                side_effect=AssertionError("CEM must be skipped"),
            ):
                result = search.MCMC(
                    Program(0),
                    {"Box": [0]},
                    [Pick],
                    config,
                    [],
                    logger=logger,
                    bmc_goal=lambda sym: True,
                )
                self.assertFalse(
                    (logger.run_dir / "artifacts/best/program.pkl").exists()
                )
                logger.finish("no_feasible_candidate")
                metrics = [
                    json.loads(line)
                    for line in (logger.run_dir / "metrics.jsonl")
                    .read_text()
                    .splitlines()
                ]
                self.assertTrue(all(m["best_cost"] is None for m in metrics[1:]))
            self.assertIsNone(result.best_program)
            self.assertFalse(result.best_checks_passed)
            self.assertEqual(result.termination_reason, "space_exhausted")

    def test_cli_defaults_and_enumeration_requires_bmc_goal(self):
        parser = run.build_parser()
        self.assertEqual(
            config_from_args(parser.parse_args([])).search_strategy, "mcmc"
        )
        self.assertEqual(
            config_from_args(
                parser.parse_args(["--search-strategy", "enumeration"])
            ).search_strategy,
            "enumeration",
        )
        for extra in ([], ["--no-bmc", "--goal-feature", "ON(1,0)"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(
                SystemExit
            ) as error:
                run.main(["--search-strategy", "enumeration", *extra])
            self.assertEqual(error.exception.code, 2)
        with self.assertRaises(ValueError):
            MCMCConfig(search_strategy="typo")
