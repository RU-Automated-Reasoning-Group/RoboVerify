"""Standalone routing through actual contract proposals, BMC and CEM."""

import contextlib
import io
import tempfile
import unittest
from functools import partial
from unittest.mock import patch

from synthesis.api.instructions import Move, Pick, Release
from synthesis.api.program import Program
from synthesis.cfg.contract_search import (
    check_contract_feasibility,
    make_contract_candidates,
)
from synthesis.cfg.test_contract_search import held_segment
from synthesis.experiment.config import MCMCConfig, config_from_args
from synthesis.experiment.mcmc import run, search
from synthesis.experiment.run_logger import RunLogger
from synthesis.experiment.test_enumeration import QuadraticRunner
from synthesis.predicates.scene import Scene
from synthesis.predicates.term import atom, block_id


class StandaloneContractTests(unittest.TestCase):
    def test_goal_feature_keeps_its_original_numeric_semantics(self):
        scene = Scene({0: [0, 0, 0.4], 1: [0, 0, 0.6]}, {})
        self.assertFalse(
            run.contract_goal("stack", run.parse_goal_feature("ON(1,0)"))(scene)
        )
        self.assertTrue(
            run.contract_goal("stack", run.parse_goal_feature("ON(0,0)"))(scene)
        )
        for seconds in (float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                MCMCConfig(search_strategy="contract", contract_seconds=seconds)

    def test_actual_proposals_are_screened_before_shared_cem(self):
        segment = held_segment()
        config = MCMCConfig(
            search_strategy="contract",
            program_slots=1,
            num_blocks=2,
            iters=2,
            contract_seconds=3,
            cem_N=2,
            cem_K=1,
            cem_iterations=1,
            video_policy="none",
            checkpoint_every=0,
            progress_every=0,
        )
        candidates = make_contract_candidates(
            [segment],
            atom("ON", block_id(1), block_id(0)),
            Program(0),
            (),
            max_length=1,
            budget=config,
            id_first=True,
        )
        events = []
        real_optimize = search.optimize_program

        def feasible(candidate):
            result = check_contract_feasibility(candidate, [segment])
            events.append(("bmc", bool(result)))
            return result

        def optimize(*args, **kwargs):
            self.assertEqual(events[-1], ("bmc", True))
            events.append(("cem", True))
            return real_optimize(*args, **kwargs)

        with tempfile.TemporaryDirectory() as root, RunLogger(
            root, "contract", {}, capture_stdout=False
        ) as logger, patch.object(
            search, "InstrumentedRunner", QuadraticRunner
        ), patch.object(
            search, "optimize_program", side_effect=optimize
        ), patch.object(
            search, "mutate_program", side_effect=AssertionError("No mutation")
        ), patch.object(
            search.random, "random", side_effect=AssertionError("No Metropolis draw")
        ):
            result = search.MCMC(
                Program(1),
                {"Box": [0, 1]},
                [Pick, Move, Release],
                config,
                [],
                logger=logger,
                candidate_programs=candidates,
                feasibility_check=feasible,
                motion_penalty=lambda p: 0,
            )
        self.assertIsNotNone(result.best_program)
        self.assertTrue(result.best_checks_passed)
        self.assertEqual(
            sum(x[0] == "cem" for x in events), sum(x == ("bmc", True) for x in events)
        )
        self.assertNotEqual(result.best_iter, -1)

    def test_missing_adapter_or_bmc_fails_clearly(self):
        config = MCMCConfig(search_strategy="contract")
        with self.assertRaisesRegex(ValueError, "candidate stream"):
            search.MCMC(Program(0), {}, [], config, [], logger=None)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            run.main(["--search-strategy", "contract", "--no-bmc"])

    def test_flags_round_trip_without_changing_default(self):
        parser = run.build_parser()
        self.assertEqual(
            config_from_args(parser.parse_args([])).search_strategy, "mcmc"
        )
        config = config_from_args(
            parser.parse_args(
                [
                    "--search-strategy",
                    "contract",
                    "--contract-beam-width",
                    "7",
                    "--contract-seconds",
                    "2",
                    "--contract-grid-only",
                    "--contract-no-memo",
                ]
            )
        )
        self.assertEqual(config.contract_beam_width, 7)
        self.assertEqual(config.contract_seconds, 2)
        self.assertFalse(config.contract_observation_proposals)
        self.assertFalse(config.contract_memoize)


if __name__ == "__main__":
    unittest.main()
