import unittest
from unittest.mock import Mock

import numpy as np

from synthesis.api.instructions import Skip
from synthesis.api.program import Program
from synthesis.cfg.demos import DemoSegment, DemoTrace
from synthesis.cfg.graph import RelationalCFG
from synthesis.cfg.region import BlockRegion
from synthesis.cfg.straightline import SearchBudget, straight_line_synthesize
from synthesis.cfg.synthesize import synthesize_cfg
from synthesis.mcmc.distance import KDEDistance, MMDDistance
from synthesis.mcmc.search_core import CandidatePool, acceptance_probability
from synthesis.predicates.scene import Scene
from synthesis.predicates.term import atom, boolean, ref


class SearchTests(unittest.TestCase):
    def test_shared_acceptance_and_pool_filters_before_ranking(self):
        self.assertEqual(acceptance_probability(1, 0.1), 1.0)
        self.assertLess(
            acceptance_probability(-1, 0.1), acceptance_probability(-1, 1.0)
        )
        pool = CandidatePool(0.1, 10)
        for program, distance in [
            ("accurate", 0.2),
            ("goal", 0.25),
            ("bad_imitation", 5.0),
        ]:
            pool.add(program, distance)
        winner, score = pool.select(
            lambda p: {"accurate": 0, "goal": 1, "bad_imitation": 2}[p]
        )
        self.assertEqual(winner, "goal")
        pool.add("new_best", 0.0)
        self.assertEqual([e.program for e in pool.entries], ["new_best"])

    def test_cached_kl_is_deterministic_and_distinguishes_shift(self):
        data = np.array([[0.0, 0.0], [0.1, 0.1], [0.2, 0.2]])
        distance = KDEDistance(data, seed=9)
        self.assertAlmostEqual(distance(data), 0.0)
        self.assertGreater(distance(data + 1), distance(data))
        self.assertEqual(distance(data + 0.1), distance(data + 0.1))

    def test_postscore_enabled_and_failed_task_not_called_success(self):
        scene = Scene({0: (0.0, 0.0, 0.425)}, {"b": 0})
        segment = DemoSegment(0, 0, 1, DemoTrace((scene, scene)), {"b": 0})
        candidate = Program(1, [Skip(0)])
        rollout = lambda p, s: list(s.states)
        result = straight_line_synthesize(
            [segment],
            boolean(True),
            candidate,
            lambda p, r: p,
            rollout=rollout,
            budget=SearchBudget(iterations=0, distance="mmd"),
        )
        self.assertTrue(result.ok)
        self.assertEqual(result.post_score, 1.0)
        failed = straight_line_synthesize(
            [segment],
            boolean(False),
            candidate,
            lambda p, r: p,
            rollout=rollout,
            budget=SearchBudget(iterations=1, cem_iterations=0, distance="mmd"),
        )
        self.assertFalse(failed.ok)

    def test_driver_runs_all_blocks_and_retains_failure(self):
        cfg = RelationalCFG.initial([], boolean(True), boolean(True))
        region = BlockRegion((Skip(0),), (Skip(0),))
        result = synthesize_cfg(cfg, lambda n, d, p: (region, True), lambda g: [])
        self.assertTrue(result)
        self.assertIs(cfg.nodes["v0"].region, region)
        result = synthesize_cfg(
            cfg, lambda n, d, p: (region, False), lambda g: [], max_refinements=0
        )
        self.assertFalse(result)
        self.assertEqual(result.status, "budget_exhausted")


if __name__ == "__main__":
    unittest.main()
