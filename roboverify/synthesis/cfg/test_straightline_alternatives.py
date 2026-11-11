"""Bounded hard-contract search retains alternatives for full-task replay."""

import unittest
from collections import Counter
from unittest.mock import patch

import numpy as np

from synthesis.api.instructions import Skip
from synthesis.api.program import Program
from synthesis.cfg.demos import DemoSegment, DemoTrace
from synthesis.cfg.endpoint_contract import EndpointResult
from synthesis.cfg.straightline import (
    SearchBudget,
    SegmentRollout,
    straight_line_synthesize,
)
from synthesis.predicates.scene import scene_from_obs
from synthesis.predicates.term import atom, ref


def program(number):
    return Program(1, [Skip(number)])


def number(candidate):
    return candidate.instructions[0].skip_steps


def examples():
    obs = np.zeros(43)
    obs[3:5] = 0.04
    obs[10:13] = (0, 0, 0.425)
    obs[22:25] = (0, 0, 0.475)
    trace = DemoTrace(
        (obs, obs), num_blocks=2, metadata={"holding_by_index": {0: None, 1: None}}
    )
    return [DemoSegment(i, 0, 1, trace, {"old": 0, "new": 1}) for i in range(2)]


class RetainedAlternativeTests(unittest.TestCase):
    def search(self, distances, *, invalid=(), candidates=None, **options):
        calls, checks = Counter(), []

        def rollout(candidate, segment):
            key = number(candidate)
            calls[key] += 1
            obs = segment.states[0].copy()
            # A controlled metric below reads the gripper coordinate. Physical
            # ON still uses the ordinary concrete predicate and endpoint audit.
            obs[0] = distances[key]
            states = SegmentRollout()
            states.observations = [obs, obs]
            states.extend(scene_from_obs(obs, 2, segment.bindings) for _ in range(2))
            states.contract_audit = {
                "reasons": ["sampled_nonmover_frame"] if key in invalid else [],
                "contract_bindings": dict(segment.bindings),
                "endpoint": {"known": True, "holding": None, "supports": [(1, 0)]},
                "instructions": [],
            }
            return states

        kwargs = dict(
            iterations=20,
            cem_iterations=0,
            search_strategy="contract",
            distance="arc",
            epsilon=1,
            theta=1,
            pool_limit=10,
            contract_endpoint_checks=True,
        )
        kwargs.update(options.pop("budget", {}))

        def feasible(candidate):
            checks.append(number(candidate))
            return number(candidate) not in options.get("rejected", ())

        with patch(
            "synthesis.cfg.straightline.make_distance",
            return_value=lambda observations, **kwargs: float(observations[0, 0]),
        ):
            result = straight_line_synthesize(
                examples(),
                atom("ON", ref("new"), ref("old")),
                program(0),
                None,
                rollout=rollout,
                budget=SearchBudget(**kwargs),
                candidates=(
                    [program(key) for key in distances]
                    if candidates is None
                    else candidates
                ),
                feasible=feasible,
                cache_rollouts=True,
                endpoint_acceptance=options.get("endpoint_acceptance"),
            )
        return result, calls, checks

    def test_first_valid_does_not_stop_better_complete_alternatives(self):
        result, calls, checks = self.search({1: 0.3, 2: 0.1, 3: 0.2})
        self.assertTrue(result.ok)
        self.assertEqual(checks, [1, 2, 3])
        self.assertEqual(number(result.program), 2)
        self.assertEqual([number(a.program) for a in result.alternatives], [2, 3, 1])
        self.assertEqual([a.distance for a in result.alternatives], [0.1, 0.2, 0.3])
        self.assertEqual(result.search_stats["retained_endpoint_candidates"], 3)
        self.assertTrue(all(a.post_score == 1 for a in result.alternatives))
        self.assertEqual(calls, {1: 2, 2: 2, 3: 2})

    def test_failure_on_one_example_and_handoff_failure_are_excluded(self):
        def handoff(candidate, segment, states, post):
            passed = number(candidate) != 2 or segment.demo_idx == 0
            return EndpointResult(passed, () if passed else ("next_action_not_ready",))

        result, _, _ = self.search(
            {1: 0.1, 2: 0.2, 3: 0.3}, invalid=(1,), endpoint_acceptance=handoff
        )
        self.assertTrue(result.ok)
        self.assertEqual([number(a.program) for a in result.alternatives], [3])
        self.assertEqual(
            result.search_stats["endpoint_rejections"],
            {"sampled_nonmover_frame": 2, "next_action_not_ready": 1},
        )

    def test_final_epsilon_pool_can_remove_earlier_valid_candidate(self):
        result, _, _ = self.search({1: 0.4, 2: 0.1}, budget={"epsilon": 0.1})
        self.assertTrue(result.ok)
        self.assertEqual([number(a.program) for a in result.alternatives], [2])

    def test_theta_and_refinement_only_filter_every_alternative(self):
        partial = program(2)
        partial._contract_refinement_only = True
        result, _, _ = self.search(
            {1: 0.7, 2: 0.2, 3: 0.4},
            candidates=[program(1), partial, program(3)],
            budget={"theta": 0.7},
        )
        self.assertEqual([number(a.program) for a in result.alternatives], [3])

    def test_iteration_and_bmc_budgets_precede_execution(self):
        result, calls, checks = self.search(
            {1: 0.3, 2: 0.2, 3: 0.1}, budget={"iterations": 1}, rejected=(1,)
        )
        self.assertTrue(result.ok)
        self.assertEqual(checks, [1, 2])
        self.assertEqual(calls, {2: 2})
        self.assertEqual((result.candidates_checked, result.bmc_rejected), (2, 1))
        self.assertEqual([number(a.program) for a in result.alternatives], [2])

    def test_retained_trajectories_survive_inferior_trial_lru_eviction(self):
        result, calls, _ = self.search(
            {1: 0.1, 2: 0.2, 3: 0.8, 4: 0.9, 5: 0.7}, budget={"pool_limit": 2}
        )
        self.assertEqual([number(a.program) for a in result.alternatives], [1, 2])
        self.assertEqual(calls, {1: 2, 2: 2, 3: 2, 4: 2, 5: 2})

    def test_no_valid_alternative_preserves_partial_result(self):
        result, _, _ = self.search({1: 0.1, 2: 0.2}, invalid=(1, 2))
        self.assertFalse(result.ok)
        self.assertEqual(result.alternatives, ())
        self.assertEqual(result.search_stats["retained_endpoint_candidates"], 0)
        self.assertEqual(number(result.program), 1)

    def test_default_search_still_returns_first_success_without_alternatives(self):
        result, calls, checks = self.search(
            {1: 0.3, 2: 0.1}, budget={"contract_endpoint_checks": False}
        )
        self.assertTrue(result.ok)
        self.assertEqual(number(result.program), 1)
        self.assertEqual(checks, [1])
        self.assertEqual(calls, {1: 2})
        self.assertEqual(result.alternatives, ())


if __name__ == "__main__":
    unittest.main()
