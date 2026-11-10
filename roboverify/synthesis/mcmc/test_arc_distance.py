"""Spatial imitation ignores speed without discarding order or scene identity."""

import unittest

import numpy as np
from synthesis.api.instructions import MoveByName, Skip
from synthesis.api.program import Program
from synthesis.cfg.demos import DemoSegment, DemoTrace
from synthesis.cfg.straightline import (
    SearchBudget,
    SegmentRollout,
    straight_line_synthesize,
)
from synthesis.mcmc.distance import ArcDistance, arc_resample, make_distance
from synthesis.predicates.scene import scene_from_obs
from synthesis.predicates.term import boolean


def observations(progress, *, origin=0.0):
    progress = np.asarray(progress, dtype=float)
    states = np.zeros((len(progress), 43))
    states[:, :3] = np.column_stack((progress + origin, progress * 0, progress * 0))
    states[:, 10:13] = states[:, :3]
    states[:, 22:25] = (origin, 0.2, 0)
    states[:, 3:5] = 0.04
    return states


def translate(states, displacement):
    states = np.array(states, copy=True)
    for first in (0, 10, 22):
        states[:, first : first + 3] += displacement
    return states


def segment(states, index=0):
    trace = DemoTrace(tuple(states), num_blocks=2)
    return DemoSegment(index, 0, len(states) - 1, trace, {"b0": 0})


class ArcDistanceTests(unittest.TestCase):
    def test_pauses_and_nonlinear_speed_do_not_change_same_path(self):
        reference = observations(np.linspace(0, 0.2, 12))
        warped = observations(
            np.r_[np.zeros(30), np.linspace(0, 1, 71) ** 3 * 0.2, [0.2] * 9]
        )
        metric = ArcDistance(reference, num_blocks=2)
        self.assertLess(metric(warped), 1e-14)
        self.assertEqual(metric(warped, samples=4), metric(warped, samples=4096))

    def test_distance_is_rms_metres_and_uses_stationary_blocks(self):
        reference = observations([0, 0.1, 0.2])
        metric = make_distance("arc", reference, num_blocks=2)
        np.testing.assert_array_equal(metric.moving_blocks, [True, False])
        self.assertAlmostEqual(metric(translate(reference, [0, 0, 0.03])), 0.03)
        changed = reference.copy()
        changed[:, 22] += 0.1
        self.assertAlmostEqual(metric(changed), 0.1 * np.sqrt(0.25 / 3.25))

    def test_block_identity_and_role_names_are_not_required(self):
        reference = observations([0, 0.1, 0.2])
        reference[:, 10:13], reference[:, 22:25] = (
            reference[:, 22:25].copy(),
            reference[:, 10:13].copy(),
        )
        metric = ArcDistance(reference, num_blocks=2)
        np.testing.assert_array_equal(metric.moving_blocks, [False, True])
        self.assertEqual(metric(reference), 0)

    def test_order_and_changed_endpoint_are_visible(self):
        reference = observations([0, 0.1, 0.2])
        metric = ArcDistance(reference, num_blocks=2)
        self.assertGreater(metric(reference[::-1]), 0.1)
        self.assertGreater(metric(observations([0, 0.1, 0.4])), 0.1)

    def test_constant_paths_and_input_validation(self):
        stationary = observations([0, 0, 0])
        metric = ArcDistance(stationary, num_blocks=2)
        self.assertEqual(metric(observations([0])), 0)
        for invalid in ([], [[np.nan] * 43], [[0] * 5]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                metric(invalid)
        with self.assertRaises(ValueError):
            arc_resample([[0, 1]], points=1)


class ArcSearchIntegrationTests(unittest.TestCase):
    def test_each_demonstration_keeps_its_scene_and_equal_aggregate_weight(self):
        demos = [
            segment(observations([0, 0.2])),
            segment(observations(np.linspace(0, 0.2, 100), origin=10), 1),
        ]
        program = Program(1, [Skip(0)])

        def rollout(candidate, demo):
            result = SegmentRollout()
            actual = np.asarray(demo.states)
            if demo.demo_idx == 1:
                actual = translate(actual, [0, 0, 0.1])
            # Predicate boundary callbacks need not be the scored observation path.
            result.extend(
                scene_from_obs(s, 2, demo.bindings)
                for s in translate(actual, [0, 0, 2])
            )
            result.observations = list(actual)
            return result

        result = straight_line_synthesize(
            demos,
            boolean(True),
            program,
            lambda p, rng: p,
            rollout=rollout,
            budget=SearchBudget(
                iterations=0, cem_iterations=0, distance="arc", theta=0.1
            ),
        )
        self.assertTrue(result.ok)
        self.assertAlmostEqual(result.distance, 0.05)
        self.assertEqual(result.post_score, 1)

    def test_changed_endpoint_cannot_pass_a_tight_spatial_threshold(self):
        demo = segment(observations([0, 0.1, 0.2]))
        result = straight_line_synthesize(
            [demo],
            boolean(True),
            Program(1, [Skip(0)]),
            lambda p, rng: p,
            rollout=lambda p, s: observations([0, 0.1, 0.4]),
            budget=SearchBudget(
                iterations=0, cem_iterations=0, distance="arc", theta=0.02
            ),
        )
        self.assertFalse(result.ok)
        self.assertGreater(result.distance, 0.02)

    def test_configured_cem_scale_is_used_and_default_is_preserved(self):
        demo = segment(observations([0, 0.1, 0.2]))
        generated = Program(
            1, [MoveByName("b0", "b0", "b0", target_offset=[0, 0, 0.05])]
        )
        seen = []

        def rollout(candidate, segment):
            seen.append(np.asarray(candidate.register_trainable_parameter()).copy())
            return segment.states

        result = straight_line_synthesize(
            [demo],
            boolean(True),
            Program(0, []),
            None,
            rollout=rollout,
            budget=SearchBudget(
                search_strategy="contract",
                iterations=0,
                cem_iterations=1,
                cem_samples=2,
                cem_elites=1,
                cem_std=0.004,
                distance="arc",
                theta=0.03,
                seed=7,
            ),
            candidates=[generated],
            feasible=lambda p: True,
        )
        self.assertTrue(result.ok)
        expected = np.random.default_rng(7).normal([0, 0, 0.05], [0.004] * 3, (2, 3))
        # Objective, final-distance and PostScore evaluation may revisit the
        # same parameters. Check the sampled vectors, not rollout call counts.
        distinct = list(dict.fromkeys(tuple(values) for values in seen))
        self.assertEqual(len(distinct), 3)
        np.testing.assert_allclose(distinct[1:], expected)
        self.assertEqual(SearchBudget().cem_std, 0.05)
        self.assertEqual(SearchBudget().distance, "kl_kde")
        for value in (0, -1, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                SearchBudget(cem_std=value)


if __name__ == "__main__":
    unittest.main()
