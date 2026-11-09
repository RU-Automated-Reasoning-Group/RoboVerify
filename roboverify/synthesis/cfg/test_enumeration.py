import unittest
from functools import partial
from unittest.mock import Mock

from synthesis.api.instructions import (
    Get,
    Move,
    Pick,
    PickByName,
    Release,
    ReleaseByName,
)
from synthesis.api.program import Program
from synthesis.cfg.bindings import close_objects
from synthesis.cfg.demos import DemoSegment, DemoTrace
from synthesis.cfg.search_feasibility import check_candidate_feasibility
from synthesis.cfg.straightline import SearchBudget, straight_line_synthesize
from synthesis.predicates.scene import Scene
from synthesis.predicates.term import boolean
from synthesis.verification_lib.highlevel_verification_lib import HighLevelContext


def segment(start=0):
    scene = Scene(
        {0: (0.0, 0.0, 0.425), 1: (0.2, 0.0, 0.425)}, {"a": 0, "alias": 0, "b": 1}
    )
    return DemoSegment(
        0,
        start,
        start + 1,
        DemoTrace((scene, scene, scene), num_blocks=2),
        scene.bindings,
    )


def program(*instructions):
    return Program(len(instructions), list(instructions))


class BMCEnumerationTests(unittest.TestCase):
    def test_actual_bmc_screens_structure_before_float_tuning(self):
        demos = [segment()]
        self.assertFalse(check_candidate_feasibility(program(Pick(0), Pick(1)), demos))
        self.assertFalse(check_candidate_feasibility(program(Release(1)), demos))
        # Existential BMC offsets, independent of the untuned candidate's values.
        candidate = program(
            Pick(0), Move(1, 0, 1, target_offset=[90, 90, 90]), Release(1)
        )
        self.assertTrue(check_candidate_feasibility(candidate, demos))
        self.assertEqual(candidate.register_trainable_parameter()[:3], [90, 90, 90])

    def test_named_aliases_and_get_are_grounded_at_each_entry(self):
        demos = [segment()]
        self.assertTrue(
            check_candidate_feasibility(
                program(PickByName("a"), ReleaseByName("alias")), demos
            )
        )
        closed = close_objects(program(Pick(1)), demos, {"a"}, HighLevelContext())
        self.assertIsInstance(closed.instructions[0], Get)
        self.assertTrue(check_candidate_feasibility(closed, demos))
        missing = program(PickByName("absent"))
        self.assertFalse(check_candidate_feasibility(missing, demos))

    def test_every_segment_is_checked_and_interior_holding_is_not_assumed(self):
        candidate = program(Release(1))
        self.assertTrue(check_candidate_feasibility(candidate, [segment(1)]))
        self.assertFalse(
            check_candidate_feasibility(candidate, [segment(1), segment(0)])
        )


class CFGEnumerationTests(unittest.TestCase):
    def test_real_bmc_then_real_cem_updates_floats_and_preserves_goal_check(self):
        demos = [segment()]
        rejected = program(Pick(0), Pick(1))
        candidate = program(Move(0, 0, 0, target_offset=[0, 0, 0]))
        events = []

        def feasible(p):
            events.append("bmc")
            return check_candidate_feasibility(p, demos)

        def rollout(p, s):
            self.assertEqual(
                len(p.instructions),
                1,
                "Rejected candidates must never reach CEM/rollout",
            )
            self.assertIn("bmc", events)
            events.append("rollout")
            value = p.register_trainable_parameter()[0]
            return [Scene({0: (value - 0.1, 0, 0.425), 1: (0.2, 0, 0.425)}, s.bindings)]

        propose = Mock(side_effect=AssertionError("Enumeration must not mutate"))
        result = straight_line_synthesize(
            demos,
            boolean(False),
            rejected,
            propose,
            rollout=rollout,
            budget=SearchBudget(
                search_strategy="enumeration",
                iterations=5,
                cem_iterations=3,
                cem_samples=16,
                cem_elites=4,
                distance="kl_kde",
                inner_samples=16,
                final_samples=32,
                seed=19,
            ),
            candidates=iter([candidate]),
            feasible=feasible,
        )
        self.assertEqual(events[:2], ["bmc", "bmc"])
        self.assertFalse(result.ok, "Imitation alone does not prove the postcondition")
        self.assertEqual(result.status, "space_exhausted")
        self.assertEqual(
            (result.candidates_checked, result.bmc_rejected, result.iterations),
            (2, 1, 1),
        )
        self.assertGreater(result.program.register_trainable_parameter()[0], 0.05)
        self.assertEqual(candidate.register_trainable_parameter(), [0, 0, 0])
        propose.assert_not_called()

    def test_rejected_candidates_never_roll_out_and_cannot_succeed(self):
        rollout = Mock(side_effect=AssertionError("Must screen before simulation"))
        result = straight_line_synthesize(
            [segment()],
            boolean(True),
            program(Pick(0), Pick(1)),
            None,
            rollout=rollout,
            budget=SearchBudget(
                search_strategy="enumeration", iterations=1, distance="mmd"
            ),
            candidates=[program(Release(1))],
            feasible=partial(check_candidate_feasibility, segments=[segment()]),
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.status, "no_feasible_candidate")
        self.assertEqual(result.bmc_rejected, 2)
        rollout.assert_not_called()

    def test_success_and_mandatory_screen(self):
        demos = [segment()]
        kwargs = dict(
            rollout=lambda p, s: s.states,
            budget=SearchBudget(
                search_strategy="enumeration", iterations=0, distance="mmd"
            ),
        )
        with self.assertRaisesRegex(ValueError, "BMC screening"):
            straight_line_synthesize(demos, boolean(True), program(), None, **kwargs)
        result = straight_line_synthesize(
            demos,
            boolean(True),
            program(),
            None,
            candidates=[],
            feasible=partial(check_candidate_feasibility, segments=demos),
            **kwargs
        )
        self.assertTrue(result.ok)
        self.assertEqual(result.candidates_checked, 1)
