"""Behavioral regressions for empirical final-state candidate contracts."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from synthesis.api.instructions import Assign, Move, Pick, Release, Skip
from synthesis.api.program import Program
from synthesis.cfg.demos import DemoSegment, DemoTrace
from synthesis.cfg.endpoint_contract import (
    EndpointRecorder,
    EndpointResult,
    audit_put_effects,
    check_endpoint_contract,
    check_next_action,
    postcondition_at_endpoint,
)
from synthesis.cfg.straightline import (
    SearchBudget,
    SegmentRollout,
    postcondition_reached,
    straight_line_synthesize,
)
from synthesis.predicates.scene import Scene, scene_from_obs
from synthesis.predicates.term import atom, boolean, ref


def observation(positions, *, empty=True):
    result = np.zeros(13 + 15 * len(positions))
    result[3:5] = 0.03 if empty else 0.02
    for i, position in enumerate(positions):
        result[10 + 12 * i : 13 + 12 * i] = position
    return result


def segment(before, after, bindings=None, holding=None):
    bindings = bindings or {"new": 1, "old": 0}
    trace = DemoTrace(
        (before, after),
        num_blocks=(len(before) - 13) // 15,
        metadata={"holding_by_index": {0: holding, 1: holding}},
    )
    return DemoSegment(0, 0, 1, trace, bindings)


def audited_states(
    seg, observations, *, holding=None, bindings=None, instructions=None, reasons=()
):
    states = SegmentRollout()
    states.observations = list(observations)
    states.extend(
        scene_from_obs(obs, seg.trace.num_blocks, bindings or seg.bindings)
        for obs in observations
    )
    states.contract_audit = {
        "reasons": list(reasons),
        "contract_bindings": dict(seg.bindings),
        "endpoint": {"known": True, "holding": holding, "supports": [(1, 0)]},
        "instructions": instructions or [],
    }
    return states


class EndpointTests(unittest.TestCase):
    def test_transient_on_does_not_count_as_endpoint_success(self):
        before = observation([(0, 0, 0.425), (0.2, 0, 0.425)])
        placed = observation([(0, 0, 0.425), (0, 0, 0.475)])
        seg = segment(before, before)
        states = [
            scene_from_obs(obs, 2, seg.bindings) for obs in (before, placed, before)
        ]
        post = atom("ON", ref("new"), ref("old"))
        self.assertTrue(postcondition_reached(states, seg, post))
        self.assertFalse(postcondition_at_endpoint(states, seg, post))

    def test_endpoint_preserves_frozen_entry_geometry(self):
        before = observation([(0, 0, 0.425), (0, 0, 0.475)])
        after = observation([(0, 0, 0.425), (0.2, 0, 0.425)])
        seg = segment(before, after)
        self.assertTrue(
            postcondition_at_endpoint(
                [after], seg, atom("ON_star_zero", ref("new"), ref("old"))
            )
        )
        self.assertFalse(
            postcondition_at_endpoint([after], seg, atom("ON", ref("new"), ref("old")))
        )

    def test_bindings_for_on_are_taken_before_pointer_update(self):
        before = observation([(0, 0, 0.425), (0.2, 0, 0.425)])
        after = observation([(0, 0, 0.425), (0, 0, 0.475)])
        seg = segment(before, after)
        states = audited_states(
            seg,
            [before, after],
            bindings={"new": 1, "old": 1},
            instructions=[{"kind": "Release", "mover": 1}],
        )
        result = check_endpoint_contract(
            Program(0, []), seg, states, atom("ON", ref("new"), ref("old"))
        )
        self.assertTrue(result, result.reasons)
        self.assertEqual(states[-1].bindings["old"], 1)

    def test_local_on_does_not_hide_scattered_effect_failure(self):
        before = Scene({0: (0, 0, 0.425), 1: (0.2, 0.2, 0.425), 2: (0, 0.10037, 0.425)})
        after = Scene(
            {0: (0, 0, 0.425), 1: (0, 0.000392, 0.475), 2: (0, 0.10037, 0.425)}
        )
        result = audit_put_effects(before, after, 1, 0)
        self.assertIn("put_effect:Scattered", result.reasons)
        self.assertIn("endpoint_domain:scattered3", result.reasons)
        self.assertNotIn("put_effect:ON_star", result.reasons)
        self.assertNotIn("put_not_enabled", result.reasons)

    def test_valid_finite_put_effect_and_frame_are_accepted(self):
        before = Scene({0: (0, 0, 0.425), 1: (0.2, 0.2, 0.425), 2: (0.3, 0, 0.425)})
        after = Scene({0: (0, 0, 0.425), 1: (0, 0, 0.475), 2: (0.3, 0, 0.425)})
        self.assertTrue(audit_put_effects(before, after, 1, 0))

    def test_invalid_premises_are_not_vacuous_success(self):
        before = Scene({0: (0, 0, 0.425), 1: (0, 0, 0.475), 2: (0.2, 0, 0.425)})
        result = audit_put_effects(before, before, 1, 0)
        self.assertIn("put_not_enabled", result.reasons)

    def test_intermediate_higher_block_can_end_holding(self):
        before = observation([(0, 0, 0.425), (0.2, 0, 0.425)], empty=False)
        after = observation([(0, 0, 0.425), (0.2, 0, 0.6)], empty=False)
        seg = segment(before, after, holding=1)
        states = audited_states(
            seg, [before, after], holding=1, instructions=[{"kind": "Move", "mover": 1}]
        )
        result = check_endpoint_contract(
            Program(0, []), seg, states, atom("Higher", ref("new"), ref("old"))
        )
        self.assertTrue(result, result.reasons)
        self.assertEqual(result.details["put_effects"], [])

    def test_audits_are_required_and_failed_physical_contract_rejects(self):
        obs = observation([(0, 0, 0.425), (0.2, 0, 0.425)])
        seg = segment(obs, obs)
        result = check_endpoint_contract(Program(0, []), seg, [obs], boolean(True))
        self.assertEqual(result.reasons, ("missing_physical_endpoint_audit",))
        states = audited_states(seg, [obs, obs], reasons=("primitive_not_converged",))
        result = check_endpoint_contract(Program(0, []), seg, states, boolean(True))
        self.assertFalse(result)

    def test_search_keeps_partial_failure_but_never_accepts_transient_post(self):
        before = observation([(0, 0, 0.425), (0.2, 0, 0.425)])
        placed = observation([(0, 0, 0.425), (0, 0, 0.475)])
        seg = segment(before, before)
        program = Program(1, [Skip(0)])
        result = straight_line_synthesize(
            [seg],
            atom("ON", ref("new"), ref("old")),
            program,
            lambda p, rng: p,
            rollout=lambda p, s: audited_states(s, [before, placed, before]),
            budget=SearchBudget(
                iterations=0, distance="mmd", theta=100, contract_endpoint_checks=True
            ),
        )
        self.assertFalse(result.ok)
        self.assertIsNotNone(result.program)
        self.assertIn(
            "postcondition_not_at_endpoint", result.search_stats["endpoint_rejections"]
        )

    def test_optional_handoff_callback_can_reject_after_endpoint_passes(self):
        obs = observation([(0, 0, 0.425), (0.2, 0, 0.425)])
        seg = segment(obs, obs)
        result = straight_line_synthesize(
            [seg],
            boolean(True),
            Program(1, [Skip(0)]),
            lambda p, rng: p,
            rollout=lambda p, s: audited_states(s, [obs, obs]),
            endpoint_acceptance=lambda *args: EndpointResult(
                False, ("next_action_not_ready",)
            ),
            budget=SearchBudget(
                iterations=0, distance="mmd", contract_endpoint_checks=True
            ),
        )
        self.assertFalse(result.ok)
        self.assertEqual(
            result.search_stats["endpoint_rejections"], {"next_action_not_ready": 1}
        )

    def test_sampled_frame_accumulates_across_whole_transfer(self):
        first = observation([(0, 0, 0.425), (0.2, 0, 0.425)])
        seg = segment(first, first)
        env = SimpleNamespace(symbolic_name_to_box_id=seg.bindings)
        recorder = EndpointRecorder(env, seg, first, frame_tolerance_m=0.005)
        empty = {"known": True, "holding": None, "supports": [], "finger_objects": []}
        held = {"known": True, "holding": 1, "supports": [], "finger_objects": [1]}
        pick, move = Pick(1), Move(0, 0, 0)
        pick.last_control_result = move.last_control_result = SimpleNamespace(
            converged=True
        )
        shifted = first.copy()
        shifted[10] = 0.004
        shifted_again = first.copy()
        shifted_again[10] = 0.008
        with patch.object(recorder, "_contact", side_effect=[empty, held, held, held]):
            recorder.instruction(pick, "start", first)
            recorder.instruction(pick, "end", shifted)
            recorder.instruction(move, "start", shifted)
            recorder.instruction(move, "end", shifted_again)
        self.assertIn("sampled_nonmover_frame", recorder.reasons)
        self.assertAlmostEqual(recorder.maximum_drift, 0.008)

    def test_on_placement_cannot_be_accepted_while_still_held(self):
        before = observation([(0, 0, 0.425), (0.2, 0, 0.425)], empty=False)
        after = observation([(0, 0, 0.425), (0, 0, 0.475)], empty=False)
        seg = segment(before, after, holding=1)
        states = audited_states(
            seg, [before, after], holding=1, instructions=[{"kind": "Move", "mover": 1}]
        )
        result = check_endpoint_contract(
            Program(0, []), seg, states, atom("ON", ref("new"), ref("old"))
        )
        self.assertIn("endpoint_placement_not_released", result.reasons)

    def test_binding_only_endpoint_keeps_assignment_semantics(self):
        obs = observation([(0, 0, 0.425), (0.2, 0, 0.425)])
        seg = segment(obs, obs)
        states = audited_states(seg, [obs, obs], bindings={"new": 1, "old": 1})
        result = check_endpoint_contract(
            Program(1, [Assign("old", "new")]),
            seg,
            states,
            atom("eq", ref("new"), ref("old")),
        )
        self.assertTrue(result, result.reasons)

    def test_nested_instruction_callbacks_cover_loop_body(self):
        import z3

        from synthesis.api.instructions import While
        from synthesis.cfg.execute import execute_current

        obs = observation([(0, 0, 0.425), (0.2, 0, 0.425)])
        env = SimpleNamespace(symbolic_name_to_box_id={}, num_blocks=2)
        loop = While(z3.BoolVal(True), [], [Skip(0)], z3.BoolVal(True))
        program = Program(1, [loop])
        calls = []
        with patch.object(
            While, "_find_and_bind_guard_exists", side_effect=[True, False]
        ):
            trajectory = execute_current(
                program,
                env,
                obs,
                on_instruction=lambda inst, phase, observation: calls.append(
                    (type(inst).__name__, phase)
                ),
            )
        self.assertEqual(
            calls,
            [("While", "start"), ("Skip", "start"), ("Skip", "end"), ("While", "end")],
        )
        self.assertEqual(len(trajectory), 1)

    def test_handoff_uses_candidate_actual_first_action_and_next_roles(self):
        from synthesis.api.instructions import PickByName

        obs = observation([(0, 0, 0.425), (0.2, 0, 0.425)])
        obs[:3] = (0, 0, 0.6)
        seg = segment(obs, obs)
        states = audited_states(seg, [obs, obs])
        candidate = Program(1, [PickByName("new")])
        result = check_next_action(candidate, seg, states, {"new": 1, "old": 0})
        self.assertTrue(result, result.reasons)
        low = obs.copy()
        low[:3] = (0, 0, 0.425)
        bad = audited_states(seg, [obs, low])
        self.assertFalse(check_next_action(candidate, seg, bad, {"new": 1, "old": 0}))
        self.assertTrue(check_next_action(candidate, seg, bad, None))

    def test_audited_simulator_rollout_preserves_trace_and_observes_nested_primitives(
        self,
    ):
        from synthesis.cfg.collection import record_execution, validate_trace
        from synthesis.cfg.program_source import load_program
        from synthesis.cfg.straightline import segment_rollout
        from synthesis.cfg.tasks import task_spec
        from synthesis.mcmc.synthesis import make_roboverify_stack_env
        from synthesis.verification_lib.highlevel_verification_lib import (
            HighLevelContext,
        )

        definition = load_program(
            "synthesis.examples.stack:build_program", HighLevelContext(), 2
        )
        trace = record_execution(
            definition, seed=38, num_blocks=2, max_loop_iterations=1
        )
        self.assertTrue(validate_trace(trace), trace.metadata)
        seg = DemoSegment(0, 0, len(trace.states) - 1, trace, {"b0": 0})
        states = segment_rollout(
            definition.program,
            seg,
            lambda _: make_roboverify_stack_env(num_blocks=2),
            audit_contract=True,
        )
        np.testing.assert_allclose(states.observations, trace.states, atol=1e-8, rtol=0)
        audit = states.contract_audit
        self.assertFalse(audit["reasons"], audit)
        self.assertGreater(len(audit["instructions"]), 2)
        self.assertTrue(all(row["converged"] for row in audit["instructions"]))
        self.assertTrue(audit["endpoint"]["known"])
        self.assertIsNone(audit["endpoint"]["holding"])
        self.assertIn((1, 0), audit["endpoint"]["supports"])
        result = check_endpoint_contract(
            definition.program, seg, states, task_spec("stack")[1]
        )
        self.assertTrue(result, result.reasons)

    def test_positive_solver_budgets(self):
        for kwargs in (
            {"contract_solver_timeout_ms": 0},
            {"contract_solver_attempts": True},
            {"contract_offset_radius": float("nan")},
        ):
            with self.assertRaises(ValueError):
                SearchBudget(**kwargs)


if __name__ == "__main__":
    unittest.main()
