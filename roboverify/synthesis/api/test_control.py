"""Controller convergence, bounded execution, and numeric/named parity."""

import unittest
from copy import deepcopy
from dataclasses import replace

import numpy as np

from synthesis.api.control import ControlConfig, PrimitiveController, get_move_action
from synthesis.api.instructions import Move, Pick, Release
from synthesis.api.program import Program
from synthesis.api.runtime import execute_instruction
from synthesis.cfg.collection import record_execution, validate_trace
from synthesis.cfg.physical import name_operands
from synthesis.cfg.program_source import (
    ProgramDefinition,
    describe_program,
    program_fingerprint,
)


class ServoEnvironment:
    """Deterministic servo; tests termination independently of MuJoCo contacts."""

    def __init__(self, *, stalled=False):
        self.env = self
        self.symbolic_name_to_box_id = {"base": 0, "other": 1}
        self.obs = np.zeros(43)
        self.obs[:3] = [0.1, 0.1, 0.6]
        self.obs[3:5] = 0.04
        self.obs[10:13] = [0, 0, 0.425]
        self.obs[22:25] = [0.2, 0, 0.425]
        self.actions, self.stalled = [], stalled

    def _get_obs(self):
        return self.obs.copy()

    def flatten_observation(self, obs):
        return obs

    def step(self, action):
        self.actions.append(action.copy())
        if not self.stalled:
            self.obs[:3] += 0.05 * np.clip(action[:3], -1, 1)
            self.obs[3:5] = np.clip(self.obs[3:5] + action[3] * 0.02, 0.01, 0.04)

    def render(self):
        return self.obs[:3].copy()


def placement(control):
    return [
        Pick(1, control=control),
        Move(1, 1, 0, target_offset=[0, 0, 0.20], control=control),
        Move(0, 0, 0, target_offset=[0, 0, 0.20], control=control),
        Move(0, 0, 0, target_offset=[0, 0, 0.05], control=control),
        Release(1, target_z=0.15, control=control),
    ]


class ControlTests(unittest.TestCase):
    def test_action_is_proportional_and_has_explicit_gripper_command(self):
        np.testing.assert_allclose(
            get_move_action([1, 2, 3], [1.1, 1.8, 3.3], gain=2, close_gripper=True),
            [0.2, -0.4, 0.6, -0.2],
        )
        np.testing.assert_array_equal(
            get_move_action([0, 0, 0], [0, 0, 0]), np.zeros(4)
        )
        with self.assertRaises(TypeError):
            get_move_action([0, 0, 0], [0, 0, 0], atol=0.1)

    def test_position_tolerance_controls_termination(self):
        counts = []
        for tolerance in (0.02, 0.002):
            env, trajectory = ServoEnvironment(), []
            controller = PrimitiveController(
                env,
                trajectory,
                control=ControlConfig(position_tolerance=tolerance, gain=10),
            )
            target = env.obs[:3] + [0.1, 0.1, 0.1]
            self.assertTrue(controller.move(target))
            self.assertLessEqual(np.linalg.norm(env.obs[:3] - target), tolerance)
            self.assertEqual(len(trajectory), controller.steps)
            np.testing.assert_array_equal(trajectory[-1], env.obs)
            counts.append(controller.steps)
        self.assertGreater(counts[1], counts[0])

    def test_exact_boundary_and_zero_budget_do_not_step(self):
        env = ServoEnvironment()
        controller = PrimitiveController(
            env, [], limit=0, control=ControlConfig(position_tolerance=0.125)
        )
        self.assertTrue(controller.move(env.obs[:3] + [0, 0, 0.125]))
        self.assertEqual(env.actions, [])
        self.assertFalse(controller.move(env.obs[:3] + [0, 0, 0.25]))
        self.assertFalse(controller.result.converged)
        self.assertEqual(controller.result.steps, 0)

    def test_step_budget_covers_entire_pick_and_reports_failed_phase(self):
        env = ServoEnvironment()
        instruction = Pick(
            1, limit=8, control=ControlConfig(position_tolerance=0.005, gain=10)
        )
        instruction.eval(env, [])
        self.assertEqual(len(env.actions), 8)
        self.assertFalse(instruction.last_control_result.converged)
        self.assertEqual(instruction.last_control_result.phase, "descend")

    def test_stalled_controller_and_runtime_event_preserve_failure(self):
        env, events = ServoEnvironment(stalled=True), []
        instruction = Move(0, 0, 0, limit=3)
        execute_instruction(
            instruction, env, [env.obs.copy()], path="0", on_event=events.append
        )
        self.assertEqual(len(env.actions), 3)
        self.assertEqual(events[-1]["control"]["steps"], 3)
        self.assertFalse(events[-1]["control"]["converged"])
        self.assertGreater(events[-1]["control"]["position_error"], 0)

    def test_release_opens_before_vertical_retreat(self):
        env = ServoEnvironment()
        env.obs[3:5] = 0.02
        instruction = Release(0, target_z=0.3)
        instruction.eval(env, [])
        self.assertTrue(instruction.last_control_result.converged)
        split = next(i for i, a in enumerate(env.actions) if a[2] != 0)
        self.assertGreater(split, 0)
        self.assertTrue(
            all(a[3] > 0 and np.all(a[:3] == 0) for a in env.actions[:split])
        )
        self.assertTrue(
            all(a[3] == 0 and np.all(a[:2] == 0) for a in env.actions[split:])
        )

    def test_named_conversion_preserves_controls_actions_observations_and_frames(self):
        config = ControlConfig(position_tolerance=0.003, gain=8)
        numeric_env, named_env = ServoEnvironment(), ServoEnvironment()
        for numeric in placement(config):
            named = name_operands(numeric, {0: "base", 1: "other"})
            self.assertEqual(named.control, config)
            numeric_states, named_states = [], []
            numeric_images = numeric.eval(numeric_env, numeric_states, True)
            named_images = named.eval(named_env, named_states, True)
            np.testing.assert_array_equal(numeric_env.actions, named_env.actions)
            np.testing.assert_array_equal(numeric_states, named_states)
            np.testing.assert_array_equal(numeric_images, named_images)
            self.assertEqual(numeric.last_control_result, named.last_control_result)

    def test_rendering_does_not_change_execution(self):
        instructions = placement(ControlConfig())
        envs = [ServoEnvironment(), ServoEnvironment()]
        for env, render in zip(envs, (False, True)):
            for inst in deepcopy(instructions):
                inst.eval(env, [], render)
        np.testing.assert_array_equal(envs[0].actions, envs[1].actions)
        np.testing.assert_array_equal(envs[0].obs, envs[1].obs)

    def test_control_settings_are_validated_and_part_of_fingerprint(self):
        for field in (
            "position_tolerance",
            "gain",
            "gripper_threshold",
            "gripper_tolerance",
        ):
            for invalid in (0, -1, float("nan"), float("inf")):
                with self.subTest(field=field, invalid=invalid), self.assertRaises(
                    ValueError
                ):
                    ControlConfig(**{field: invalid})
        original = Program(1, [Move()])
        changed = deepcopy(original)
        changed.instructions[0].control = replace(
            original.instructions[0].control, position_tolerance=0.004
        )
        self.assertNotEqual(program_fingerprint(original), program_fingerprint(changed))
        self.assertEqual(
            describe_program(changed)[0]["control"]["position_tolerance"], 0.004
        )
        # Runtime diagnostics must not change executable identity.
        before = program_fingerprint(changed)
        changed.instructions[0].eval(ServoEnvironment(), [])
        self.assertEqual(program_fingerprint(changed), before)

    def test_unconverged_control_cannot_be_an_accepted_demo(self):
        from synthesis.cfg.test_collection import example_trace

        trace = example_trace()
        trace.events = (
            {
                "kind": "instruction_end",
                "path": "0",
                "control": {"converged": False, "steps": 50, "phase": "move"},
            },
        )
        self.assertFalse(validate_trace(trace))
        self.assertEqual(trace.metadata["status"], "incomplete")
        self.assertIn("step limit", trace.metadata["reason"])


class SimulatorControlTests(unittest.TestCase):
    def test_stack_example_finishes_in_one_iteration_per_nonbase_block(self):
        from synthesis.cfg.program_source import load_program
        from synthesis.verification_lib.highlevel_verification_lib import (
            HighLevelContext,
        )

        for num_blocks in (3, 4):
            with self.subTest(num_blocks=num_blocks):
                definition = load_program(
                    "synthesis.examples.stack:build_program",
                    HighLevelContext(),
                    num_blocks,
                )
                trace = record_execution(
                    definition,
                    seed=3,
                    num_blocks=num_blocks,
                    max_loop_iterations=num_blocks - 1,
                )
                self.assertTrue(validate_trace(trace), trace.metadata["reason"])
                selected = [
                    e["bindings"]["b_prime"]
                    for e in trace.events
                    if e["kind"] == "loop_head"
                ]
                self.assertEqual(selected, list(range(1, num_blocks)))
                self.assertTrue(
                    all(
                        e["control"]["converged"] and e["control"]["steps"] <= 50
                        for e in trace.events
                        if "control" in e
                    )
                )

    def test_numeric_and_named_programs_have_identical_mujoco_rollouts(self):
        config = ControlConfig(position_tolerance=0.003)
        numeric = placement(config)
        named = [name_operands(i, {0: "base", 1: "other"}) for i in numeric]
        definitions = [
            ProgramDefinition(
                Program(5, rows), {"b0": 0, "base": 0, "other": 1}, "control parity"
            )
            for rows in (numeric, named)
        ]
        traces = [record_execution(d, seed=0, num_blocks=2) for d in definitions]
        self.assertEqual(traces[0].metadata["status"], "completed")
        self.assertEqual(traces[1].metadata["status"], "completed")
        np.testing.assert_array_equal(traces[0].actions[0], traces[1].actions[0])
        np.testing.assert_array_equal(traces[0].states, traces[1].states)
        self.assertEqual(traces[0].events, traces[1].events)
        self.assertTrue(
            all(e["control"]["converged"] for e in traces[0].events if "control" in e)
        )


if __name__ == "__main__":
    unittest.main()
