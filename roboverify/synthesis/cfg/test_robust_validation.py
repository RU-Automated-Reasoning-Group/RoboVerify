"""Full-task validation must observe failures between successful endpoints."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import z3

from synthesis.api.instructions import Assign, Pick, Release, Skip, While
from synthesis.api.program import Program
from synthesis.cfg.demos import DemoTrace
from synthesis.cfg.program_source import ProgramDefinition, program_fingerprint
from synthesis.cfg.recordings import load_traces
from synthesis.cfg.reset import capture
from synthesis.cfg.robust_validation import ValidationBudget, validate_training_program
from synthesis.predicates.term import atom, ref


class FakeEnv:
    def __init__(self, *, on_step=None, task_success=None):
        self.observation = np.zeros(34)
        self.observation[:3] = (0, 0, 0.7)
        self.observation[3:5] = 0.04
        self.observation[10:13] = (0, 0, 0.425)
        self.observation[22:25] = (0, 0, 0.475)
        self.symbolic_name_to_box_id = {"b0": 0, "teacher_only": 1}
        self.sim = SimpleNamespace(data=SimpleNamespace(), forward=lambda: None)
        self.action_space = SimpleNamespace(shape=(4,))
        self.num_blocks = 2
        self.unwrapped = self.env = self
        self.on_step, self.task_success = on_step, task_success
        self.steps, self.actions, self.closed, self.held = 0, [], False, None

    def get_GT_state(self):
        return self.observation.copy()

    def set_GT_state(self, state):
        self.observation = state.copy()

    def flatten_observation(self, value):
        return value.copy()

    def _get_obs(self):
        return self.observation.copy()

    def _is_success(self, observation):
        return True if self.task_success is None else self.task_success(self)

    def step(self, action):
        self.steps += 1
        self.actions.append(np.array(action, copy=True))
        if self.on_step is not None:
            self.on_step(self)
        return self.observation.copy(), 0, False, {}

    def close(self):
        self.closed = True


def source_trace():
    env = FakeEnv()
    return DemoTrace(
        (env.observation.copy(),),
        (capture(env),),
        ((), (0,)),
        num_blocks=2,
        task="stack",
        events=({"kind": "teacher_event", "index": 0},),
    )


def definition(*instructions):
    return ProgramDefinition(
        Program(len(instructions), list(instructions)), {"b0": 0}, "test"
    )


def contact(env, observation, count):
    return {
        "known": True,
        "holding": env.held,
        "supports": [(1, 0)],
        "aperture": float(sum(observation[3:5])),
        "finger_objects": [],
    }


class FullValidationTests(unittest.TestCase):
    def validate(self, env, program=None, **options):
        trace = options.pop("trace", None) or source_trace()
        with patch(
            "synthesis.cfg.endpoint_contract.contact_state", side_effect=contact
        ):
            return validate_training_program(
                program or definition(Skip(0)), [trace], lambda trace: env, **options
            )

    def test_actual_block_on_and_full_put_effects_use_pre_update_bindings(self):
        source = FakeEnv()
        source.observation[22:25] = (0.2, 0, 0.425)
        trace = DemoTrace(
            (source.observation.copy(),),
            (capture(source),),
            ((), (0,)),
            num_blocks=2,
            task="stack",
        )
        candidate = ProgramDefinition(
            Program(3, [Pick(1), Release(0), Assign("old", "new")]),
            {"b0": 0, "old": 0, "new": 1},
            "block-test",
        )

        def pick(instruction, env, trajectory, return_image=False):
            env.held = 1
            trajectory.append(env.step(np.zeros(4))[0])
            instruction.last_control_result = SimpleNamespace(converged=True)
            return []

        def release(instruction, env, trajectory, return_image=False):
            env.held = None
            env.observation[22:25] = (0, 0, 0.475)
            trajectory.append(env.step(np.zeros(4))[0])
            instruction.last_control_result = SimpleNamespace(converged=True)
            return []

        contract = {
            "body": {
                "paths": ("0", "1", "2"),
                "post": atom("ON", ref("new"), ref("old")),
            }
        }
        env = FakeEnv()
        with patch.object(Pick, "eval", pick), patch.object(Release, "eval", release):
            result = self.validate(
                env,
                candidate,
                trace=trace,
                block_contracts=contract,
                budget=ValidationBudget(settling_steps=2),
            )
        self.assertTrue(result["accepted"], result)
        block = result["demos"][0]["block_contracts"][0]
        self.assertTrue(block["ok"])
        self.assertEqual(block["bindings"]["old"], 0)
        self.assertEqual(env.symbolic_name_to_box_id["old"], 1)
        self.assertEqual(len(block["details"]["put_effects"]), 1)

    def test_actual_block_post_failure_cannot_hide_behind_global_task_success(self):
        contract = {
            "identity": {
                "paths": ("0",),
                "post": atom("Higher", ref("b0"), ref("upper")),
            }
        }
        candidate = ProgramDefinition(
            Program(1, [Skip(0)]), {"b0": 0, "upper": 1}, "block-test"
        )
        result = self.validate(
            FakeEnv(),
            candidate,
            block_contracts=contract,
            budget=ValidationBudget(settling_steps=0),
        )
        row = result["demos"][0]
        self.assertFalse(row["accepted"])
        self.assertTrue(row["checkpoints"][0]["logical_post"])
        self.assertTrue(row["checkpoints"][0]["simulator_goal"])
        self.assertIn("block:identity:postcondition_not_at_endpoint", row["reasons"])

    def test_exact_start_candidate_aliases_and_zero_action_settling_are_saved(self):
        env = FakeEnv()
        with tempfile.TemporaryDirectory() as directory:
            result = self.validate(env, artifact_dir=directory)
            self.assertTrue(result["accepted"], result)
            self.assertEqual(result["demos"][0]["settling_completed"], 50)
            self.assertEqual(len(result["demos"][0]["checkpoints"]), 51)
            self.assertEqual(env.symbolic_name_to_box_id, {"b0": 0})
            self.assertTrue(
                all(np.array_equal(action, np.zeros(4)) for action in env.actions)
            )
            archived = load_traces(result["demos"][0]["replay_archive"])[0]
            self.assertEqual(len(archived.actions[0]), 50)
            self.assertEqual(archived.snapshots[0].bindings, {"b0": 0})
            self.assertFalse(
                any(event["kind"] == "teacher_event" for event in archived.events)
            )
            self.assertEqual(
                json.loads((Path(directory) / "summary.json").read_text())["passed"], 1
            )
        self.assertTrue(env.closed)

    def test_transient_logical_failure_during_settling_rejects_later_recovery(self):
        def drift(env):
            env.observation[22] = 0.1 if env.steps == 3 else 0

        result = self.validate(
            FakeEnv(on_step=drift), budget=ValidationBudget(settling_steps=5)
        )
        row = result["demos"][0]
        self.assertFalse(result["accepted"])
        self.assertTrue(row["checkpoints"][0]["logical_post"])
        self.assertFalse(row["checkpoints"][3]["logical_post"])
        self.assertTrue(row["checkpoints"][-1]["logical_post"])
        self.assertIn("logical_postcondition_lost", row["reasons"])

    def test_canonical_simulator_goal_is_independent_of_logical_goal(self):
        result = self.validate(
            FakeEnv(task_success=lambda env: env.steps != 2),
            budget=ValidationBudget(settling_steps=4),
        )
        self.assertFalse(result["accepted"])
        self.assertTrue(
            all(row["logical_post"] for row in result["demos"][0]["checkpoints"])
        )
        self.assertIn("simulator_task_goal_lost", result["demos"][0]["reasons"])

    def test_nested_primitive_callbacks_and_last_transfer_frame_include_settling(self):
        inner = While(
            z3.BoolVal(True),
            [],
            [Pick(1), Release(0, target_z=0.2)],
            None,
            max_iters=None,
        )
        outer = While(z3.BoolVal(True), [], [inner], None, max_iters=None)
        candidate = definition(outer)

        def once(loop, env, trajectory):
            previous = getattr(loop, "test_entered", False)
            loop.test_entered = True
            return not previous

        def pick(instruction, env, trajectory, return_image=False):
            env.held = 1
            trajectory.append(env.step(np.zeros(4))[0])
            instruction.last_control_result = SimpleNamespace(converged=True)
            return []

        def release(instruction, env, trajectory, return_image=False):
            env.held = None
            trajectory.append(env.step(np.zeros(4))[0])
            instruction.last_control_result = SimpleNamespace(converged=True)
            return []

        def drift(env):
            if env.steps >= 3:
                env.observation[10] = 0.006

        with patch.object(While, "_find_and_bind_guard_exists", once), patch.object(
            Pick, "eval", pick
        ), patch.object(Release, "eval", release):
            result = self.validate(
                FakeEnv(on_step=drift),
                candidate,
                budget=ValidationBudget(settling_steps=2),
            )
        row = result["demos"][0]
        self.assertFalse(result["accepted"])
        self.assertTrue(
            all(
                point["logical_post"] and point["simulator_goal"]
                for point in row["checkpoints"]
            )
        )
        self.assertEqual(
            [item["kind"] for item in row["physical_audit"]["instructions"]],
            ["Pick", "Release"],
        )
        self.assertIn("sampled_transfer_frame", row["reasons"])
        self.assertAlmostEqual(row["max_nonmover_drift_m"], 0.006)
        self.assertIsNone(candidate.program.instructions[0].max_iters)

    def test_loop_budget_does_not_turn_nontermination_into_acceptance_or_mutate_program(
        self,
    ):
        loop = While(z3.BoolVal(True), [], [Skip(0)], None, max_iters=None)
        candidate = definition(loop)
        original = program_fingerprint(candidate.program)
        result = self.validate(
            FakeEnv(),
            candidate,
            budget=ValidationBudget(max_loop_iterations=2, settling_steps=0),
        )
        self.assertFalse(result["accepted"])
        self.assertIn("budget_exhausted", result["demos"][0]["reasons"])
        self.assertEqual(program_fingerprint(candidate.program), original)
        self.assertIsNone(loop.max_iters)

    def test_action_budget_retains_rejected_program_and_exact_partial_actions(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.validate(
                FakeEnv(),
                artifact_dir=directory,
                budget=ValidationBudget(max_actions=2, settling_steps=5),
            )
            row = result["demos"][0]
            self.assertFalse(row["accepted"])
            self.assertEqual(row["settling_completed"], 2)
            self.assertEqual(row["actions"], 2)
            self.assertTrue((Path(directory) / "program.json").exists())
            archived = load_traces(row["replay_archive"])[0]
            self.assertEqual(len(archived.actions[0]), 2)
            self.assertEqual(archived.actions[1][-1], 2)

    def test_external_timeout_propagates_and_closes_environment(self):
        def timeout(env):
            raise TimeoutError("enclosing CLI deadline")

        env = FakeEnv(on_step=timeout)
        with self.assertRaisesRegex(TimeoutError, "enclosing CLI"):
            self.validate(env)
        self.assertTrue(env.closed)

    def test_expired_budget_marks_all_examples_unaccepted_without_running(self):
        factory = unittest.mock.Mock()
        result = validate_training_program(
            definition(Skip(0)),
            [source_trace(), source_trace()],
            factory,
            budget=ValidationBudget(seconds=1e-9),
        )
        self.assertEqual(result["passed"], 0)
        self.assertEqual(result["total"], 2)
        factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
