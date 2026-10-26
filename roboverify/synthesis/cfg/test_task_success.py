"""Shared task-success checker: settling, criteria and recorded-trace replay."""

import unittest
from types import SimpleNamespace

import numpy as np

from synthesis.cfg.task_success import (
    CRITERIA,
    SETTLING_STEPS,
    check_task_success,
    criterion,
    settle,
)


class ScriptedEnv:
    """Observation-only stand-in: each step applies ``motion`` to the blocks."""

    def __init__(self, blocks, *, motion=None, goals=None):
        self.num_blocks = len(blocks)
        self.state = np.zeros(10 + 12 * self.num_blocks)
        for i, xyz in enumerate(blocks):
            self.state[10 + 12 * i : 13 + 12 * i] = xyz
        self.state[:3] = (1.3, 0.75, 0.6)
        self.motion = motion or (lambda state, step: None)
        self.actions = []
        if goals is not None:
            self.goal_layout = SimpleNamespace(positions=np.asarray(goals, float))

    unwrapped = property(lambda self: self)

    def flatten_observation(self, obs):
        return obs

    def _get_obs(self):
        return self.state.copy()

    def _is_success(self, obs):
        return False

    def step(self, action):
        self.actions.append(np.asarray(action).copy())
        self.motion(self.state, len(self.actions))
        return self.state.copy(), 0.0, False, {}

    def block(self, i):
        return self.state[10 + 12 * i : 13 + 12 * i]


def falls_off(state, step):
    """Block 1 slides off block 0 during the first settling steps."""
    if step <= 5:
        state[10 + 12 : 13 + 12] += (0.02, 0.0, -0.01)


def drops_onto(state, step):
    """Block 1 drops 25 mm onto block 0 and stays aligned."""
    if step <= 5:
        state[10 + 12 + 2] -= 0.005


class ScriptedCheckerTests(unittest.TestCase):
    def check(self, env, task="stack", steps=SETTLING_STEPS, **options):
        entry = env._get_obs()
        if task == "stack":
            # Equal-height Scattered entry scene for Stack's precondition.
            entry[10:13], entry[22:25] = (1.30, 0.75, 0.425), (1.45, 0.75, 0.425)
        return check_task_success(
            env,
            task,
            entry_observation=entry,
            bindings={"b0": 0},
            completed=options.get("completed", True),
            unconverged=options.get("unconverged", 0),
            post_settling_steps=steps,
        )

    def test_criterion_names(self):
        self.assertEqual(criterion(0), CRITERIA[0])
        self.assertEqual(criterion(0), "task-spec-v1")
        self.assertEqual(criterion(50), "task-spec-v2")
        with self.assertRaises(ValueError):
            self.check(ScriptedEnv([(1.3, 0.75, 0.425)] * 2), steps=-1)

    def test_falling_block_passes_at_end_but_fails_after_settling(self):
        env = ScriptedEnv([(1.30, 0.75, 0.425), (1.31, 0.75, 0.50)], motion=falls_off)
        result = self.check(env)
        self.assertTrue(result.pre)
        self.assertTrue(result.post_at_end)
        self.assertFalse(result.post_settled)
        self.assertFalse(result.success)
        self.assertEqual(result.criterion, "task-spec-v2")
        self.assertEqual(len(env.actions), SETTLING_STEPS)
        # The same final state passes the original last-state criterion.
        env = ScriptedEnv([(1.30, 0.75, 0.425), (1.31, 0.75, 0.50)], motion=falls_off)
        original = self.check(env, steps=0)
        self.assertTrue(original.success)
        self.assertIsNone(original.post_settled)
        self.assertEqual(env.actions, [])

    def test_transient_tower_that_settles_correctly_passes(self):
        env = ScriptedEnv([(1.30, 0.75, 0.425), (1.30, 0.75, 0.50)], motion=drops_onto)
        result = self.check(env)
        self.assertTrue(result.post_at_end and result.post_settled and result.success)
        np.testing.assert_allclose(result.positions[1], (1.30, 0.75, 0.475))
        self.assertEqual(result.issues, [])

    def test_completion_and_convergence_gate_success(self):
        blocks = [(1.30, 0.75, 0.425), (1.30, 0.75, 0.475)]
        self.assertFalse(self.check(ScriptedEnv(blocks), completed=False).success)
        failed = self.check(ScriptedEnv(blocks), unconverged=1)
        self.assertFalse(failed.success)
        self.assertTrue(failed.post_settled)

    def test_goal_tasks_use_settled_occupancy(self):
        goals = [(1.2, 0.6, 0.425), (1.3, 0.6, 0.425)]

        def slide(state, step):
            if step == 1:
                state[22:25] = (1.40, 0.6, 0.425)  # leaves its goal

        entry = [(1.2, 0.9, 0.425), (1.3, 0.9, 0.425)]
        env = ScriptedEnv(goals, motion=slide, goals=goals)
        start = ScriptedEnv(entry)._get_obs()
        result = check_task_success(
            env,
            "pyramid",
            entry_observation=start,
            bindings={},
            completed=True,
            unconverged=0,
            post_settling_steps=10,
        )
        self.assertTrue(result.pre and result.post_at_end)
        self.assertFalse(result.post_settled)
        self.assertEqual(result.goal_occupied, [True, False])
        self.assertFalse(result.success)

    def test_settle_holds_the_initial_gripper_position(self):
        env = ScriptedEnv([(1.3, 0.75, 0.425)])
        settle(env, 3)
        self.assertEqual(len(env.actions), 3)
        for action in env.actions:
            np.testing.assert_array_equal(action, np.zeros(4))


try:
    import mujoco_py  # noqa: F401

    SIMULATOR = True
except Exception:  # pragma: no cover - environment dependent
    SIMULATOR = False


@unittest.skipUnless(SIMULATOR, "MuJoCo is unavailable")
class SimulatorCheckerTests(unittest.TestCase):
    def make_stack(self, seed=0):
        from synthesis.mcmc.synthesis import make_roboverify_env

        np.random.seed(seed)
        env = make_roboverify_env("stack", num_blocks=2)
        return env, settle(env, SETTLING_STEPS, env.reset()[0])

    def place(self, env, block, xyz, velocity=(0, 0, 0)):
        from synthesis.cfg.reset import inner_env

        inner = inner_env(env)
        name = f"{inner.object_names[block]}:joint"
        qpos = inner.sim.data.get_joint_qpos(name).copy()
        qpos[:3] = xyz
        qpos[3:] = (1, 0, 0, 0)
        inner.sim.data.set_joint_qpos(name, qpos)
        inner.sim.data.set_joint_qvel(name, np.r_[velocity, 0, 0, 0])
        inner.sim.forward()
        return inner.flatten_observation(inner._get_obs())

    def test_settle_matches_the_previous_inlined_loops(self):
        from synthesis.api.control import get_move_action
        from synthesis.cfg.reset import inner_env

        for task in ("stack", "unstack"):
            from synthesis.mcmc.synthesis import make_roboverify_env

            envs, starts = [], []
            for _ in range(3):
                np.random.seed(7)
                envs.append(make_roboverify_env(task, num_blocks=3))
                starts.append(envs[-1].reset()[0])
            try:
                helper = settle(envs[0], SETTLING_STEPS, starts[0])
                # Collection's loop: re-read the observation after each step.
                first, inner = starts[1], inner_env(envs[1])
                target = first[:3].copy()
                for _ in range(SETTLING_STEPS):
                    envs[1].step(get_move_action(first, target, close_gripper=False))
                    first = inner.flatten_observation(inner._get_obs())
                # CaP's loop: use the observation returned by step.
                obs = starts[2]
                target = obs[:3].copy()
                for _ in range(SETTLING_STEPS):
                    obs = envs[2].step(get_move_action(obs, target))[0]
                for other in (first, obs):
                    np.testing.assert_array_equal(helper, other)
                states = [inner_env(e).get_GT_state() for e in envs]
                np.testing.assert_array_equal(states[0], states[1])
                np.testing.assert_array_equal(states[0], states[2])
            finally:
                for env in envs:
                    env.close()

    def test_moving_block_satisfies_post_at_end_but_not_after_settling(self):
        from synthesis.cfg.demos import DemoTrace
        from synthesis.cfg.reset import capture, inner_env
        from synthesis.cfg.task_success import check_recorded_task_success

        env, entry = self.make_stack()
        try:
            base = entry[10:13].copy()
            # Block 1 is 2 cm off-centre above b0, moving away at 1 m/s: ON*(1, b0)
            # holds now, but the block lands on the table beside b0.
            final = self.place(
                env, 1, base + (0.02, 0.0, 0.08), velocity=(1.0, 0.0, 0.0)
            )
            snapshot = capture(env)
            trace = DemoTrace(
                (entry, final),
                (snapshot, snapshot),
                ((), (0, 0)),
                0,
                "stack",
                2,
                (),
                dict(
                    status="valid",
                    initial_bindings={"b0": 0},
                    higher_tolerance=0.001,
                ),
            )
            live = check_task_success(
                env,
                "stack",
                entry_observation=entry,
                bindings={"b0": 0},
                completed=True,
                unconverged=0,
                post_settling_steps=SETTLING_STEPS,
            )
        finally:
            env.close()
        self.assertTrue(live.pre)
        self.assertTrue(live.post_at_end)
        self.assertFalse(live.post_settled)
        self.assertFalse(live.success)
        self.assertGreater(live.positions[1][0] - base[0], 0.05)
        self.assertLess(abs(live.positions[1][2] - base[2]), 0.01)
        # Restoring the recorded final snapshot gives the same settled result.
        recorded = check_recorded_task_success(trace, SETTLING_STEPS)
        self.assertEqual(recorded.as_dict(), live.as_dict())
        original = check_recorded_task_success(trace, 0)
        self.assertTrue(original.success)

    def test_dropped_block_that_settles_on_b0_passes(self):
        env, entry = self.make_stack()
        try:
            self.place(env, 1, entry[10:13] + (0.005, 0.0, 0.08))
            result = check_task_success(
                env,
                "stack",
                entry_observation=entry,
                bindings={"b0": 0},
                completed=True,
                unconverged=0,
                post_settling_steps=SETTLING_STEPS,
            )
        finally:
            env.close()
        self.assertTrue(result.post_at_end and result.post_settled and result.success)
        self.assertAlmostEqual(result.positions[1][2] - entry[12], 0.05, delta=0.005)

    def test_collection_settles_unstack_and_keeps_recordings_unchanged(self):
        from synthesis.cfg.collection import record_execution, validate_trace
        from synthesis.cfg.program_source import load_program
        from synthesis.verification_lib.highlevel_verification_lib import (
            HighLevelContext,
        )

        context = HighLevelContext(mode="declare", use_tbl=True)
        traces = {}
        for steps in (0, SETTLING_STEPS):
            definition = load_program(
                "synthesis.examples.unstack:build_program",
                context,
                2,
                allow_observation_references=True,
            )
            traces[steps] = record_execution(
                definition,
                seed=3,
                num_blocks=2,
                task="unstack",
                post_settling_steps=steps,
            )
        original, settled = traces[0], traces[SETTLING_STEPS]
        for trace in (original, settled):
            self.assertEqual(
                trace.metadata["initialization"],
                {"source": "reset", "settling_steps": SETTLING_STEPS},
            )
        # Final settling is unrecorded: states, actions and indices are identical.
        self.assertEqual(len(original.states), len(settled.states))
        np.testing.assert_array_equal(
            np.asarray(original.states), np.asarray(settled.states)
        )
        self.assertEqual(original.actions[1], settled.actions[1])
        self.assertNotIn("validation", original.metadata)
        stored = settled.metadata["validation"]["settled"]
        self.assertEqual(stored["criterion"], "task-spec-v2")
        self.assertTrue(stored["success"])
        self.assertTrue(validate_trace(original))
        self.assertTrue(validate_trace(settled))
        self.assertNotIn("criterion", original.metadata["validation"])
        self.assertEqual(settled.metadata["validation"]["criterion"], "task-spec-v2")
        self.assertEqual(settled.metadata["validation"]["settled"], stored)


if __name__ == "__main__":
    unittest.main()
