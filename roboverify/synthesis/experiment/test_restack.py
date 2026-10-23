"""Restack collection: lateral entry, both phases, and faithful archive replay."""

import copy
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from synthesis.cfg.collection import record_execution, validate_trace
from synthesis.cfg.demos import DemoSegment
from synthesis.cfg.program_source import load_program
from synthesis.cfg.recordings import load_traces, save_traces
from synthesis.cfg.reset import inner_env, reset_segment
from synthesis.cfg.tasks import task_spec
from synthesis.entry.collect_demos import main
from synthesis.environment.restack_reset import (
    resolve_tower_blocks,
    sample_restack_layout,
)
from synthesis.mcmc.synthesis import make_roboverify_restack_env
from synthesis.predicates.scene import evaluate, scene_from_obs
from synthesis.util.on import get_block_pos, on_star_implementation
from synthesis.verification_lib.highlevel_verification_lib import HighLevelContext

SOURCE = "synthesis.examples.restack:build_program"


def positions(observation, count):
    return np.array([get_block_pos(observation, i) for i in range(count)])


class RestackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.traces = {}
        for count, height, seed in ((2, 2, 0), (4, 3, 0), (5, 3, 1), (5, 4, 2)):
            definition = load_program(SOURCE, HighLevelContext(use_tbl=True), count)
            trace = record_execution(
                definition,
                task="restack",
                num_blocks=count,
                seed=seed,
                environment_config=dict(initial_tower_blocks=height),
            )
            if not validate_trace(trace):
                raise AssertionError(trace.metadata)
            cls.traces[count, height] = trace

    def test_lateral_layouts_are_separated_reproducible_and_bounded(self):
        rng = np.random.get_state()
        self.addCleanup(np.random.set_state, rng)
        base, gripper = np.array([0.7, 0.75]), np.array([1.3, 0.75])
        seen = []
        for seed in range(30):
            np.random.seed(seed)
            xy, levels = sample_restack_layout(5, 3, 0, base, gripper)
            seen.append(xy[0].copy())
            self.assertEqual(levels[0], 1)
            self.assertEqual(sorted(levels.tolist()), [0, 0, 0, 1, 2])
            self.assertTrue(
                np.all((xy[:, 0] - base[0] >= 0.40) & (xy[:, 0] - base[0] <= 0.70))
            )
            self.assertTrue(
                np.all((xy[:, 1] - base[1] >= 0.15) & (xy[:, 1] - base[1] <= 0.25))
            )
            self.assertTrue(0.54 <= xy[0, 0] - base[0] <= 0.70)
            columns = np.unique(xy, axis=0)
            self.assertEqual(len(columns), 3)
            for i, p in enumerate(columns):
                self.assertGreaterEqual(np.linalg.norm(p - gripper), 0.10)
                for q in columns[i + 1 :]:
                    self.assertGreaterEqual(np.max(np.abs(p - q)), 0.105 - 1e-12)
            np.random.seed(seed)
            again = sample_restack_layout(5, 3, 0, base, gripper)
            np.testing.assert_array_equal(xy, again[0])
            np.testing.assert_array_equal(levels, again[1])
        self.assertGreater(np.ptp(np.array(seen)[:, 0]), 0.05)
        self.assertEqual(resolve_tower_blocks(4), 3)
        for count, height in ((1, None), (4, 1), (4, 5), (4, 2.5)):
            with self.assertRaises(ValueError):
                resolve_tower_blocks(count, height)
        with self.assertRaisesRegex(ValueError, "cannot fit"):
            sample_restack_layout(6, 3, 0, base, gripper)

    def test_two_phases_free_b0_once_then_stack_every_other_block(self):
        for (count, height), trace in self.traces.items():
            with self.subTest(count=count, height=height):
                initial = positions(trace.states[0], count)
                expected = sorted(
                    [
                        i
                        for i in range(count)
                        if on_star_implementation(initial[i], initial[0])
                    ],
                    key=lambda i: -initial[i, 2],
                )
                heads = [e for e in trace.events if e["kind"] == "loop_head"]
                self.assertEqual(
                    [e["bindings"]["b_prime"] for e in heads if e["path"] == "1"],
                    expected,
                )
                stacked = [e["bindings"]["b_prime"] for e in heads if e["path"] == "3"]
                self.assertCountEqual(stacked, range(1, count))
                self.assertEqual(len(stacked), count - 1)
                exits = [e for e in trace.events if e["kind"] == "loop_exit"]
                self.assertEqual([e["path"] for e in exits], ["1", "3"])
                self.assertEqual(exits[0]["bindings"]["p"], 0)
                self.assertEqual(trace.metadata["initialization"]["settling_steps"], 50)
                controls = [e["control"] for e in trace.events if "control" in e]
                self.assertEqual(len(controls), 6 * (len(expected) + count - 1))
                self.assertTrue(all(c["converged"] for c in controls))
                env = make_roboverify_restack_env(count, initial_tower_blocks=height)
                try:
                    final = positions(trace.states[-1], count)
                    self.assertTrue(inner_env(env)._roboverify_task_complete(final))
                    self.assertTrue(
                        np.all(final[:, 1] < inner_env(env).robot_base_xy[1])
                    )
                    self.assertEqual(int(np.argmin(final[:, 2])), 0)
                finally:
                    env.close()

    def test_precondition_allows_towers_and_post_requires_the_designated_base(self):
        trace = self.traces[4, 3]
        pre, post = task_spec("restack")
        bindings = trace.metadata["initial_bindings"]
        scene = scene_from_obs(trace.states[0], 4, bindings, include_table=True)
        self.assertTrue(evaluate(pre, scene))
        self.assertFalse(evaluate(post, scene))
        final = scene_from_obs(trace.states[-1], 4, bindings, include_table=True)
        self.assertTrue(evaluate(post, final))
        final.bindings["b0"] = int(np.argmax(positions(trace.states[-1], 4)[:, 2]))
        self.assertFalse(evaluate(post, final))
        # An unrelated block too close to the source column violates separation.
        loose = next(
            i
            for i in range(1, 4)
            if abs(scene.positions[i][0] - scene.positions[0][0]) > 0.1
        )
        scene.positions[loose] = (
            scene.positions[0][0] + 0.075,
            scene.positions[0][1],
            0.425,
        )
        self.assertFalse(evaluate(pre, scene))

    def test_archive_restores_both_loop_phases_and_actions(self):
        trace = self.traces[4, 3]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "demonstrations.npz"
            save_traces(path, [trace])
            loaded = load_traces(path, require_valid=True)[0]
            self.assertEqual(loaded.events, trace.events)
            env = make_roboverify_restack_env(4)
            try:
                for phase in ("1", "3"):
                    event = next(
                        e
                        for e in loaded.events
                        if e["kind"] == "loop_head" and e["path"] == phase
                    )
                    t = event["index"]
                    segment = DemoSegment(
                        0, t, len(trace.states) - 1, loaded, event["bindings"]
                    )
                    next_states = []
                    for mode in ("reset", "replay"):
                        obs = reset_segment(env, segment, mode=mode)
                        np.testing.assert_allclose(
                            obs, loaded.states[t], atol=1e-8, rtol=0
                        )
                        next_states.append(
                            env.step(np.array(loaded.actions[0][loaded.actions[1][t]]))[
                                0
                            ]
                        )
                    np.testing.assert_allclose(*next_states, atol=1e-8, rtol=0)
                definition = load_program(SOURCE, HighLevelContext(use_tbl=True), 4)
                replay = record_execution(
                    definition,
                    task="restack",
                    num_blocks=4,
                    seed=999,
                    initial_snapshot=loaded.snapshots[0],
                )
                self.assertTrue(validate_trace(replay), replay.metadata)
                self.assertEqual(replay.metadata["initialization"]["settling_steps"], 0)
                np.testing.assert_array_equal(replay.states, trace.states)
            finally:
                env.close()

    def test_failed_controller_or_nonfinal_tower_is_rejected(self):
        trace = copy.deepcopy(self.traces[4, 3])
        trace.metadata["status"] = "completed"
        trace.states = (*trace.states, trace.states[0])
        self.assertFalse(validate_trace(trace))
        trace = copy.deepcopy(self.traces[4, 3])
        trace.metadata["status"] = "completed"
        next(e for e in trace.events if "control" in e)["control"]["converged"] = False
        self.assertFalse(validate_trace(trace))
        self.assertEqual(trace.metadata["status"], "incomplete")

    def test_cli_uses_provided_program_and_retains_budget_failure(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            output = Path(directory) / "failed"
            code = main(
                [
                    "--task",
                    "restack",
                    "--seeds",
                    "0",
                    "--max-loop-iterations",
                    "1",
                    "--output-dir",
                    str(output),
                ]
            )
            self.assertEqual(code, 2)
            self.assertFalse((output / "demonstrations.npz").exists())
            trace = load_traces(output / "diagnostics" / "seed_0000.npz")[0]
            self.assertEqual(trace.metadata["source"], SOURCE)
            self.assertEqual(
                trace.metadata["environment"], dict(initial_tower_blocks=3)
            )
            self.assertEqual(trace.metadata["status"], "incomplete")


if __name__ == "__main__":
    unittest.main()
