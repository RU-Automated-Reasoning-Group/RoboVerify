"""Groundtruth collection and faithful replay for nested goal programs."""

import copy
import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np

from synthesis.cfg.collection import record_execution, validate_trace
from synthesis.cfg.demos import DemoSegment
from synthesis.cfg.goal_collection import (
    PROVIDED_PROGRAMS,
    make_goal_env,
    resolve_goal_configuration,
)
from synthesis.cfg.program_source import load_program, program_fingerprint
from synthesis.cfg.recordings import load_traces, loop_store, save_traces
from synthesis.cfg.reset import capture, reset_segment, restore
from synthesis.entry.collect_demos import main
from synthesis.environment.goal_layout import encode_goal_bindings
from synthesis.verification_lib.highlevel_verification_lib import HighLevelContext


class GoalCollectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.traces = {}
        for task, config in [
            ("grid", dict(grid_rows=2, grid_cols=3)),
            ("pyramid", dict(bottom_row_blocks=4)),
        ]:
            count, config = resolve_goal_configuration(task, **config)
            definition = load_program(
                PROVIDED_PROGRAMS[task],
                HighLevelContext(),
                count,
                allow_goal_program=True,
            )
            trace = record_execution(
                definition,
                task=task,
                num_blocks=count,
                seed=1,
                environment_config=config,
            )
            if not validate_trace(trace):
                raise AssertionError(trace.metadata)
            cls.traces[task] = trace

    def test_dimensions_defaults_and_conflicting_counts(self):
        self.assertEqual(
            resolve_goal_configuration("grid"), (6, dict(grid_rows=3, grid_cols=2))
        )
        self.assertEqual(
            resolve_goal_configuration("pyramid", bottom_row_blocks=4),
            (10, dict(bottom_row_blocks=4)),
        )
        for task, kwargs in [
            ("pyramid", dict(bottom_row_blocks=4, num_blocks=4)),
            ("pyramid", dict(grid_rows=2)),
            ("grid", dict(bottom_row_blocks=3)),
            ("grid", dict(grid_cols=0)),
        ]:
            with self.assertRaises(ValueError):
                resolve_goal_configuration(task, **kwargs)

    def test_goal_program_loader_is_collection_opt_in_and_fingerprints_traversal(self):
        source = PROVIDED_PROGRAMS["pyramid"]
        with self.assertRaisesRegex(ValueError, "collection only"):
            load_program(source, HighLevelContext(), 10)
        definition = load_program(
            source, HighLevelContext(), 10, allow_goal_program=True
        )
        self.assertEqual(definition.initial_bindings, {})
        changed = copy.deepcopy(definition.program)
        changed.instructions[1].body[-1].var_name = "r"
        self.assertNotEqual(
            program_fingerprint(definition.program), program_fingerprint(changed)
        )
        self.assertIn("AtGoal", definition.metadata["program"][1]["body"][1]["guard"])

    def test_archives_preserve_graph_cursors_and_nested_invocations(self):
        for task, trace in self.traces.items():
            with self.subTest(task=task), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "demonstrations.npz"
                save_traces(path, [trace])
                loaded = load_traces(path, require_valid=True)[0]
                np.testing.assert_array_equal(loaded.states, trace.states)
                self.assertEqual(loaded.events, trace.events)
                for original, saved in zip(trace.snapshots, loaded.snapshots):
                    self.assertEqual(saved.goal_layout, original.goal_layout)
                    self.assertEqual(saved.goal_bindings, original.goal_bindings)
                inner_exits = [
                    e
                    for e in loaded.events
                    if e["kind"] == "loop_exit" and e["path"] == "1.1"
                ]
                self.assertEqual(
                    [e["invocation"] for e in inner_exits],
                    list(range(len(inner_exits))),
                )
                self.assertEqual(
                    [e["iteration"] for e in inner_exits],
                    list(loaded.snapshots[0].goal_layout.row_lengths),
                )
                self.assertEqual(
                    loaded.metadata["initialization"]["settling_steps"], 50
                )
                self.assertEqual(loaded.actions[1][0], 0)
                with self.assertRaisesRegex(ValueError, "invariant inference"):
                    loop_store([loaded])

    def test_snapshot_and_action_replay_preserve_goal_cursors_and_next_action(self):
        for task, trace in self.traces.items():
            with self.subTest(task=task), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "record.npz"
                save_traces(path, [trace])
                loaded = load_traces(path)[0]
                env = make_goal_env(task, trace.metadata["environment"])
                try:
                    np.testing.assert_array_equal(
                        restore(env, loaded.snapshots[0]), loaded.states[0]
                    )
                    # A loop head follows logical cursor/binding changes with no action.
                    head = next(
                        e
                        for e in loaded.events
                        if e["kind"] == "loop_head"
                        and e["path"] == "1.1"
                        and e["invocation"] == 1
                    )
                    held = next(
                        e
                        for e in loaded.events
                        if e["kind"] == "instruction_end" and e["path"] == "1.1.1"
                    )
                    for event in (head, held):
                        t = event["index"]
                        segment = DemoSegment(
                            0,
                            t,
                            len(loaded.states) - 1,
                            loaded,
                            event["bindings"],
                            goal_bindings=event["goal_bindings"],
                        )
                        next_states = []
                        for mode in ("reset", "replay"):
                            restored = reset_segment(env, segment, mode=mode)
                            np.testing.assert_allclose(
                                restored, loaded.states[t], atol=1e-8, rtol=0
                            )
                            self.assertEqual(
                                encode_goal_bindings(env.env.symbolic_name_to_goal),
                                event["goal_bindings"],
                            )
                            np.testing.assert_array_equal(
                                capture(env).arrays["mocap_pos"],
                                loaded.snapshots[t].arrays["mocap_pos"],
                            )
                            action_index = loaded.actions[1][t]
                            next_states.append(
                                env.step(np.array(loaded.actions[0][action_index]))[0]
                            )
                        np.testing.assert_allclose(
                            next_states[0], next_states[1], atol=1e-8, rtol=0
                        )
                finally:
                    env.close()

    def test_wrong_grid_topology_and_corrupt_bindings_are_rejected(self):
        trace = self.traces["grid"]
        env = make_goal_env("grid", dict(grid_rows=3, grid_cols=2))
        try:
            with self.assertRaisesRegex(ValueError, "row layout"):
                restore(env, trace.snapshots[0])
        finally:
            env.close()
        bad = copy.deepcopy(trace)
        bad.snapshots[-1].goal_bindings["r"] = 999
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(
            ValueError, "goal reference"
        ):
            save_traces(Path(directory) / "bad.npz", [bad])

    def test_transient_coverage_and_failed_primitive_are_not_valid_demos(self):
        trace = copy.deepcopy(self.traces["grid"])
        trace.metadata["status"] = "completed"
        trace.states = (*trace.states, trace.states[0])
        trace.snapshots = (*trace.snapshots, trace.snapshots[0])
        trace.actions = (trace.actions[0], (*trace.actions[1], trace.actions[1][-1]))
        self.assertFalse(validate_trace(trace))
        self.assertEqual(trace.metadata["validation"]["post_reached"], 1)
        self.assertEqual(trace.metadata["validation"]["post_at_end"], 0)
        trace = copy.deepcopy(self.traces["pyramid"])
        trace.metadata["status"] = "completed"
        next(e for e in trace.events if "control" in e)["control"]["converged"] = False
        self.assertFalse(validate_trace(trace))
        self.assertEqual(trace.metadata["status"], "incomplete")

    def test_release_height_reference_need_not_name_the_occupied_goal(self):
        trace = copy.deepcopy(self.traces["grid"])
        trace.metadata["status"] = "completed"
        release = trace.metadata["program"][1]["body"][1]["body"][5]
        self.assertEqual(release["instruction"], "ReleaseByName")
        # All Grid goals have the same Z, so h0 gives the same retreat action.
        release["operands"][0]["val"] = "h0"
        self.assertTrue(validate_trace(trace))

    def test_collector_defaults_to_groundtruth_and_keeps_budget_failure_diagnostic(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            output = Path(directory) / "failed"
            code = main(
                [
                    "--task",
                    "grid",
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
            failed = load_traces(output / "diagnostics/seed_0000.npz")[0]
            self.assertEqual(failed.metadata["source"], PROVIDED_PROGRAMS["grid"])
            self.assertEqual(failed.metadata["status"], "incomplete")
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(["--task", "pyramid", "--bottom-row-blocks", "4", "--num-blocks", "4"])


if __name__ == "__main__":
    unittest.main()
