"""Build3D groundtruth collection, three nested loops, and exact archive replay."""

import copy
import io
import tempfile
import unittest
from contextlib import redirect_stdout
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
from synthesis.cfg.reset import reset_segment, restore
from synthesis.cfg.tasks import task_identity
from synthesis.entry.audit_demos import audit_archive
from synthesis.entry.collect_demos import main
from synthesis.verification_lib.highlevel_verification_lib import HighLevelContext


class Build3DCollectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.traces = []
        for rows, cols, layers in [(2, 2, 2), (2, 3, 2), (2, 2, 3)]:
            count, config = resolve_goal_configuration(
                "build3d", build_rows=rows, build_cols=cols, build_layers=layers
            )
            definition = load_program(
                PROVIDED_PROGRAMS["build3d"],
                HighLevelContext(),
                count,
                allow_goal_program=True,
            )
            trace = record_execution(
                definition,
                task="build3d",
                num_blocks=count,
                seed=0,
                environment_config=config,
            )
            if not validate_trace(trace):
                raise AssertionError(trace.metadata)
            trace.metadata["task_spec"] = task_identity("build3d")
            cls.traces.append(trace)

    def test_program_places_every_block_in_layer_row_column_order(self):
        for trace in self.traces:
            config = trace.metadata["environment"]
            with self.subTest(config=config):
                events = trace.events
                placements = [
                    e
                    for e in events
                    if e["kind"] == "instruction_end" and e["path"] == "1.1.1.5"
                ]
                self.assertEqual(
                    [e["goal_bindings"]["r"] for e in placements],
                    list(range(trace.num_blocks)),
                )
                self.assertEqual(
                    len({e["bindings"]["b_prime"] for e in placements}),
                    trace.num_blocks,
                )
                controls = [e["control"] for e in events if "control" in e]
                self.assertEqual(len(controls), 6 * trace.num_blocks)
                self.assertTrue(all(c["converged"] for c in controls))
                exits = [e for e in events if e["kind"] == "loop_exit"]
                for path, iterations, invocations in [
                    ("1", config["build_layers"], 1),
                    ("1.1", config["build_rows"], config["build_layers"]),
                    (
                        "1.1.1",
                        config["build_cols"],
                        config["build_rows"] * config["build_layers"],
                    ),
                ]:
                    rows = [e for e in exits if e["path"] == path]
                    self.assertEqual(
                        [e["iteration"] for e in rows], [iterations] * invocations
                    )
                self.assertEqual(trace.metadata["initialization"]["settling_steps"], 50)

    def test_archive_preserves_layer_heads_and_replays_full_physics(self):
        trace = self.traces[0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "demonstrations.npz"
            save_traces(path, [trace])
            saved = load_traces(path, require_valid=True)[0]
            self.assertEqual(saved.events, trace.events)
            self.assertEqual(
                saved.snapshots[-1].goal_layout, trace.snapshots[-1].goal_layout
            )
            # Cursor updates after the last physical action live in events, not
            # an invented simulator step/snapshot.
            self.assertEqual(saved.snapshots[-1].goal_bindings["u"], 4)
            final_exit = next(
                e
                for e in reversed(saved.events)
                if e["kind"] == "loop_exit" and e["path"] == "1"
            )
            self.assertIsNone(final_exit["goal_bindings"]["u"])
            env = make_goal_env("build3d", trace.metadata["environment"])
            try:
                event = next(
                    e
                    for e in saved.events
                    if e["kind"] == "loop_head"
                    and e["path"] == "1.1"
                    and e["invocation"] == 1
                )
                index = event["index"]
                segment = DemoSegment(
                    0,
                    index,
                    len(saved.states) - 1,
                    saved,
                    event["bindings"],
                    goal_bindings=event["goal_bindings"],
                )
                for mode in ("reset", "replay"):
                    obs = reset_segment(env, segment, mode=mode)
                    np.testing.assert_allclose(
                        obs, saved.states[index], atol=1e-8, rtol=0
                    )
                    self.assertEqual(env.env.symbolic_name_to_goal["u"].index, 4)
            finally:
                env.close()
            report = audit_archive(path, seeds=[0], settle_steps=50)
            self.assertTrue(report["valid"], report)
            with self.assertRaisesRegex(ValueError, "invariant inference"):
                loop_store([saved])

    def test_layer_grouping_mismatch_is_rejected_even_when_flat_rows_match(self):
        trace = self.traces[0]
        env = make_goal_env("build3d", dict(build_rows=1, build_cols=2, build_layers=4))
        try:
            with self.assertRaisesRegex(ValueError, "row layout"):
                restore(env, trace.snapshots[0])
        finally:
            env.close()
        bad = copy.deepcopy(trace)
        bad.metadata["status"] = "completed"
        bad.metadata["environment"].update(build_rows=1, build_layers=4)
        self.assertFalse(validate_trace(bad))

    def test_layer_traversal_is_fingerprinted_and_loader_requires_opt_in(self):
        source = PROVIDED_PROGRAMS["build3d"]
        with self.assertRaisesRegex(ValueError, "collection only"):
            load_program(source, HighLevelContext(), 8)
        definition = load_program(
            source, HighLevelContext(), 8, allow_goal_program=True
        )
        changed = copy.deepcopy(definition.program)
        changed.instructions[1].body[-1].var_name = "d"
        self.assertNotEqual(
            program_fingerprint(definition.program), program_fingerprint(changed)
        )

    def test_shared_cli_defaults_to_build3d_groundtruth(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            output = Path(directory) / "build3d"
            code = main(
                [
                    "--task",
                    "build3d",
                    "--build-rows",
                    "1",
                    "--build-cols",
                    "2",
                    "--build-layers",
                    "2",
                    "--seeds",
                    "1",
                    "--output-dir",
                    str(output),
                ]
            )
            self.assertEqual(code, 0)
            trace = load_traces(output / "demonstrations.npz", require_valid=True)[0]
            self.assertEqual(trace.num_blocks, 4)
            self.assertEqual(trace.metadata["source"], PROVIDED_PROGRAMS["build3d"])
            self.assertEqual(trace.snapshots[0].goal_layout.layer_row_counts, (1, 1))


if __name__ == "__main__":
    unittest.main()
