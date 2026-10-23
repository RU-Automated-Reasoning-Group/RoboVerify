"""Lateral Reverse resets and top-first physical reversal of an entire tower."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from synthesis.api.instructions import Assign, While
from synthesis.cfg.collection import record_execution, validate_trace
from synthesis.cfg.demos import DemoSegment
from synthesis.cfg.program_source import load_program
from synthesis.cfg.recordings import load_traces, save_traces
from synthesis.cfg.reset import inner_env, reset_segment
from synthesis.examples.reverse import DESTINATION_OFFSET
from synthesis.mcmc.synthesis import make_roboverify_env, make_roboverify_reverse_env
from synthesis.util.on import BLOCK_LENGTH, get_block_pos
from synthesis.verification_lib.highlevel_verification_lib import HighLevelContext


class ReverseTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(np.random.set_state, np.random.get_state())

    def test_reset_uses_one_side_for_every_block_count_and_factory(self):
        for count in (2, 3, 4):
            expected = {}
            for factory in (
                make_roboverify_reverse_env,
                lambda num_blocks: make_roboverify_env("reverse", num_blocks),
            ):
                env = factory(num_blocks=count)
                try:
                    inner = inner_env(env)
                    for seed in (0, 4, 73):
                        with self.subTest(count=count, seed=seed, factory=factory):
                            np.random.seed(seed)
                            obs, _ = env.reset()
                            xyz = np.array(
                                [get_block_pos(obs, i) for i in range(count)]
                            )
                            offsets = xyz[:, :2] - inner.robot_base_xy
                            self.assertTrue(
                                np.all(
                                    (0.54 <= offsets[:, 0]) & (offsets[:, 0] <= 0.70)
                                )
                            )
                            self.assertTrue(
                                np.all(
                                    (0.15 <= offsets[:, 1]) & (offsets[:, 1] <= 0.25)
                                )
                            )
                            np.testing.assert_array_equal(
                                xyz[:, :2], np.tile(xyz[0, :2], (count, 1))
                            )
                            np.testing.assert_array_equal(
                                xyz[:, 2],
                                (
                                    inner.height_offset
                                    + BLOCK_LENGTH * np.arange(count)
                                ).astype(xyz.dtype),
                            )
                            self.assertEqual(inner.symbolic_name_to_box_id, {"b0": 0})
                            self.assertFalse(inner._roboverify_task_complete(xyz))
                            reversed_xyz = xyz.copy()
                            reversed_xyz[:, 2] = xyz[::-1, 2]
                            self.assertTrue(
                                inner._roboverify_task_complete(reversed_xyz)
                            )
                            if seed in expected:
                                np.testing.assert_array_equal(obs, expected[seed])
                            else:
                                expected[seed] = obs.copy()
                    sampled = np.array(
                        [get_block_pos(o, 0)[:2] for o in expected.values()]
                    )
                    sampled -= inner.robot_base_xy
                    self.assertGreater(sampled[:, 0].max(), 0.65)
                    self.assertGreater(sampled[:, 1].max(), 0.20)
                finally:
                    env.close()

    def test_groundtruth_moves_each_block_top_first_to_the_opposite_side(self):
        context = HighLevelContext(use_tbl=True)
        for count, base_id in ((2, 0), (3, 0), (4, 0), (4, 2)):
            with self.subTest(count=count, base_id=base_id):
                definition = load_program(
                    "synthesis.examples.reverse:build_program", context, count
                )
                definition.initial_bindings["b0"] = base_id
                self.assertEqual(definition.initial_bindings["tbl"], "tbl")
                self.assertEqual(len(definition.program.instructions), 2)
                initialization, loop = definition.program.instructions
                self.assertIsInstance(initialization, Assign)
                self.assertEqual(
                    (initialization.left, initialization.right), ("b", "tbl")
                )
                self.assertIsInstance(loop, While)
                geometry = {}

                def factory():
                    env = make_roboverify_reverse_env(num_blocks=count)
                    inner = inner_env(env)
                    inner.roboverify_base_block_id = base_id
                    inner.symbolic_name_to_box_id = {"b0": base_id}
                    geometry["table"] = inner.table_position
                    geometry["base"] = inner.robot_base_xy.copy()
                    return env

                trace = record_execution(
                    definition,
                    seed=0,
                    num_blocks=count,
                    task="reverse",
                    env_factory=factory,
                    max_loop_iterations=count,
                    timeout_seconds=12,
                )
                self.assertTrue(validate_trace(trace), trace.metadata)
                final = np.array(
                    [get_block_pos(trace.states[-1], i) for i in range(count)]
                )
                expected_order = list(
                    reversed([base_id] + [i for i in range(count) if i != base_id])
                )
                self.assertEqual(np.argsort(final[:, 2]).tolist(), expected_order)
                selected = [
                    e["bindings"]["b_prime"]
                    for e in trace.events
                    if e["kind"] == "instruction_end"
                    and e.get("control", {}).get("phase") == "close"
                ]
                self.assertEqual(selected, expected_order)
                destination_xy = geometry["table"][:2] + DESTINATION_OFFSET
                # The fixed column is a gripper target. Payload centering has no
                # 1 cm guarantee; grasp drift is measured separately by the
                # tower-workspace diagnostic, including the wider reset range.
                placements = [
                    trace.states[e["index"]][:2]
                    for e in trace.events
                    if e["kind"] == "instruction_end" and e["path"] == "1.4"
                ]
                self.assertEqual(len(placements), count)
                np.testing.assert_allclose(
                    placements, np.tile(destination_xy, (count, 1)), atol=0.002, rtol=0
                )
                self.assertTrue(np.all(final[:, 1] < geometry["base"][1]))
                heads = [e for e in trace.events if e["kind"] == "loop_head"]
                self.assertEqual(len(heads), count)
                self.assertEqual(
                    [e["bindings"]["b"] for e in heads], ["tbl", *expected_order[:-1]]
                )
                self.assertTrue(
                    all(
                        e["path"].startswith("1.")
                        for e in trace.events
                        if "control" in e
                    )
                )
                self.assertEqual(sum(e["kind"] == "loop_exit" for e in trace.events), 1)

    def test_archive_and_segment_replay_preserve_table_and_block_support_bindings(self):
        definition = load_program(
            "synthesis.examples.reverse:build_program",
            HighLevelContext(use_tbl=True),
            2,
        )
        trace = record_execution(
            definition,
            seed=0,
            num_blocks=2,
            task="reverse",
            max_loop_iterations=2,
            timeout_seconds=12,
        )
        self.assertTrue(validate_trace(trace), trace.metadata)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reverse.npz"
            save_traces(path, [trace])
            loaded = load_traces(path, require_valid=True)[0]
            env = make_roboverify_reverse_env(num_blocks=2)
            try:
                table = inner_env(env).table_position
                heads = [e for e in loaded.events if e["kind"] == "loop_head"]
                self.assertEqual([e["bindings"]["b"] for e in heads], ["tbl", 1])
                for head in heads:
                    segment = DemoSegment(
                        0,
                        head["index"],
                        len(loaded.states) - 1,
                        loaded,
                        head["bindings"],
                    )
                    for mode in ("reset", "replay"):
                        obs = reset_segment(env, segment, mode=mode)
                        np.testing.assert_array_equal(obs, loaded.states[head["index"]])
                        self.assertEqual(
                            inner_env(env).symbolic_name_to_box_id["b"],
                            head["bindings"]["b"],
                        )
                        np.testing.assert_array_equal(
                            inner_env(env).table_position, table
                        )
            finally:
                env.close()


if __name__ == "__main__":
    unittest.main()
