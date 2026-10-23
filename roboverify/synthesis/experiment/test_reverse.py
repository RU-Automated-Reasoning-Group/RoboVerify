"""Lateral Reverse resets and top-first physical reversal of an entire tower."""

import unittest

import numpy as np

from synthesis.cfg.collection import record_execution, validate_trace
from synthesis.cfg.program_source import load_program
from synthesis.cfg.reset import inner_env
from synthesis.environment.tower_reset import REVERSE_TOWER_Y_OFFSET
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
                                    (0.54 <= offsets[:, 0]) & (offsets[:, 0] <= 0.58)
                                )
                            )
                            self.assertTrue(
                                np.all(
                                    abs(offsets[:, 1] - REVERSE_TOWER_Y_OFFSET) <= 0.01
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

                def factory():
                    env = make_roboverify_reverse_env(num_blocks=count)
                    inner = inner_env(env)
                    inner.roboverify_base_block_id = base_id
                    inner.symbolic_name_to_box_id = {"b0": base_id}
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
                initial = np.array(
                    [get_block_pos(trace.states[0], i) for i in range(count)]
                )
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
                destination_xy = initial[base_id, :2] - [0, 2 * REVERSE_TOWER_Y_OFFSET]
                np.testing.assert_allclose(
                    final[:, :2], np.tile(destination_xy, (count, 1)), atol=0.01, rtol=0
                )
                self.assertEqual(
                    sum(e["kind"] == "loop_head" for e in trace.events), count - 1
                )
                self.assertEqual(sum(e["kind"] == "loop_exit" for e in trace.events), 1)


if __name__ == "__main__":
    unittest.main()
