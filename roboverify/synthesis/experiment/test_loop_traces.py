"""Exercise loop-head collection on a real three-iteration MuJoCo rollout."""

import unittest

from synthesis.entry.collect_stack_loop_traces import collect_stack_loop_traces
from synthesis.inference_lib.demo_store import to_inference_inputs, tower_vocabulary
from synthesis.util import on
from synthesis.verification_lib.highlevel_verification_lib import HighLevelContext


class SimulatorLoopTraces(unittest.TestCase):
    def test_stack_rollout_captures_three_heads_and_all_four_blocks(self):
        store, outcomes = collect_stack_loop_traces(
            num_blocks=4, seeds=[0], max_iters=3
        )
        rows = store.for_loop("1")
        self.assertEqual(len(rows), 3)
        self.assertEqual(outcomes[0]["loop_heads"], 3)
        for row in rows:
            self.assertEqual(set(row.positions), {"x1", "x2", "x3", "x4", "tbl"})
            self.assertEqual(row.entry_positions, rows[0].entry_positions)
            self.assertEqual(row.constants["b0"], "x1")
            self.assertIn("b_prime", row.constants)
            self.assertIs(row.positions["tbl"], on.TABLE)
        self.assertNotEqual(rows[0].positions, rows[1].positions)
        self.assertNotEqual(rows[1].positions, rows[2].positions)
        context = HighLevelContext(mode="declare")
        zeros, states, mappings = to_inference_inputs(
            store, "1", tower_vocabulary("stack"), context
        )
        self.assertEqual([len(zeros), len(states), len(mappings)], [3, 3, 3])
        self.assertEqual(set(states[0]), {"x1", "x2", "x3", "x4"})


if __name__ == "__main__":
    unittest.main()
