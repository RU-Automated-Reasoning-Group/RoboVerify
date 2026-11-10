import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from synthesis.api.instructions import Skip
from synthesis.cfg.demos import DemoAssignment, DemoSegment, DemoTrace
from synthesis.cfg.execute import execute_cfg
from synthesis.cfg.graph import Edge, Node, RelationalCFG
from synthesis.cfg.region import BlockRegion
from synthesis.predicates.term import boolean


class FailedPrefixExecutionTests(unittest.TestCase):
    def test_empty_region_stops_execution_and_retains_reached_prefix(self):
        obs = np.zeros(22)
        obs[10:13] = (0, 0, 0.425)
        segment = DemoSegment(0, 0, 1, DemoTrace((obs, obs), num_blocks=1), {"b0": 0})
        for empty_first in (True, False):
            with self.subTest(empty_first=empty_first):
                names = (
                    ["empty", "later"] if empty_first else ["prefix", "empty", "later"]
                )
                cfg = RelationalCFG(
                    {
                        n: Node(
                            n, BlockRegion(None, () if n == "empty" else (Skip(0),))
                        )
                        for n in names
                    },
                    [
                        Edge(a, b, boolean(True))
                        for a, b in zip(["entry"] + names, names + ["exit"])
                    ],
                    names,
                    DemoAssignment({n: [segment] for n in names}),
                )
                env = SimpleNamespace(symbolic_name_to_box_id={"b0": 0}, close=Mock())
                executed = []

                def run(program, environment, initial, on_state):
                    executed.extend(program.instructions)
                    on_state(initial, environment.symbolic_name_to_box_id)
                    return [initial]

                with patch(
                    "synthesis.cfg.execute.reset_segment", return_value=obs
                ), patch("synthesis.cfg.execute.execute_current", side_effect=run):
                    negatives = execute_cfg(cfg, None, lambda trace: env)
                self.assertEqual(len(executed), 0 if empty_first else 1)
                self.assertEqual(len(negatives), 1)
                self.assertEqual(negatives[0].positions[0], (0, 0, 0.425))
                env.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
