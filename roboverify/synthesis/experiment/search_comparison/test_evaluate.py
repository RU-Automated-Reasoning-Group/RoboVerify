"""Regression checks for independent full-loop evaluation semantics."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from synthesis.api.instructions import Assign, While
from synthesis.cfg.reset import Snapshot
from synthesis.experiment.search_comparison.evaluate import (
    LoopMonitor,
    build_loop_program,
    canonical_goal,
    evaluate_case,
    snapshot_digest,
)


def observation(points):
    obs = np.zeros(10 + 12 * len(points))
    obs[3:5] = 0.045
    for i, p in enumerate(points):
        obs[10 + 12 * i : 13 + 12 * i] = p
    return obs


class LoopEvaluationTests(unittest.TestCase):
    def test_body_lengths_do_not_change_guard_or_pointer_scaffold(self):
        for task in ("stack", "unstack"):
            for count in (2, 7):
                rows = [
                    {"kind": "Move", "operands": ["b0"] * 3, "offsets": [0, 0, 0.1]}
                ] * (count - 1)
                rows += [{"kind": "Release", "operands": ["b0"], "offsets": [0.15]}]
                p, path = build_loop_program(task, rows, 3)
                loop = p.instructions[int(path)]
                self.assertIsInstance(loop, While)
                self.assertEqual(len(loop.body), count + int(task == "stack"))
                if task == "stack":
                    self.assertIsInstance(loop.body[-1], Assign)
                    self.assertEqual(
                        (loop.body[-1].left, loop.body[-1].right), ("b", "b_prime")
                    )
                self.assertIn("b_prime", str(loop.cond))

    def test_canonical_goal_rejects_accumulated_stack_drift(self):
        self.assertTrue(
            canonical_goal(
                "stack",
                observation([[0, 0, 0.425], [0.01, 0, 0.475], [0.02, 0, 0.525]]),
                3,
                0.4,
            )
        )
        self.assertFalse(
            canonical_goal(
                "stack",
                observation([[0, 0, 0.425], [0.02, 0, 0.475], [0.04, 0, 0.525]]),
                3,
                0.4,
            )
        )
        self.assertFalse(
            canonical_goal(
                "unstack", observation([[0, 0, 0.425], [0.2, 0, 0.50]]), 2, 0.4
            )
        )
        self.assertTrue(
            canonical_goal(
                "unstack", observation([[0, 0, 0.425], [0.2, 0, 0.425]]), 2, 0.4
            )
        )

    def test_snapshot_pairing_covers_auxiliary_state_and_bindings(self):
        a = Snapshot(
            np.array([1.0, 2.0]), {"qacc_warmstart": np.array([0.0])}, {"b0": 0}
        )
        b = Snapshot(a.gt_state.copy(), {"qacc_warmstart": np.array([1.0])}, {"b0": 0})
        c = Snapshot(a.gt_state.copy(), a.arrays, {"b0": 1})
        self.assertNotEqual(snapshot_digest(a), snapshot_digest(b))
        self.assertNotEqual(snapshot_digest(a), snapshot_digest(c))

    def test_control_count_and_missing_body_end_fail_closed(self):
        for count, completed, recorded, expected in (
            (2, 2, True, True),
            (7, 7, True, True),
            (7, 6, True, False),
            (2, 2, False, False),
        ):
            monitor = LoopMonitor(None, "stack", 3, [None] * count, "1")
            report = dict(
                no_unintended_contacts=True,
                frame_within_tolerance=True,
                saw_mover_finger_contact=True,
                lift_observed=True,
                max_nonmover_displacement_m=0.0,
            )
            monitor.active = SimpleNamespace(
                __exit__=lambda *a: None, report=lambda: report
            )
            monitor.states = [np.zeros(46)]
            monitor.info = {"controls": [{"converged": True}] * completed}
            if recorded:
                monitor.info["body_end"] = {"released": True, "local_goal": True}
            with patch(
                "synthesis.experiment.search_comparison.evaluate.local_goal",
                return_value=True,
            ):
                monitor.finish_transfer()
            self.assertEqual(monitor.transfers[0]["physical_pass"], expected)

    def test_supplied_controls_observer_parity(self):
        # No saved demonstration dependency; these are instrumentation controls,
        # never seeds offered to either synthesis algorithm.
        from synthesis.examples.stack import build_program as stack
        from synthesis.examples.unstack import build_program as unstack
        from synthesis.verification_lib.highlevel_verification_lib import (
            HighLevelContext,
        )

        for task, factory in (("stack", stack), ("unstack", unstack)):
            p = factory(HighLevelContext(use_tbl=task == "unstack"), num_blocks=3)
            loop = next(i for i in p.instructions if isinstance(i, While))
            rows = []
            for i in loop.body:
                if isinstance(i, Assign):
                    continue
                kind = type(i).__name__.replace("ByName", "")
                values = []
                i.register_trainable_parameter(values)
                rows.append(
                    dict(
                        kind=kind,
                        operands=[o["val"] for o in i.get_operand()],
                        offsets=values,
                    )
                )
            result, _ = evaluate_case(task, rows, 3, 1, parity=True)
            self.assertTrue(result["full_program_success"], result.get("error"))
            self.assertEqual(result["observer_parity_max_error"], 0.0)
            self.assertEqual(result["final_settling_steps"], 50)


if __name__ == "__main__":
    unittest.main()
