"""Generalization test-case sets: identities, manifests and initial states."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from ..generalization import testset as ts


def scored_row(task, layout, seed, feasible=True):
    return dict(
        task=task,
        layout=layout,
        seed=seed,
        feasible=feasible,
        initial_drift=0.0,
        goal_drift=0.001,
        goal_settle_shift=0.002,
        reach_margin_initial=None,
        reach_margin_goal=0.5,
        initial_tops=[[1.0, 0.7, 0.42]],
    )


class GeneralizationTestsetTests(unittest.TestCase):
    def test_case_ids_and_run_configs(self):
        cases = {
            "stack-n22-s3": ts.GeneralizationCase("stack", {"num_blocks": 22}, 3),
            "restack-n12-t4-s0": ts.GeneralizationCase(
                "restack", {"num_blocks": 12, "initial_tower_blocks": 4}, 0
            ),
            "grid-n24-6x4-s2": ts.GeneralizationCase(
                "grid", {"num_blocks": 24, "grid_rows": 6, "grid_cols": 4}, 2
            ),
            "pyramid-n28-k7-s1": ts.GeneralizationCase(
                "pyramid", {"num_blocks": 28, "bottom_row_blocks": 7}, 1
            ),
        }
        for case_id, case in cases.items():
            with self.subTest(case_id=case_id):
                self.assertEqual(case.case_id, case_id)
                config = ts.run_config(case)
                self.assertTrue(config.relax_boundaries_for_generalization_test)
                self.assertEqual(config.num_blocks, case.layout["num_blocks"])
                self.assertEqual(config.seed, case.seed)

    def test_only_feasible_rows_of_selected_tasks_become_cases(self):
        rows = [
            scored_row("stack", {"num_blocks": 4}, 0),
            scored_row("stack", {"num_blocks": 4}, 1, feasible=False),
            scored_row("grid", {"num_blocks": 6, "grid_rows": 3, "grid_cols": 2}, 0),
        ]
        pairs = ts.cases_from_scored(rows, {"stack"})
        self.assertEqual([c.case_id for c, _ in pairs], ["stack-n4-s0"])
        case, tops = pairs[0]
        self.assertEqual(case.evidence["reach_margin_goal"], 0.5)
        self.assertIsNone(case.evidence["reach_margin_initial"])
        self.assertEqual(tops, [[1.0, 0.7, 0.42]])

    def test_manifest_round_trip_and_validation(self):
        cases = [
            ts.GeneralizationCase("stack", {"num_blocks": 4}, s, "ab" * 32, {"x": 1})
            for s in range(3)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "set.json"
            ts.write_manifest(
                path, "set", cases, environment={}, feasibility={}, git_commit=None
            )
            payload, loaded = ts.load_manifest(path)
            self.assertEqual(payload["counts"], {"stack": {"4": 3}})
            self.assertEqual(loaded, cases)
            self.assertEqual(loaded[2].evidence, {"x": 1})
            ts.write_manifest(
                path,
                "set",
                cases + cases[:1],
                environment={},
                feasibility={},
                git_commit=None,
            )
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                ts.load_manifest(path)
            path.write_text(path.read_text().replace('"stack-n4-s0"', '"stack-n5-s0"'))
            with self.assertRaisesRegex(ValueError, "Case id mismatch"):
                ts.load_manifest(path)

    def test_snapshot_encoding_round_trips_and_hash_is_stable(self):
        from synthesis.cfg.reset import Snapshot
        from synthesis.environment.goal_layout import GoalLayout

        layout = GoalLayout.from_positions([[1.0, 0.7, 0.42], [1.1, 0.7, 0.42]], (2,))
        snapshot = Snapshot(
            np.arange(5.0),
            {"ctrl": np.zeros(2), "mocap_pos": np.ones((1, 3))},
            {"b0": 0},
            layout,
            {"h0": 0, "null": None},
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "case.npz"
            path.write_bytes(ts.snapshot_bytes(snapshot))
            loaded = ts.load_snapshot(path)
        self.assertEqual(ts.state_sha256(loaded), ts.state_sha256(snapshot))
        np.testing.assert_array_equal(loaded.gt_state, snapshot.gt_state)
        self.assertEqual(loaded.goal_layout, layout)
        moved = Snapshot(
            snapshot.gt_state + 1e-12,
            snapshot.arrays,
            snapshot.bindings,
            layout,
            snapshot.goal_bindings,
        )
        self.assertNotEqual(ts.state_sha256(moved), ts.state_sha256(snapshot))

    def test_committed_relaxed_v1_manifest(self):
        payload, cases = ts.load_manifest(ts.TESTSET_DIR / "relaxed_v1.json")
        self.assertTrue(
            payload["environment"]["relax_boundaries_for_generalization_test"]
        )
        self.assertEqual(
            {c.task for c in cases},
            {"stack", "unstack", "reverse", "restack", "grid", "pyramid"},
        )
        self.assertTrue(all(len(c.initial_state_sha256) == 64 for c in cases))
        largest = {}
        for case in cases:
            largest[case.task] = max(
                largest.get(case.task, 0), case.layout["num_blocks"]
            )
        self.assertGreaterEqual(largest["stack"], 22)
        self.assertGreaterEqual(largest["pyramid"], 28)
        for case in cases[:: max(1, len(cases) // 50)]:
            ts.run_config(case)

    def test_simulator_rebuild_matches_the_committed_hash(self):
        _, cases = ts.load_manifest(ts.TESTSET_DIR / "relaxed_v1.json")
        case = next(c for c in cases if c.case_id == "stack-n4-s0")
        snapshot, _ = ts.initial_snapshot(case)
        self.assertEqual(ts.state_sha256(snapshot), case.initial_state_sha256)


if __name__ == "__main__":
    unittest.main()
