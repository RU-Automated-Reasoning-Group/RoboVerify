import json
import tempfile
import unittest
from pathlib import Path

from .report import accepted_trial, read_json, summarize


class ReportTests(unittest.TestCase):
    def test_deduplication_keeps_search_failure_in_denominator(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def write(path, value):
                path = root / path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(value))

            write(
                "protocol.json",
                dict(
                    tasks=["stack"],
                    methods=["beam"],
                    seeds=[0, 1, 2],
                    heldout=dict(block_counts=[3], seeds=[60]),
                ),
            )
            for seed in range(3):
                write(
                    f"main/stack/beam/seed{seed}/result.json",
                    dict(
                        config=dict(task="stack", method="beam", seed=seed),
                        pilot=False,
                        status="shortlist_exhausted",
                        training_accepted=seed < 2,
                        selected=(
                            dict(program_sha256="same", discovery_seconds=10)
                            if seed < 2
                            else None
                        ),
                        evaluation_counts=dict(
                            simulator_rollouts=15 if seed < 2 else 0
                        ),
                    ),
                )
                write(
                    f"main/stack/beam/seed{seed}/process.json",
                    dict(returncode=0, wall_seconds=20),
                )
            write(
                "frozen-candidates.json",
                dict(
                    candidates=[
                        dict(
                            task="stack",
                            program_sha256="same",
                            file_sha256="filehash",
                            aliases=[
                                dict(method="beam", seed=s, training_accepted=True)
                                for s in (0, 1)
                            ],
                        )
                    ]
                ),
            )
            write(
                "heldout/stack/same/n3/seed60.json",
                dict(candidate_sha256="filehash", full_program_success=True),
            )
            report = summarize(root)
            self.assertEqual(report["actual_loop_executions"], 1)
            row = report["groups"][0]
            self.assertEqual(row["training_accepted_trials"], 2)
            self.assertEqual(row["unique_selected_programs"], 1)
            self.assertEqual(row["loops"]["3"]["selected_cases"], 2)
            self.assertEqual(row["loops"]["3"]["accepted_search_successes"], 2)
            self.assertEqual(row["loops"]["3"]["end_to_end_denominator"], 3)

    def test_large_summaries_are_rejected_before_reading(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "huge.json"
            path.write_text(" " * 100)
            with self.assertRaises(ValueError):
                read_json(path, limit=10)

    def test_error_or_outer_timeout_cannot_promote_retained_candidate(self):
        record = dict(training_accepted=True, status="budget_exhausted_with_candidate")
        self.assertTrue(accepted_trial(record, dict(returncode=0, wall_seconds=48)))
        self.assertFalse(accepted_trial(record, dict(returncode=124, wall_seconds=55)))
        self.assertFalse(
            accepted_trial(
                dict(record, status="error"), dict(returncode=0, wall_seconds=30)
            )
        )
        self.assertFalse(accepted_trial(record, None))


if __name__ == "__main__":
    unittest.main()
