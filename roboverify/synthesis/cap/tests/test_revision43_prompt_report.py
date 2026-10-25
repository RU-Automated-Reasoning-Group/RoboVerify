import json
import tempfile
import unittest
from pathlib import Path

from ..reports.revision43_prompt_report import cells, trials, wilson


def write(root, job, cell, round0, final, method="cap_method_1"):
    path = Path(root) / "trials" / job
    path.mkdir(parents=True)
    (path / "summary.json").write_text(
        json.dumps(
            {
                "job_id": job,
                "variant": cell,
                "method": method,
                "first_attempt_success": round0,
                "task_success": final,
            }
        )
    )


class PromptReportTest(unittest.TestCase):
    def test_rerun_replaces_listed_jobs_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            main, rerun = Path(tmp) / "main", Path(tmp) / "rerun"
            write(main, "a", "p0001", True, True)
            write(main, "b", "p0001", False, False)
            write(main, "c", "p0000", False, True)
            write(rerun, "b", "p0001", True, True)
            (rerun / "comparison_plan.json").write_text(
                json.dumps({"rerun_job_ids": ["b"]})
            )
            table = cells(trials(main, rerun))
        self.assertEqual(table["p0001"], {"n": 2, "round0": 2, "final": 2})
        self.assertEqual(table["p0000"], {"n": 1, "round0": 0, "final": 1})

    def test_cells_by_method(self):
        with tempfile.TemporaryDirectory() as tmp:
            main = Path(tmp) / "main"
            write(main, "a", "p0001", True, True, "cap_google_baseline")
            write(main, "b", "p0001", False, True, "cap_method_2")
            rows = trials(main)
        self.assertEqual(
            cells(rows, "cap_method_2")["p0001"], {"n": 1, "round0": 0, "final": 1}
        )
        self.assertEqual(cells(rows)["p0001"], {"n": 2, "round0": 1, "final": 2})

    def test_wilson_interval(self):
        low, high = wilson(105, 150)
        self.assertAlmostEqual(low, 0.6224, places=3)
        self.assertAlmostEqual(high, 0.7676, places=3)
        self.assertEqual(wilson(0, 10)[0], 0.0)


if __name__ == "__main__":
    unittest.main()
