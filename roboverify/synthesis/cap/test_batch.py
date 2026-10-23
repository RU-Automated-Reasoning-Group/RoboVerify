"""Process isolation, crash accounting, and no-paid-rerun resume semantics."""

import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from .batch import atomic_json, create_batch, execute_batch
from .config import ModelConfig, RunConfig


class BatchTests(unittest.TestCase):
    def test_serial_parallel_parity_and_completed_resume(self):
        import numpy as np

        from synthesis.cfg.recordings import load_traces

        with tempfile.TemporaryDirectory() as tmp:
            actions = []
            for workers in (1, 2):
                root = Path(tmp) / f"batch{workers}"
                create_batch(
                    root,
                    RunConfig(
                        num_blocks=1, save_video=False, prompt_version="typed_v3"
                    ),
                    ModelConfig(),
                    methods=("cap_google_baseline", "cap_method_1", "cap_method_2"),
                    modes=("dsl_only",),
                    trials=1,
                    max_workers=workers,
                    responses=["Pick(b0)"],
                    fixture=True,
                )
                with contextlib.redirect_stdout(io.StringIO()):
                    summary = execute_batch(root)
                self.assertEqual(summary["finished"], 3)
                for row in summary["trials"]:
                    self.assertTrue(row["success"])
                    self.assertEqual(row["dsl_compliance"], "pass")
                    path = (
                        Path(row["run_dir"])
                        / "artifacts/trial_001/execution/rollout_001/trajectory.npz"
                    )
                    actions.append(load_traces(path)[0].actions[0])
                saved = {
                    p: str(p.stat().st_mtime_ns)
                    for p in root.glob("trials/*/worker_result.json")
                }
                with contextlib.redirect_stdout(io.StringIO()):
                    again = execute_batch(root, resume=True)
                self.assertEqual(again["finished"], 3)
                self.assertEqual(saved, {p: str(p.stat().st_mtime_ns) for p in saved})
            for action in actions[1:]:
                np.testing.assert_array_equal(actions[0], action)

    def test_worker_crash_does_not_drop_a_trial(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "batch"
            create_batch(
                root,
                RunConfig(num_blocks=1, save_video=False, prompt_version="typed_v3"),
                ModelConfig(),
                methods=("cap_method_1",),
                modes=("dsl_only",),
                trials=2,
                max_workers=2,
                responses=["Pick(b0)"],
                fixture=True,
            )
            original = subprocess.Popen

            def launch(command, **kwargs):
                if command[-1].endswith("001/job.json"):
                    command = [sys.executable, "-c", "raise SystemExit(7)"]
                return original(command, **kwargs)

            with (
                patch("synthesis.cap.batch.subprocess.Popen", side_effect=launch),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                summary = execute_batch(root)
            self.assertEqual(summary["finished"], 2)
            self.assertEqual(
                sum(r.get("success") is True for r in summary["trials"]), 1
            )
            self.assertTrue(
                any(r["process_status"] == "worker_failed" for r in summary["trials"])
            )

    def test_resume_skips_interrupted_and_runs_only_pending(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "batch"
            manifest = create_batch(
                root,
                RunConfig(num_blocks=1, save_video=False, prompt_version="typed_v3"),
                ModelConfig(),
                methods=("cap_method_1",),
                modes=("dsl_only",),
                trials=2,
                max_workers=2,
                responses=["Pick(b0)"],
                fixture=True,
            )
            schedule = json.loads((root / "schedule.json").read_text())
            first = manifest["jobs"][0]["id"]
            schedule[first] = {"status": "running", "pid": 999999999}
            atomic_json(root / "schedule.json", schedule)
            with contextlib.redirect_stdout(io.StringIO()):
                summary = execute_batch(root, resume=True)
            self.assertEqual(summary["finished"], 2)
            row = next(r for r in summary["trials"] if r["job_id"] == first)
            self.assertEqual(row["status"], "interrupted")
            self.assertFalse((root / "trials" / first / "worker_result.json").exists())

    def test_resume_records_overridden_worker_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "batch"
            create_batch(
                root,
                RunConfig(num_blocks=1, save_video=False, prompt_version="typed_v3"),
                ModelConfig(),
                methods=("cap_method_1",),
                modes=("dsl_only",),
                trials=2,
                max_workers=1,
                responses=["Pick(b0)"],
                fixture=True,
            )
            with contextlib.redirect_stdout(io.StringIO()):
                summary = execute_batch(root, resume=True, max_workers=2)
            self.assertEqual(summary["finished"], 2)
            metadata = json.loads((root / "execution_metadata.json").read_text())
            self.assertEqual(metadata["sessions"][-1]["max_workers"], 2)
            self.assertEqual(metadata["requested_workers"], 1)
            self.assertEqual(metadata["peak_active_workers"], 2)

    def test_recovery_comparison_keeps_six_groups_and_all_attempt_usage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "batch"
            create_batch(
                root,
                RunConfig(num_blocks=1, save_video=False, prompt_version="typed_v3"),
                ModelConfig(),
                modes=("dsl_only",),
                trials=1,
                max_workers=3,
                recovery_variants=(False, True),
                fixture=True,
                responses=["x=get(lambda x: not ON_star(x,x))", "pass"],
            )
            with contextlib.redirect_stdout(io.StringIO()):
                summary = execute_batch(root)
            self.assertEqual(summary["finished"], 6)
            for row in summary["trials"]:
                self.assertEqual(row["success"], row["recovery_enabled"])
                self.assertEqual(
                    row["model_calls"], 2 if row["recovery_enabled"] else 1
                )
                self.assertEqual(row["recovery_rounds"], int(row["recovery_enabled"]))
            report = (root / "report.md").read_text()
            self.assertIn("First-attempt success", report)
            self.assertIn("# of tokens used", report)

    def test_modified_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "batch"
            create_batch(
                root, RunConfig(prompt_version="typed_v3"), ModelConfig(), trials=1
            )
            with (root / "manifest.json").open("a") as handle:
                handle.write(" ")
            with self.assertRaisesRegex(ValueError, "manifest changed"):
                execute_batch(root)


if __name__ == "__main__":
    unittest.main()
