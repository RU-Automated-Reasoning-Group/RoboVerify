"""Deferred rendering preserves rollouts and owns independent resumable jobs."""

import contextlib
import fcntl
import io
import json
import shutil
import subprocess
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
from synthesis.cfg.recordings import load_traces
from synthesis.cfg.reset import restore

from .batch import create_batch, execute_batch
from .config import ModelConfig, RecoveryConfig, RunConfig
from .deferred_video import (
    RenderPool,
    digest,
    drain_queue,
    read,
    render_job,
    resolved_video,
)
from .gpu_scheduler import atomic_json
from .llm import ScriptedModel
from .runner import run
from .test_adapter import RobotFixture

METHODS = ("cap_google_baseline", "cap_method_1", "cap_method_2")


class VideoConfigTests(unittest.TestCase):
    def test_default_legacy_aliases_and_conflicts(self):
        self.assertEqual(RunConfig().video_mode, "deferred")
        self.assertEqual(RunConfig(save_video=True).video_mode, "inline")
        self.assertEqual(RunConfig(save_video=False).video_mode, "off")
        for mode in ("inline", "deferred", "off"):
            config = RunConfig(video_mode=mode)
            self.assertEqual(asdict(config), asdict(RunConfig(**asdict(config))))
        for settings in (
            {"video_mode": "bad"},
            {"video_mode": "off", "save_video": True},
            {"video_mode": "deferred", "save_video": False},
        ):
            with self.assertRaises(ValueError):
                RunConfig(**settings)


@unittest.skipUnless(
    shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg/ffprobe required"
)
class DeferredVideoTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    @staticmethod
    def factory(config, snapshot):
        env = RobotFixture(config.num_blocks, success=True)
        if snapshot:
            restore(env, snapshot)
        return env

    def trial(self, mode, method="cap_method_1", recovery=True):
        backend = ScriptedModel(["Pick(b0)\nMove(b0,b0,b0,[10,0,0.2])", "pass"])
        with contextlib.redirect_stdout(io.StringIO()):
            result = run(
                RunConfig(
                    method=method,
                    num_blocks=1,
                    video_mode=mode,
                    output_root=str(self.root),
                    prompt_version="typed_v3",
                    recovery=RecoveryConfig(enabled=recovery),
                ),
                ModelConfig(max_output_tokens=16384),
                backend=backend,
                environment_factory=self.factory,
                capture_stdout=False,
            )
        return result, backend

    @staticmethod
    def execution(result):
        return Path(result["run_dir"]) / "artifacts/trial_001/execution/rollout_001"

    def test_three_methods_modes_trajectories_and_feedback_match(self):
        for method in METHODS:
            modes = {}
            for mode in ("inline", "deferred", "off"):
                result, backend = self.trial(mode, method)
                self.assertTrue(result["success"], result)
                self.assertEqual(result["rollout_count"], 2)
                self.assertEqual(result["attempts"][0]["status"], "primitive_failed")
                traces = [
                    load_traces(self.execution(a) / "trajectory.npz")[0]
                    for a in result["attempts"]
                ]
                modes[mode] = (result, traces, backend)
                if mode == "deferred":

                    def canonical_feedback(item):
                        evidence = read(
                            Path(item["run_dir"])
                            / "artifacts/trial_001/rounds/000/feedback.json"
                        )
                        mapping = evidence["evidence"]["object_handle_to_physical_id"]

                        def canonical(value):
                            if isinstance(value, dict):
                                return {
                                    canonical(k): canonical(v) for k, v in value.items()
                                }
                            if isinstance(value, list):
                                return [canonical(v) for v in value]
                            if isinstance(value, str) and value in mapping:
                                return f"physical_{mapping[value]}"
                            return value

                        return canonical(evidence)

                    self.assertEqual(
                        canonical_feedback(modes["inline"][0]),
                        canonical_feedback(result),
                    )
                    raw = Path(result["run_dir"]) / "result.json"
                    before = raw.read_bytes()
                    summary = drain_queue(result["run_dir"], max_workers=2)
                    self.assertEqual(summary["counts"], {"completed": 2})
                    self.assertEqual(before, raw.read_bytes())
                    self.assertTrue(resolved_video(result["video"])["frames"])
                    self.assertEqual(
                        drain_queue(result["run_dir"])["counts"], {"completed": 2}
                    )
            for mode in ("deferred", "off"):
                for expected, actual in zip(modes["inline"][1], modes[mode][1]):
                    np.testing.assert_array_equal(expected.states, actual.states)
                    np.testing.assert_array_equal(
                        expected.actions[0], actual.actions[0]
                    )
                    for a, b in zip(expected.snapshots, actual.snapshots):
                        np.testing.assert_array_equal(a.gt_state, b.gt_state)
                        for key in a.arrays:
                            np.testing.assert_array_equal(a.arrays[key], b.arrays[key])
                # Feedback contains randomly allocated handle IDs and source hashes;
                # compare invariant facts rather than those run-specific identities.
                for kind in ("primitive_failed", "Move", "position_error"):
                    self.assertIn(kind, modes[mode][2].prompts[1])
            for ai in range(2):
                frames = []
                for mode in ("inline", "deferred"):
                    video = self.execution(modes[mode][0]["attempts"][ai]) / "video.mp4"
                    frames.append(
                        subprocess.check_output(
                            [
                                "ffmpeg",
                                "-v",
                                "error",
                                "-i",
                                str(video),
                                "-f",
                                "rawvideo",
                                "-pix_fmt",
                                "rgb24",
                                "-threads",
                                "1",
                                "-",
                            ]
                        )
                    )
                self.assertEqual(frames[0], frames[1])

    def test_hash_failure_retry_lock_and_resume(self):
        result, _ = self.trial("deferred", recovery=False)
        path = self.execution(result) / "render_job.json"
        trajectory = path.with_name("trajectory.npz")
        original = trajectory.read_bytes()
        trajectory.write_bytes(original + b"changed")
        self.assertEqual(render_job(path)["status"], "failed")
        self.assertIn(
            "hash mismatch", read(path.with_name("render_status.json"))["reason"]
        )
        trajectory.write_bytes(original)
        self.assertEqual(drain_queue(result["run_dir"])["counts"], {"failed": 1})
        with path.with_name(".render.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):
                render_job(path)
        self.assertEqual(
            drain_queue(result["run_dir"], retry_failed=True)["counts"],
            {"completed": 1},
        )
        video = path.with_name("video.mp4")
        before = digest(video)
        self.assertEqual(render_job(path)["status"], "completed")
        self.assertEqual(digest(video), before)
        self.assertEqual(len(read(path.with_name("render_status.json"))["attempts"]), 2)

    def test_missing_archive_and_metadata_mismatch_do_not_change_result(self):
        for mismatch in ("missing", "metadata"):
            result, _ = self.trial("deferred", recovery=False)
            original = (Path(result["run_dir"]) / "result.json").read_bytes()
            path = self.execution(result) / "render_job.json"
            if mismatch == "missing":
                path.with_name("trajectory.npz").unlink()
            else:
                job = read(path)
                job["environment"]["fps"] = 21
                atomic_json(path, job)
                status = read(path.with_name("render_status.json"))
                status["job_sha256"] = digest(path)
                atomic_json(path.with_name("render_status.json"), status)
            self.assertEqual(render_job(path)["status"], "failed")
            self.assertEqual(
                (Path(result["run_dir"]) / "result.json").read_bytes(), original
            )

    def test_interrupted_job_and_coordinator_lock(self):
        result, _ = self.trial("deferred", recovery=False)
        path = self.execution(result) / "render_job.json"
        status = read(path.with_name("render_status.json"))
        status.update(status="running", owner={"pid": 99999999, "start_ticks": 0})
        atomic_json(path.with_name("render_status.json"), status)
        pool = RenderPool(result["run_dir"])
        try:
            with self.assertRaises(BlockingIOError):
                RenderPool(result["run_dir"])
            self.assertEqual(pool.pending(), [path])
        finally:
            pool.close()
        self.assertEqual(drain_queue(result["run_dir"])["counts"], {"completed": 1})

    def test_actual_worker_interruption_can_resume_without_duplicate_dispatch(self):
        result, _ = self.trial("deferred", recovery=False)
        pool = RenderPool(result["run_dir"], max_workers=1)
        pool.tick(1)
        self.assertEqual(len(pool.active), 1)
        pool.close()
        self.assertEqual(drain_queue(result["run_dir"])["counts"], {"completed": 1})

    def test_render_launch_failure_is_video_only(self):
        result, _ = self.trial("deferred", recovery=False)
        original = (Path(result["run_dir"]) / "result.json").read_bytes()
        pool = RenderPool(result["run_dir"])
        try:
            with patch(
                "synthesis.cap.deferred_video.subprocess.Popen",
                side_effect=OSError("test launch failure"),
            ):
                pool.tick(1)
            self.assertEqual(pool.report()["counts"], {"failed": 1})
            self.assertFalse(pool.active)
        finally:
            pool.close()
        self.assertEqual(
            (Path(result["run_dir"]) / "result.json").read_bytes(), original
        )
        self.assertEqual(
            drain_queue(result["run_dir"], retry_failed=True)["counts"],
            {"completed": 1},
        )

    def test_live_unclaimed_worker_blocks_resume(self):
        result, _ = self.trial("deferred", recovery=False)
        pool = RenderPool(result["run_dir"])
        try:
            with patch(
                "synthesis.cap.deferred_video.unclaimed_worker_alive", return_value=True
            ):
                with self.assertRaisesRegex(RuntimeError, "still alive"):
                    pool.pending()
        finally:
            pool.close()

    def test_batch_pool_respects_total_slots_and_reports_separate_video_status(self):
        directory = self.root / "batch"
        create_batch(
            directory,
            RunConfig(num_blocks=1, prompt_version="typed_v3"),
            ModelConfig(),
            modes=("dsl_only",),
            trials=1,
            max_workers=2,
            responses=["Pick(b0)"],
            fixture=True,
        )
        summary = execute_batch(directory)
        self.assertEqual(summary["finished"], 3)
        self.assertEqual(summary["video_rendering"]["counts"], {"completed": 3})
        metadata = read(directory / "execution_metadata.json")
        self.assertLessEqual(metadata["peak_combined_workers"], 2)
        for row in summary["trials"]:
            self.assertTrue(row["success"])
            self.assertEqual(row["video"]["status"], "completed")
        self.assertEqual(execute_batch(directory, resume=True)["finished"], 3)


if __name__ == "__main__":
    unittest.main()
