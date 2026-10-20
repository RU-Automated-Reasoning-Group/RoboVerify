"""GPU admission and crash safety without requiring graphics hardware."""

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from .config import GPUConfig, ModelConfig, RunConfig
from .gpu_scheduler import GPUScheduler, GPUUnavailable, process_identity

A = "GPU-00000000-0000-0000-0000-000000000001"
B = "GPU-00000000-0000-0000-0000-000000000002"
MAPPING = {u: {"uuid": u, "egl_index": i + 3} for i, u in enumerate((A, B))}


class Monitor:
    def __init__(self, free=2000, load=5):
        self.free, self.load = free, load
        self.error = False

    def sample(self, allowlist):
        if self.error:
            raise GPUUnavailable("monitor offline")
        return {
            u: {
                "healthy": True,
                "free_memory_mb": self.free,
                "used_memory_mb": 100,
                "utilization_percent": self.load,
            }
            for u in allowlist
        }


def race_reserve(config, job, start, result, finish):
    scheduler = GPUScheduler(config, mapping=MAPPING, monitor=Monitor())
    deadline = time.monotonic() + 10
    while not Path(start).exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assignment = scheduler.reserve(job)
    Path(result).write_text(json.dumps(bool(assignment)))
    while not Path(finish).exists() and time.monotonic() < deadline:
        time.sleep(0.01)


class GPUTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.config = GPUConfig(
            [A], self.tmp.name, min_free_memory_mb=200, worker_memory_mb=600
        )
        self.monitor = Monitor()
        self.scheduler = GPUScheduler(
            self.config, mapping=MAPPING, monitor=self.monitor
        )

    def test_allowlist_and_startup_slots(self):
        first = self.scheduler.reserve(Path(self.tmp.name) / "a.json")
        self.assertEqual(first["uuid"], A)
        self.assertEqual(first["egl_index"], 3)
        self.assertIsNone(self.scheduler.reserve(Path(self.tmp.name) / "b.json"))
        self.scheduler.release(first["token"])
        self.assertIsNotNone(self.scheduler.reserve(Path(self.tmp.name) / "b.json"))

    def test_memory_reservations_prevent_startup_overcommit(self):
        self.config.max_workers_per_gpu = 4
        self.assertIsNotNone(self.scheduler.reserve("a"))
        self.assertIsNotNone(self.scheduler.reserve("b"))
        self.assertIsNotNone(self.scheduler.reserve("c"))
        self.assertIsNone(self.scheduler.reserve("d"))

    def test_shared_load_window_uses_recent_peak_memory(self):
        self.monitor.free = 100
        self.assertIsNone(self.scheduler.reserve("a"))
        self.monitor.free = 10000
        self.scheduler.last_sample = None
        self.assertIsNone(self.scheduler.reserve("a"))

    def test_monitoring_unavailable_requires_explicit_fixed_mode(self):
        self.monitor.error = True
        with self.assertRaises(GPUUnavailable):
            self.scheduler.reserve("a")
        self.config.gpu_policy = "fixed"
        assignment = self.scheduler.reserve("a")
        self.assertIn("monitoring_unavailable", assignment["admission_sample"])

    def test_mapping_failure_and_configuration(self):
        with self.assertRaisesRegex(GPUUnavailable, "mapping"):
            GPUScheduler(self.config, mapping={})
        with self.assertRaises(ValueError):
            GPUConfig([A, B], self.tmp.name, gpu_policy="fixed")
        with self.assertRaises(ValueError):
            GPUConfig([A], self.tmp.name)

    def test_process_start_time_and_live_child_survive_parent_death(self):
        assignment = self.scheduler.reserve("a")
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        try:
            self.scheduler.bind(assignment["token"], child.pid)
            with self.scheduler.locked() as data:
                r = data["reservations"][assignment["token"]]
                r["parent"] = {"pid": 999999999}
                r["reserved_epoch"] = 0
            self.assertIsNone(self.scheduler.reserve("b"))
            with self.assertRaisesRegex(GPUUnavailable, "live worker"):
                self.scheduler.release(assignment["token"])
        finally:
            child.kill()
            child.wait()
        self.assertIsNotNone(self.scheduler.reserve("b"))

    def test_pid_reuse_does_not_pin_stale_slot(self):
        assignment = self.scheduler.reserve("a")
        with self.scheduler.locked() as data:
            data["reservations"][assignment["token"]]["worker"] = process_identity() | {
                "start_ticks": "impossible"
            }
        self.assertIsNotNone(self.scheduler.reserve("b"))

    def test_live_unbound_child_is_not_reclaimed(self):
        path = str(Path(self.tmp.name) / "a.json")
        assignment = self.scheduler.reserve(path)
        child = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)", path]
        )
        try:
            with self.scheduler.locked() as data:
                data["reservations"][assignment["token"]]["parent"] = {"pid": 999999999}
            self.assertIsNone(self.scheduler.reserve("b"))
        finally:
            child.kill()
            child.wait()
        self.assertIsNotNone(self.scheduler.reserve("b"))

    def test_concurrent_launchers_reserve_only_one_slot(self):
        # Fresh interpreters also work when the enclosing suite is run from stdin.
        start = Path(self.tmp.name) / "start"
        finish = Path(self.tmp.name) / "finish"
        outputs = [Path(self.tmp.name) / f"result{i}.json" for i in range(3)]
        code = "import json,sys; from synthesis.cap.test_gpu_scheduler import race_reserve; race_reserve(*json.loads(sys.argv[1]))"
        children = []
        try:
            for i, output in enumerate(outputs):
                args = [
                    asdict(self.config),
                    f"job{i}",
                    str(start),
                    str(output),
                    str(finish),
                ]
                children.append(
                    subprocess.Popen([sys.executable, "-c", code, json.dumps(args)])
                )
            start.touch()
            deadline = time.monotonic() + 10
            while (
                not all(p.exists() and p.stat().st_size for p in outputs)
                and time.monotonic() < deadline
            ):
                time.sleep(0.01)
            self.assertTrue(all(p.exists() for p in outputs))
            self.assertEqual(sum(json.loads(p.read_text()) for p in outputs), 1)
        finally:
            finish.touch()
            for child in children:
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
        self.assertIsNotNone(self.scheduler.reserve("after_exit"))

    def test_cross_allowlist_samples_do_not_hide_devices(self):
        second = GPUScheduler(
            GPUConfig([B], self.tmp.name, min_free_memory_mb=200, worker_memory_mb=600),
            mapping=MAPPING,
            monitor=Monitor(),
        )
        self.scheduler.poll()
        second.poll()
        self.assertEqual(self.scheduler.reserve("a")["uuid"], A)

    def test_duplicate_trial_refused(self):
        self.config.max_workers_per_gpu = 2
        self.scheduler.reserve("same")
        with self.assertRaisesRegex(GPUUnavailable, "already"):
            self.scheduler.reserve("same")

    def test_launch_failure_releases_slot_and_retains_denominator(self):
        from .batch import create_batch, execute_batch

        root = Path(self.tmp.name) / "batch"
        create_batch(
            root,
            RunConfig(prompt_version="typed_v3"),
            ModelConfig(),
            methods=("cap_method_1",),
            modes=("dsl_only",),
            trials=2,
            gpu_config=self.config,
            fixture=True,
            responses=["pass"],
        )
        with (
            patch(
                "synthesis.cap.gpu_scheduler.GPUScheduler", return_value=self.scheduler
            ),
            patch(
                "synthesis.cap.batch.subprocess.Popen",
                side_effect=OSError("launch failed"),
            ),
        ):
            result = execute_batch(root)
        self.assertEqual(result["finished"], 2)
        self.assertTrue(all(r["status"] == "worker_failed" for r in result["trials"]))
        with patch(
            "synthesis.cap.gpu_scheduler.GPUScheduler",
            side_effect=GPUUnavailable("offline"),
        ):
            self.assertEqual(execute_batch(root, resume=True)["finished"], 2)
        with self.scheduler.locked() as data:
            self.assertEqual(data["reservations"], {})

    def test_no_eligible_gpu_leaves_trials_pending_and_resumable(self):
        from .batch import create_batch, execute_batch

        self.monitor.free = 0
        self.config.admission_timeout_seconds = 0.05
        root = Path(self.tmp.name) / "batch"
        create_batch(
            root,
            RunConfig(prompt_version="typed_v3"),
            ModelConfig(),
            methods=("cap_method_1",),
            modes=("dsl_only",),
            trials=1,
            gpu_config=self.config,
            fixture=True,
            responses=["pass"],
        )
        with patch(
            "synthesis.cap.gpu_scheduler.GPUScheduler", return_value=self.scheduler
        ):
            result = execute_batch(root)
            self.assertEqual(result["finished"], 0)
            result = execute_batch(root, resume=True)
            self.assertEqual(result["finished"], 0)
        self.assertFalse(list(root.glob("trials/*/worker_start.json")))

    def test_gpu_worker_crashes_release_slots_without_changing_parent_environment(self):
        from .batch import create_batch, execute_batch

        root = Path(self.tmp.name) / "crash"
        create_batch(
            root,
            RunConfig(prompt_version="typed_v3"),
            ModelConfig(),
            methods=("cap_method_1",),
            modes=("dsl_only",),
            trials=2,
            max_workers=2,
            gpu_config=self.config,
            fixture=True,
            responses=["pass"],
        )
        original = subprocess.Popen
        environment = dict(os.environ)

        def crash(command, **kwargs):
            self.assertEqual(kwargs["env"]["GPUS"], "3")
            return original([sys.executable, "-c", "raise SystemExit(7)"], **kwargs)

        with (
            patch(
                "synthesis.cap.gpu_scheduler.GPUScheduler", return_value=self.scheduler
            ),
            patch("synthesis.cap.batch.subprocess.Popen", side_effect=crash),
        ):
            result = execute_batch(root)
        self.assertEqual(result["finished"], 2)
        self.assertTrue(all(r["worker_exit_code"] == 7 for r in result["trials"]))
        self.assertEqual(dict(os.environ), environment)
        with self.scheduler.locked() as data:
            self.assertFalse(data["reservations"])

    def test_monitoring_loss_drains_live_worker_and_pauses_new_admission(self):
        from .batch import create_batch, execute_batch

        self.config.admission_timeout_seconds = 0.05
        self.config.gpu_poll_interval_seconds = 0.001
        root = Path(self.tmp.name) / "monitor-loss"
        create_batch(
            root,
            RunConfig(prompt_version="typed_v3"),
            ModelConfig(),
            methods=("cap_method_1",),
            modes=("dsl_only",),
            trials=2,
            max_workers=1,
            gpu_config=self.config,
            fixture=True,
            responses=["pass"],
        )
        original = subprocess.Popen

        def launch(command, **kwargs):
            self.monitor.error = True
            return original(
                [sys.executable, "-c", "import time; time.sleep(.3)"], **kwargs
            )

        with (
            patch(
                "synthesis.cap.gpu_scheduler.GPUScheduler", return_value=self.scheduler
            ),
            patch("synthesis.cap.batch.subprocess.Popen", side_effect=launch),
        ):
            result = execute_batch(root)
        self.assertEqual(result["finished"], 1)
        self.assertEqual(result["trials"][0]["worker_exit_code"], 0)
        schedule = json.loads((root / "schedule.json").read_text())
        self.assertEqual(sum(s["status"] == "pending" for s in schedule.values()), 1)

    def test_worker_mapping_failure_records_zero_model_calls(self):
        from .batch_worker import main

        path = Path(self.tmp.name) / "job.json"
        assignment = self.scheduler.reserve(path)
        path.write_text(json.dumps({"gpu_assignment": assignment}))
        with patch.dict(os.environ, {"GPUS": "0"}):
            with self.assertRaises(GPUUnavailable):
                main(path)
        result = json.loads(path.with_name("worker_result.json").read_text())
        self.assertEqual(result["status"], "gpu_initialization_failed")
        self.assertEqual(result["model_calls"], 0)

    def test_worker_checks_environment_before_gpu_use(self):
        from .gpu_scheduler import initialize_worker

        assignment = self.scheduler.reserve("job")
        with patch.dict(os.environ, {"GPUS": "0"}):
            with self.assertRaisesRegex(GPUUnavailable, "GPUS"):
                initialize_worker(assignment, "job")
        with (
            patch.dict(os.environ, {"GPUS": "3"}),
            patch("synthesis.cap.gpu_scheduler.egl_uuid_mapping", return_value=MAPPING),
        ):
            self.assertTrue(initialize_worker(assignment, "job")["mapping_verified"])


@unittest.skipUnless(
    os.environ.get("CAP_GPU_TEST_UUIDS"), "Opt-in real EGL/simulator integration"
)
class GPUIntegrationTests(unittest.TestCase):
    def test_serial_parallel_recovery_and_video_parity(self):
        import numpy as np
        from synthesis.cfg.recordings import load_traces

        from .batch import create_batch, execute_batch

        root = Path(os.environ["CAP_GPU_TEST_OUTPUT"])
        snapshot = os.environ["CAP_GPU_TEST_SNAPSHOT"]
        allowlist = os.environ["CAP_GPU_TEST_UUIDS"].split(",")
        source = """def policy():
    top = b0
    while exists(lambda x: not ON_star(x, b0)):
        source = get(lambda x: not ON_star(x, b0) and forall(lambda y: eq(y, x) or not ON_star(y, x)))
        Pick(source)
        Move(source, source, top, [0, 0, 0.10])
        Move(b0, b0, top, [0, 0, 0.10])
        Move(b0, b0, top, [0, 0, 0.05])
        Release(source, 0.15)
        top = source
policy()"""
        # The upstream baseline may request a helper even for an inline
        # definition. Keep this scheduling fixture top-level and self-contained.
        source = "\n".join(line[4:] for line in source.splitlines()[1:-1])
        results = []
        for workers in (1, 2):
            batch = root / f"workers-{workers}"
            if not batch.exists():
                create_batch(
                    batch,
                    RunConfig(prompt_version="typed_v3", num_blocks=3),
                    ModelConfig(max_output_tokens=16384),
                    initial_state=snapshot,
                    modes=("dsl_only",),
                    trials=1,
                    max_workers=workers,
                    recovery_variants=(False, True),
                    responses=["Pick(b0)\nx=get(lambda x: not ON_star(x,x))", source],
                    gpu_config=GPUConfig(
                        allowlist,
                        str(root / "reservations"),
                        min_free_memory_mb=2048,
                        worker_memory_mb=1024,
                        gpu_poll_interval_seconds=0.2,
                        gpu_sample_window_seconds=1,
                    ),
                )
            schedule = json.loads((batch / "schedule.json").read_text())
            summary = execute_batch(
                batch, resume=any(s["status"] != "pending" for s in schedule.values())
            )
            self.assertEqual(summary["finished"], 6)
            traces = {}
            for row in summary["trials"]:
                self.assertEqual(row["success"], row["recovery_enabled"], row)
                self.assertEqual(
                    row["model_calls"], 2 if row["recovery_enabled"] else 1
                )
                self.assertEqual(row["gpu_assignment"]["uuid"] in allowlist, True)
                job = batch / "trials" / row["job_id"]
                probe = json.loads((job / "gpu_probe.json").read_text())
                self.assertEqual(probe["render_probe"]["shape"], [32, 32, 3])
                attempts = row.get("attempts") or [{"run_dir": row["run_dir"]}]
                for index, attempt in enumerate(attempts):
                    run = Path(attempt["run_dir"])
                    config = json.loads((run / "config.json").read_text())
                    self.assertEqual(
                        config["config"]["batch_context"]["gpu_assignment"]["token"],
                        row["gpu_assignment"]["token"],
                    )
                    execution = run / "artifacts/trial_001/execution/rollout_001"
                    trace = load_traces(execution / "trajectory.npz")[0]
                    video = json.loads((run / "result.json").read_text())["video"]
                    self.assertEqual(video["status"], "saved")
                    self.assertEqual(video["frames"], len(trace.states))
                    frames = subprocess.run(
                        [
                            "ffprobe",
                            "-v",
                            "error",
                            "-count_frames",
                            "-select_streams",
                            "v:0",
                            "-show_entries",
                            "stream=nb_read_frames",
                            "-of",
                            "csv=p=0",
                            str(execution / "video.mp4"),
                        ],
                        capture_output=True,
                        text=True,
                        check=True,
                    )
                    self.assertEqual(int(frames.stdout.strip()), len(trace.states))
                    self.assertTrue(
                        list(
                            (run / "artifacts/trial_001/checkpoints").glob(
                                "*/manifest.json"
                            )
                        )
                    )
                    expected_source = (
                        "Pick(b0)\nx=get(lambda x: not ON_star(x,x))"
                        if index == 0
                        else source
                    )
                    root_artifacts = run / "artifacts/trial_001"
                    self.assertEqual(
                        (root_artifacts / "lmp/call_0001/response.txt")
                        .read_text()
                        .strip(),
                        expected_source.strip(),
                    )
                    outcome = json.loads((run / "result.json").read_text())
                    self.assertEqual(
                        (
                            root_artifacts
                            / "checkpoints"
                            / outcome["active_checkpoint_id"]
                            / "policy.py"
                        )
                        .read_text()
                        .strip(),
                        expected_source.strip(),
                    )
                    traces[(row["job_id"], index)] = (
                        np.array(trace.states),
                        np.array(trace.actions[0]),
                    )
                if row["recovery_enabled"]:
                    for index in range(1, len(attempts)):
                        np.testing.assert_array_equal(
                            traces[(row["job_id"], 0)][0][0],
                            traces[(row["job_id"], index)][0][0],
                        )
            saved = {
                p: p.stat().st_mtime_ns
                for p in batch.glob("trials/*/worker_result.json")
            }
            self.assertEqual(execute_batch(batch, resume=True)["finished"], 6)
            self.assertEqual(saved, {p: p.stat().st_mtime_ns for p in saved})
            results.append(traces)
        self.assertEqual(results[0].keys(), results[1].keys())
        for key in results[0]:
            np.testing.assert_array_equal(results[0][key][0], results[1][key][0])
            np.testing.assert_array_equal(results[0][key][1], results[1][key][1])
        self.assertEqual(
            json.loads((root / "reservations/reservations.json").read_text())[
                "reservations"
            ],
            {},
        )


if __name__ == "__main__":
    unittest.main()
