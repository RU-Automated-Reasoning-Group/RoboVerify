"""Request admission keeps started trials funded and pending trials resumable."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from ..batch import create_batch, execute_batch
from ..config import ModelConfig, RunConfig
from ..request_control import SharedRequestGate, dispatch_reservation


def spend_requests(path, ident):
    gate = SharedRequestGate(path, interval_seconds=0, request_limit=20, job_id=ident)
    for _ in range(8):
        try:
            gate.reserve(3)
        except OSError:
            return


class RequestControlTests(unittest.TestCase):
    def test_parallel_processes_cannot_overspend_shared_limit(self):
        import multiprocessing

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "requests.json"
            ctx = multiprocessing.get_context("fork")
            processes = [
                ctx.Process(target=spend_requests, args=(path, str(i)))
                for i in range(3)
            ]
            try:
                for process in processes:
                    process.start()
                for process in processes:
                    process.join(timeout=5)
                    self.assertEqual(process.exitcode, 0)
            finally:
                for process in processes:
                    if process.is_alive():
                        process.terminate()
                        process.join()
            state = json.loads(path.read_text())
            self.assertEqual(state["admitted"], 20)
            self.assertEqual(sum(state["by_job"].values()), 20)

    def test_separate_clients_share_counter_and_enforce_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "requests.json"
            first = SharedRequestGate(
                path, interval_seconds=0, request_limit=2, job_id="a"
            )
            second = SharedRequestGate(
                path, interval_seconds=0, request_limit=2, job_id="b"
            )
            first.reserve(1)
            second.reserve(1)
            with self.assertRaises(OSError):
                first.reserve(1)
            state = json.loads(path.read_text())
            self.assertEqual(state["admitted"], 2)
            self.assertEqual(state["by_job"], {"a": 1, "b": 1})

    def test_pacing_timeout_does_not_dispatch_or_debit_another_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "requests.json"
            calls = []
            gate = SharedRequestGate(
                path, interval_seconds=1, opener=lambda r, timeout: calls.append(r)
            )
            gate("first", timeout=1)
            with self.assertRaises(TimeoutError):
                gate("second", timeout=0.01)
            self.assertEqual(calls, ["first"])
            self.assertEqual(json.loads(path.read_text())["admitted"], 1)

    def test_reservations_account_for_active_trial_remaining_calls(self):
        active = {
            "id": "on",
            "run_config": {"recovery": {"enabled": True, "total_calls": 76}},
        }
        pending = {"id": "off", "run_config": {"max_model_calls": 12}}
        state = {"admitted": 700, "by_job": {"on": 10}}
        result = dispatch_reservation(
            pending, [active], state, request_limit=800, free_remaining=78
        )
        self.assertTrue(result["allowed"])
        self.assertEqual(result["required_remaining_calls"], 78)
        self.assertFalse(
            dispatch_reservation(
                pending, [active], state, request_limit=800, free_remaining=77
            )["allowed"]
        )
        self.assertFalse(
            dispatch_reservation(
                pending, [active], state, request_limit=777, free_remaining=1000
            )["allowed"]
        )

    def test_allowing_gate_does_not_stop_after_a_worker_finishes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "batch"
            create_batch(
                root,
                RunConfig(num_blocks=1, save_video=False, prompt_version="typed_v3"),
                ModelConfig(),
                methods=("cap_method_2",),
                modes=("dsl_only",),
                trials=2,
                max_workers=1,
                responses=["pass"],
                fixture=True,
            )
            with contextlib.redirect_stdout(io.StringIO()):
                result = execute_batch(root, dispatch_gate=lambda job, active: True)
            self.assertEqual(result["finished"], 2)

    def test_paused_dispatch_preserves_pending_jobs_and_resume_runs_them_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "batch"
            create_batch(
                root,
                RunConfig(num_blocks=1, save_video=False, prompt_version="typed_v3"),
                ModelConfig(),
                methods=("cap_method_2",),
                modes=("dsl_only",),
                trials=2,
                max_workers=2,
                responses=["pass"],
                fixture=True,
            )
            with contextlib.redirect_stdout(io.StringIO()):
                first = execute_batch(
                    root, dispatch_gate=lambda job, active: job["trial"] == 1
                )
            self.assertEqual(first["finished"], 1)
            schedule = json.loads((root / "schedule.json").read_text())
            self.assertEqual(
                sum(s["status"] == "pending" for s in schedule.values()), 1
            )
            result_file = next(root.glob("trials/*/worker_result.json"))
            stamp = result_file.stat().st_mtime_ns
            with contextlib.redirect_stdout(io.StringIO()):
                second = execute_batch(root, resume=True)
            self.assertEqual(second["finished"], 2)
            self.assertEqual(stamp, result_file.stat().st_mtime_ns)


class RotatingCredentialTests(unittest.TestCase):
    def test_rotation_caps_and_artifacts_never_contain_credentials(self):
        import os
        import urllib.request
        from unittest.mock import patch

        from ..request_control import RotatingRequestGate

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.dict(
                os.environ, {"TEST_KEY_A": "secret-one", "TEST_KEY_B": "secret-two"}
            ),
        ):
            path = Path(tmp) / "requests.json"
            headers = []
            gate = RotatingRequestGate(
                path,
                credential_limits={"TEST_KEY_A": 3, "TEST_KEY_B": 1},
                request_limit=4,
                interval_seconds=0,
                opener=lambda request, timeout: headers.append(
                    request.get_header("Authorization")
                ),
            )
            for _ in range(4):
                gate(urllib.request.Request("https://example.invalid"), timeout=1)
            self.assertEqual(headers[:2], ["Bearer secret-one", "Bearer secret-two"])
            self.assertEqual(headers[2:], ["Bearer secret-one"] * 2)
            with self.assertRaises(OSError):
                gate(urllib.request.Request("https://example.invalid"), timeout=1)
            state = json.loads(path.read_text())
            self.assertEqual(state["by_credential"], {"TEST_KEY_A": 3, "TEST_KEY_B": 1})
            for item in Path(tmp).iterdir():
                self.assertNotIn("secret-one", item.read_text())
                self.assertNotIn("secret-two", item.read_text())

    def test_pacing_and_dynamic_quota_caps(self):
        import os
        from unittest.mock import patch

        from ..request_control import RotatingRequestGate

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.dict(os.environ, {"TEST_KEY_A": "a", "TEST_KEY_B": "b"}),
        ):
            path = Path(tmp) / "requests.json"
            gate = RotatingRequestGate(
                path,
                credential_limits={"TEST_KEY_A": 3, "TEST_KEY_B": 3},
                interval_seconds=1,
            )
            self.assertEqual(gate.reserve(1)[1], "TEST_KEY_A")
            self.assertEqual(gate.reserve(1)[1], "TEST_KEY_B")
            with self.assertRaises(TimeoutError):
                gate.reserve(0.01)
            state = json.loads(path.read_text())
            self.assertEqual(state["admitted"], 2)
            state["credential_caps"] = {"TEST_KEY_A": 1, "TEST_KEY_B": 1}
            path.write_text(json.dumps(state))
            with self.assertRaises(OSError):
                gate.reserve(1)

    def test_rotated_key_is_redacted_from_provider_error(self):
        import os
        import urllib.error
        from unittest.mock import patch

        from ..llm import HTTPModel, TransportError
        from ..request_control import RotatingRequestGate

        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.dict(
                os.environ, {"TEST_KEY_A": "secret-one", "TEST_KEY_B": "secret-two"}
            ),
        ):

            def error(request, timeout):
                raise urllib.error.HTTPError(
                    request.full_url, 500, "failed", {}, io.BytesIO(b"bad secret-two")
                )

            gate = RotatingRequestGate(
                Path(tmp) / "requests.json",
                credential_limits={"TEST_KEY_B": 1},
                interval_seconds=0,
                opener=error,
            )
            model = HTTPModel(ModelConfig(api_key_env="TEST_KEY_A"), opener=gate)
            with self.assertRaises(TransportError) as caught:
                model.generate("pass", 1)
            self.assertNotIn("secret-two", str(caught.exception))
            self.assertIn("[REDACTED]", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
