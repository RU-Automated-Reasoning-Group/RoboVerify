import json
import tempfile
import unittest
from pathlib import Path

from .batch import atomic_json, create_batch
from .config import ModelConfig, RunConfig
from .revision43_prompt_experiment import HealthGate, create_rerun


class RerunTest(unittest.TestCase):
    def test_rerun_copies_selected_jobs_with_new_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = Path(tmp) / "state.npz"
            state.write_bytes(b"snapshot")
            source = Path(tmp) / "source"
            manifest = create_batch(
                source,
                RunConfig(num_blocks=1, save_video=False, prompt_version="typed_v3"),
                ModelConfig(),
                methods=("cap_method_1",),
                modes=("dsl_only",),
                trials=3,
                initial_state=state,
                responses=["Pick(b0)"],
                fixture=True,
                transport_control={"state_path": str(source / "request_gate.json")},
            )
            atomic_json(source / "comparison_plan.json", {"planned": 3})
            chosen = [manifest["jobs"][0]["id"], manifest["jobs"][2]["id"]]
            target = Path(tmp) / "rerun"
            rerun = create_rerun(target, source=source, job_ids=chosen, reason="test")
            self.assertEqual([j["id"] for j in rerun["jobs"]], chosen)
            for job, original in zip(rerun["jobs"], manifest["jobs"][::2]):
                self.assertEqual(
                    {k: v for k, v in job["run_config"].items() if k != "output_root"},
                    {
                        k: v
                        for k, v in original["run_config"].items()
                        if k != "output_root"
                    },
                )
                self.assertTrue(
                    job["run_config"]["output_root"].startswith(str(target))
                )
                self.assertTrue(job["initial_state"].startswith(str(target)))
                self.assertEqual(
                    job["transport_control"]["state_path"],
                    str(target / "request_gate.json"),
                )
            plan = json.loads((target / "comparison_plan.json").read_text())
            self.assertEqual(plan["planned"], 2)
            self.assertEqual(plan["rerun_job_ids"], sorted(chosen))
            with self.assertRaises(ValueError):
                create_rerun(
                    Path(tmp) / "bad", source=source, job_ids=["missing"], reason="x"
                )


class HealthGateTest(unittest.TestCase):
    def test_unreachable_endpoint_blocks_dispatch(self):
        calls = []
        gate = HealthGate(
            lambda job, active: calls.append(job) or True,
            "http://127.0.0.1:9/v1/models",
        )
        self.assertFalse(gate({"id": "a"}, []))
        self.assertEqual(gate.reason, "model_endpoint_unavailable")
        self.assertEqual(calls, [])

    def test_dispatch_limit_file_caps_active_trials_and_logs_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            limit = Path(tmp) / "dispatch_limit.json"
            gate = HealthGate(None, "http://127.0.0.1:9/v1/models", limit_path=limit)
            gate.healthy, gate.checked_at = True, float("inf")
            self.assertTrue(gate({"id": "a"}, [{}] * 5))
            limit.write_text(json.dumps({"max_active": 2}))
            self.assertFalse(gate({"id": "a"}, [{}, {}]))
            self.assertEqual(gate.reason, "dispatch_limit")
            self.assertTrue(gate({"id": "a"}, [{}]))
            limit.write_text("not json")
            self.assertFalse(gate({"id": "a"}, [{}, {}]))
            limit.write_text(json.dumps({"max_active": 0}))
            self.assertFalse(gate({"id": "a"}, []))
            events = [
                json.loads(line)["max_active"]
                for line in (Path(tmp) / "dispatch_limit_events.jsonl")
                .read_text()
                .splitlines()
            ]
            self.assertEqual(events, [2, 0])


if __name__ == "__main__":
    unittest.main()
