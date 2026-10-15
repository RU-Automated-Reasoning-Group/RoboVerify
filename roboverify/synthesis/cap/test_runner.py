"""Trial outcomes, source artifacts, and exact archive replay without MuJoCo."""

import contextlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from synthesis.cfg.recordings import load_traces
from synthesis.cfg.reset import restore
from synthesis.experiment.report import RunSummary

from .config import ModelConfig, RunConfig
from .llm import ScriptedModel
from .runner import run
from .test_adapter import RobotFixture


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.envs = []

    def tearDown(self):
        self.tmp.cleanup()

    def factory(self, config, snapshot):
        env = RobotFixture(config.num_blocks, success=True)
        if snapshot:
            restore(env, snapshot)
        self.envs.append(env)
        return env

    def execute(self, method, responses, **options):
        config = RunConfig(
            method=method,
            num_blocks=1,
            save_video=options.pop("save_video", False),
            output_root=self.tmp.name,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            return run(
                config,
                ModelConfig(),
                backend=ScriptedModel(responses),
                environment_factory=self.factory,
                capture_stdout=False,
                **options
            )

    def test_both_methods_record_the_shared_primitive_path(self):
        for method in ["cap_method_1", "cap_google_baseline"]:
            with self.subTest(method=method):
                result = self.execute(method, ["Pick(b0)"])
                self.assertTrue(result["success"])
                root = Path(result["run_dir"])
                trace_path = (
                    root / "artifacts/trial_001/execution/rollout_001/trajectory.npz"
                )
                traces = load_traces(trace_path)
                self.assertEqual(len(traces[0].states), len(traces[0].actions[0]) + 1)
                self.assertGreater(len(traces[0].actions[0]), 0)
                self.assertTrue(self.envs[-1].closed)
                self.assertIn("primitive_end", (root / "stdout.log").read_text())
                self.assertIn(method, RunSummary(root).render())

    def test_failure_preserves_partial_recording_and_cannot_become_success(self):
        result = self.execute("cap_google_baseline", ["Pick(b0)\nmissing_value"])
        self.assertFalse(result["success"])
        self.assertTrue(result["environment_success"])
        root = Path(result["run_dir"]) / "artifacts/trial_001"
        self.assertTrue((root / "execution/traceback.txt").exists())
        trace = load_traces(root / "execution/rollout_001/trajectory.npz")[0]
        self.assertGreater(len(trace.actions[0]), 0)

    def test_exact_saved_initial_state_is_reused(self):
        first = self.execute("cap_method_1", ["pass"])
        path = Path(first["run_dir"]) / "artifacts/trial_001/initial_state.npz"
        second = self.execute("cap_google_baseline", ["pass"], initial_state=path)
        new_path = Path(second["run_dir"]) / "artifacts/trial_001/initial_state.npz"
        import numpy as np

        np.testing.assert_array_equal(
            load_traces(path)[0].states[0], load_traces(new_path)[0].states[0]
        )

    def test_generation_diagnostic_does_not_report_rollout_success(self):
        result = self.execute("cap_method_1", ["Pick(b0)"], generate_only=True)
        self.assertEqual(result["status"], "generated")
        self.assertIsNone(result["success"])
        self.assertEqual(self.envs, [])

    def test_generation_failure_reports_the_invalid_checkpoint(self):
        result = self.execute(
            "cap_method_1", ["if Pick(b0):\n    pass"] * 3, generate_only=True
        )
        self.assertEqual(result["status"], "constraint_failed")
        self.assertIsNone(result["success"])
        self.assertIsNotNone(result["checkpoint_id"])
        path = (
            Path(result["run_dir"])
            / "artifacts/trial_001/checkpoints"
            / result["checkpoint_id"]
            / "validation.json"
        )
        self.assertEqual(json.loads(path.read_text())["status"], "invalid")

    def test_query_after_primitive_references_current_state(self):
        result = self.execute("cap_method_1", ["Pick(b0)\nif Higher(b0,b0):\n    pass"])
        root = Path(result["run_dir"]) / "artifacts/trial_001"
        trace = load_traces(root / "execution/rollout_001/trajectory.npz")[0]
        events = [
            json.loads(line)
            for line in (root / "timeline.jsonl").read_text().splitlines()
        ]
        query = next(e for e in events if e["kind"] == "predicate")
        self.assertEqual(query["state_index"], len(trace.states) - 1)

    def test_goal_adapter_gap_is_explicit(self):
        config = RunConfig(task="grid", output_root=self.tmp.name)
        with contextlib.redirect_stdout(io.StringIO()):
            result = run(
                config, ModelConfig(), backend=ScriptedModel([]), capture_stdout=False
            )
        self.assertEqual(result["status"], "adapter_gap")
        self.assertEqual(result["model_calls"], 0)
        self.assertIsNone(result["success"])

    def test_environment_and_dsl_scores_are_distinct(self):
        config = RunConfig(num_blocks=2, save_video=False, output_root=self.tmp.name)
        with contextlib.redirect_stdout(io.StringIO()):
            result = run(
                config,
                ModelConfig(),
                backend=ScriptedModel(["pass"]),
                environment_factory=self.factory,
                capture_stdout=False,
            )
        # The fixture's evaluator deliberately reports true independently of geometry.
        self.assertTrue(result["success"])
        self.assertFalse(result["dsl_postcondition"])

    @unittest.skipUnless(
        shutil.which("ffmpeg"), "ffmpeg is required for video integration"
    )
    def test_failed_rollout_keeps_video_with_frame_to_state_index(self):
        result = self.execute(
            "cap_google_baseline", ["Pick(b0)\nmissing_value"], save_video=True
        )
        root = Path(result["run_dir"]) / "artifacts/trial_001/execution/rollout_001"
        self.assertGreater((root / "video.mp4").stat().st_size, 0)
        frames = [
            json.loads(line)
            for line in (root / "frame_index.jsonl").read_text().splitlines()
        ]
        trace = load_traces(root / "trajectory.npz")[0]
        self.assertEqual(len(frames), len(trace.states))
        self.assertEqual([f["frame"] for f in frames], list(range(len(frames))))
        self.assertEqual(frames[0]["step"], 0)
        self.assertIn("checkpoint_id", frames[-1])


if __name__ == "__main__":
    unittest.main()
