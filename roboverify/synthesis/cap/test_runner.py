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
            condition_mode=options.pop("condition_mode", "dsl_only"),
            task=options.pop("task", "stack"),
            prompt_version=options.pop("prompt_version", "legacy_v1"),
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
                **options,
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

    def test_four_combinations_share_position_reads_actions_and_mode_metadata(self):
        import numpy as np

        source = "p=get_block_position(b0)\ng=get_gripper_position()\nPick(b0)\nduring=get_gripper_position()\nMove(b0,b0,b0,[0,0,g[2]-p[2]])\nafter=get_gripper_position()"
        actions, reads, snapshot = [], [], None
        for method in ("cap_google_baseline", "cap_method_1"):
            for mode in ("dsl_only", "coordinate_predicates"):
                with self.subTest(method=method, mode=mode):
                    result = self.execute(
                        method, [source], condition_mode=mode, initial_state=snapshot
                    )
                    self.assertEqual(result["status"], "completed")
                    self.assertEqual(result["condition_mode"], mode)
                    directory = Path(result["run_dir"])
                    root = directory / "artifacts/trial_001"
                    if snapshot is None:
                        snapshot = root / "initial_state.npz"
                    events = [
                        json.loads(line)
                        for line in (root / "timeline.jsonl").read_text().splitlines()
                    ]
                    reads.append(
                        [e["result"] for e in events if e["kind"] == "coordinate_query"]
                    )
                    actions.append(np.array(self.envs[-1].actions))
                    self.assertIn(
                        f"condition_mode: {mode}", RunSummary(directory).render()
                    )
                    for path in root.glob("checkpoints/*/manifest.json"):
                        manifest = json.loads(path.read_text())
                        self.assertEqual(manifest["condition_mode"], mode)
                        self.assertEqual(
                            manifest["validation_enabled"], method == "cap_method_1"
                        )
        for action in actions[1:]:
            np.testing.assert_array_equal(actions[0], action)
        for read in reads[1:]:
            self.assertEqual(reads[0], read)
        self.assertNotEqual(reads[0][1], reads[0][2])

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

    def test_table_tasks_bind_tbl_for_every_method(self):
        source = "if ON_star(tbl, tbl):\n    Pick(b0)"
        for method in ("cap_method_1", "cap_method_2", "cap_google_baseline"):
            with self.subTest(method=method):
                result = self.execute(
                    method, [source], task="unstack", prompt_version="typed_v3"
                )
                self.assertEqual(result["status"], "completed")
                root = Path(result["run_dir"])
                config = RunSummary(root).config["config"]
                self.assertIn("tbl is the table", config["input_context"])
                self.assertIn(["tbl", "object"], config["dsl_contract"]["input_types"])
                adapter_config = json.loads(
                    (root / "artifacts/trial_001/adapter_config.json").read_text()
                )
                self.assertEqual(
                    adapter_config["table_handle"]["version"], "cap-table-handle-v1"
                )
        stack = self.execute("cap_method_1", [source] * 3)
        self.assertEqual(stack["status"], "constraint_failed")

    def test_task_spec_success_follows_shared_pre_and_postcondition(self):
        result = self.execute("cap_method_1", ["pass"])
        self.assertTrue(result["task_spec_success"])
        self.assertEqual(result["success_criterion"], "task-spec-v1")
        self.assertTrue(result["task_spec_precondition"])
        self.assertTrue(result["task_spec_postcondition"])
        self.assertTrue(result["settled_environment_success"])
        self.assertEqual(result["unconverged_primitives"], 0)

        config = RunConfig(num_blocks=2, save_video=False, output_root=self.tmp.name)
        with contextlib.redirect_stdout(io.StringIO()):
            scattered = run(
                config,
                ModelConfig(),
                backend=ScriptedModel(["pass"]),
                environment_factory=self.factory,
                capture_stdout=False,
            )
        # Physically "successful" fixture, but the blocks never form a tower.
        self.assertTrue(scattered["success"])
        self.assertFalse(scattered["task_spec_postcondition"])
        self.assertFalse(scattered["task_spec_success"])

    def test_task_spec_success_requires_normal_completion(self):
        result = self.execute("cap_google_baseline", ["pass\nmissing_value"])
        self.assertEqual(result["status"], "failed")
        self.assertTrue(result["task_spec_postcondition"])
        self.assertFalse(result["task_spec_success"])

    def test_caught_primitive_failure_is_not_task_spec_success(self):
        result = self.execute(
            "cap_google_baseline",
            ["try:\n    Move(b0,b0,b0,[99,99,99])\nexcept Exception:\n    pass"],
        )
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["unconverged_primitives"], 1)
        self.assertFalse(result["task_spec_success"])

    def test_generation_diagnostic_has_no_task_spec_outcome(self):
        result = self.execute("cap_method_1", ["Pick(b0)"], generate_only=True)
        self.assertIsNone(result["task_spec_success"])

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


class TaskSpecEvaluationTests(unittest.TestCase):
    """Shared task_spec evaluation for tasks whose formulas include the table."""

    def evaluate(self, task, entry_positions, final_positions):
        from .tasks import evaluate_terminal

        env = RobotFixture(len(entry_positions), success=False)
        for i, position in enumerate(entry_positions):
            env.obs[10 + 12 * i : 13 + 12 * i] = position
        entry = env.obs.copy()
        for i, position in enumerate(final_positions):
            env.obs[10 + 12 * i : 13 + 12 * i] = position
        return evaluate_terminal(env, task, entry)

    def test_reverse_uses_frozen_entry_order(self):
        tower = [[0, 0, 0.025], [0, 0, 0.075]]
        reversed_tower = [[0, 0, 0.075], [0, 0, 0.025]]
        result = self.evaluate("reverse", tower, reversed_tower)
        self.assertTrue(result["task_spec_precondition"])
        self.assertTrue(result["task_spec_postcondition"])
        # Earlier results had no DSL postcondition for Reverse; keep it that way.
        self.assertIsNone(result["dsl_postcondition"])
        self.assertFalse(
            self.evaluate("reverse", tower, tower)["task_spec_postcondition"]
        )

    def test_restack_requires_every_block_over_b0(self):
        loose = [[0, 0, 0.025], [0.2, 0, 0.025]]
        stacked = [[0, 0, 0.025], [0, 0, 0.075]]
        result = self.evaluate("restack", loose, stacked)
        self.assertTrue(result["task_spec_precondition"])
        self.assertTrue(result["task_spec_postcondition"])
        self.assertFalse(
            self.evaluate("restack", loose, loose)["task_spec_postcondition"]
        )


class StartValidationTests(unittest.TestCase):
    """Every tower task checks its shared precondition before any model call."""

    def validate(self, task, positions):
        from .tasks import validate_start

        env = RobotFixture(len(positions))
        for i, position in enumerate(positions):
            env.obs[10 + 12 * i : 13 + 12 * i] = position
        validate_start(env, task, env.obs.copy())

    def test_unstack_requires_one_tower_over_b0(self):
        from .contracts import CapError

        self.validate("unstack", [[0, 0, 0.025], [0, 0, 0.075]])
        with self.assertRaisesRegex(CapError, "shared task precondition"):
            self.validate("unstack", [[0, 0, 0.025], [0.2, 0, 0.025]])

    def test_restack_requires_b0_inside_a_source_tower(self):
        from .contracts import CapError

        self.validate("restack", [[0, 0, 0.025], [0, 0, 0.075], [0.2, 0, 0.025]])
        self.validate("restack", [[0, 0, 0.075], [0, 0, 0.025], [0.2, 0, 0.025]])
        with self.assertRaisesRegex(CapError, "not inside a source tower"):
            self.validate("restack", [[0.2, 0, 0.025], [0, 0, 0.025], [0, 0, 0.075]])


class EnvironmentOptionTests(unittest.TestCase):
    def test_settling_follows_upstream_collection(self):
        from .tasks import settling_steps

        for task, steps in (("stack", 50), ("restack", 50), ("unstack", 0)):
            self.assertEqual(settling_steps(RunConfig(task=task)), steps)
        self.assertEqual(settling_steps(RunConfig(task="reverse")), 0)
        self.assertEqual(settling_steps(RunConfig(), initial_snapshot=object()), 0)

    def test_layout_options_are_resolved_and_task_specific(self):
        from .tasks import environment_options

        self.assertEqual(
            environment_options(RunConfig(task="restack", num_blocks=4)),
            {"initial_tower_blocks": 3},
        )
        self.assertEqual(environment_options(RunConfig()), {})
        with self.assertRaises(ValueError):
            environment_options(
                RunConfig(task="restack", num_blocks=3, initial_tower_blocks=4)
            )
        with self.assertRaisesRegex(ValueError, "applies only to restack"):
            RunConfig(task="stack", initial_tower_blocks=2)
        with self.assertRaisesRegex(ValueError, "applies only to pyramid"):
            RunConfig(task="grid", bottom_row_blocks=3)


if __name__ == "__main__":
    unittest.main()
