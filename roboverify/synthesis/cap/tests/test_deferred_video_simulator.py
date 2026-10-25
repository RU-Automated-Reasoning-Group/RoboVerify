"""Opt-in physical comparison; no model requests or policy search."""

import json
import os
import subprocess
import time
import unittest
from pathlib import Path

import numpy as np

from synthesis.cfg.recordings import load_traces

from ..batch import METHODS, create_batch, execute_batch
from ..config import GPUConfig, ModelConfig, RecoveryConfig, RunConfig
from ..deferred_video import atomic_json, read

POLICY = """top = b0
while exists(lambda x: not ON_star(x, b0)):
    selected = get(lambda x: not ON_star(x, b0) and forall(lambda y: eq(y, x) or not ON_star(y, x)))
    Pick(selected)
    Move(selected, selected, top, [0, 0, 0.10])
    Move(b0, b0, top, [0, 0, 0.10])
    Move(b0, b0, top, [0, 0, 0.05])
    Release(selected, 0.15)
    top = selected"""


def canonical_feedback(run):
    value = read(Path(run) / "artifacts/trial_001/rounds/000/feedback.json")
    mapping = value["evidence"]["object_handle_to_physical_id"]

    def normalize(item):
        if isinstance(item, dict):
            return {normalize(k): normalize(v) for k, v in item.items()}
        if isinstance(item, list):
            return [normalize(v) for v in item]
        if isinstance(item, str) and item in mapping:
            return f"physical_{mapping[item]}"
        return item

    return normalize(value)


@unittest.skipUnless(
    os.environ.get("CAP_DEFERRED_GPU_UUIDS"), "Opt-in GPU/video replay validation"
)
class DeferredSimulatorTests(unittest.TestCase):
    def test_three_methods_recovery_video_modes(self):
        root = Path(os.environ["CAP_DEFERRED_OUTPUT"]).resolve()
        snapshot = os.environ["CAP_DEFERRED_SNAPSHOT"]
        allowlist = os.environ["CAP_DEFERRED_GPU_UUIDS"].split(",")
        rows, timings = {}, {}
        for mode in ("inline", "deferred", "off"):
            batch = root / mode
            create_batch(
                batch,
                RunConfig(
                    prompt_version="typed_v3",
                    video_mode=mode,
                    recovery=RecoveryConfig(enabled=True),
                ),
                ModelConfig(max_output_tokens=16384),
                initial_state=snapshot,
                modes=("dsl_only",),
                trials=1,
                max_workers=3,
                render_workers=2,
                responses=["Pick(b0)\nMove(b0,b0,b0,[10,0,0.2])", POLICY],
                gpu_config=GPUConfig(
                    allowlist,
                    str(root / "reservations"),
                    max_workers_per_gpu=2,
                    min_free_memory_mb=4096,
                    worker_memory_mb=1024,
                    gpu_poll_interval_seconds=0.3,
                    gpu_sample_window_seconds=1,
                ),
            )
            started = time.monotonic()
            result = execute_batch(batch)
            metadata = read(batch / "execution_metadata.json")
            timings[mode] = {
                "all_artifacts_seconds": time.monotonic() - started,
                "experiment_seconds": metadata["experiment_finished_epoch"]
                - metadata["sessions"][0]["started_epoch"],
                "peak_combined_workers": metadata["peak_combined_workers"],
            }
            self.assertEqual(result["finished"], 3)
            self.assertLessEqual(metadata["peak_combined_workers"], 3)
            self.assertTrue(all(r["success"] for r in result["trials"]))
            if mode == "deferred":
                self.assertEqual(result["video_rendering"]["counts"], {"completed": 6})
            rows[mode] = {r["method"]: r for r in result["trials"]}
        checks = []
        for method in METHODS:
            self.assertEqual(
                canonical_feedback(rows["inline"][method]["run_dir"]),
                canonical_feedback(rows["deferred"][method]["run_dir"]),
            )
            for index in (0, 1):
                directories = {
                    mode: Path(rows[mode][method]["attempts"][index]["run_dir"])
                    / "artifacts/trial_001/execution/rollout_001"
                    for mode in rows
                }
                traces = {
                    mode: load_traces(path / "trajectory.npz")[0]
                    for mode, path in directories.items()
                }
                reference = traces["inline"]
                record = {
                    "method": method,
                    "round": index,
                    "frames": len(reference.states),
                }
                for mode in ("deferred", "off"):
                    trace = traces[mode]
                    np.testing.assert_array_equal(
                        reference.actions[0], trace.actions[0]
                    )
                    np.testing.assert_allclose(
                        reference.states, trace.states, atol=1e-8, rtol=0
                    )
                    record[mode + "_max_observation_error"] = float(
                        np.max(np.abs(np.array(reference.states) - trace.states))
                    )
                    self.assertEqual(len(reference.snapshots), len(trace.snapshots))
                    for a, b in zip(reference.snapshots, trace.snapshots):
                        np.testing.assert_allclose(
                            a.gt_state, b.gt_state, atol=1e-8, rtol=0
                        )
                        self.assertEqual(a.bindings, b.bindings)
                        for name in a.arrays:
                            tolerance = 1e-6 if name == "qacc_warmstart" else 1e-8
                            np.testing.assert_allclose(
                                a.arrays[name], b.arrays[name], atol=tolerance, rtol=0
                            )
                np.testing.assert_array_equal(
                    traces["deferred"].states, traces["off"].states
                )
                frames = []
                for mode in ("inline", "deferred"):
                    frames.append(
                        np.frombuffer(
                            subprocess.check_output(
                                [
                                    "ffmpeg",
                                    "-v",
                                    "error",
                                    "-i",
                                    str(directories[mode] / "video.mp4"),
                                    "-f",
                                    "rawvideo",
                                    "-pix_fmt",
                                    "rgb24",
                                    "-threads",
                                    "1",
                                    "-",
                                ]
                            ),
                            dtype=np.uint8,
                        )
                    )
                self.assertEqual(frames[0].shape, frames[1].shape)
                delta = np.abs(frames[0].astype(float) - frames[1].astype(float))
                record.update(
                    pixel_mean_absolute_error=float(delta.mean()),
                    pixel_max_absolute_error=float(delta.max()),
                )
                self.assertLessEqual(record["pixel_mean_absolute_error"], 1.0)
                self.assertLessEqual(record["pixel_max_absolute_error"], 8.0)
                checks.append(record)
        self.assertEqual(
            read(root / "reservations/reservations.json")["reservations"], {}
        )
        atomic_json(
            root / "validation.json",
            {
                "trials": 9,
                "attempts": 18,
                "videos": 12,
                "normalized_inline_deferred_feedback_equal": True,
                "timing_scope": "one scripted batch per mode; not a statistical throughput estimate",
                "timings": timings,
                "parity": checks,
                "reservations_released": True,
            },
        )


if __name__ == "__main__":
    unittest.main()
