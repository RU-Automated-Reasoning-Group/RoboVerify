"""One process owns one trial; only the parent owns batch summaries."""

import json
import sys
import time
from pathlib import Path

from .batch import atomic_json
from .config import ModelConfig, RunConfig
from .llm import HTTPModel, ScriptedModel


def main(path):
    spec = json.loads(Path(path).read_text())
    assignment = spec.get("gpu_assignment")
    if assignment:
        from .gpu_scheduler import initialize_worker, rendering_probe

        try:
            evidence = initialize_worker(assignment, path)
            if (
                not spec.get("fixture")
                and RunConfig(**spec["run_config"]).video_mode == "inline"
            ):
                evidence["render_probe"] = rendering_probe()
            else:
                evidence["render_probe"] = "skipped: no inline rendering"
            atomic_json(Path(path).with_name("gpu_probe.json"), evidence)
        except Exception as exc:
            atomic_json(
                Path(path).with_name("worker_result.json"),
                {
                    "status": "gpu_initialization_failed",
                    "success": False,
                    "reason": f"{type(exc).__name__}: {exc}",
                    "model_calls": 0,
                    "known_tokens": 0,
                    "gpu_assignment": assignment,
                },
            )
            raise
    # Device assignment and mapping checks precede imports of trial machinery.
    from .runner import run

    config = RunConfig(**spec["run_config"])
    model = ModelConfig(**spec["model_config"])
    backend = (
        ScriptedModel(spec["responses"]) if spec.get("responses") is not None else None
    )
    if backend is None and spec.get("transport_control"):
        from .request_control import RotatingRequestGate, SharedRequestGate

        gate = (
            RotatingRequestGate
            if "credential_limits" in spec["transport_control"]
            else SharedRequestGate
        )
        backend = HTTPModel(
            model,
            opener=gate(**spec["transport_control"], job_id=spec["id"]),
        )
    factory = None
    if spec.get("fixture"):
        from .tests.test_adapter import RobotFixture

        def factory(config, snapshot):
            from synthesis.cfg.reset import restore

            env = RobotFixture(config.num_blocks, success=True)
            if snapshot:
                restore(env, snapshot)
            return env

    started = time.time()
    atomic_json(Path(path).with_name("worker_start.json"), {"started_epoch": started})
    result = run(
        config,
        model,
        instruction=spec.get("instruction"),
        backend=backend,
        initial_state=spec.get("initial_state"),
        environment_factory=factory,
        queue_wait_seconds=max(0.0, started - spec["enqueued_epoch"]),
        batch_context={
            "batch_dir": spec["batch_dir"],
            "job_id": spec["id"],
            "trial": spec["trial"],
            "variant": spec.get("variant", "default"),
            "transport_control": spec.get("transport_control"),
            "gpu_assignment": assignment,
        },
        **({"seed_attempt": spec["seed_attempt"]} if spec.get("seed_attempt") else {}),
    )
    atomic_json(Path(path).with_name("worker_result.json"), result)


if __name__ == "__main__":
    main(sys.argv[1])
