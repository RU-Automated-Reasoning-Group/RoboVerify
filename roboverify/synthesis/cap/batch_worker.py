"""One process owns one trial; only the parent owns batch summaries."""

import json
import sys
import time
from pathlib import Path

from .batch import atomic_json
from .config import ModelConfig, RunConfig
from .llm import ScriptedModel
from .runner import run


def main(path):
    spec = json.loads(Path(path).read_text())
    config = RunConfig(**spec["run_config"])
    model = ModelConfig(**spec["model_config"])
    backend = (
        ScriptedModel(spec["responses"]) if spec.get("responses") is not None else None
    )
    factory = None
    if spec.get("fixture"):
        from .test_adapter import RobotFixture

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
        },
    )
    atomic_json(Path(path).with_name("worker_result.json"), result)


if __name__ == "__main__":
    main(sys.argv[1])
