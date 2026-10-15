"""Record the same CaP execution's actions, full states, and optional video."""

import json
import sys
import time

from .contracts import BudgetExceeded
from .logger import json_value


class Recorder:
    def __init__(self, env, logger, budget, *, video=False, seed=0, task="stack"):
        self.env, self.logger, self.budget = env, logger, budget
        self.seed, self.task = seed, task
        self.directory = logger.root / "execution" / "rollout_001"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.steps = (self.directory / "steps.jsonl").open("a", buffering=1)
        self.frames = (self.directory / "frame_index.jsonl").open("a", buffering=1)
        self.states, self.snapshots, self.actions = [], [], []
        self.original_step = env.step
        self.video = None
        self.video_error = None
        self.video_metadata = {"status": "pending" if video else "disabled"}
        if video:
            from synthesis.cfg.collection import VideoRecorder

            try:
                self.video = VideoRecorder(self.directory / "video.mp4")
            except Exception as exc:
                self.video_error = str(exc)
                self.video_metadata = {"status": "failed", "reason": str(exc)}
                logger.emit("video_error", error=str(exc))
                logger.register(
                    self.directory / "video.mp4", "missing", reason=str(exc)
                )

    def state(self):
        from synthesis.cfg.reset import capture, inner_env

        inner = inner_env(self.env)
        self.states.append(inner.flatten_observation(inner._get_obs()).copy())
        self.snapshots.append(capture(self.env))
        self.logger.state_context.update(
            state_index=len(self.states) - 1, sim_step=len(self.actions)
        )
        if self.video is not None and not self.video.error:
            try:
                self.video.append(inner.render(mode="rgb_array"))
                if self.video.error:
                    self.logger.emit("video_error", error=self.video.error)
                    return
                row = self.logger.context | {
                    "frame": self.video.frames - 1,
                    "state_index": len(self.states) - 1,
                    "step": len(self.actions),
                    "elapsed_seconds": time.monotonic() - self.logger.started,
                }
                self.frames.write(json.dumps(row, default=json_value) + "\n")
            except (BudgetExceeded, TimeoutError):
                raise
            except Exception as exc:
                self.video_error = str(exc)
                self.video.error = str(exc)
                self.logger.emit("video_error", error=str(exc))

    def __enter__(self):
        try:
            self.state()
            self.save(self.logger.root / "initial_state.npz", "initial")
            self.env.step = self.step
            return self
        except BaseException:
            self.__exit__(*sys.exc_info())
            raise

    def step(self, action):
        import numpy as np

        self.budget.tick()
        result = self.original_step(action)
        self.actions.append(np.array(action, copy=True))
        self.state()
        row = self.logger.context | {
            "step": len(self.actions),
            "state_index": len(self.states) - 1,
            "action": action,
            "reward": result[1],
            "terminated": result[2],
            "truncated": result[3],
            "info": result[4],
        }
        self.steps.write(json.dumps(row, default=json_value, allow_nan=False) + "\n")
        self.logger.emit("sim_step", **row)
        return result

    def save(self, path, status):
        from synthesis.cfg.demos import DemoTrace
        from synthesis.cfg.recordings import save_traces

        count = (len(self.states[0]) - 13) // 15
        trace = DemoTrace(
            tuple(self.states),
            tuple(self.snapshots),
            (tuple(self.actions), tuple(range(len(self.states)))),
            self.seed,
            self.task,
            count,
            (),
            {
                "status": status,
                "source_kind": "cap",
                "checkpoint_id": self.logger.context.get("checkpoint_id"),
            },
        )
        save_traces(path, [trace])
        self.logger.register(path)

    def __exit__(self, exc_type, exc, tb):
        self.env.step = self.original_step
        try:
            if self.states:
                self.save(
                    self.directory / "trajectory.npz",
                    "completed" if exc is None else "failed",
                )
            if self.video is not None:
                try:
                    self.budget.remaining()
                except Exception:
                    # Avoid the existing encoder's 15-second wait after the trial deadline.
                    if self.video.process is not None:
                        self.video.process.kill()
                    self.video.error = (
                        self.video.error or "Trial deadline exhausted during recording"
                    )
                metadata = self.video.close()
                self.video_metadata = metadata
                self.logger.emit("video_end", metadata=metadata)
                self.logger.register(
                    self.video.path,
                    "completed" if not self.video.error else "failed",
                    **{k: v for k, v in metadata.items() if k not in {"path", "status"}}
                )
                encoder = self.video.path.with_suffix(".encoder.log")
                if encoder.exists():
                    self.logger.register(encoder)
        finally:
            self.steps.close()
            self.frames.close()
            self.logger.register(self.directory / "steps.jsonl")
            self.logger.register(self.directory / "frame_index.jsonl")
