"""Unthrottled CaP events and text output layered on the existing RunLogger."""

import json
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .contracts import GoalHandle, ObjectHandle
from .timing import timed


def json_value(value):
    if isinstance(value, ObjectHandle):
        return {"object_handle": value.token}
    if isinstance(value, GoalHandle):
        return {"goal_handle": value.token}
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return f"<{type(value).__name__}>"


class CapLogger:
    def __init__(
        self, run_dir, *, run=None, trial_id="trial_001", console=True, timings=None
    ):
        self.timings = timings
        self.run_dir = Path(run_dir).resolve()
        self.root = self.run_dir / "artifacts" / trial_id
        self.root.mkdir(parents=True, exist_ok=True)
        self.trial_id, self.run, self.console = trial_id, run, console
        self.timeline = (self.root / "timeline.jsonl").open("a", buffering=1)
        self.started = time.monotonic()
        self.sequence = 0
        self.context = {}
        self.prompt_provenance = {}
        self.source_generations = {}
        self.state_context = {}
        self.paths = {}
        self.emit("trial_log_start", run_directory=str(self.run_dir))

    def relative(self, path):
        return str(Path(path).resolve().relative_to(self.run_dir))

    @timed("logging_io")
    def text(self, text):
        text = text.rstrip() + "\n"
        if self.run is not None and self.run._saved_fds is not None:
            print(text, end="", flush=True)
            if self.console:
                self.run.progress_line(text)
        else:
            with (self.run_dir / "stdout.log").open("a") as handle:
                handle.write(text)
            if self.console:
                print(text, end="", flush=True)

    @timed("logging_io")
    def emit(self, kind, **fields):
        self.sequence += 1
        event = dict(self.context)
        event.update(self.state_context)
        event.update(fields)
        event.update(
            event_id=self.sequence,
            kind=kind,
            trial_id=self.trial_id,
            run_id=self.run_dir.name,
            ts=datetime.now(timezone.utc).isoformat(),
            elapsed_seconds=time.monotonic() - self.started,
        )
        line = json.dumps(event, default=json_value, allow_nan=False)
        self.timeline.write(line + "\n")
        self.text(line)
        if self.run:
            details = {
                key: event[key] for key in ("checkpoint_id", "helper") if key in event
            }
            self.run.set_status(
                phase=kind,
                trial_id=self.trial_id,
                last_cap_event=self.sequence,
                **details,
            )
        return self.sequence

    @contextmanager
    def scoped(self, **fields):
        old = self.context
        self.context = old | fields
        try:
            yield
        finally:
            self.context = old

    @timed("logging_io")
    def artifact(self, relative, value, *, print_content=False):
        path = self.root / relative
        if not path.resolve().is_relative_to(self.root.resolve()):
            raise ValueError("Artifact path escapes the trial directory")
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(f"Immutable artifact already exists: {path}")
        content = (
            value
            if isinstance(value, str)
            else json.dumps(value, indent=2, default=json_value, allow_nan=False) + "\n"
        )
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(content)
        temporary.replace(path)
        self.register(path)
        if print_content:
            self.text(content)
        return path

    def register(self, path, status="completed", **metadata):
        relative = self.relative(path)
        self.paths[relative] = {"status": status, **metadata}
        self.emit("artifact", path=relative, status=status, **metadata)
        return relative

    @timed("logging_io")
    def close(self):
        if self.timeline.closed:
            return
        path = self.root / "manifest.json"
        path.write_text(
            json.dumps({"trial_id": self.trial_id, "artifacts": self.paths}, indent=2)
            + "\n"
        )
        self.emit("trial_log_end", manifest_path=self.relative(path))
        self.timeline.close()
