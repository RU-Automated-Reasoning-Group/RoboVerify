"""Optional process-shared admission pacing for rate-limited model routes."""

import fcntl
import json
import math
import os
import time
import urllib.request
from pathlib import Path


class SharedRequestGate:
    """Reserve request starts atomically; never store prompts or credentials."""

    def __init__(
        self,
        state_path,
        *,
        interval_seconds=6.0,
        request_limit=800,
        job_id="probe",
        opener=None,
    ):
        if not math.isfinite(interval_seconds) or interval_seconds < 0:
            raise ValueError("Request interval must be finite and nonnegative")
        if type(request_limit) is not int or request_limit < 1:
            raise ValueError("Request limit must be a positive integer")
        self.path = Path(state_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.interval, self.limit, self.job_id = interval_seconds, request_limit, job_id
        self.opener = opener or urllib.request.urlopen

    def reserve(self, timeout):
        started = time.monotonic()
        with self.path.with_suffix(".lock").open("a") as lock:
            while True:
                remaining = timeout - (time.monotonic() - started)
                if remaining <= 0:
                    raise TimeoutError("Request pacing exhausted the request timeout")
                fcntl.flock(lock, fcntl.LOCK_EX)
                try:
                    state = (
                        json.loads(self.path.read_text())
                        if self.path.exists()
                        else {"admitted": 0, "by_job": {}, "next_at": 0.0}
                    )
                    if state["admitted"] >= self.limit:
                        raise OSError("Shared batch request allowance exhausted")
                    wait = state["next_at"] - time.time()
                    if wait <= 0:
                        now = time.time()
                        state["admitted"] += 1
                        state["by_job"][self.job_id] = (
                            state["by_job"].get(self.job_id, 0) + 1
                        )
                        state["next_at"] = now + self.interval
                        temporary = self.path.with_suffix(".tmp")
                        temporary.write_text(json.dumps(state, sort_keys=True) + "\n")
                        temporary.replace(self.path)
                        with self.path.with_suffix(".events.jsonl").open("a") as events:
                            events.write(
                                json.dumps(
                                    {
                                        "request": state["admitted"],
                                        "job_id": self.job_id,
                                        "pid": os.getpid(),
                                        "admitted_epoch": now,
                                        "wait_seconds": time.monotonic() - started,
                                    }
                                )
                                + "\n"
                            )
                        return timeout - (time.monotonic() - started)
                finally:
                    fcntl.flock(lock, fcntl.LOCK_UN)
                time.sleep(min(wait, remaining, 0.2))

    def __call__(self, request, timeout):
        remaining = self.reserve(timeout)
        if remaining <= 0:
            raise TimeoutError("Request pacing exhausted the request timeout")
        return self.opener(request, timeout=remaining)


def request_allowance(job):
    config = job["run_config"]
    recovery = config.get("recovery", {})
    return (
        recovery["total_calls"]
        if recovery.get("enabled")
        else config["max_model_calls"]
    )


def dispatch_reservation(job, active_jobs, state, *, request_limit, free_remaining):
    """Keep enough calls for every active trial; do not cut a trial off mid-rollout."""
    by_job = state.get("by_job", {})
    reserved = sum(
        max(0, request_allowance(active) - by_job.get(active["id"], 0))
        for active in active_jobs
    )
    needed = reserved + request_allowance(job)
    return {
        "allowed": needed
        <= min(request_limit - state.get("admitted", 0), free_remaining),
        "active_reserved_calls": reserved,
        "candidate_reserved_calls": request_allowance(job),
        "required_remaining_calls": needed,
        "batch_admitted_calls": state.get("admitted", 0),
        "free_remaining_calls": free_remaining,
        "batch_request_limit": request_limit,
    }


class RotatingRequestGate(SharedRequestGate):
    """Rotate authorized credentials under separate pacing/caps and one wire ledger.

    Configuration and artifacts contain environment-variable names only. Caps are
    supplied from verified provider quotas; this class never increases them.
    """

    def __init__(self, state_path, *, credential_limits, **kwargs):
        import re

        super().__init__(state_path, **kwargs)
        if not credential_limits or any(
            not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name)
            or type(limit) is not int
            or limit < 0
            for name, limit in credential_limits.items()
        ):
            raise ValueError(
                "Credential limits must map environment names to nonnegative request caps"
            )
        self.credential_limits = dict(credential_limits)
        self.credential_envs = list(credential_limits)

    def reserve(self, timeout):
        started = time.monotonic()
        with self.path.with_suffix(".lock").open("a") as lock:
            while True:
                remaining = timeout - (time.monotonic() - started)
                if remaining <= 0:
                    raise TimeoutError(
                        "Credential pacing exhausted the request timeout"
                    )
                fcntl.flock(lock, fcntl.LOCK_EX)
                try:
                    state = (
                        json.loads(self.path.read_text())
                        if self.path.exists()
                        else {
                            "admitted": 0,
                            "by_job": {},
                            "by_credential": {},
                            "credential_next_at": {},
                            "rotation_cursor": 0,
                        }
                    )
                    if state["admitted"] >= self.limit:
                        raise OSError("Shared batch request allowance exhausted")
                    names = self.credential_envs
                    cursor = state.get("rotation_cursor", 0) % len(names)
                    order = names[cursor:] + names[:cursor]
                    candidates = []
                    for index, name in enumerate(order):
                        cap = min(
                            self.credential_limits[name],
                            state.get("credential_caps", {}).get(
                                name, self.credential_limits[name]
                            ),
                        )
                        if state.get("by_credential", {}).get(name, 0) < cap:
                            if not os.environ.get(name):
                                raise OSError(
                                    f"Missing credential environment variable: {name}"
                                )
                            candidates.append(
                                (
                                    state.get("credential_next_at", {}).get(name, 0),
                                    index,
                                    name,
                                )
                            )
                    if not candidates:
                        raise OSError(
                            "All configured credential request allowances exhausted"
                        )
                    next_at, _, selected = min(candidates)
                    wait = next_at - time.time()
                    if wait <= 0:
                        now = time.time()
                        state["admitted"] += 1
                        state.setdefault("by_job", {})[self.job_id] = (
                            state["by_job"].get(self.job_id, 0) + 1
                        )
                        state.setdefault("by_credential", {})[selected] = (
                            state["by_credential"].get(selected, 0) + 1
                        )
                        state.setdefault("credential_next_at", {})[selected] = (
                            now + self.interval
                        )
                        state["rotation_cursor"] = (names.index(selected) + 1) % len(
                            names
                        )
                        temporary = self.path.with_suffix(".tmp")
                        temporary.write_text(json.dumps(state, sort_keys=True) + "\n")
                        temporary.replace(self.path)
                        with self.path.with_suffix(".events.jsonl").open("a") as events:
                            events.write(
                                json.dumps(
                                    {
                                        "request": state["admitted"],
                                        "job_id": self.job_id,
                                        "credential_env": selected,
                                        "pid": os.getpid(),
                                        "admitted_epoch": now,
                                        "wait_seconds": time.monotonic() - started,
                                    }
                                )
                                + "\n"
                            )
                        return timeout - (time.monotonic() - started), selected
                finally:
                    fcntl.flock(lock, fcntl.LOCK_UN)
                time.sleep(min(wait, remaining, 0.2))

    def __call__(self, request, timeout):
        remaining, name = self.reserve(timeout)
        if remaining <= 0:
            raise TimeoutError("Credential pacing exhausted the request timeout")
        request.add_header("Authorization", "Bearer " + os.environ[name])
        return self.opener(request, timeout=remaining)
