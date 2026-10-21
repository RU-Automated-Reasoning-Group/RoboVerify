"""Reproducible Revision 4.2 main/ablation batch, using existing isolated workers."""

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import time
import urllib.request
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .batch import METHODS, atomic_json, create_batch, execute_batch
from .config import (
    GPUConfig,
    ModelConfig,
    PrimitiveFeedbackConfig,
    PrimitivePromptConfig,
    RunConfig,
)
from .request_control import dispatch_reservation

PRIOR = Path("runs/cap-revision4-minimax31flash/20261020-152924-free-300")
GPU_UUIDS = [
    "GPU-4e45a91e-060f-7724-1e8d-c8e03021a651",
    "GPU-9e9f9166-3050-3b33-3900-55a1c3267159",
]
CREDENTIALS = [
    "OPENROUTER_API_KEY",
    "OPENROUTER_API_KEY_2",
    "OPENROUTER_API_KEY_3",
    "OPENROUTER_API_KEY_4",
]
PROFILES = ("full", "no_prompt", "no_target", "no_phases", "no_context")


def variants():
    full_prompt = dict.fromkeys(asdict(PrimitivePromptConfig()), True)
    result = []
    for name in PROFILES:
        prompt = dict(full_prompt)
        feedback = asdict(
            PrimitiveFeedbackConfig(
                target_derivation=True, controller_phases=True, action_context=True
            )
        )
        if name == "no_prompt":
            prompt = asdict(PrimitivePromptConfig())
        elif name != "full":
            feedback[
                {
                    "no_target": "target_derivation",
                    "no_phases": "controller_phases",
                    "no_context": "action_context",
                }[name]
            ] = False
        result.append(
            {
                "name": name,
                "methods": list(METHODS if name == "full" else METHODS[1:]),
                "run_config": {
                    "primitive_prompt": prompt,
                    "primitive_feedback": feedback,
                },
            }
        )
    return result


def quota_snapshot():
    result = {}
    for name in CREDENTIALS:
        if not os.environ.get(name):
            continue
        request = urllib.request.Request(
            "https://openrouter.ai/api/v1/key",
            headers={"Authorization": "Bearer " + os.environ[name]},
        )
        with urllib.request.urlopen(request, timeout=20) as response:
            result[name] = json.load(response)["data"]["free_model_daily_requests"]
    return result


class Admission:
    def __init__(self, directory, limits, request_limit):
        self.directory, self.limits, self.request_limit = (
            directory,
            limits,
            request_limit,
        )
        self.checked_at = 0
        self.quota = None
        self.checked_counts = {}
        self.reason = "ready"

    def __call__(self, job, active):
        gate_path = self.directory / "request_gate.json"
        state = (
            json.loads(gate_path.read_text())
            if gate_path.exists()
            else {"admitted": 0, "by_job": {}, "by_credential": {}}
        )
        now = time.time()
        if self.quota is None or now - self.checked_at >= 30:
            try:
                self.quota = quota_snapshot()
                self.checked_at = now
                self.checked_counts = dict(state.get("by_credential", {}))
                atomic_json(
                    self.directory / "latest_free_quota.json",
                    {"checked_epoch": now, "credentials": self.quota},
                )
            except Exception as exc:
                self.reason = "quota_lookup_failed"
                atomic_json(
                    self.directory / "admission.json",
                    {
                        "allowed": False,
                        "reason": self.reason,
                        "error_type": type(exc).__name__,
                    },
                )
                return False
        remaining = 0
        caps = {}
        for name, limit in self.limits.items():
            admitted = state.get("by_credential", {}).get(name, 0)
            remote = max(
                0,
                self.quota[name]["remaining"]
                - max(0, admitted - self.checked_counts.get(name, 0))
                - 2,
            )
            available = min(max(0, limit - admitted), remote)
            caps[name] = admitted + available
            remaining += available
        with gate_path.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            current = (
                json.loads(gate_path.read_text())
                if gate_path.exists()
                else {"admitted": 0, "by_job": {}, "by_credential": {}}
            )
            current["credential_caps"] = caps
            atomic_json(gate_path, current)
        result = dispatch_reservation(
            job,
            active,
            current,
            request_limit=self.request_limit,
            free_remaining=max(0, remaining - len(active)),
        )
        self.reason = (
            "allowed" if result["allowed"] else "quota_or_batch_request_reservation"
        )
        atomic_json(
            self.directory / "admission.json",
            result
            | {
                "reason": self.reason,
                "candidate": job["id"],
                "active_workers": len(active),
                "credential_caps": caps,
                "epoch": now,
            },
        )
        return result["allowed"]


def base_configuration():
    previous = json.loads((PRIOR / "manifest.json").read_text())
    job = previous["jobs"][0]
    config = RunConfig(**job["run_config"])
    config.recovery.enabled = True
    model = ModelConfig(**job["model_config"])
    return previous, job, config, model


SUCCESS_SOURCE = """top = b0
while exists(lambda x: not ON_star(x, b0)):
    cand = get(lambda x: not ON_star(x, b0))
    Pick(cand)
    Move(cand, cand, b0, [0.0, 0.0, 0.2])
    Move(b0, b0, b0, [0.0, 0.0, 0.2])
    Move(b0, b0, top, [0.0, 0.0, 0.055])
    Release(cand, 0.1)
    top = cand
"""


def create(directory, *, preflight=False, max_workers=18):
    previous, job, config, model = base_configuration()
    limits = {}
    quotas = {}
    if not preflight:
        quotas = quota_snapshot()
        if set(quotas) != set(CREDENTIALS):
            raise ValueError("All four configured credential variables are required")
        limits = {
            name: max(0, min(950 if name == CREDENTIALS[0] else 45, q["remaining"] - 5))
            for name, q in quotas.items()
        }
    request_limit = min(1085, sum(limits.values())) if limits else 1
    control = (
        None
        if preflight
        else {
            "state_path": str(directory / "request_gate.json"),
            "interval_seconds": 6.0,
            "request_limit": request_limit,
            "credential_limits": limits,
        }
    )
    gpu = GPUConfig(
        GPU_UUIDS,
        "/tmp/roboverify-cap-gpu-reservations",
        max_workers_per_gpu=max(1, (max_workers + 1) // 2),
        min_free_memory_mb=4096,
        worker_memory_mb=1024,
        gpu_poll_interval_seconds=2,
        gpu_sample_window_seconds=10,
        admission_timeout_seconds=60,
    )
    responses = (
        [
            "candidate=get(lambda x: not eq(x,b0))\nPick(candidate)\nMove(b0,b0,b0,[3,0,0.2])",
            SUCCESS_SOURCE,
        ]
        if preflight
        else None
    )
    manifest = create_batch(
        directory,
        config,
        model,
        initial_state=job["initial_state"],
        instruction=job["instruction"],
        modes=["dsl_only"],
        trials=1 if preflight else 20,
        max_workers=max_workers,
        schedule_seed=42,
        recovery_variants=[True],
        experiment_variants=variants(),
        gpu_config=gpu,
        transport_control=control,
        responses=responses,
    )
    assert manifest["snapshot_sha256"] == previous["snapshot_sha256"]
    sources = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in Path("synthesis/cap").rglob("*")
        if p.is_file()
        and p.suffix in {".py", ".json"}
        and "docs" not in p.parts
        and "__pycache__" not in p.parts
    }
    atomic_json(
        directory / "comparison_plan.json",
        {
            "planned": 11 if preflight else 220,
            "preflight": preflight,
            "prior_batch": str(PRIOR.resolve()),
            "historical_comparison": "unpaired, fixed scene; provider load/time differs",
            "snapshot_sha256": manifest["snapshot_sha256"],
            "source_sha256": sources,
            "git_sha": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "credential_limits": limits,
            "request_limit": request_limit,
            "quota_at_creation": quotas,
            "gpu_policy": asdict(gpu),
            "no_failed_trial_replacements": True,
            "ablation_unit": "Section 3 candidate group; no_prompt disables all four prompt Booleans",
            "wire_pacing": "six seconds between starts for each credential; common admission reserves active-trial call allowances",
        },
    )
    preflight_path = Path("/tmp/revision42-provider-preflight")
    if preflight_path.exists() and not preflight:
        shutil.copytree(preflight_path, directory / "provider_preflight")
    return manifest


def execute(directory, *, resume=False):
    plan = json.loads((directory / "comparison_plan.json").read_text())
    gate = (
        None
        if plan["preflight"]
        else Admission(directory, plan["credential_limits"], plan["request_limit"])
    )
    while True:
        result = execute_batch(directory, resume=resume, dispatch_gate=gate)
        if result["finished"] == result["planned"]:
            atomic_json(
                directory / "control_status.json",
                {
                    "status": "completed",
                    "finished": result["finished"],
                    "planned": result["planned"],
                },
            )
            return result
        schedule = json.loads((directory / "schedule.json").read_text())
        manifest = json.loads((directory / "manifest.json").read_text())
        pending = [
            j for j in manifest["jobs"] if schedule[j["id"]]["status"] == "pending"
        ]
        if gate and pending and gate(pending[0], []):
            resume = True
            continue
        atomic_json(
            directory / "control_status.json",
            {
                "status": "paused",
                "finished": result["finished"],
                "planned": result["planned"],
                "reason": gate.reason if gate else "gpu_admission",
            },
        )
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-workers", type=int, default=18)
    parser.add_argument("--create-only", action="store_true")
    args = parser.parse_args()
    directory = args.directory.resolve()
    if not args.resume:
        create(directory, preflight=args.preflight, max_workers=args.max_workers)
    if not args.create_only:
        result = execute(directory, resume=args.resume)
        print(
            json.dumps(
                {
                    "directory": str(directory),
                    "finished": result["finished"],
                    "planned": result["planned"],
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
