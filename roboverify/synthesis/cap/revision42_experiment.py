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
DEFAULT_MODEL = "minimax/minimax-m3.1-flash:free"
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
PROFILES = ("full", "no_prompt", "no_target", "no_phases", "no_context", "only_prompt")
FEEDBACK_ABLATIONS = {
    "no_target": "target_derivation",
    "no_phases": "controller_phases",
    "no_context": "action_context",
}
# The original 220-trial design: full for every method, ablations for Method 1/2.
DEFAULT_SELECTION = {
    name: list(METHODS if name == "full" else METHODS[1:]) for name in PROFILES[:5]
}


def variants(selection=None):
    """Return batch variants for ``{profile: [methods]}`` in PROFILES order."""
    selection = DEFAULT_SELECTION if selection is None else selection
    if not set(selection) <= set(PROFILES) or any(
        not methods or not set(methods) <= set(METHODS)
        for methods in selection.values()
    ):
        raise ValueError("Invalid profile/method selection")
    full_prompt = dict.fromkeys(asdict(PrimitivePromptConfig()), True)
    result = []
    for name in (p for p in PROFILES if p in selection):
        prompt = dict(full_prompt)
        feedback = asdict(
            PrimitiveFeedbackConfig(
                target_derivation=True, controller_phases=True, action_context=True
            )
        )
        if name == "no_prompt":
            prompt = asdict(PrimitivePromptConfig())
        elif name == "only_prompt":
            feedback.update(dict.fromkeys(FEEDBACK_ABLATIONS.values(), False))
        elif name != "full":
            feedback[FEEDBACK_ABLATIONS[name]] = False
        result.append(
            {
                "name": name,
                "methods": [m for m in METHODS if m in selection[name]],
                "run_config": {
                    "primitive_prompt": prompt,
                    "primitive_feedback": feedback,
                },
            }
        )
    return result


def quota_snapshot(credentials=CREDENTIALS):
    result = {}
    for name in credentials:
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
    def __init__(self, directory, limits, request_limit, *, paid=False):
        self.directory, self.limits, self.request_limit = (
            directory,
            limits,
            request_limit,
        )
        # Paid models have no free daily quota; only the recorded caps apply.
        self.paid = paid
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
        if not self.paid and (self.quota is None or now - self.checked_at >= 30):
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
            remote = (
                limit
                if self.paid
                else max(
                    0,
                    self.quota[name]["remaining"]
                    - max(0, admitted - self.checked_counts.get(name, 0))
                    - 2,
                )
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


def base_configuration(model_name=None, video_mode=None):
    previous = json.loads((PRIOR / "manifest.json").read_text())
    job = previous["jobs"][0]
    run_config = dict(job["run_config"])
    if video_mode is not None:
        run_config.update(save_video=None, video_mode=video_mode)
    config = RunConfig(**run_config)
    config.recovery.enabled = True
    model = ModelConfig(
        **(job["model_config"] | {"model": model_name or DEFAULT_MODEL})
    )
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


def create(
    directory,
    *,
    preflight=False,
    max_workers=18,
    model_name=None,
    credentials=None,
    request_interval=6.0,
    paid_request_limit=1085,
    video_mode=None,
    selection=None,
    prior_batches=(),
):
    previous, job, config, model = base_configuration(model_name, video_mode)
    experiment_variants = variants(selection)
    planned = (
        11 if preflight else 20 * sum(len(v["methods"]) for v in experiment_variants)
    )
    paid = not model.model.endswith(":free")
    credentials = list(credentials or (CREDENTIALS[:1] if paid else CREDENTIALS))
    limits = {}
    quotas = {}
    if not preflight and paid:
        missing = [name for name in credentials if not os.environ.get(name)]
        if missing:
            raise ValueError(f"Missing credential environment variables: {missing}")
        limits = dict.fromkeys(credentials, paid_request_limit)
    elif not preflight:
        quotas = quota_snapshot(credentials)
        if set(quotas) != set(credentials):
            raise ValueError("All configured credential variables are required")
        limits = {
            name: max(0, min(950 if name == credentials[0] else 45, q["remaining"] - 5))
            for name, q in quotas.items()
        }
    request_limit = (
        min(paid_request_limit if paid else 1085, sum(limits.values())) if limits else 1
    )
    control = (
        None
        if preflight
        else {
            "state_path": str(directory / "request_gate.json"),
            "interval_seconds": request_interval,
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
        experiment_variants=experiment_variants,
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
            "planned": planned,
            "preflight": preflight,
            "prior_batch": str(PRIOR.resolve()),
            "model": model.model,
            "admission": "paid_request_cap" if paid else "free_daily_quota",
            "video_mode": config.video_mode,
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
            "ablation_unit": "Section 3 candidate group; no_prompt disables all four prompt Booleans; only_prompt keeps them and disables all three feedback enrichments",
            "profile_methods": {v["name"]: v["methods"] for v in experiment_variants},
            "accumulates_with": [str(Path(b).resolve()) for b in prior_batches],
            "wire_pacing": f"{request_interval:g} seconds between starts for each credential; common admission reserves active-trial call allowances",
        },
    )
    preflight_path = Path("/tmp/revision42-provider-preflight")
    if preflight_path.exists() and not preflight:
        shutil.copytree(preflight_path, directory / "provider_preflight")
    return manifest


def execute(directory, *, resume=False, max_workers=None, wrap_gate=None):
    plan = json.loads((directory / "comparison_plan.json").read_text())
    gate = (
        None
        if plan["preflight"]
        else Admission(
            directory,
            plan["credential_limits"],
            plan["request_limit"],
            paid=plan.get("admission") == "paid_request_cap",
        )
    )
    if wrap_gate is not None:
        gate = wrap_gate(gate)
    while True:
        result = execute_batch(
            directory, resume=resume, dispatch_gate=gate, max_workers=max_workers
        )
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
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--credential",
        action="append",
        help="credential environment-variable name; repeat to rotate",
    )
    parser.add_argument("--request-interval", type=float, default=6.0)
    parser.add_argument("--paid-request-limit", type=int, default=1085)
    parser.add_argument(
        "--variant",
        action="append",
        metavar="PROFILE=METHOD[,METHOD]",
        help="select profile/method groups; default is the original 220-trial design",
    )
    parser.add_argument(
        "--accumulates-with",
        action="append",
        default=[],
        type=Path,
        help="earlier batch whose groups this batch complements, recorded in the plan",
    )
    parser.add_argument(
        "--video-mode",
        choices=("inline", "deferred", "off"),
        help="override the prior configuration's video setting",
    )
    args = parser.parse_args()
    directory = args.directory.resolve()
    if not args.resume:
        create(
            directory,
            preflight=args.preflight,
            max_workers=args.max_workers,
            model_name=args.model,
            credentials=args.credential,
            request_interval=args.request_interval,
            paid_request_limit=args.paid_request_limit,
            video_mode=args.video_mode,
            selection=(
                {
                    name: methods.split(",")
                    for name, methods in (v.split("=", 1) for v in args.variant)
                }
                if args.variant
                else None
            ),
            prior_batches=args.accumulates_with,
        )
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
