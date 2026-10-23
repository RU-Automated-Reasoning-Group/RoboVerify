"""Revision 4.3 primitive prompt factorial on locally served models.

All 16 combinations of the four Revision 4.2 prompt switches, for every
method, with Revision 4 default recovery feedback. One batch per model; the
model is served by a local OpenAI-compatible vLLM endpoint.
"""

import argparse
import copy
import hashlib
import itertools
import json
import os
import shutil
import subprocess
import time
import urllib.request
from dataclasses import asdict
from pathlib import Path

from .batch import METHODS, atomic_json, create_batch
from .config import GPUConfig, ModelConfig, PrimitiveFeedbackConfig, RunConfig
from .revision42_experiment import GPU_UUIDS, PRIOR, Admission, execute

SWITCHES = (
    "explain_target_offsets",
    "explain_convergence_criteria",
    "explain_partial_execution",
    "explain_motion_planning_limits",
)
MODELS = {
    "qwen3.8-27b-fp8": {
        "port": 8106,
        "checkpoint": "Qwen/Qwen3.8-27B-FP8",
        "revision": "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a",
        # The template defaults to xhigh, which exceeded 32k output tokens on
        # every smoke round 0; medium matches the OpenRouter reasoning lengths.
        "template": {"enable_thinking": True, "reasoning_effort": "medium"},
        # Served with this vLLM logits processor from 2026-10-23 on (Revision
        # 4.3.1, Section 10); earlier batches ran without it.
        "logits_processors": [
            "synthesis.cap.serving.think_close_guard:ThinkCloseGuard"
        ],
    },
    "qwen3.5-9b": {
        # Local least-in-flight balancer; GPU 7 first, GPU 6 added after the 27B run.
        "port": 8117,
        "checkpoint": "Qwen/Qwen3.5-9B",
        "revision": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        # The Qwen3.5 template has no reasoning-effort option.
        "template": {"enable_thinking": True},
    },
}
CREDENTIAL = "VLLM_API_KEY"
OUTPUT_TOKENS = 32768
# Local decoding runs at roughly 25-75 tokens/s per request under load, so a
# 32k-token response can take over 1,000 s; time limits scale accordingly.
REQUEST_TIMEOUT = 2400
ROUND_TIMEOUT = 3600
TRIAL_TIMEOUT = 18000
# Output reservations double with the 32k cap, so token budgets double too.
INITIAL_TOKENS = 100000
ROUND_TOKENS = 200000
TRIAL_TOKENS = 900000


class HealthGate:
    """Dispatch only while the model endpoint answers; otherwise pause the batch.

    A dead local server turns every new trial into an immediate transport
    failure, so dispatch stops instead of consuming the pending trials.

    ``limit_path`` optionally names a JSON file ``{"max_active": N}`` that is
    re-read before every dispatch. It lowers concurrency while a batch runs
    (for example before serving GPUs are released) without interrupting
    running trials; 0 pauses dispatch. Each change is appended to
    ``dispatch_limit_events.jsonl`` next to the file.
    """

    def __init__(self, gate, url, *, interval=10, limit_path=None):
        self.gate, self.url, self.interval = gate, url, interval
        self.limit_path = Path(limit_path) if limit_path else None
        self.checked_at, self.healthy = 0.0, False
        self.limit, self.limited = None, False

    @property
    def reason(self):
        if not self.healthy:
            return "model_endpoint_unavailable"
        if self.limited:
            return "dispatch_limit"
        return self.gate.reason if self.gate else "ready"

    def _read_limit(self):
        if self.limit_path is None or not self.limit_path.exists():
            limit = None
        else:
            try:
                limit = int(json.loads(self.limit_path.read_text())["max_active"])
            except (ValueError, KeyError, TypeError):
                return self.limit  # Keep the last valid value on a bad edit.
        if limit != self.limit:
            with (self.limit_path.parent / "dispatch_limit_events.jsonl").open(
                "a"
            ) as events:
                events.write(json.dumps({"epoch": time.time(), "max_active": limit}))
                events.write("\n")
            self.limit = limit
        return limit

    def __call__(self, job, active):
        limit = self._read_limit()
        self.limited = limit is not None and len(active) >= limit
        if self.limited:
            return False
        if time.time() - self.checked_at >= self.interval:
            request = urllib.request.Request(
                self.url,
                headers={"Authorization": f"Bearer {os.environ.get(CREDENTIAL, '')}"},
            )
            try:
                with urllib.request.urlopen(request, timeout=10) as response:
                    self.healthy = response.status == 200
            except OSError:
                self.healthy = False
            self.checked_at = time.time()
        return self.healthy and (self.gate is None or self.gate(job, active))


def health_gate(model_name):
    url = f"http://127.0.0.1:{MODELS[model_name]['port']}/v1/models"
    return lambda gate: HealthGate(gate, url)


def serving_record(model_name):
    server = MODELS[model_name]
    return {
        "engine": "vllm",
        "endpoint": f"http://127.0.0.1:{server['port']}/v1",
        "checkpoint": server["checkpoint"],
        "revision": server["revision"],
        "sampling": "checkpoint generation_config defaults",
        "chat_template_kwargs": server["template"],
        "logits_processors": server.get("logits_processors", []),
    }


def cell_name(flags):
    return "p" + "".join("1" if flags[s] else "0" for s in SWITCHES)


def variants(cells=None, methods=None):
    """All 16 prompt cells for every method; feedback stays at Revision 4.

    ``cells``/``methods`` restrict the design for smoke runs only.
    """
    feedback = asdict(PrimitiveFeedbackConfig())
    result = []
    for bits in itertools.product((False, True), repeat=len(SWITCHES)):
        flags = dict(zip(SWITCHES, bits))
        if cells and cell_name(flags) not in cells:
            continue
        result.append(
            {
                "name": cell_name(flags),
                "methods": [m for m in METHODS if not methods or m in methods],
                "run_config": {
                    "primitive_prompt": flags,
                    "primitive_feedback": feedback,
                },
            }
        )
    return result


def configuration(model_name):
    previous = json.loads((PRIOR / "manifest.json").read_text())
    job = previous["jobs"][0]
    run_config = dict(job["run_config"])
    run_config.update(
        save_video=None,
        video_mode="deferred",
        max_tokens=INITIAL_TOKENS,
        timeout=ROUND_TIMEOUT,
    )
    config = RunConfig(**run_config)
    config.recovery.enabled = True
    config.recovery.round_tokens = ROUND_TOKENS
    config.recovery.total_tokens = TRIAL_TOKENS
    config.recovery.round_timeout = ROUND_TIMEOUT
    config.recovery.total_timeout = TRIAL_TIMEOUT
    server = MODELS[model_name]
    model = ModelConfig(
        provider="vllm",
        model=model_name,
        base_url=f"http://127.0.0.1:{server['port']}/v1",
        api_key_env=CREDENTIAL,
        max_output_tokens=OUTPUT_TOKENS,
        request_timeout=REQUEST_TIMEOUT,
        transport_retries=1,
        extra_body={"chat_template_kwargs": dict(server["template"])},
    )
    return previous, job, config, model


def create(
    directory,
    *,
    model_name,
    trials=50,
    max_workers=32,
    request_limit=None,
    workers_per_gpu=48,
    worker_memory_mb=128,
    cells=None,
    methods=None,
):
    previous, job, config, model = configuration(model_name)
    experiment_variants = variants(cells, methods)
    planned = trials * sum(len(v["methods"]) for v in experiment_variants)
    # Local serving has no provider quota; the cap only bounds runaway retries.
    request_limit = request_limit or planned * 76
    limits = {CREDENTIAL: request_limit}
    control = {
        "state_path": str(directory / "request_gate.json"),
        "interval_seconds": 0.0,
        "request_limit": request_limit,
        "credential_limits": limits,
    }
    gpu = GPUConfig(
        GPU_UUIDS,
        "/tmp/roboverify-cap-gpu-reservations",
        # The scheduler caps each GPU at the smallest slot limit among live
        # reservations, so concurrent batches must share one per-GPU value.
        max_workers_per_gpu=workers_per_gpu,
        min_free_memory_mb=2048,
        worker_memory_mb=worker_memory_mb,
        gpu_poll_interval_seconds=2,
        gpu_sample_window_seconds=10,
        admission_timeout_seconds=60,
    )
    manifest = create_batch(
        directory,
        config,
        model,
        initial_state=job["initial_state"],
        instruction=job["instruction"],
        modes=["dsl_only"],
        trials=trials,
        max_workers=max_workers,
        schedule_seed=43,
        recovery_variants=[True],
        experiment_variants=experiment_variants,
        gpu_config=gpu,
        transport_control=control,
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
    server = MODELS[model_name]
    atomic_json(
        directory / "comparison_plan.json",
        {
            "planned": planned,
            "preflight": False,
            "experiment": "revision43_prompt_factorial",
            "prior_batch": str(PRIOR.resolve()),
            "model": model.model,
            "serving": {
                "engine": "vllm",
                "endpoint": model.base_url,
                "checkpoint": server["checkpoint"],
                "revision": server["revision"],
                "sampling": "checkpoint generation_config defaults",
                "chat_template_kwargs": server["template"],
            },
            "admission": "paid_request_cap",
            "video_mode": config.video_mode,
            "snapshot_sha256": manifest["snapshot_sha256"],
            "source_sha256": sources,
            "git_sha": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "credential_limits": limits,
            "request_limit": request_limit,
            "gpu_policy": asdict(gpu),
            "no_failed_trial_replacements": True,
            "factors": list(SWITCHES),
            "cells": [v["name"] for v in experiment_variants],
            "feedback": "Revision 4 defaults (cap-recovery-v1); primitive enrichments off",
            "budgets": {
                "output_tokens": OUTPUT_TOKENS,
                "request_timeout_seconds": REQUEST_TIMEOUT,
                "initial_tokens": INITIAL_TOKENS,
                "round_tokens": ROUND_TOKENS,
                "trial_tokens": TRIAL_TOKENS,
                "round_timeout_seconds": ROUND_TIMEOUT,
                "trial_timeout_seconds": TRIAL_TIMEOUT,
            },
            "primary_outcome": "round-0 task success",
            "restricted_design": bool(cells or methods),
        },
    )
    return manifest


def create_rerun(directory, *, source, job_ids, reason, max_workers=64):
    """Copy selected job definitions of ``source`` into a new batch.

    Used for trials invalidated by infrastructure failures. The copied jobs keep
    their identifiers and configuration; only paths change. The source batch is
    read only, and analyses replace its listed trials with this batch's.
    """
    source = Path(source).resolve()
    manifest = json.loads((source / "manifest.json").read_text())
    if (
        hashlib.sha256((source / "manifest.json").read_bytes()).hexdigest()
        != (source / "manifest.sha256").read_text().strip()
    ):
        raise ValueError("Source manifest changed")
    selected = set(job_ids)
    jobs = [copy.deepcopy(j) for j in manifest["jobs"] if j["id"] in selected]
    if len(jobs) != len(selected):
        raise ValueError("Unknown job identifiers in rerun list")
    directory.mkdir(parents=True)
    snapshot = directory / "inputs/initial_state.npz"
    snapshot.parent.mkdir()
    shutil.copyfile(source / "inputs/initial_state.npz", snapshot)
    if hashlib.sha256(snapshot.read_bytes()).hexdigest() != manifest["snapshot_sha256"]:
        raise ValueError("Source snapshot changed")
    plan = json.loads((source / "comparison_plan.json").read_text())
    request_limit = len(jobs) * 76
    for job in jobs:
        job["run_config"]["output_root"] = str(directory / "trials" / job["id"])
        job["initial_state"] = str(snapshot)
        job["transport_control"] = dict(
            job["transport_control"],
            state_path=str(directory / "request_gate.json"),
            request_limit=request_limit,
            credential_limits={CREDENTIAL: request_limit},
        )
        Path(job["run_config"]["output_root"]).mkdir(parents=True)
    rerun = dict(manifest, created_epoch=time.time(), max_workers=max_workers)
    rerun["render_workers"] = min(manifest["render_workers"], max_workers)
    rerun["jobs"] = jobs
    atomic_json(directory / "manifest.json", rerun)
    (directory / "manifest.sha256").write_text(
        hashlib.sha256((directory / "manifest.json").read_bytes()).hexdigest() + "\n"
    )
    atomic_json(
        directory / "schedule.json", {j["id"]: {"status": "pending"} for j in jobs}
    )
    atomic_json(
        directory / "comparison_plan.json",
        plan
        | {
            "planned": len(jobs),
            "request_limit": request_limit,
            "credential_limits": {CREDENTIAL: request_limit},
            "rerun_of": str(source),
            "rerun_reason": reason,
            "rerun_job_ids": sorted(selected),
            "git_sha": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
        },
    )
    return rerun


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--model", choices=sorted(MODELS), required=True)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument(
        "--max-workers",
        type=int,
        help="concurrent trials (default 32); on --resume, overrides the manifest",
    )
    parser.add_argument("--workers-per-gpu", type=int, default=48)
    parser.add_argument("--worker-memory-mb", type=int, default=128)
    parser.add_argument("--cell", action="append", help="smoke runs only")
    parser.add_argument("--method", action="append", choices=METHODS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--rerun-of", type=Path, help="source batch to rerun jobs from")
    parser.add_argument("--rerun-jobs", type=Path, help="JSON list of job ids")
    parser.add_argument("--rerun-reason")
    parser.add_argument("--create-only", action="store_true")
    args = parser.parse_args()
    directory = args.directory.resolve()
    if args.rerun_of:
        create_rerun(
            directory,
            source=args.rerun_of,
            job_ids=json.loads(args.rerun_jobs.read_text()),
            reason=args.rerun_reason,
            max_workers=args.max_workers or 64,
        )
    elif not args.resume:
        create(
            directory,
            model_name=args.model,
            trials=args.trials,
            max_workers=args.max_workers or 32,
            workers_per_gpu=args.workers_per_gpu,
            worker_memory_mb=args.worker_memory_mb,
            cells=args.cell,
            methods=args.method,
        )
    if not args.create_only:
        result = execute(
            directory,
            resume=args.resume,
            max_workers=args.max_workers if args.resume else None,
            wrap_gate=health_gate(args.model),
        )
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
