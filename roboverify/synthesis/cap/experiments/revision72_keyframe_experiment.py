"""Revision 7.2: legacy vs RDP recovery keyframes on shared failed round-0 attempts.

The legacy arm is an existing ``revision6_experiment --frozen v6`` batch set
(demonstrations on): its recovery rounds already used the legacy keyframes. The
RDP arm restarts recovery from the same archived round-0 attempts, so both arms
share round 0 and differ only in the recovery keyframes::

    python -m synthesis.cap.experiments.revision72_keyframe_experiment select \\
        --source runs/cap-revision7-1/20261026-demo-ablation-v6/demos-on \\
        --manifest ROOT/seeds.json
    python -m synthesis.cap.experiments.revision72_keyframe_experiment run \\
        --manifest ROOT/seeds.json --root ROOT/rdp
    python -m synthesis.cap.experiments.revision72_keyframe_experiment report \\
        --manifest ROOT/seeds.json --root ROOT/rdp

``run`` creates one batch per task with one seeded job per failed round 0 and
the overrides in ``RDP``; everything else is the source batch's ``config_frozen_v6``
configuration. Credentials come from ``OPENROUTER_API_KEY`` only.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from ..batch import _run_directory, atomic_json, create_batch, execute_batch
from ..config import GPUConfig
from .revision6_experiment import GPU_UUIDS, TASKS, configuration

# Only the keyframe method changes. The larger keyframe budget and evidence cap
# let every recorded rollout fit; with the fixed_items layout the cap is a guard
# and does not resize any other item.
RDP = {
    "recovery": {
        "feedback": {
            "keyframe_method": "rdp",
            "keyframe_bytes": 65536,
            "evidence_bytes": 98304,
        }
    }
}


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select(source, manifest_path):
    """Failed round-0 attempts of every task in ``source`` (read only)."""
    source = Path(source).resolve()
    seeds = []
    for task in TASKS:
        for directory in sorted(source.glob(f"*/{task}")):
            manifest = json.loads((directory / "manifest.json").read_text())
            rows = {
                r["job_id"]: r
                for r in json.loads((directory / "summary.json").read_text())["trials"]
            }
            plan = json.loads((directory / "comparison_plan.json").read_text())
            for job in manifest["jobs"]:
                row = rows[job["id"]]
                if row.get("first_attempt_task_spec_success"):
                    continue
                result = (
                    _run_directory(job) / "artifacts/trial_001/rounds/000/result.json"
                )
                seeds.append(
                    {
                        "seed_id": f"{task}-{job['run_config']['seed']:02d}",
                        "task": task,
                        "model_label": plan["model_label"],
                        "frozen_config": plan["frozen_config"],
                        "demonstrations": plan["demonstrations"],
                        "scene_seed": job["run_config"]["seed"],
                        "source_batch": str(directory),
                        "source_job": job["id"],
                        "seed_result": str(result),
                        "seed_result_sha256": file_sha256(result),
                        "legacy": {
                            k: row.get(k)
                            for k in (
                                "task_spec_success",
                                "task_spec_success_round",
                                "recovery_rounds",
                                "stop_reason",
                                "status",
                                "known_tokens",
                                "demonstration_tokens",
                                "model_calls",
                            )
                        },
                    }
                )
    manifest = {"source": str(source), "rdp_overrides": RDP, "seeds": seeds}
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("x") as handle:
        handle.write(json.dumps(manifest, indent=2) + "\n")
    manifest_path.chmod(0o444)
    return manifest


def load_manifest(path):
    manifest = json.loads(Path(path).read_text())
    for seed in manifest["seeds"]:
        if file_sha256(seed["seed_result"]) != seed["seed_result_sha256"]:
            raise ValueError(f"Seed result changed: {seed['seed_id']}")
    return manifest


def create(directory, task, seeds, *, workers, manifest_path):
    first = seeds[0]
    config, model, instruction, overrides = configuration(
        task,
        first["model_label"],
        demonstrations=first["demonstrations"],
        frozen="v6",
    )
    variants = [
        {
            "name": f"seed{seed['scene_seed']:02d}",
            "methods": [config.method],
            "seed_attempt": seed["seed_result"],
            "run_config": RDP,
        }
        for seed in seeds
    ]
    request_limit = len(variants) * config.recovery.total_calls
    create_batch(
        directory,
        config,
        model,
        instruction=instruction,
        methods=[config.method],
        modes=[config.condition_mode],
        trials=1,
        max_workers=workers,
        schedule_seed=7200,
        experiment_variants=variants,
        gpu_config=GPUConfig(
            GPU_UUIDS,
            "/tmp/roboverify-cap-gpu-reservations",
            max_workers_per_gpu=48,
            min_free_memory_mb=2048,
            worker_memory_mb=128,
            gpu_poll_interval_seconds=2,
            gpu_sample_window_seconds=10,
            admission_timeout_seconds=60,
        ),
        transport_control={
            "state_path": str(directory / "request_gate.json"),
            "interval_seconds": 0.0,
            "request_limit": request_limit,
            "credential_limits": {model.api_key_env: request_limit},
        },
    )
    atomic_json(
        directory / "comparison_plan.json",
        {
            "experiment": "revision7_2_rdp_recovery_keyframes",
            "frozen_config": "cap-frozen-v6",
            "frozen_config_source": "synthesis/cap/configs/config_frozen_v6.py",
            "driver": "synthesis/cap/experiments/revision72_keyframe_experiment.py",
            "overrides": overrides | {"variant_run_config": RDP},
            "seed_manifest": str(Path(manifest_path).resolve()),
            "seed_manifest_sha256": file_sha256(manifest_path),
            "task": task,
            "model_label": first["model_label"],
            "demonstrations": first["demonstrations"],
            "trials": len(variants),
            "scene_seeds": [s["scene_seed"] for s in seeds],
            "worker_gpus": GPU_UUIDS,
            "request_limit": request_limit,
            "git_sha": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "git_dirty": bool(
                subprocess.check_output(
                    ["git", "status", "--porcelain", "synthesis"], text=True
                ).strip()
            ),
            "run_config": asdict(config),
            "model_config": asdict(model),
            "no_failed_trial_replacements": True,
        },
    )


def run_task(root, task, manifest_path, workers):
    manifest = load_manifest(manifest_path)
    seeds = [s for s in manifest["seeds"] if s["task"] == task]
    directory = Path(root).resolve() / seeds[0]["model_label"] / task
    if not directory.exists():
        create(directory, task, seeds, workers=workers, manifest_path=manifest_path)
    return execute_batch(directory, resume=(directory / "summary.json").exists())


def rdp_rows(root, manifest):
    """Final RDP-arm rows by seed_id (None while a job has not finished)."""
    rows = {}
    for task in {s["task"] for s in manifest["seeds"]}:
        label = next(s["model_label"] for s in manifest["seeds"] if s["task"] == task)
        directory = Path(root) / label / task
        summary = directory / "summary.json"
        if not summary.exists():
            continue
        jobs = json.loads((directory / "manifest.json").read_text())["jobs"]
        names = {j["id"]: j["variant"] for j in jobs}
        for row in json.loads(summary.read_text())["trials"]:
            rows[f"{task}-{int(names[row['job_id']][4:]):02d}"] = row | {
                "batch": str(directory)
            }
    return rows


def report(root, manifest_path):
    manifest = load_manifest(manifest_path)
    rows = rdp_rows(root, manifest)
    lines = [
        "| Task | Seeds | Legacy final | RDP final | Both | Only legacy | Only RDP | Neither | Legacy mean rounds | RDP mean rounds | Legacy mean effort tokens | RDP mean effort tokens | RDP keyframes partial |",
        "| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    per_seed = []
    for task in TASKS:
        seeds = [s for s in manifest["seeds"] if s["task"] == task]
        if not seeds:
            continue
        done = [(s, rows[s["seed_id"]]) for s in seeds if s["seed_id"] in rows]
        legacy = [bool(s["legacy"]["task_spec_success"]) for s, _ in done]
        rdp = [bool(r.get("task_spec_success")) for _, r in done]
        pairs = list(zip(legacy, rdp))
        partial = 0
        for s, r in done:
            partial += any(
                isinstance(d, dict) for d in keyframe_delivery(r.get("run_dir"))
            )
            per_seed.append((s, r))
        n = len(done)
        mean = lambda xs: sum(xs) / max(1, len(xs))
        lines.append(
            f"| {task} | {n}/{len(seeds)} | {sum(legacy)}/{n} | {sum(rdp)}/{n} | "
            f"{sum(a and b for a, b in pairs)} | {sum(a and not b for a, b in pairs)} | "
            f"{sum(b and not a for a, b in pairs)} | {sum(not a and not b for a, b in pairs)} | "
            f"{mean([s['legacy']['recovery_rounds'] for s, _ in done]):.2f} | "
            f"{mean([r.get('recovery_rounds', 0) for _, r in done]):.2f} | "
            f"{mean([s['legacy']['known_tokens'] for s, _ in done]):,.0f} | "
            f"{mean([r.get('known_tokens', 0) for _, r in done]):,.0f} | {partial} |"
        )
    lines += [
        "",
        "| Seed | Legacy | Legacy round | RDP | RDP round | RDP stop reason |",
        "| --- | --- | ---: | --- | ---: | --- |",
    ]
    for s, r in per_seed:
        lines.append(
            f"| {s['seed_id']} | {s['legacy']['task_spec_success']} | {s['legacy']['task_spec_success_round']} | "
            f"{r.get('task_spec_success')} | {r.get('task_spec_success_round')} | {r.get('stop_reason')} |"
        )
    text = "\n".join(lines) + "\n"
    (Path(root) / "paired_report.md").write_text(text)
    return text


def keyframe_delivery(run_dir):
    """Per-round keyframe delivery of one recovery run (empty if unknown)."""
    if not run_dir:
        return []
    found = []
    for path in sorted(
        Path(run_dir).glob("artifacts/trial_001/rounds/*/feedback.json")
    ):
        feedback = json.loads(path.read_text())
        if "item_delivery" in feedback:
            found.append(feedback["item_delivery"].get("keyframes"))
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("select", "run", "report"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--workers", type=int, default=5, help="per batch")
    parser.add_argument("--task", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.command == "select":
        manifest = select(args.source, args.manifest)
        print(json.dumps({"seeds": len(manifest["seeds"])}))
        return
    if args.command == "report":
        print(report(args.root, args.manifest))
        return
    if args.task:
        summary = run_task(args.root, args.task, args.manifest, args.workers)
        print(json.dumps({"task": args.task, "finished": summary["finished"]}))
        return
    if not os.environ.get("OPENROUTER_API_KEY"):
        parser.error("the RDP arm needs the OPENROUTER_API_KEY environment variable")
    manifest = load_manifest(args.manifest)
    args.root.mkdir(parents=True, exist_ok=True)
    processes = []
    for task in TASKS:
        if not any(s["task"] == task for s in manifest["seeds"]):
            continue
        log = (args.root / f"launch_{task}.log").open("a")
        command = [
            sys.executable,
            "-m",
            __spec__.name,
            "run",
            "--manifest",
            str(args.manifest),
            "--root",
            str(args.root),
            "--workers",
            str(args.workers),
            "--task",
            task,
        ]
        processes.append(
            (task, subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT))
        )
    for task, process in processes:
        print(json.dumps({"task": task, "exit_code": process.wait()}), flush=True)
    print(report(args.root, args.manifest))


if __name__ == "__main__":
    main()
