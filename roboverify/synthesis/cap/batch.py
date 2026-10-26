"""Bounded process-isolated CaP batches with immutable manifests and safe resume."""

import fcntl
import hashlib
import json
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from .config import GPUConfig, ModelConfig, RunConfig
from .prompt_package import package

METHODS = ("cap_google_baseline", "cap_method_1", "cap_method_2")
MODES = ("dsl_only", "coordinate_predicates")


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def variant_config(base, overrides):
    """Apply variant overrides; recovery feedback/history blocks merge per field."""
    config = base | {k: v for k, v in overrides.items() if k != "recovery"}
    recovery = dict(base["recovery"])
    for block, values in overrides.get("recovery", {}).items():
        recovery[block] = dict(recovery[block]) | values
    return config | {"recovery": recovery}


def create_batch(
    directory,
    config,
    model,
    *,
    initial_state=None,
    instruction=None,
    methods=METHODS,
    modes=MODES,
    trials=5,
    max_workers=2,
    schedule_seed=0,
    responses=None,
    fixture=False,
    recovery_variants=None,
    transport_control=None,
    gpu_config=None,
    experiment_variants=None,
    render_workers=2,
    render_timeout=600.0,
    seed_per_trial=False,
):
    """Plan a batch. ``seed_per_trial`` gives trial k the scene seed
    ``config.seed + k - 1`` (fresh resets); otherwise every trial uses ``config.seed``.
    """
    directory = Path(directory).resolve()
    if (
        type(render_workers) is not int
        or render_workers < 1
        or not 0 < render_timeout < float("inf")
    ):
        raise ValueError("Invalid rendering worker count or timeout")
    if directory.exists():
        raise FileExistsError("Batch directory already exists; use resume")
    if seed_per_trial and initial_state:
        raise ValueError("Per-trial seeds apply to fresh resets, not a fixed snapshot")
    if (
        trials < 1
        or max_workers < 1
        or not set(methods) <= set(METHODS)
        or not set(modes) <= set(MODES)
    ):
        raise ValueError("Invalid batch configuration")
    if (
        not methods
        or not modes
        or len(set(methods)) != len(methods)
        or len(set(modes)) != len(modes)
    ):
        raise ValueError("Methods and modes must be nonempty and unique")
    if "cap_method_2" in methods and not package(config.prompt_version).get(
        "method2_templates"
    ):
        raise ValueError("Method 2 requires a prompt package with method2_templates")
    if recovery_variants is not None and (
        not recovery_variants
        or len(set(recovery_variants)) != len(recovery_variants)
        or any(type(v) is not bool for v in recovery_variants)
    ):
        raise ValueError("Recovery variants must be unique Boolean settings")
    variants = experiment_variants or [{"name": "default", "run_config": {}}]
    if experiment_variants is not None:
        if not isinstance(experiment_variants, list) or not experiment_variants:
            raise ValueError("Experiment variants must be a nonempty list")
        names = [v["name"] for v in variants]
        if len(set(names)) != len(names) or any(
            not re.fullmatch(r"[a-z][a-z0-9_]*", n) for n in names
        ):
            raise ValueError("Experiment variant names must be unique safe identifiers")
        for v in variants:
            allowed = v.get("methods", list(methods))
            if (
                not allowed
                or len(set(allowed)) != len(allowed)
                or not set(allowed) <= set(methods)
            ):
                raise ValueError("Invalid variant methods")
            overrides = v.get("run_config", {})
            if set(overrides) - {
                "primitive_prompt",
                "primitive_feedback",
                "recovery",
            } or set(overrides.get("recovery", {})) - {"feedback", "history"}:
                raise ValueError(
                    "Variant overrides are restricted to primitive prompt/feedback "
                    "and recovery feedback/history settings"
                )
            RunConfig(**variant_config(asdict(config), overrides))
            seed = v.get("seed_attempt")
            if seed is not None and (
                not Path(seed).is_file()
                or (recovery_variants or [config.recovery.enabled]) != [True]
            ):
                raise ValueError(
                    "A seeded variant needs an existing seed result and recovery on"
                )
    if gpu_config is not None:
        gpu_config = (
            gpu_config if isinstance(gpu_config, GPUConfig) else GPUConfig(**gpu_config)
        )
        gpu_config.reservation_directory = str(
            Path(gpu_config.reservation_directory).resolve()
        )
    directory.mkdir(parents=True)
    snapshot = None
    snapshot_hash = None
    if initial_state:
        snapshot = directory / "inputs/initial_state.npz"
        snapshot.parent.mkdir()
        shutil.copyfile(initial_state, snapshot)
        snapshot_hash = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    rng = random.Random(schedule_seed)
    jobs = []
    for trial in range(1, trials + 1):
        groups = [
            (method, mode, recovery, variant)
            for variant in variants
            for method in methods
            if method in variant.get("methods", methods)
            for mode in modes
            for recovery in (
                recovery_variants
                if recovery_variants is not None
                else [config.recovery.enabled]
            )
        ]
        rng.shuffle(groups)
        for method, mode, recovery, variant in groups:
            ident = f"{method}-{mode}-{trial:03d}"
            if recovery_variants is not None:
                ident += "-recovery-" + ("on" if recovery else "off")
            if experiment_variants is not None:
                ident += "-" + variant["name"]
            job_config = variant_config(asdict(config), variant.get("run_config", {}))
            job_config |= {
                "method": method,
                "condition_mode": mode,
                "seed": config.seed + trial - 1 if seed_per_trial else config.seed,
                "recovery": job_config["recovery"] | {"enabled": recovery},
                "output_root": str(directory / "trials" / ident),
            }
            jobs.append(
                {
                    "id": ident,
                    "variant": variant["name"],
                    "trial": trial,
                    "run_config": job_config,
                    "model_config": asdict(model),
                    "instruction": instruction,
                    "initial_state": str(snapshot) if snapshot else None,
                    "responses": responses,
                    "fixture": fixture,
                    "transport_control": transport_control,
                }
                # Branched recovery: round 0 is the archived seed attempt.
                | (
                    {"seed_attempt": str(Path(variant["seed_attempt"]).resolve())}
                    if variant.get("seed_attempt")
                    else {}
                )
            )
    manifest = {
        "schema": "cap-batch-v1",
        "created_epoch": time.time(),
        "max_workers": max_workers,
        "render_workers": min(render_workers, max_workers),
        "render_timeout": render_timeout,
        "gpu_config": asdict(gpu_config) if gpu_config else None,
        "experiment_variants": experiment_variants,
        "schedule_seed": schedule_seed,
        "seed_per_trial": seed_per_trial,
        "snapshot_sha256": snapshot_hash,
        "fixture": fixture,
        "scripted": responses is not None,
        "jobs": jobs,
    }
    for job in jobs:
        Path(job["run_config"]["output_root"]).mkdir(parents=True)
    atomic_json(directory / "manifest.json", manifest)
    (directory / "manifest.sha256").write_text(
        hashlib.sha256((directory / "manifest.json").read_bytes()).hexdigest() + "\n"
    )
    atomic_json(
        directory / "schedule.json", {job["id"]: {"status": "pending"} for job in jobs}
    )
    return manifest


def _run_directory(job):
    configs = list(Path(job["run_config"]["output_root"]).glob("cap/*/config.json"))
    configs = [p for p in configs if p.parent.name != "latest"]
    return configs[0].parent if len(configs) == 1 else None


def collect_job(job, state):
    from synthesis.experiment.report import RunSummary

    from .audit import audit_run

    path = Path(job["run_config"]["output_root"])
    result_file = path / "worker_result.json"
    result = (
        json.loads(result_file.read_text())
        if result_file.exists()
        else {
            "status": state["status"],
            "success": False,
            "task_spec_success": False,
            "reason": state.get("reason", "Worker did not produce a result"),
        }
    )
    run_dir = Path(result["run_dir"]) if result.get("run_dir") else _run_directory(job)
    result.update(
        method=job["run_config"]["method"],
        condition_mode=job["run_config"]["condition_mode"],
        trial=job["trial"],
        job_id=job["id"],
        variant=job.get("variant", "default"),
    )
    if run_dir:
        summary = RunSummary(run_dir)
        if not result_file.exists() and summary.result:
            # Recover the immutable runner outcome, without rerunning a trial.
            result.update(
                {k: v for k, v in summary.result.items() if k not in {"status"}}
            )
        result["run_dir"] = str(run_dir)
        attempt_dirs = [Path(a["run_dir"]) for a in result.get("attempts", [])] or [
            run_dir
        ]
        calls = sorted(
            p
            for d in attempt_dirs
            for p in (d / "artifacts/trial_001/lmp").glob("call_*")
        )
        total, missing = 0, 0
        for call in calls:
            response = call / "response.json"
            usage = (
                json.loads(response.read_text()).get("usage", {})
                if response.exists()
                else {}
            )
            if usage.get("total_tokens") is None:
                missing += 1
            else:
                total += int(usage["total_tokens"])
        result.update(
            model_calls=len(calls), known_tokens=total, unknown_usage_calls=missing
        )
        if (
            result["method"] == "cap_google_baseline"
            and job["run_config"]["baseline_audit"]
            and not result.get("recovery_enabled")
        ):
            audit_path = run_dir / "artifacts/trial_001/dsl_compliance.json"
            audit = (
                json.loads(audit_path.read_text())
                if audit_path.exists()
                else audit_run(run_dir, timeout=job["run_config"]["audit_timeout"])
            )
            result.update(
                dsl_compliance=audit["status"], audit_seconds=audit["elapsed_seconds"]
            )
    result.setdefault(
        "recovery_enabled", job["run_config"].get("recovery", {}).get("enabled", False)
    )
    result.setdefault(
        "first_attempt_success",
        result.get("success") is True and result.get("dsl_postcondition") is not False,
    )
    result.setdefault(
        "first_attempt_task_spec_success", result.get("task_spec_success") is True
    )
    result.setdefault("recovery_rounds", 0)
    if run_dir:
        (path / "report.txt").write_text(RunSummary(run_dir).render() + "\n")
    result["process_status"] = state["status"]
    result["worker_exit_code"] = state.get("exit_code")
    if state.get("gpu_assignment"):
        result["gpu_assignment"] = state["gpu_assignment"]
    return result


def write_report(directory, manifest, schedule):
    rows = []
    for job in manifest["jobs"]:
        state = schedule[job["id"]]
        if state["status"] not in {"pending", "running"}:
            cached = directory / "trials" / job["id"] / "summary.json"
            if cached.exists():
                rows.append(json.loads(cached.read_text()))
    from .deferred_video import resolved_video

    for row in rows:
        row["video"] = resolved_video(row.get("video"))
        for attempt in row.get("attempts", []):
            attempt["video"] = resolved_video(attempt.get("video"))
    video_summary = (
        json.loads((directory / "video_summary.json").read_text())
        if (directory / "video_summary.json").exists()
        else {}
    )
    atomic_json(
        directory / "summary.json",
        {
            "planned": len(manifest["jobs"]),
            "finished": len(rows),
            "trials": rows,
            "video_rendering": video_summary,
        },
    )
    groups = defaultdict(list)
    for row in rows:
        groups[
            (
                row["method"],
                row["condition_mode"],
                row.get("recovery_enabled", False),
                row.get("variant", "default"),
            )
        ].append(row)
    planned = defaultdict(int)
    for job in manifest["jobs"]:
        planned[
            (
                job["run_config"]["method"],
                job["run_config"]["condition_mode"],
                job["run_config"].get("recovery", {}).get("enabled", False),
                job.get("variant", "default"),
            )
        ] += 1
    lines = [
        "# CaP batch report",
        "",
        "Status: "
        + ("complete" if len(rows) == len(manifest["jobs"]) else "in progress"),
        "",
    ]
    if video_summary.get("jobs"):
        lines += [
            f"Deferred videos: {video_summary['counts']}. Video completion is separate from experiment status.",
            "",
        ]
    if manifest["fixture"]:
        lines += ["Scripted fixture diagnostic; not a physical task benchmark.", ""]
    if not manifest["fixture"] and manifest.get(
        "scripted", manifest["jobs"][0].get("responses") is not None
    ):
        lines += [
            "Scripted-response simulator integration diagnostic; no live model generation.",
            "",
        ]
    first = manifest["jobs"][0]
    c, m = first["run_config"], first["model_config"]
    lines += [
        f"Model: `{m['model']}`. Prompt: `{c['prompt_version']}`. Output limit: {m['max_output_tokens']}. Reasoning/options: `{json.dumps(m['extra_body'], sort_keys=True)}`.",
        f"Workers: {manifest['max_workers']}; schedule seed: {manifest['schedule_seed']}. Each trial: {c['timeout']} s, {c['max_model_calls']} calls, {c['max_tokens']} reported tokens, {c['max_sim_steps']} simulation steps.",
        "Recovery-enabled trials allow four additional rounds (100k tokens, 16 calls, 600 s each), with whole-trial caps from recovery configuration. Per-request output remains capped; all attempts count toward tokens and calls.",
        "All trials use the manifest snapshot; queue waiting is separate from active trial time. Task-spec success is RoboVerify's criterion (normal completion, converged primitives, shared pre/postcondition); physical success and DSL compliance are separate metrics. Failed/incomplete trials remain in the denominator.",
        "",
        "| Method | Variant | Condition mode | Recovery | First-attempt task-spec success | Final task-spec success | First-attempt success | Final success | DSL pass | Finished / planned |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for key, count in sorted(planned.items()):
        data = groups[key]
        lines.append(
            f"| {key[0]} | {key[3]} | {key[1]} | {'on' if key[2] else 'off'} | {sum(r.get('first_attempt_task_spec_success') is True for r in data)}/{count} | {sum(r.get('task_spec_success') is True for r in data)}/{count} | {sum(r.get('first_attempt_success', r.get('success')) is True for r in data)}/{count} | {sum(r.get('success') is True for r in data)}/{count} | {sum(r.get('dsl_compliance') == 'pass' for r in data)}/{count} | {len(data)}/{count} |"
        )
    lines += [
        "",
        "| Method | Variant | Mode | Recovery | Trial | Status | Task-spec success | Success | DSL compliance | # of tokens used | Calls | Recovery rounds | Active s | Queue s | Failure / stop reason |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in sorted(
        rows,
        key=lambda r: (
            r["method"],
            r.get("variant", "default"),
            r["condition_mode"],
            r["trial"],
        ),
    ):
        token = str(row.get("known_tokens", 0)) + (
            "*" if row.get("unknown_usage_calls", 0) else ""
        )
        reason = (
            str(row.get("stop_reason", row.get("reason", "")))
            .replace("|", "/")
            .replace("\n", " ")
        )
        link = (
            f"[trial {row['trial']}](trials/{row['job_id']}/report.txt)"
            if row.get("run_dir")
            else str(row["trial"])
        )
        lines.append(
            f"| {row['method']} | {row.get('variant', 'default')} | {row['condition_mode']} | {'on' if row.get('recovery_enabled') else 'off'} | {link} | {row['status']} | {row.get('task_spec_success')} | {row.get('success')} | {row.get('dsl_compliance','not_available')} | {token} | {row.get('model_calls',0)} | {row.get('recovery_rounds',0)} | {row.get('active_wall_seconds',0):.2f} | {row.get('queue_wait_seconds',0):.2f} | {reason} |"
        )
    lines += [
        "",
        "Tokens sum returned input/output usage across policy/helper/repair/transport attempts; reasoning tokens are not added twice. `*` marks incomplete provider usage. Baseline compliance checks final output only. Method 1 uses local repair; Method 2 repairs whole candidates. Static compliance does not guarantee task success.",
        "",
    ]
    temporary = directory / "report.md.tmp"
    temporary.write_text("\n".join(lines))
    temporary.replace(directory / "report.md")


def _execute_batch(directory, *, resume=False, dispatch_gate=None, max_workers=None):
    directory = Path(directory).resolve()
    if (
        hashlib.sha256((directory / "manifest.json").read_bytes()).hexdigest()
        != (directory / "manifest.sha256").read_text().strip()
    ):
        raise ValueError("Immutable batch manifest changed")
    manifest = json.loads((directory / "manifest.json").read_text())
    schedule = json.loads((directory / "schedule.json").read_text())
    if manifest["snapshot_sha256"]:
        actual = hashlib.sha256(
            Path(manifest["jobs"][0]["initial_state"]).read_bytes()
        ).hexdigest()
        if actual != manifest["snapshot_sha256"]:
            raise ValueError("Batch snapshot changed")
    if not resume and any(s["status"] != "pending" for s in schedule.values()):
        raise ValueError("Batch has already started; use resume")
    for job in manifest["jobs"]:
        state = schedule[job["id"]]
        if state["status"] == "running":
            pid = state.get("pid")
            if not pid:
                for candidate in Path("/proc").glob("[0-9]*/cmdline"):
                    try:
                        if (
                            str(directory / "trials" / job["id"] / "job.json").encode()
                            in candidate.read_bytes()
                        ):
                            raise RuntimeError(
                                "An unrecorded prior worker is still running"
                            )
                    except (FileNotFoundError, PermissionError, ProcessLookupError):
                        pass
            if pid and Path(f"/proc/{pid}/cmdline").exists():
                cmdline = Path(f"/proc/{pid}/cmdline").read_bytes()
                if (
                    str(directory / "trials" / job["id"] / "job.json").encode()
                    in cmdline
                ):
                    raise RuntimeError(
                        "A prior worker is still running; refuse duplicate dispatch"
                    )
            state.update(
                status="interrupted",
                reason="Recovered an interrupted worker; no automatic retry",
            )
            row = collect_job(job, state)
            atomic_json(directory / "trials" / job["id"] / "summary.json", row)
    pending = [j for j in manifest["jobs"] if schedule[j["id"]]["status"] == "pending"]
    gpu = None
    if manifest.get("gpu_config") and (
        pending or (directory / "render_queue").exists()
    ):
        from .gpu_scheduler import GPUScheduler

        gpu = GPUScheduler(manifest["gpu_config"])
    from .deferred_video import RenderPool

    renders = RenderPool(
        directory,
        gpu=gpu,
        max_workers=manifest.get("render_workers", 2),
        timeout=manifest.get("render_timeout", 600),
    )
    active = {}
    metadata_path = directory / "execution_metadata.json"
    metadata = (
        json.loads(metadata_path.read_text())
        if metadata_path.exists()
        else {
            "requested_workers": manifest["max_workers"],
            "peak_active_workers": 0,
            "sessions": [],
        }
    )
    # Concurrency is a scheduling choice, not part of a trial's definition, so a
    # resumed session may change it; each session records the value it used.
    if max_workers is not None and max_workers < 1:
        raise ValueError("max_workers must be positive")
    workers = max_workers or manifest["max_workers"]
    metadata["sessions"].append(
        {"started_epoch": time.time(), "resume": resume, "max_workers": workers}
    )
    atomic_json(metadata_path, metadata)
    blocked_since = None
    saved_gpu_sample = None
    try:
        while pending or active or renders.pending() or renders.active:
            monitoring_error = None
            if gpu:
                from .gpu_scheduler import GPUUnavailable

                try:
                    gpu.poll()
                except GPUUnavailable as exc:
                    monitoring_error = str(exc)
                    metadata["gpu_wait_reason"] = monitoring_error
                    blocked_since = (
                        time.monotonic() if blocked_since is None else blocked_since
                    )
                metadata["gpu_telemetry"] = gpu.telemetry
                metadata["worker_resource_samples"] = gpu.worker_telemetry
                if gpu.latest_sample and gpu.latest_sample["epoch"] != saved_gpu_sample:
                    with (directory / "gpu_samples.jsonl").open("a") as samples:
                        samples.write(json.dumps(gpu.latest_sample) + "\n")
                    saved_gpu_sample = gpu.latest_sample["epoch"]
                atomic_json(metadata_path, metadata)
            if not monitoring_error:
                renders.tick(workers - len(active))
            metadata["peak_combined_workers"] = max(
                metadata.get("peak_combined_workers", 0),
                len(active) + len(renders.active),
            )
            admission_blocked = False
            while pending and len(active) + len(renders.active) < workers:
                if monitoring_error:
                    break
                if dispatch_gate is not None and not dispatch_gate(
                    pending[0], [entry[2] for entry in active.values()]
                ):
                    admission_blocked = True
                    break
                job = pending[0]
                path = directory / "trials" / job["id"]
                path.mkdir(parents=True, exist_ok=True)
                assignment = gpu.reserve(path / "job.json") if gpu else None
                if gpu and assignment is None:
                    blocked_since = (
                        time.monotonic() if blocked_since is None else blocked_since
                    )
                    metadata["gpu_wait_reason"] = gpu.last_reason
                    atomic_json(metadata_path, metadata)
                    break
                blocked_since = None
                pending.pop(0)
                spec = job | {
                    "enqueued_epoch": manifest["created_epoch"],
                    "batch_dir": str(directory),
                    "gpu_assignment": assignment,
                }
                atomic_json(path / "job.json", spec)
                handle = (path / "worker.log").open("ab")
                state = schedule[job["id"]]
                state.update(status="running", pid=None, dispatched_epoch=time.time())
                if assignment:
                    state["gpu_assignment"] = assignment
                atomic_json(directory / "schedule.json", schedule)
                try:
                    process = subprocess.Popen(
                        [
                            sys.executable,
                            "-m",
                            "synthesis.cap.batch_worker",
                            str(path / "job.json"),
                        ],
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                        env=(
                            dict(os.environ, GPUS=str(assignment["egl_index"]))
                            if assignment
                            else None
                        ),
                    )
                except OSError as exc:
                    handle.close()
                    if assignment:
                        gpu.release(assignment["token"])
                    state.update(
                        status="worker_failed",
                        reason=f"Worker launch failed: {exc}",
                        ended_epoch=time.time(),
                    )
                    atomic_json(path / "summary.json", collect_job(job, state))
                    atomic_json(directory / "schedule.json", schedule)
                    write_report(directory, manifest, schedule)
                    continue
                state = schedule[job["id"]]
                state.update(
                    status="running", pid=process.pid, dispatched_epoch=time.time()
                )
                active[job["id"]] = (process, handle, job, time.monotonic())
                if assignment:
                    gpu.bind(assignment["token"], process.pid)
                metadata["peak_combined_workers"] = max(
                    metadata.get("peak_combined_workers", 0),
                    len(active) + len(renders.active),
                )
                metadata["peak_active_workers"] = max(
                    metadata["peak_active_workers"], len(active)
                )
                atomic_json(metadata_path, metadata)
                atomic_json(directory / "schedule.json", schedule)
            for ident, (process, handle, job, started) in list(active.items()):
                timeout = (
                    (
                        job["run_config"].get("recovery", {}).get("total_timeout", 3000)
                        if job["run_config"].get("recovery", {}).get("enabled")
                        else job["run_config"]["timeout"]
                    )
                    + job["run_config"]["audit_timeout"]
                    + 60
                )
                expired = time.monotonic() - started > timeout
                if expired and process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                code = process.poll()
                if code is not None:
                    handle.close()
                    if gpu:
                        gpu.release(schedule[ident]["gpu_assignment"]["token"])
                    state = schedule[ident]
                    state.update(
                        status="completed" if code == 0 else "worker_failed",
                        exit_code=code,
                        ended_epoch=time.time(),
                    )
                    if code:
                        state["reason"] = (
                            "Worker hard deadline"
                            if expired
                            else f"Worker exited with code {code}"
                        )
                    row = collect_job(job, state)
                    atomic_json(directory / "trials" / ident / "summary.json", row)
                    del active[ident]
                    atomic_json(directory / "schedule.json", schedule)
                    write_report(directory, manifest, schedule)
                    print(
                        f"Finished {ident}: {row['status']}; DSL={row.get('dsl_compliance')}",
                        flush=True,
                    )
            if pending and not active and admission_blocked:
                # Unstarted trials remain pending in the schedule; drain available videos.
                pending = []
            if gpu and pending and not active and blocked_since is not None:
                if (
                    time.monotonic() - blocked_since
                    >= gpu.config.admission_timeout_seconds
                ):
                    metadata["pause_reason"] = (
                        "GPU admission timeout; unstarted trials remain pending"
                    )
                    atomic_json(metadata_path, metadata)
                    break
            if not pending and not active:
                if "experiment_finished_epoch" not in metadata and all(
                    v["status"] not in {"pending", "running"} for v in schedule.values()
                ):
                    metadata["experiment_finished_epoch"] = time.time()
                    atomic_json(metadata_path, metadata)
                if (
                    renders.blocked_since
                    and not renders.active
                    and gpu
                    and (
                        time.monotonic() - renders.blocked_since
                        >= gpu.config.admission_timeout_seconds
                    )
                ):
                    metadata["video_pause_reason"] = (
                        "GPU admission timeout; video jobs remain resumable"
                    )
                    break
                if (
                    monitoring_error
                    and not renders.active
                    and blocked_since is not None
                    and gpu
                    and (
                        time.monotonic() - blocked_since
                        >= gpu.config.admission_timeout_seconds
                    )
                ):
                    break
            if active or pending or renders.active or renders.pending():
                time.sleep(0.2)
    finally:
        for ident, (process, handle, job, _) in active.items():
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            handle.close()
            if gpu:
                gpu.release(schedule[ident]["gpu_assignment"]["token"])
            schedule[ident].update(
                status="interrupted",
                reason="Parent interrupted; no automatic retry",
                ended_epoch=time.time(),
            )
            atomic_json(
                directory / "trials" / ident / "summary.json",
                collect_job(job, schedule[ident]),
            )
        renders.close()
        metadata["video_rendering"] = renders.report()
        if metadata["video_rendering"]["all_videos_completed"]:
            metadata["all_videos_finished_epoch"] = time.time()
        atomic_json(metadata_path, metadata)
        atomic_json(directory / "schedule.json", schedule)
        write_report(directory, manifest, schedule)
    return json.loads((directory / "summary.json").read_text())


def execute_batch(directory, *, resume=False, dispatch_gate=None, max_workers=None):
    directory = Path(directory).resolve()
    with (directory / ".batch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _execute_batch(
            directory,
            resume=resume,
            dispatch_gate=dispatch_gate,
            max_workers=max_workers,
        )


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True)
    parser.add_argument(
        "--config", help="JSON containing run_config, model_config, and batch options"
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.resume:
        if args.config:
            parser.error("Resume uses the immutable manifest; omit --config")
    else:
        if not args.config:
            parser.error("A new batch requires --config")
        options = json.loads(Path(args.config).read_text())
        config = RunConfig(**options.pop("run_config"))
        model = ModelConfig(**options.pop("model_config"))
        create_batch(args.directory, config, model, **options)
    summary = execute_batch(args.directory, resume=args.resume)
    print(json.dumps({"finished": summary["finished"], "planned": summary["planned"]}))


if __name__ == "__main__":
    main()
