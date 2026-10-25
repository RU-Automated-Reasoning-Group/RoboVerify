"""Revision 4.3 feedback ablation: shared round 0, branched recovery.

Subcommands follow docs/revision_4_3_ablation_experiment_plan.md:

- ``stage-a``: round-0-only attempts (recovery on, ``max_rounds=0``) that supply
  failed seed candidates, with a fixed primitive prompt set.
- ``select-seeds``: classify failed round-0 attempts from Stage A batches or
  existing batches (read only) and sample an immutable, stratified seed manifest.
- ``stage-b``: one recovery branch per seed x feedback profile, every branch
  starting from the same archived round-0 attempt.
- ``audit``: check shared seeds, profile settings and per-item delivery.
- ``summarize``: paired per-seed outcomes per profile against ``full``.
"""

import argparse
import hashlib
import json
import math
import random
import re
import subprocess
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

from ..batch import METHODS, _run_directory, atomic_json, create_batch
from ..config import FEEDBACK_ITEMS, GPUConfig, PrimitivePromptConfig
from .revision42_experiment import GPU_UUIDS, execute
from .revision43_prompt_experiment import (
    CREDENTIAL,
    MODELS,
    SWITCHES,
    HealthGate,
    configuration,
    serving_record,
)

CLASSES = (
    "move_nonconvergence",
    "pick_nonconvergence",
    "release_nonconvergence",
    "execution_budget",
    "no_witness",
    "goal_failure",
    "other_runtime",
)
QUOTAS = {
    "move_nonconvergence": 15,
    "pick_nonconvergence": 8,
    "release_nonconvergence": 8,
    "execution_budget": 6,
    "no_witness": 5,
    "goal_failure": 4,
    "other_runtime": 4,
}


def scaled_quotas(total, base=QUOTAS):
    """Scale the per-method class quotas to ``total`` seeds (largest remainder).

    Ties go to the class listed first in ``base``, so the result is deterministic.
    """
    if total < 1:
        raise ValueError("Seed total must be positive")
    scale = total / sum(base.values())
    exact = {k: v * scale for k, v in base.items()}
    quotas = {k: math.floor(v) for k, v in exact.items()}
    order = sorted(base, key=lambda k: -(exact[k] - quotas[k]))
    for name in order[: total - sum(quotas.values())]:
        quotas[name] += 1
    return quotas


ENRICHMENTS = ("target_derivation", "controller_phases", "action_context")
SUPPORTING_EVENTS = 40
EVIDENCE_GUARD = 32768
SEED_SCHEMA = "cap-revision43-seeds-v1"
SHORT = {"cap_google_baseline": "baseline", "cap_method_1": "m1", "cap_method_2": "m2"}
PROFILES = (
    "full",
    "no_current_source",
    "no_failure_location",
    "no_failure_events",
    "no_keyframes",
    "no_goal_clauses",
    "no_supporting_events",
    "no_metadata",
    "no_target",
    "no_phases",
    "no_context",
    "history_summaries",
    "history_none",
    "minimal",
    "legacy_v1",
)
# Items removed by each leave-one-out profile (plan Section 4).
REMOVED = {
    "no_current_source": ("current_source",),
    "no_failure_location": ("failure_location",),
    "no_failure_events": ("failure_events", "query_candidates"),
    "no_keyframes": ("keyframes",),
    "no_goal_clauses": ("goal_clauses",),
    "no_supporting_events": ("supporting_events",),
    "no_metadata": ("event_counts", "selection_metadata"),
    "no_target": ("target_derivation",),
    "no_phases": ("controller_phases",),
    "no_context": ("action_context",),
}


def profile_settings(
    name, *, supporting_events=SUPPORTING_EVENTS, guard=EVIDENCE_GUARD
):
    """Resolved feedback items, history mode and enrichments for one profile."""
    if name not in PROFILES:
        raise ValueError(f"Unknown profile: {name}")
    items = dict.fromkeys(FEEDBACK_ITEMS, True)
    enrichments = dict.fromkeys(ENRICHMENTS, True)
    history = "executed_programs"
    for item in REMOVED.get(name, ()):
        if item in items:
            items[item] = False
        else:
            enrichments[item] = False
    if name == "no_current_source":
        # executed_programs needs the current source in its registry; without
        # it, earlier attempts can only appear as summaries.
        history = "summaries"
    elif name == "history_summaries":
        history = "summaries"
    elif name == "history_none":
        history = "none"
    elif name == "minimal":
        # Core plus current source; DSL diagnostics are held on in every profile.
        items = {item: item in {"current_source", "dsl_diagnostics"} for item in items}
        enrichments = dict.fromkeys(ENRICHMENTS, False)
        history = "none"
    elif name == "legacy_v1":
        enrichments = dict.fromkeys(ENRICHMENTS, False)
        history = "summaries"
    feedback = items | {
        "supporting_event_window": supporting_events,
        "evidence_layout": "fixed_items",
        "evidence_bytes": guard,
    }
    return {
        "recovery": {"feedback": feedback, "history": {"mode": history}},
        "primitive_feedback": enrichments,
    }


def prompt_flags(text):
    """Parse ``--prompt-flags`` as JSON or four 0/1 characters (SWITCHES order)."""
    if re.fullmatch(r"[01]{4}", text):
        flags = {s: c == "1" for s, c in zip(SWITCHES, text)}
    else:
        flags = json.loads(text)
    return asdict(PrimitivePromptConfig(**flags))


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _local_plan(directory, *, planned, request_limit, extra):
    limits = {CREDENTIAL: request_limit}
    plan = {
        "planned": planned,
        "preflight": False,
        "admission": "paid_request_cap",
        "credential_limits": limits,
        "request_limit": request_limit,
        "git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_sha256": {
            str(p): file_sha256(p)
            for p in Path("synthesis/cap").rglob("*")
            if p.is_file()
            and p.suffix in {".py", ".json"}
            and "docs" not in p.parts
            and "__pycache__" not in p.parts
        },
        "no_failed_trial_replacements": True,
    } | extra
    atomic_json(Path(directory) / "comparison_plan.json", plan)
    return plan


def _transport(directory, request_limit):
    return {
        "state_path": str(Path(directory) / "request_gate.json"),
        "interval_seconds": 0.0,
        "request_limit": request_limit,
        "credential_limits": {CREDENTIAL: request_limit},
    }


def _gpu(gpu=None):
    """Worker GPU policy.

    The scheduler caps each device at the smallest slot limit among live
    reservations, so batches sharing a GPU must use one per-GPU value.
    """
    gpu = gpu or {}
    return GPUConfig(
        list(gpu.get("uuids") or GPU_UUIDS),
        "/tmp/roboverify-cap-gpu-reservations",
        max_workers_per_gpu=gpu.get("workers_per_gpu", 48),
        min_free_memory_mb=2048,
        worker_memory_mb=gpu.get("worker_memory_mb", 128),
        gpu_poll_interval_seconds=2,
        gpu_sample_window_seconds=10,
        admission_timeout_seconds=60,
    )


# --------------------------------------------------------------------- Stage A


def create_stage_a(directory, *, model_name, flags, trials, max_workers=32, gpu=None):
    directory = Path(directory).resolve()
    previous, job, config, model = configuration(model_name)
    config.recovery.max_rounds = 0
    planned = trials * len(METHODS)
    # Recovery stays enabled (max_rounds=0), so the request gate reserves the
    # full recovery call allowance for every active trial.
    request_limit = planned * config.recovery.total_calls
    manifest = create_batch(
        directory,
        config,
        model,
        initial_state=job["initial_state"],
        instruction=job["instruction"],
        modes=["dsl_only"],
        trials=trials,
        max_workers=max_workers,
        schedule_seed=4301,
        recovery_variants=[True],
        experiment_variants=[
            {"name": "stage_a", "run_config": {"primitive_prompt": flags}}
        ],
        gpu_config=_gpu(gpu),
        transport_control=_transport(directory, request_limit),
    )
    assert manifest["snapshot_sha256"] == previous["snapshot_sha256"]
    _local_plan(
        directory,
        planned=planned,
        request_limit=request_limit,
        extra={
            "experiment": "revision43_feedback_stage_a",
            "model": model.model,
            "serving": serving_record(model.model),
            "prompt_flags": flags,
            "max_rounds": 0,
            "snapshot_sha256": manifest["snapshot_sha256"],
        },
    )
    return manifest


# -------------------------------------------------------------- seed selection


def rolled_out(run_dir):
    timeline = Path(run_dir) / "artifacts/trial_001/timeline.jsonl"
    if not timeline.is_file():
        return False
    with timeline.open() as handle:
        return any('"policy_rollout_start"' in line for line in handle)


def classify(result, rolled):
    """Failure class of a round-0 attempt; None for success."""
    if result.get("task_success") is True:
        return None
    if not rolled:
        return "preparation"
    status = result.get("status")
    reason = result.get("reason") or ""
    if status == "primitive_failed":
        match = re.match(r"PrimitiveFailure: (Move|Pick|Release)\b", reason)
        return f"{match.group(1).lower()}_nonconvergence" if match else "other_runtime"
    if reason.startswith("CapError: No object satisfies get condition"):
        return "no_witness"
    if status == "budget_exhausted":
        exhausted = result.get("budget_exhausted_by")
        if exhausted in {"execution_ticks", "simulation_steps"}:
            return "execution_budget"
        return "other_runtime"
    if status in {"goal_failed", "completed"}:
        return "goal_failure"
    return "other_runtime"


def candidates(source, variant=None):
    """Round-0 attempts of one batch (read only), optionally one variant."""
    source = Path(source).resolve()
    manifest = json.loads((source / "manifest.json").read_text())
    found = []
    for job in manifest["jobs"]:
        if variant and job.get("variant") != variant:
            continue
        run_dir = _run_directory(job)
        result_path = (
            run_dir / "artifacts/trial_001/rounds/000/result.json" if run_dir else None
        )
        if result_path is None or not result_path.is_file():
            continue
        result = json.loads(result_path.read_text())
        attempt = Path(result["run_dir"])
        state = attempt / "artifacts/trial_001/initial_state.npz"
        found.append(
            {
                "source_batch": str(source),
                "job_id": job["id"],
                "variant": job.get("variant"),
                "model": job["model_config"]["model"],
                "method": job["run_config"]["method"],
                "prompt_flags": asdict(
                    PrimitivePromptConfig(
                        **job["run_config"].get("primitive_prompt", {})
                    )
                ),
                "seed_result": str(result_path),
                "seed_result_sha256": file_sha256(result_path),
                "attempt_run_dir": str(attempt),
                "initial_state_sha256": file_sha256(state) if state.is_file() else None,
                "status": result.get("status"),
                "reason": (result.get("reason") or "")[:200],
                "class": classify(result, rolled_out(attempt)),
            }
        )
    return found


def sample(pool, quotas, rng):
    """Deterministic stratified sample with proportional refill of shortfalls."""
    by_class = defaultdict(list)
    for item in sorted(pool, key=lambda c: c["seed_result"]):
        by_class[item["class"]].append(item)
    for items in by_class.values():
        rng.shuffle(items)
    taken = {c: min(q, len(by_class[c])) for c, q in quotas.items()}
    shortfall = sum(quotas.values()) - sum(taken.values())
    surplus = {
        c: len(by_class[c]) - taken[c] for c in quotas if len(by_class[c]) > taken[c]
    }
    refill = dict.fromkeys(quotas, 0)
    if shortfall and surplus:
        total = sum(surplus.values())
        share = min(shortfall, total)
        exact = {c: share * n / total for c, n in surplus.items()}
        refill.update({c: math.floor(v) for c, v in exact.items()})
        order = sorted(exact, key=lambda c: (-(exact[c] - refill[c]), CLASSES.index(c)))
        for c in order[: share - sum(refill.values())]:
            refill[c] += 1
    selected = []
    for c in quotas:
        selected += [i | {"quota_role": "quota"} for i in by_class[c][: taken[c]]]
        extra = by_class[c][taken[c] : taken[c] + refill[c]]
        selected += [i | {"quota_role": "refill"} for i in extra]
    report = {
        c: {
            "available": len(by_class[c]),
            "quota": quotas[c],
            "taken": taken[c],
            "refill": refill[c],
            "shortfall": quotas[c] - taken[c],
        }
        for c in quotas
    }
    return selected, report


def select_seeds(output, sources, *, model, flags, quotas=QUOTAS, rng_seed=43):
    """Write an immutable seed manifest; refuse to overwrite an existing one."""
    output = Path(output)
    if output.exists():
        raise FileExistsError("Seed manifest already exists; it is immutable")
    pool = []
    for source in sources:
        batch, _, variant = str(source).partition(":")
        pool += candidates(batch, variant or None)
    pool = [c for c in pool if c["model"] == model and c["prompt_flags"] == flags]
    counts = Counter((c["method"], c["class"]) for c in pool)
    states = {c["initial_state_sha256"] for c in pool if c["class"]}
    if len(states - {None}) > 1:
        raise ValueError("Seed candidates use different initial states")
    seeds, strata = [], {}
    for method in METHODS:
        failed = [
            c
            for c in pool
            if c["method"] == method and c["class"] not in {None, "preparation"}
        ]
        chosen, report = sample(
            failed, quotas, random.Random(f"{rng_seed}:{model}:{method}")
        )
        for index, item in enumerate(chosen):
            seeds.append(item | {"seed_id": f"{method}-s{index:03d}"})
        strata[method] = {
            "candidates": sum(n for (m, _), n in counts.items() if m == method),
            "successes": counts[(method, None)],
            "preparation_failures": counts[(method, "preparation")],
            "classes": report,
            "selected": len(chosen),
        }
    manifest = {
        "schema": SEED_SCHEMA,
        "model": model,
        "prompt_flags": flags,
        "quotas": quotas,
        "rng_seed": rng_seed,
        "sources": [str(s) for s in sources],
        "strata": strata,
        "seeds": seeds,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        handle.write(json.dumps(manifest, indent=2) + "\n")
    output.with_suffix(".sha256").write_text(file_sha256(output) + "\n")
    output.chmod(0o444)
    return manifest


def load_seed_manifest(path):
    """Read a seed manifest after checking it and every seed file is unchanged."""
    path = Path(path)
    if file_sha256(path) != path.with_suffix(".sha256").read_text().strip():
        raise ValueError("Seed manifest changed after selection")
    manifest = json.loads(path.read_text())
    for seed in manifest["seeds"]:
        if file_sha256(seed["seed_result"]) != seed["seed_result_sha256"]:
            raise ValueError(f"Seed result changed: {seed['seed_id']}")
    return manifest


# --------------------------------------------------------------------- Stage B


def stage_b_variants(manifest, *, profiles=PROFILES, **sizes):
    variants = []
    for seed in manifest["seeds"]:
        for profile in profiles:
            settings = profile_settings(profile, **sizes)
            variants.append(
                {
                    "name": f"{SHORT[seed['method']]}_{seed['seed_id'].split('-')[-1]}_{profile}",
                    "methods": [seed["method"]],
                    "seed_attempt": seed["seed_result"],
                    "seed_id": seed["seed_id"],
                    "profile": profile,
                    "run_config": {
                        "primitive_prompt": manifest["prompt_flags"],
                        "primitive_feedback": settings["primitive_feedback"],
                        "recovery": settings["recovery"],
                    },
                }
            )
    return variants


def create_stage_b(
    directory,
    seed_manifest,
    *,
    max_workers=32,
    profiles=PROFILES,
    supporting_events=SUPPORTING_EVENTS,
    guard=EVIDENCE_GUARD,
    gpu=None,
):
    directory = Path(directory).resolve()
    manifest = load_seed_manifest(seed_manifest)
    previous, job, config, model = configuration(manifest["model"])
    config.recovery.max_rounds = 4
    variants = stage_b_variants(
        manifest, profiles=profiles, supporting_events=supporting_events, guard=guard
    )
    request_limit = len(variants) * config.recovery.total_calls
    batch = create_batch(
        directory,
        config,
        model,
        initial_state=job["initial_state"],
        instruction=job["instruction"],
        methods=sorted({s["method"] for s in manifest["seeds"]}, key=METHODS.index),
        modes=["dsl_only"],
        trials=1,
        max_workers=max_workers,
        schedule_seed=4302,
        recovery_variants=[True],
        experiment_variants=variants,
        gpu_config=_gpu(gpu),
        transport_control=_transport(directory, request_limit),
    )
    assert batch["snapshot_sha256"] == previous["snapshot_sha256"]
    _local_plan(
        directory,
        planned=len(variants),
        request_limit=request_limit,
        extra={
            "experiment": "revision43_feedback_stage_b",
            "model": model.model,
            "serving": serving_record(model.model),
            "seed_manifest": str(Path(seed_manifest).resolve()),
            "seed_manifest_sha256": file_sha256(seed_manifest),
            "prompt_flags": manifest["prompt_flags"],
            "profiles": list(profiles),
            "supporting_events": supporting_events,
            "evidence_guard_bytes": guard,
            "snapshot_sha256": batch["snapshot_sha256"],
        },
    )
    return batch


# ----------------------------------------------------------------- audit/report


def _branches(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    variants = {v["name"]: v for v in manifest["experiment_variants"]}
    for job in manifest["jobs"]:
        yield job, variants[job["variant"]], _run_directory(job)


def audit(directory):
    """Issues across all finished branches; an empty list means the batch passes."""
    issues, seeds = [], defaultdict(set)
    for job, variant, run_dir in _branches(directory):
        if run_dir is None:
            continue
        name = job["id"]
        seed_file = run_dir / "artifacts/trial_001/seed.json"
        if not seed_file.is_file():
            issues.append(f"{name}: missing seed.json")
            continue
        seed = json.loads(seed_file.read_text())
        seeds[variant["seed_id"]].add(
            (
                seed["seed_result_sha256"],
                seed["initial_state_sha256"],
                seed["seed_run_dir"],
            )
        )
        if seed["seed_result"] != job.get("seed_attempt"):
            issues.append(f"{name}: seed path differs from the manifest")
        expected = variant["run_config"]
        config = json.loads((run_dir / "config.json").read_text())["config"]
        recovery = config["recovery"]
        for item, value in expected["recovery"]["feedback"].items():
            if recovery["feedback"].get(item) != value:
                issues.append(f"{name}: feedback {item} differs from profile")
        if recovery["history"]["mode"] != expected["recovery"]["history"]["mode"]:
            issues.append(f"{name}: history mode differs from profile")
        for item, value in expected["primitive_feedback"].items():
            if config["primitive_feedback"].get(item) != value:
                issues.append(f"{name}: enrichment {item} differs from profile")
        if config["primitive_prompt"] != expected["primitive_prompt"]:
            issues.append(f"{name}: prompt flags differ")
        enabled = {k for k, v in expected["recovery"]["feedback"].items() if v is True}
        enabled |= {k for k, v in expected["primitive_feedback"].items() if v is True}
        for path in sorted(
            (run_dir / "artifacts/trial_001/rounds").glob("*/feedback.json")
        ):
            feedback = json.loads(path.read_text())
            if not feedback.get("ready"):
                continue
            label = f"{name} round {path.parent.name}"
            if feedback.get("history_mode") != expected["recovery"]["history"]["mode"]:
                issues.append(f"{label}: delivered history mode differs")
            for item, state in feedback.get("item_delivery", {}).items():
                # Partial deliveries are recorded as dicts, never as plain states.
                if item in enabled and state not in ("complete", "not_applicable"):
                    issues.append(f"{label}: {item} delivered {state}")
                if item not in enabled and state != "disabled":
                    issues.append(f"{label}: disabled {item} delivered {state}")
    for seed_id, identities in seeds.items():
        if len(identities) != 1:
            issues.append(f"{seed_id}: branches do not share one round-0 attempt")
    return issues


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value from discordant counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2**n
    return min(1.0, 2 * tail)


def summarize(directory):
    seed_manifest = json.loads(
        Path(
            json.loads((Path(directory) / "comparison_plan.json").read_text())[
                "seed_manifest"
            ]
        ).read_text()
    )
    seed_class = {s["seed_id"]: s["class"] for s in seed_manifest["seeds"]}
    outcomes = {}
    for job, variant, run_dir in _branches(directory):
        path = run_dir / "artifacts/trial_001/recovery_result.json" if run_dir else None
        if path is None or not path.is_file():
            continue
        result = json.loads(path.read_text())
        outcomes[(variant["seed_id"], variant["profile"])] = {
            "recovered": bool(result.get("success")),
            "rounds": result.get("recovery_rounds"),
            "calls": result.get("model_calls"),
            "tokens": result.get("known_tokens"),
            "stop_reason": result.get("stop_reason"),
        }
    profiles = sorted({p for _, p in outcomes}, key=PROFILES.index)
    table = {}
    for profile in profiles:
        rows = {s: o for (s, p), o in outcomes.items() if p == profile}
        paired = [s for s in rows if (s, "full") in outcomes]
        b = sum(
            outcomes[(s, "full")]["recovered"] and not rows[s]["recovered"]
            for s in paired
        )
        c = sum(
            rows[s]["recovered"] and not outcomes[(s, "full")]["recovered"]
            for s in paired
        )
        by_class = Counter()
        for s, o in rows.items():
            by_class[(seed_class[s], o["recovered"])] += 1
        table[profile] = {
            "finished": len(rows),
            "recovered": sum(o["recovered"] for o in rows.values()),
            "paired_with_full": len(paired),
            "full_only_recovered": b,
            "profile_only_recovered": c,
            "mcnemar_p": mcnemar_exact(b, c),
            "by_class": {
                cls: {
                    "recovered": by_class[(cls, True)],
                    "total": by_class[(cls, True)] + by_class[(cls, False)],
                }
                for cls in CLASSES
                if by_class[(cls, True)] + by_class[(cls, False)]
            },
            "mean_tokens": (
                sum(o["tokens"] or 0 for o in rows.values()) / len(rows)
                if rows
                else None
            ),
            "stop_reasons": dict(Counter(o["stop_reason"] for o in rows.values())),
        }
    summary = {"profiles": table, "branches": len(outcomes)}
    atomic_json(Path(directory) / "stage_b_summary.json", summary)
    return summary


# ------------------------------------------------------------------------- CLI


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    a = commands.add_parser("stage-a")
    a.add_argument("--directory", type=Path, required=True)
    a.add_argument("--model", choices=sorted(MODELS), required=True)
    a.add_argument("--prompt-flags", required=True, help="JSON or four 0/1 digits")
    a.add_argument("--trials", type=int, default=100, help="per method")
    a.add_argument("--max-workers", type=int, default=32)
    a.add_argument("--gpu-uuid", action="append", help="worker GPU allowlist")
    a.add_argument("--workers-per-gpu", type=int, default=48)
    a.add_argument("--worker-memory-mb", type=int, default=128)
    a.add_argument("--resume", action="store_true")
    a.add_argument("--create-only", action="store_true")

    s = commands.add_parser("select-seeds")
    s.add_argument("--output", type=Path, required=True)
    s.add_argument(
        "--source",
        action="append",
        required=True,
        help="batch directory, optionally BATCH:VARIANT (read only)",
    )
    s.add_argument("--model", choices=sorted(MODELS), required=True)
    s.add_argument("--prompt-flags", required=True)
    s.add_argument("--rng-seed", type=int, default=43)
    s.add_argument(
        "--seeds-per-method",
        type=int,
        default=sum(QUOTAS.values()),
        help="class quotas are scaled proportionally",
    )

    b = commands.add_parser("stage-b")
    b.add_argument("--directory", type=Path, required=True)
    b.add_argument("--seeds", type=Path, required=True)
    b.add_argument("--max-workers", type=int, default=32)
    b.add_argument("--gpu-uuid", action="append", help="worker GPU allowlist")
    b.add_argument("--workers-per-gpu", type=int, default=48)
    b.add_argument("--worker-memory-mb", type=int, default=128)
    b.add_argument("--supporting-events", type=int, default=SUPPORTING_EVENTS)
    b.add_argument("--evidence-guard", type=int, default=EVIDENCE_GUARD)
    b.add_argument("--profile", action="append", choices=PROFILES)
    b.add_argument("--resume", action="store_true")
    b.add_argument("--create-only", action="store_true")

    for name in ("audit", "summarize"):
        commands.add_parser(name).add_argument("--directory", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "select-seeds":
        manifest = select_seeds(
            args.output,
            args.source,
            model=args.model,
            flags=prompt_flags(args.prompt_flags),
            quotas=scaled_quotas(args.seeds_per_method),
            rng_seed=args.rng_seed,
        )
        print(json.dumps(manifest["strata"], indent=2))
        return
    if args.command == "audit":
        issues = audit(args.directory)
        print(json.dumps({"issues": issues, "count": len(issues)}, indent=2))
        raise SystemExit(1 if issues else 0)
    if args.command == "summarize":
        print(json.dumps(summarize(args.directory), indent=2))
        return
    directory = args.directory.resolve()
    gpu = {
        "uuids": args.gpu_uuid,
        "workers_per_gpu": args.workers_per_gpu,
        "worker_memory_mb": args.worker_memory_mb,
    }
    if not args.resume:
        if args.command == "stage-a":
            create_stage_a(
                directory,
                model_name=args.model,
                flags=prompt_flags(args.prompt_flags),
                trials=args.trials,
                max_workers=args.max_workers,
                gpu=gpu,
            )
        else:
            create_stage_b(
                directory,
                args.seeds,
                max_workers=args.max_workers,
                gpu=gpu,
                profiles=tuple(args.profile or PROFILES),
                supporting_events=args.supporting_events,
                guard=args.evidence_guard,
            )
    if not args.create_only:
        # Pause dispatch while the batch's model endpoint is down.
        manifest = json.loads((directory / "manifest.json").read_text())
        url = manifest["jobs"][0]["model_config"]["base_url"].rstrip("/") + "/models"
        result = execute(
            directory,
            resume=args.resume,
            wrap_gate=lambda gate: HealthGate(
                gate, url, limit_path=directory / "dispatch_limit.json"
            ),
        )
        print(json.dumps({"directory": str(directory)} | result), flush=True)


if __name__ == "__main__":
    main()
