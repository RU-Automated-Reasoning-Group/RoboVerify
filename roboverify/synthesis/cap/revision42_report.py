"""Artifact-only reporting/auditing for the Revision 4.2 factorial subset.

Batches of one or more models on the same scene and configuration can be
reported together: batches with the same model accumulate, different models are
shown side by side. Each batch keeps its own per-trial audit cache.
"""

import argparse
import csv
import hashlib
import json
import os
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

from .batch import atomic_json
from .primitive_prompts import normalize, render
from .revision42_experiment import METHODS, PROFILES

NAMES = dict(zip(METHODS, ("Baseline", "Method 1", "Method 2")))
LABELS = {
    "full": "Full",
    "no_prompt": "− Prompt",
    "no_target": "− Targets",
    "no_phases": "− Phases",
    "no_context": "− Context",
    "only_prompt": "Prompt only",
}
MODEL_LABELS = {
    "minimax/minimax-m3.1-flash:free": "MiniMax M3.1 Flash free",
    "anthropic/claude-haiku-5.5": "Claude Haiku 5.5",
}


def read(path):
    return json.loads(Path(path).read_text())


def audit_trial(job, row, *, videos=False):
    issues = []
    attempts = row.get("attempts", []) or ([row] if row.get("run_dir") else [])
    results = []
    tokens = Counter()
    calls = 0
    recovery_payloads = 0
    video_count = 0
    for index, attempt in enumerate(attempts):
        run = Path(attempt["run_dir"])
        root = run / "artifacts/trial_001"
        events = [
            json.loads(line)
            for line in (root / "timeline.jsonl").read_text().splitlines()
        ]
        rolled = any(e["kind"] == "policy_rollout_start" for e in events)
        restorations = [e for e in events if e["kind"] == "initial_state_restoration"]
        if not restorations or any(not e["exact_match"] for e in restorations):
            issues.append(f"round {index}: initial-state restoration")
        failed = next(
            (
                e
                for e in reversed(events)
                if e["kind"] == "primitive_end" and e.get("converged") is False
            ),
            None,
        )
        failure = failed["primitive"] if failed else None
        results.append(
            {
                "round": index,
                "status": attempt["status"],
                "success": bool(attempt.get("task_success", attempt.get("success"))),
                "rolled_out": rolled,
                "nonconvergence": attempt["status"] == "primitive_failed"
                and failed is not None,
                "primitive": failure,
                "run_dir": str(run),
            }
        )
        feedback = (
            read(root / "recovery_feedback.json")
            if (root / "recovery_feedback.json").exists()
            else None
        )
        if feedback:
            recovery_payloads += 1
            parent = (
                Path(row["run_dir"])
                / "artifacts/trial_001/rounds"
                / f"{index-1:03d}"
                / "feedback.json"
            )
            if feedback != read(parent):
                issues.append(f"round {index}: prepared/delivered feedback mismatch")
            if feedback["feedback_token_bound"] > 8192:
                issues.append(f"round {index}: evidence bound")
            if len(feedback["evidence"]["earlier_failures"]) != index - 1:
                issues.append(f"round {index}: history length")
            diag = feedback["evidence"].get("primitive_diagnostics", {})
            flags = job["run_config"]["primitive_feedback"]
            if feedback["evidence"]["latest"]["status"] == "primitive_failed":
                for key in ("target_derivation", "controller_phases", "action_context"):
                    if (key in diag) != flags[key]:
                        issues.append(f"round {index}: wrong feedback group {key}")
        for call in sorted((root / "lmp").glob("call_*")):
            calls += 1
            prompt = (call / "prompt.txt").read_text()
            meta = read(call / "prompt_metadata.json")
            request = read(call / "request.json")
            if meta["prompt_sha256"] != hashlib.sha256(prompt.encode()).hexdigest():
                issues.append(f"{call.name}: prompt hash")
            matches_request = (
                request.get("prompt") == prompt
                if request.get("model") == "scripted"
                else any(
                    m.get("content") == prompt for m in request.get("messages", [])
                )
            )
            if not matches_request:
                issues.append(f"{call.name}: request prompt differs")
            if meta.get("primitive_prompt") != job["run_config"]["primitive_prompt"]:
                issues.append(f"{call.name}: prompt flags")
            supplement = render(job["run_config"]["primitive_prompt"])
            if supplement and supplement not in prompt:
                issues.append(f"{call.name}: missing general supplement")
            if (
                not supplement
                and "Additional primitive contract explanations:" in prompt
            ):
                issues.append(f"{call.name}: unexpected general supplement")
            if feedback and meta["role"] == "recovery_policy":
                decoder = json.JSONDecoder()
                source = decoder.raw_decode(
                    prompt.split("CURRENT EXECUTED SOURCE AND BINDINGS\n", 1)[1]
                )[0]
                evidence = decoder.raw_decode(
                    prompt.split("OBSERVED FAILURE AND ACCUMULATED HISTORY\n", 1)[1]
                )[0]
                if (
                    source
                    != {
                        "sources": feedback["current_sources"],
                        "bindings": feedback["execution_bindings"],
                    }
                    or evidence != feedback["evidence"]
                ):
                    issues.append(
                        f"{call.name}: delivered feedback differs from request"
                    )
            response = call / "response.json"
            usage = read(response).get("usage", {}) if response.exists() else {}
            if usage.get("total_tokens") is None:
                tokens["unknown_usage_calls"] += 1
            else:
                tokens["total_tokens"] += usage["total_tokens"]
                tokens["input_tokens"] += usage.get("prompt_tokens", 0)
                tokens["output_tokens"] += usage.get("completion_tokens", 0)
        from .deferred_video import resolved_video

        video = resolved_video(attempt.get("video", {}))
        if video.get("status") in {"saved", "completed"}:
            video_count += 1
            if videos:
                frames = subprocess.run(
                    [
                        "ffprobe",
                        "-v",
                        "error",
                        "-count_frames",
                        "-select_streams",
                        "v:0",
                        "-show_entries",
                        "stream=nb_read_frames",
                        "-of",
                        "csv=p=0",
                        video["path"],
                    ],
                    capture_output=True,
                    text=True,
                )
                if frames.returncode or frames.stdout.strip() != str(video["frames"]):
                    issues.append(f"round {index}: video frame count")
                frame_index = root / "execution/rollout_001/frame_index.jsonl"
                if len(frame_index.read_text().splitlines()) != video["frames"]:
                    issues.append(f"round {index}: video state/frame mapping")
        elif rolled:
            issues.append(f'round {index}: video status {video.get("status")}')
    if tokens["total_tokens"] != row.get("known_tokens", 0):
        issues.append("trial token total")
    if calls != row.get("model_calls", 0):
        issues.append("trial model-call total")
    return {
        "audit_version": 2,
        "job_id": job["id"],
        "method": job["run_config"]["method"],
        "variant": job["variant"],
        "trial": job["trial"],
        "success": row.get("success") is True,
        "status": row["status"],
        "stop_reason": row.get("stop_reason"),
        "attempts": results,
        "calls": calls,
        **dict(tokens),
        "issues": issues,
        "verified_feedbacks": recovery_payloads,
        "saved_videos": video_count,
        "video_frames_checked": videos,
        "run_dir": row.get("run_dir"),
        "active_seconds": row.get("active_wall_seconds", 0),
        "queue_seconds": row.get("queue_wait_seconds", 0),
        "dsl_compliance": row.get("dsl_compliance"),
    }


def aggregate(records):
    groups = defaultdict(Counter)
    for r in records:
        g = groups[(r["method"], r["variant"])]
        g["finished"] += 1
        g["success"] += r["success"]
        g["calls"] += r["calls"]
        g["tokens"] += r.get("total_tokens", 0)
        g["unknown_usage_calls"] += r.get("unknown_usage_calls", 0)
        g["audit_issues"] += len(r["issues"])
        attempts = r["attempts"]
        if not attempts:
            continue
        g["initial_success"] += attempts[0]["success"]
        g["initial_nonconv"] += attempts[0]["nonconvergence"]
        g["final_nonconv"] += attempts[-1]["nonconvergence"]
        g["any_nonconv"] += any(a["nonconvergence"] for a in attempts)
        for a in attempts:
            g["rollouts"] += a["rolled_out"]
            g["nonconv_rollouts"] += a["nonconvergence"]
            if a["nonconvergence"]:
                g[a["primitive"]] += 1
            if not a["rolled_out"]:
                g["preparation_attempts"] += 1
        for a, b in zip(attempts, attempts[1:]):
            if not a["nonconvergence"]:
                continue
            g["nonconv_transitions"] += 1
            if b["nonconvergence"]:
                g["repeat_nonconv"] += 1
                g["same_primitive"] += a["primitive"] == b["primitive"]
            elif b["success"]:
                g["transition_success"] += 1
            elif not b["rolled_out"]:
                g["transition_preparation"] += 1
            else:
                g["transition_other_failure"] += 1
        g["terminal_nonconv_without_next"] += attempts[-1]["nonconvergence"]
        final = attempts[-1]
        g["final_budget_stop"] += (
            final["rolled_out"] and final["status"] == "budget_exhausted"
        )
        g["final_goal_failure"] += final["status"] == "goal_failed"
        g["final_runtime_failure"] += (
            final["rolled_out"] and final["status"] == "failed"
        )
        g["preparation_stop"] += (
            r.get("stop_reason") == "preparation_failure_no_rollout"
        )
        g["feedback_size_stop"] += (
            r.get("stop_reason") == "required_feedback_exceeds_token_limit"
        )
    return groups


def ablation_layout(groups):
    """Profiles present for each method that has at least one non-full group."""
    layout = {}
    for m in METHODS:
        present = [p for p in PROFILES if groups.get((m, p), {}).get("planned")]
        if any(p != "full" for p in present):
            layout[m] = present
    return layout


HATCHES = ("", "//", "..", "xx")


def plots(cohorts, directory, *, prefix="revision42"):
    """Draw one bar per model; color encodes the measure, hatching the model."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Patch

    directory.mkdir(parents=True, exist_ok=True)
    models = list(cohorts)
    hatch = {m: HATCHES[i % len(HATCHES)] for i, m in enumerate(models)}

    def rate(model, method, profile, field):
        g = cohorts[model].get((method, profile))
        return g[field] / g["finished"] if g and g["finished"] else float("nan")

    def label(ax, bar, model, method, profile, field, size=7):
        g = cohorts[model].get((method, profile))
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            (bar.get_height() if g and g["finished"] else 0) + 0.015,
            f"{g[field]}/{g['finished']}" if g and g["finished"] else "n/a",
            ha="center",
            va="bottom",
            fontsize=size,
            rotation=90,
            color="black" if g and g["finished"] else "gray",
        )

    def legend(ax, measures):
        handles = [Patch(facecolor=c, label=l) for l, c in measures] + [
            Patch(
                facecolor="white",
                edgecolor="black",
                hatch=hatch[m],
                label=MODEL_LABELS.get(m, m),
            )
            for m in models
        ]
        ax.legend(handles=handles, loc="upper right", fontsize=7, ncol=2)

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(directory / name, dpi=180, bbox_inches="tight")
        plt.close(fig)

    measures = [
        ("success", "Final task success", "#2077b4"),
        ("initial_nonconv", "Initial nonconvergence", "#de8f05"),
        ("final_nonconv", "Final nonconvergence", "#d62728"),
    ]
    bars_per_method = len(measures) * len(models)
    width = 0.84 / bars_per_method
    fig, ax = plt.subplots(figsize=(max(8, 1.8 * bars_per_method), 4.8))
    x = np.arange(len(METHODS))
    for i, (field, _, color) in enumerate(measures):
        for j, model in enumerate(models):
            k = i * len(models) + j
            bars = ax.bar(
                x + (k - (bars_per_method - 1) / 2) * width,
                [rate(model, m, "full", field) for m in METHODS],
                width,
                color=color,
                edgecolor="black",
                linewidth=0.4,
                hatch=hatch[model],
            )
            for m, bar in zip(METHODS, bars):
                label(ax, bar, model, m, "full", field)
    ax.set_xticks(x, [NAMES[m] for m in METHODS])
    ax.set_ylim(0, 1.45)
    ax.set_ylabel("Fraction of finished trials")
    ax.set_title("Full treatment")
    legend(ax, [(l, c) for _, l, c in measures])
    save(fig, f"{prefix}_full_outcomes.png")

    layouts = {m: ablation_layout(cohorts[m]) for m in models}
    panels = [m for m in METHODS if any(m in layout for layout in layouts.values())]
    if not panels:
        return

    def profiles_for(method):
        return [
            p for p in PROFILES if any(p in layouts[m].get(method, ()) for m in models)
        ]

    def ablation_figure(name, fields, ylabel):
        fig, axes = plt.subplots(
            len(panels), 1, figsize=(11, 3.4 * len(panels)), sharey=True, squeeze=False
        )
        for ax, method in zip(axes[:, 0], panels):
            profiles = profiles_for(method)
            x = np.arange(len(profiles))
            count = len(fields) * len(models)
            width = 0.84 / count
            for i, (field, _, color) in enumerate(fields):
                for j, model in enumerate(models):
                    k = i * len(models) + j
                    bars = ax.bar(
                        x + (k - (count - 1) / 2) * width,
                        [rate(model, method, p, field) for p in profiles],
                        width,
                        color=color,
                        edgecolor="black",
                        linewidth=0.4,
                        hatch=hatch[model],
                    )
                    for p, bar in zip(profiles, bars):
                        label(ax, bar, model, method, p, field)
            ax.set_xticks(x, [LABELS[p] for p in profiles])
            ax.set_title(NAMES[method], fontsize=10)
            ax.set_ylim(0, 1.5)
            ax.set_ylabel(ylabel)
        legend(axes[0, 0], [(l, c) for _, l, c in fields])
        save(fig, name)

    ablation_figure(
        f"{prefix}_ablation_success.png",
        [("success", "Final task success", "#2077b4")],
        "Final success / finished",
    )
    ablation_figure(
        f"{prefix}_ablation_nonconvergence.png",
        [
            ("initial_nonconv", "Initial nonconvergence", "#de8f05"),
            ("final_nonconv", "Final nonconvergence", "#d62728"),
        ],
        "Nonconvergence / finished",
    )


def load_records(directory, *, videos=False):
    manifest = read(directory / "manifest.json")
    summary = read(directory / "summary.json")
    jobs = {j["id"]: j for j in manifest["jobs"]}
    cache = directory / "revision42_analysis"
    cache.mkdir(exist_ok=True)
    records = []
    for row in summary["trials"]:
        path = cache / (row["job_id"] + ".json")
        if (
            path.exists()
            and read(path).get("audit_version") == 2
            and not any(
                a.get("video", {}).get("mode") == "deferred"
                for a in row.get("attempts", [row])
            )
            and (not videos or read(path).get("video_frames_checked"))
        ):
            record = read(path)
        else:
            record = audit_trial(jobs[row["job_id"]], row, videos=videos)
            atomic_json(path, record)
        record["batch"] = str(directory)
        record["model"] = jobs[row["job_id"]]["model_config"]["model"]
        records.append(record)
    return manifest, records


def generate(
    directories,
    report_path,
    *,
    videos=False,
    output=None,
    plot_prefix="revision42",
    interpretation=None,
):
    """Audit batches and write one report, accumulating batches by model.

    Batches with the same model are pooled; different models are reported side
    by side. All batches must replay the same initial snapshot. Single-batch
    analysis files stay in that batch; otherwise they are written to ``output``.
    ``interpretation`` is hand-written Markdown placed before the per-trial
    table, so regenerated numbers never overwrite it.
    """
    if isinstance(directories, (str, Path)):
        directories = [directories]
    directories = [Path(d).resolve() for d in directories]
    if len(directories) > 1 and output is None:
        raise ValueError("An accumulated report needs an output directory")
    output = Path(output).resolve() if output else directories[0]
    output.mkdir(parents=True, exist_ok=True)
    report_path = Path(report_path).resolve()
    batches = {}
    records = []
    for directory in directories:
        manifest, batch_records = load_records(directory, videos=videos)
        model = {j["model_config"]["model"] for j in manifest["jobs"]}
        if len(model) != 1:
            raise ValueError(f"{directory} mixes models")
        batches[directory] = (model.pop(), manifest)
        records += batch_records
    if len({m["snapshot_sha256"] for _, m in batches.values()}) > 1:
        raise ValueError("Accumulated batches must replay the same initial snapshot")
    models = list(dict.fromkeys(model for model, _ in batches.values()))
    jobs = {
        model: [
            j
            for m, manifest in batches.values()
            if m == model
            for j in manifest["jobs"]
        ]
        for model in models
    }
    cohorts = {}
    for model in models:
        if len(
            {(j["run_config"]["method"], j["variant"], j["trial"]) for j in jobs[model]}
        ) != len(jobs[model]):
            raise ValueError(f"Duplicate trial identities for {model}")
        cohorts[model] = aggregate([r for r in records if r["model"] == model])
        for job in jobs[model]:
            cohorts[model][(job["run_config"]["method"], job["variant"])][
                "planned"
            ] += 1
    planned = sum(len(j) for j in jobs.values())

    def name(model):
        return MODEL_LABELS.get(model, model)

    def ordered(model):
        return sorted(
            cohorts[model].items(),
            key=lambda x: (METHODS.index(x[0][0]), PROFILES.index(x[0][1])),
        )

    atomic_json(
        output / "revision42_analysis.json",
        {
            "batches": {str(d): model for d, (model, _) in batches.items()},
            "finished": len(records),
            "planned": planned,
            "groups": [
                {"model": model, "method": m, "variant": v, **dict(g)}
                for model in models
                for (m, v), g in ordered(model)
            ],
            "trials": records,
        },
    )
    with (output / "revision42_trials.csv").open("w") as f:
        columns = [
            "model",
            "method",
            "variant",
            "trial",
            "success",
            "status",
            "stop_reason",
            "calls",
            "total_tokens",
            "input_tokens",
            "output_tokens",
            "unknown_usage_calls",
            "run_dir",
        ]
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    plots(cohorts, report_path.parent / "plots", prefix=plot_prefix)

    def link(label, p):
        return f"[{label}]({os.path.relpath(p,report_path.parent)})"

    finished = {m: sum(r["model"] == m for r in records) for m in models}
    complete = len(records) == planned
    scripted = any(m.get("scripted", False) for _, m in batches.values())
    lines = [
        "# Revision 4.2: primitive prompt and feedback experiment",
        "",
        f"**Status: {'complete' if complete else 'partial'}; "
        + "; ".join(f"{name(m)} {finished[m]}/{len(jobs[m])}" for m in models)
        + " trials finished.** Pending trials are not observed failures.",
        "",
        f"Models: {', '.join(name(m) for m in models)}; each uses medium reasoning; 16,384 output tokens; Stack with three blocks; `dsl_only`; `typed_v3` plus the selected primitive prompt supplement. All trials enable recovery, with up to four repairs/five rollouts from the same full initial state. Model identity is the only experimental difference between cohorts.",
        "",
    ]
    if scripted:
        lines += [
            "Scripted simulator preflight only; these trials and usage are excluded from the live-trial design.",
            "",
        ]
    for model in models:
        groups = cohorts[model]
        design = [
            f"**{name(model)}:** {sum(groups[(m, 'full')]['planned'] for m in METHODS)} full-treatment trials cover all three methods."
        ]
        for m, profiles in ablation_layout(groups).items():
            ablations = [p for p in profiles if p != "full"]
            design.append(
                f"{NAMES[m]} ablations: {', '.join(f'`{p}`' for p in ablations)} "
                f"({sum(groups[(m, p)]['planned'] for p in ablations)} trials)."
            )
        lines += [" ".join(design), ""]
        for directory, (batch_model, _) in batches.items():
            if batch_model == model:
                plan = read(directory / "comparison_plan.json")
                lines.append(
                    f"- `{directory.name}` ({plan['planned']} planned): "
                    + link("Immutable manifest", directory / "manifest.json")
                    + " · "
                    + link(
                        "Configuration and source hashes",
                        directory / "comparison_plan.json",
                    )
                )
        lines.append("")
    lines += [
        "The full groups are reused as controls; no duplicate control cohort is added. "
        + link("Machine-readable analysis", output / "revision42_analysis.json")
        + " · "
        + link("Per-trial CSV", output / "revision42_trials.csv"),
        "",
        "## 1. Main comparison",
        "",
        "| Model | Method | Finished / planned | Initial success | Final success | Initial nonconvergence | Final nonconvergence | # of tokens used |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model in models:
        for m in METHODS:
            g = cohorts[model][(m, "full")]
            n = g["finished"]
            lines.append(
                f"| {name(model)} | {NAMES[m]} | {n}/{g['planned']} | {g['initial_success']}/{n} | {g['success']}/{n} | {g['initial_nonconv']}/{n} | {g['final_nonconv']}/{n} | {g['tokens']:,} |"
            )
    lines += [
        "",
        "Ratios use finished trials; planned counts remain visible until completion. Nonconvergence includes Pick, Move and Release, not all runtime failures. In the figures, color identifies the measure and hatching identifies the model.",
        "",
        f"![Full treatment outcomes](plots/{plot_prefix}_full_outcomes.png)",
        "",
        "## 2. Candidate-level ablations",
        "",
        "`no_prompt` disables all four general-prompt additions. `no_target`, `no_phases`, and `no_context` each disable only their named feedback enrichment. `only_prompt` keeps the four general-prompt additions and disables all three feedback enrichments. Legacy API text and legacy feedback facts remain present.",
        "",
        "| Model | Method | Profile | Finished / planned | Initial success | Final success | Initial nonconvergence | Final nonconvergence | Any nonconvergence | Nonconverged / executed rollouts | Tokens |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for m in METHODS:
        for model in models:
            for v in ablation_layout(cohorts[model]).get(m, ()):
                g = cohorts[model][(m, v)]
                n = g["finished"]
                lines.append(
                    f"| {name(model)} | {NAMES[m]} | {v} | {n}/{g['planned']} | {g['initial_success']}/{n} | {g['success']}/{n} | {g['initial_nonconv']}/{n} | {g['final_nonconv']}/{n} | {g['any_nonconv']}/{n} | {g['nonconv_rollouts']}/{g['rollouts']} | {g['tokens']:,} |"
                )
    lines += [
        "",
        f"![Ablation task success](plots/{plot_prefix}_ablation_success.png)",
        "",
        f"![Ablation nonconvergence](plots/{plot_prefix}_ablation_nonconvergence.png)",
        "",
        "### 2.1 Unsuccessful trials by terminal path",
        "",
        "| Model | Method | Profile | Unsuccessful | Final nonconvergence | Final execution/query budget stop | Final goal failure | Final other runtime failure | Preparation stop | Feedback-size stop |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model in models:
        for (m, v), g in ordered(model):
            lines.append(
                f"| {name(model)} | {NAMES[m]} | {v} | {g['finished'] - g['success']} | {g['final_nonconv']} | {g['final_budget_stop']} | {g['final_goal_failure']} | {g['final_runtime_failure']} | {g['preparation_stop']} | {g['feedback_size_stop']} |"
            )
    lines += [
        "",
        "The feedback-size column can overlap a final rollout failure: that stop occurs while preparing feedback for another attempt. A preparation stop has no final rollout.",
        "",
        "## 3. Recovery transitions and failure exposure",
        "",
        "| Model | Method | Profile | Nonconv → nonconv / eligible transitions | Same primitive | Next task success | Next other execution failure | Next preparation failure | Terminal nonconv without next attempt | Pick / Move / Release failed rollouts |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for model in models:
        for (m, v), g in ordered(model):
            lines.append(
                f"| {name(model)} | {NAMES[m]} | {v} | {g['repeat_nonconv']}/{g['nonconv_transitions']} | {g['same_primitive']} | {g['transition_success']} | {g['transition_other_failure']} | {g['transition_preparation']} | {g['terminal_nonconv_without_next']} | {g['Pick']} / {g['Move']} / {g['Release']} |"
            )
    issues = [
        {"job": r["job_id"], "batch": r["batch"], "issues": r["issues"]}
        for r in records
        if r["issues"]
    ]
    lines += [
        "",
        "A transition is eligible only when an actual next attempt exists. Repeated failure labels do not establish the same defect, and disappearance of nonconvergence is not necessarily success. Attempts/transitions within a trial are dependent; these plots do not assign independent-sample confidence intervals to them.",
        "",
        "## 4. Validation, resources, and comparison scope",
        "",
    ]
    for model in models:
        rows = [r for r in records if r["model"] == model]
        count = sum(
            len(x["issues"]) for x in issues if x["batch"] in {r["batch"] for r in rows}
        )
        lines += [
            f"- **{name(model)}** — artifact audit issues: **{count}**; verified delivered recovery payloads: {sum(r['verified_feedbacks'] for r in rows)}; saved videos: {sum(r['saved_videos'] for r in rows)}. Reported tokens: {sum(r.get('total_tokens',0) for r in rows):,}; calls: {sum(r['calls'] for r in rows)}; calls without returned usage: {sum(r.get('unknown_usage_calls',0) for r in rows)}.",
        ]
    lines += [
        f"- Independent video decoding checks: {'enabled' if videos else 'pending final audit'}. Reasoning is included in output usage once.",
        "- Source compliance, physical task success, and logical postconditions remain distinct. Baseline retains the pinned original executor and no pre-execution DSL gate. No preparation/API repair mechanism was added.",
        "- GPU assignments are fixed through each trial; request pacing and quota reservations are independent controls. Credential values are absent from manifests/logs; only environment-variable identities are recorded.",
        "- All groups use independent policy generations on one fixed scene. Ablating general prompts can change the initial candidate; this is an end-to-end ablation, not a paired repair comparison from identical failed programs.",
    ]
    if "minimax/minimax-m3.1-flash:free" in models:
        lines.append(
            "- Historical MiniMax recovery-on results were Baseline 42/50, Method 1 40/50, Method 2 40/50 final success; initial nonconvergence 24/50, 28/50, 19/50; final nonconvergence 3/50, 6/50, 7/50. These are a separate historical cohort with different execution dates/request concurrency, not a contemporaneous control."
        )
    lines.append(
        f"- The scripted physical preflights and minimal credential probes are excluded from the {planned} live trials. Full validation logs and preflight audits are retained with the experiment artifacts."
    )
    for directory, (model, _) in batches.items():
        label = f" (`{directory.name}`)"
        if (directory / "execution_metadata.json").exists():
            meta = read(directory / "execution_metadata.json")
            lines.append(
                f"- Requested workers{label}: {meta['requested_workers']}; observed peak active workers: {meta['peak_active_workers']}. "
                + link("GPU/resource telemetry", directory / "execution_metadata.json")
            )
        if (directory / "request_gate.json").exists():
            gate = read(directory / "request_gate.json")
            lines.append(
                f"- Wire requests by credential environment{label}: `"
                + json.dumps(gate.get("by_credential", {}), sort_keys=True)
                + "`."
            )
    if issues:
        lines += [
            "",
            "Audit issue details:",
            "",
            "```json",
            json.dumps(issues, indent=2),
            "```",
        ]
    section = 5
    if interpretation:
        text = Path(interpretation).read_text().strip()
        lines += ["", text]
        section += sum(
            1
            for line in text.splitlines()
            if line.startswith("## ") and line[3:4].isdigit()
        )
    lines += [
        "",
        f"## {section}. Per-trial results",
        "",
        "| Model | Method | Profile | Trial | Initial status | Final status | Success | # of tokens used | Calls | Artifacts |",
        "| --- | --- | --- | ---: | --- | --- | --- | ---: | ---: | --- |",
    ]
    for r in sorted(
        records,
        key=lambda r: (
            models.index(r["model"]),
            r["method"],
            PROFILES.index(r["variant"]),
            r["trial"],
        ),
    ):
        initial = r["attempts"][0]["status"] if r["attempts"] else "not_started"
        tokens = str(r.get("total_tokens", 0)) + (
            "*" if r.get("unknown_usage_calls") else ""
        )
        lines.append(
            f"| {name(r['model'])} | {NAMES[r['method']]} | {r['variant']} | {r['trial']} | {initial} | {r['status']} | {r['success']} | {tokens} | {r['calls']} | {link('Run',Path(r['run_dir'])) if r['run_dir'] else 'Unavailable'} |"
        )
    lines += [
        "",
        "`*` marks incomplete returned token usage. No failed trial is replaced.",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_suffix(".md.tmp")
    temporary.write_text("\n".join(lines) + "\n")
    temporary.replace(report_path)
    atomic_json(
        output / "revision42_audit.json",
        {
            "batches": [str(d) for d in directories],
            "issues": issues,
            "finished": len(records),
            "planned": planned,
            "video_frames_checked": videos,
        },
    )
    return {
        "finished": len(records),
        "planned": planned,
        "audit_issues": sum(len(x["issues"]) for x in issues),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--directory",
        required=True,
        type=Path,
        action="append",
        help="batch directory; repeat to accumulate batches (grouped by model)",
    )
    parser.add_argument(
        "--report", type=Path, default=Path("synthesis/cap/docs/revision_4_2_report.md")
    )
    parser.add_argument(
        "--output", type=Path, help="analysis directory for an accumulated report"
    )
    parser.add_argument("--plot-prefix", default="revision42")
    parser.add_argument(
        "--interpretation", type=Path, help="hand-written Markdown sections to retain"
    )
    parser.add_argument("--videos", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            generate(
                args.directory,
                args.report,
                videos=args.videos,
                output=args.output,
                plot_prefix=args.plot_prefix,
                interpretation=args.interpretation,
            )
        )
    )


if __name__ == "__main__":
    main()
