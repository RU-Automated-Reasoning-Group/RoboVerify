"""Artifact-only reporting/auditing for the Revision 4.2 factorial subset."""

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
    return groups


def plots(groups, directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    directory.mkdir(parents=True, exist_ok=True)

    def rate(method, profile, field):
        g = groups[(method, profile)]
        return g[field] / g["finished"] if g["finished"] else float("nan")

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(directory / name, dpi=180, bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(3)
    for i, (field, label, color) in enumerate(
        [
            ("success", "Final task success", "#2077b4"),
            ("initial_nonconv", "Initial nonconvergence", "#de8f05"),
            ("final_nonconv", "Final nonconvergence", "#d62728"),
        ]
    ):
        bars = ax.bar(
            x + (i - 1) * 0.25,
            [rate(m, "full", field) for m in METHODS],
            0.25,
            label=label,
            color=color,
        )
        for m, bar in zip(METHODS, bars):
            g = groups[(m, "full")]
            if g["finished"]:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.015,
                    f"{g[field]}/{g['finished']}",
                    ha="center",
                    fontsize=8,
                )
    ax.set_xticks(x, [NAMES[m] for m in METHODS])
    ax.set_ylim(0, 1.16)
    ax.set_ylabel("Fraction of finished trials")
    ax.set_title("MiniMax: full treatment")
    ax.legend(loc="upper right", fontsize=8)
    save(fig, "revision42_full_outcomes.png")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, method in zip(axes, METHODS[1:]):
        bars = ax.bar(
            range(5),
            [rate(method, p, "success") for p in PROFILES],
            color=["#2077b4"] + ["#8baec8"] * 4,
        )
        for profile, bar in zip(PROFILES, bars):
            g = groups[(method, profile)]
            if g["finished"]:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.015,
                    f"{g['success']}/{g['finished']}",
                    ha="center",
                    fontsize=8,
                )
        ax.set_xticks(range(5), [LABELS[p] for p in PROFILES], rotation=25, ha="right")
        ax.set_title(NAMES[method])
        ax.set_ylim(0, 1.14)
    axes[0].set_ylabel("Final task success / finished trials")
    save(fig, "revision42_ablation_success.png")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, method in zip(axes, METHODS[1:]):
        x = np.arange(5)
        for offset, field, label, color in [
            (-0.18, "initial_nonconv", "Initial", "#de8f05"),
            (0.18, "final_nonconv", "Final", "#d62728"),
        ]:
            bars = ax.bar(
                x + offset,
                [rate(method, p, field) for p in PROFILES],
                0.36,
                label=label,
                color=color,
            )
            for profile, bar in zip(PROFILES, bars):
                g = groups[(method, profile)]
                if g["finished"]:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.015,
                        f"{g[field]}/{g['finished']}",
                        ha="center",
                        fontsize=7,
                        rotation=90,
                    )
        ax.set_xticks(x, [LABELS[p] for p in PROFILES], rotation=25, ha="right")
        ax.set_title(NAMES[method])
        ax.set_ylim(0, 1.22)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Nonconvergence / finished trials")
    save(fig, "revision42_ablation_nonconvergence.png")


def generate(directory, report_path, *, videos=False):
    directory = Path(directory).resolve()
    report_path = Path(report_path).resolve()
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
        records.append(record)
    groups = aggregate(records)
    for job in manifest["jobs"]:
        groups[(job["run_config"]["method"], job["variant"])]["planned"] += 1
    atomic_json(
        directory / "revision42_analysis.json",
        {
            "finished": len(records),
            "planned": len(jobs),
            "groups": [
                {"method": m, "variant": v, **dict(g)} for (m, v), g in groups.items()
            ],
            "trials": records,
        },
    )
    with (directory / "revision42_trials.csv").open("w") as f:
        columns = [
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
    plot_dir = report_path.parent / "plots"
    plots(groups, plot_dir)

    def link(label, p):
        return f"[{label}]({os.path.relpath(p,report_path.parent)})"

    complete = len(records) == len(jobs)
    scripted = manifest.get("scripted", False)
    lines = [
        "# Revision 4.2: primitive prompt and feedback experiment",
        "",
        f"**Status: {'complete' if complete else 'partial'}; {len(records)}/{len(jobs)} trials finished.** Pending trials are not observed failures.",
        "",
        "MiniMax M3.1 Flash free; medium reasoning; 16,384 output tokens; Stack with three blocks; `dsl_only`; `typed_v3` plus the selected primitive prompt supplement. All trials enable recovery, with up to four repairs/five rollouts from the same full initial state.",
        (
            "Scripted simulator preflight only; these trials and usage are excluded from the 220 live-trial design."
            if scripted
            else "The 60 full-treatment trials cover all three methods. The 160 ablation trials cover Method 1/2 with one Section 3 candidate group removed. The full groups are reused as controls; no duplicate control cohort is added."
        ),
        "",
        link("Immutable manifest", directory / "manifest.json")
        + " · "
        + link("Configuration and source hashes", directory / "comparison_plan.json")
        + " · "
        + link("Machine-readable analysis", directory / "revision42_analysis.json")
        + " · "
        + link("Per-trial CSV", directory / "revision42_trials.csv"),
        "",
        "## 1. Main comparison",
        "",
        "| Method | Finished / planned | Initial success | Final success | Initial nonconvergence | Final nonconvergence | # of tokens used |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for m in METHODS:
        g = groups[(m, "full")]
        n = g["finished"]
        lines.append(
            f"| {NAMES[m]} | {n}/{g['planned']} | {g['initial_success']}/{n} | {g['success']}/{n} | {g['initial_nonconv']}/{n} | {g['final_nonconv']}/{n} | {g['tokens']:,} |"
        )
    lines += [
        "",
        "Ratios use finished trials; planned counts remain visible until completion. Nonconvergence includes Pick, Move and Release, not all runtime failures.",
        "",
        "![Full treatment outcomes](plots/revision42_full_outcomes.png)",
        "",
        "## 2. Candidate-level ablations",
        "",
        "`no_prompt` disables all four general-prompt additions. `no_target`, `no_phases`, and `no_context` each disable only their named feedback enrichment. Legacy API text and legacy feedback facts remain present.",
        "",
        "| Method | Profile | Finished / planned | Final success | Initial nonconvergence | Final nonconvergence | Any nonconvergence | Nonconverged / executed rollouts | Tokens |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for m in METHODS[1:]:
        for v in PROFILES:
            g = groups[(m, v)]
            n = g["finished"]
            lines.append(
                f"| {NAMES[m]} | {v} | {n}/{g['planned']} | {g['success']}/{n} | {g['initial_nonconv']}/{n} | {g['final_nonconv']}/{n} | {g['any_nonconv']}/{n} | {g['nonconv_rollouts']}/{g['rollouts']} | {g['tokens']:,} |"
            )
    lines += [
        "",
        "![Ablation task success](plots/revision42_ablation_success.png)",
        "",
        "![Ablation nonconvergence](plots/revision42_ablation_nonconvergence.png)",
        "",
        "## 3. Recovery transitions and failure exposure",
        "",
        "| Method | Profile | Nonconv → nonconv / eligible transitions | Same primitive | Next task success | Next other execution failure | Next preparation failure | Terminal nonconv without next attempt | Pick / Move / Release failed rollouts |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for (m, v), g in sorted(groups.items()):
        lines.append(
            f"| {NAMES[m]} | {v} | {g['repeat_nonconv']}/{g['nonconv_transitions']} | {g['same_primitive']} | {g['transition_success']} | {g['transition_other_failure']} | {g['transition_preparation']} | {g['terminal_nonconv_without_next']} | {g['Pick']} / {g['Move']} / {g['Release']} |"
        )
    issues = [
        {"job": r["job_id"], "issues": r["issues"]} for r in records if r["issues"]
    ]
    lines += [
        "",
        "A transition is eligible only when an actual next attempt exists. Repeated failure labels do not establish the same defect, and disappearance of nonconvergence is not necessarily success. Attempts/transitions within a trial are dependent; these plots do not assign independent-sample confidence intervals to them.",
        "",
        "## 4. Validation, resources, and comparison scope",
        "",
        f"- Artifact audit issues: **{sum(len(x['issues']) for x in issues)}**; verified delivered recovery payloads: {sum(r['verified_feedbacks'] for r in records)}; saved videos: {sum(r['saved_videos'] for r in records)}. Independent video decoding checks: {'enabled' if videos else 'pending final audit'}.",
        f"- Reported tokens: {sum(r.get('total_tokens',0) for r in records):,}; calls: {sum(r['calls'] for r in records)}; calls without returned usage: {sum(r.get('unknown_usage_calls',0) for r in records)}. Reasoning is included in output usage once.",
        "- Source compliance, physical task success, and logical postconditions remain distinct. Baseline retains the pinned original executor and no pre-execution DSL gate. No preparation/API repair mechanism was added.",
        "- GPU assignments are fixed through each trial; request pacing and quota reservations are independent controls. Credential values are absent from manifests/logs; only environment-variable identities are recorded.",
        "- All groups use independent policy generations on one fixed scene. Ablating general prompts can change the initial candidate; this is an end-to-end ablation, not a paired repair comparison from identical failed programs.",
        "- Historical MiniMax recovery-on results were Baseline 42/50, Method 1 40/50, Method 2 40/50 final success; initial nonconvergence 24/50, 28/50, 19/50; final nonconvergence 3/50, 6/50, 7/50. These are a separate historical cohort with different execution dates/request concurrency, not a contemporaneous control.",
        "- The 11-setting scripted physical preflight and minimal credential probes are excluded from the 220 live trials. Full validation logs and preflight audits are retained with the experiment artifacts.",
    ]
    if (directory / "execution_metadata.json").exists():
        meta = read(directory / "execution_metadata.json")
        lines.append(
            f"- Requested workers: {meta['requested_workers']}; observed peak active workers: {meta['peak_active_workers']}. "
            + link("GPU/resource telemetry", directory / "execution_metadata.json")
        )
    if (directory / "request_gate.json").exists():
        gate = read(directory / "request_gate.json")
        lines.append(
            "- Wire requests by credential environment: `"
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
    lines += [
        "",
        "## 5. Per-trial results",
        "",
        "| Method | Profile | Trial | Initial status | Final status | Success | # of tokens used | Calls | Artifacts |",
        "| --- | --- | ---: | --- | --- | --- | ---: | ---: | --- |",
    ]
    for r in sorted(
        records, key=lambda r: (r["method"], PROFILES.index(r["variant"]), r["trial"])
    ):
        initial = r["attempts"][0]["status"] if r["attempts"] else "not_started"
        tokens = str(r.get("total_tokens", 0)) + (
            "*" if r.get("unknown_usage_calls") else ""
        )
        lines.append(
            f"| {NAMES[r['method']]} | {r['variant']} | {r['trial']} | {initial} | {r['status']} | {r['success']} | {tokens} | {r['calls']} | {link('Run',Path(r['run_dir'])) if r['run_dir'] else 'Unavailable'} |"
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
        directory / "revision42_audit.json",
        {
            "issues": issues,
            "finished": len(records),
            "planned": len(jobs),
            "video_frames_checked": videos,
        },
    )
    return {
        "finished": len(records),
        "planned": len(jobs),
        "audit_issues": sum(len(x["issues"]) for x in issues),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument(
        "--report", type=Path, default=Path("synthesis/cap/docs/revision_4_2_report.md")
    )
    parser.add_argument("--videos", action="store_true")
    args = parser.parse_args()
    print(json.dumps(generate(args.directory, args.report, videos=args.videos)))


if __name__ == "__main__":
    main()
