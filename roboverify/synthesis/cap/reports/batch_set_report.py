"""One report for a set of CaP batches under a common root.

Each batch directory (``manifest.json``, ``schedule.json``, ``summary.json``)
already gets its own ``report.md`` from ``batch.write_report``. This module
combines every batch below ``ROOT`` (for example ``ROOT/<model>/<task>``) into
``ROOT/report.md`` with the same columns plus the task, model, scene seed,
demonstration and settled-physical fields, and a per-round table.

    python -m synthesis.cap.reports.batch_set_report ROOT [--title TITLE] [--note TEXT ...]

It only reads batch outputs; rerunning it after more batches finish refreshes
the report.
"""

import argparse
import json
from pathlib import Path

TASK_ORDER = ("stack", "unstack", "reverse", "restack", "grid", "pyramid")


def find_batches(root):
    """Batch directories below ``root`` (or ``root`` itself), in a stable order."""
    root = Path(root)
    found = [
        p.parent
        for p in root.rglob("manifest.json")
        if (p.parent / "schedule.json").exists() and "trials" not in p.parent.parts
    ]

    def key(directory):
        manifest = json.loads((directory / "manifest.json").read_text())
        task = manifest["jobs"][0]["run_config"]["task"]
        order = TASK_ORDER.index(task) if task in TASK_ORDER else len(TASK_ORDER)
        return (str(directory.relative_to(root).parent), order, str(directory))

    return sorted(found, key=key)


def load_batch(root, directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    summary_path = directory / "summary.json"
    summary = (
        json.loads(summary_path.read_text())
        if summary_path.exists()
        else {"trials": [], "planned": len(manifest["jobs"])}
    )
    plan_path = directory / "comparison_plan.json"
    plan = json.loads(plan_path.read_text()) if plan_path.exists() else {}
    videos_path = directory / "video_summary.json"
    videos = json.loads(videos_path.read_text()) if videos_path.exists() else {}
    first = manifest["jobs"][0]
    return {
        "label": str(directory.relative_to(root)) if directory != root else ".",
        "path": directory,
        "manifest": manifest,
        "plan": plan,
        "rows": summary["trials"],
        "planned": len(manifest["jobs"]),
        "run": first["run_config"],
        "model": first["model_config"],
        "seeds": {j["id"]: j["run_config"].get("seed") for j in manifest["jobs"]},
        "videos": videos.get("counts", {}),
    }


def _count(rows, field, value=True):
    return sum(r.get(field) == value for r in rows)


def _cell(text):
    return str(text).replace("|", "/").replace("\n", " ")


def render(root, batches, *, title=None, notes=()):
    rows_total = sum(len(b["rows"]) for b in batches)
    planned_total = sum(b["planned"] for b in batches)
    videos = {}
    for b in batches:
        for k, v in b["videos"].items():
            videos[k] = videos.get(k, 0) + v
    lines = [
        f"# CaP batch report{': ' + title if title else ''}",
        "",
        "Status: "
        + ("complete" if rows_total == planned_total else "in progress")
        + f" ({rows_total}/{planned_total} trials finished in {len(batches)} batches)",
        "",
    ]
    if videos:
        lines += [
            f"Deferred videos: {videos}. Video completion is separate from experiment status.",
            "",
        ]
    configs = sorted(
        {
            (
                b["plan"].get("frozen_config", "-"),
                b["plan"].get("frozen_config_source", "-"),
                b["plan"].get("git_sha", "-")[:7]
                + (" (dirty)" if b["plan"].get("git_dirty") else ""),
            )
            for b in batches
        }
    )
    for frozen, source, sha in configs:
        lines.append(f"Configuration: `{frozen}` (`{source}`), git `{sha}`.")
    models = sorted(
        {
            (
                b["model"]["model"],
                b["model"].get("provider", "-"),
                json.dumps(b["model"].get("extra_body", {}), sort_keys=True),
                b["model"]["max_output_tokens"],
            )
            for b in batches
        }
    )
    for name, provider, options, output in models:
        lines.append(
            f"Model: `{name}` ({provider}). Output limit: {output}. Reasoning/options: `{options}`."
        )
    run = batches[0]["run"]
    recovery = run.get("recovery", {})
    lines += [
        f"Method: `{run['method']}`, condition mode `{run['condition_mode']}`, prompt `{run['prompt_version']}`. Each trial: {run['timeout']} s, {run['max_model_calls']} calls, {run['max_tokens']} reported tokens, {run['max_sim_steps']} simulation steps.",
    ]
    if recovery.get("enabled"):
        lines.append(
            f"Recovery: up to {recovery['max_rounds']} additional rounds ({recovery['round_tokens']} effort tokens, {recovery['round_calls']} calls, {recovery['round_timeout']} s each); whole trial {recovery['total_tokens']} effort tokens, {recovery['total_calls']} calls, {recovery['total_timeout']} s; context allowance {recovery['context_tokens']} tokens; success field `{recovery['success_field']}`."
        )
    demos = {
        (
            b["run"].get("demonstrations", {}).get("enabled", False),
            b["run"].get("demonstrations", {}).get("token_counter"),
            b["run"].get("demonstrations", {}).get("waypoint_tolerance"),
        )
        for b in batches
    }
    for enabled, counter, tolerance in sorted(demos, key=str):
        if enabled:
            lines.append(
                f"Demonstrations: on (waypoint tolerance {tolerance} m, counter `{counter}`); the block is the first part of every request and is charged separately from the effort budgets."
            )
        else:
            lines.append("Demonstrations: off.")
    criteria = sorted(
        {r.get("success_criterion", "task-spec-v1") for b in batches for r in b["rows"]}
    )
    lines += [
        f"Task-spec success ({', '.join(criteria) or 'task-spec-v1'}) is RoboVerify's criterion; physical success, settled physical success and DSL compliance are separate metrics. Failed/incomplete trials remain in the denominator.",
        "",
        "| Batch | Task | Model | Blocks | Scene seeds | Recovery | First-attempt task-spec success | Final task-spec success | First-attempt success | Final success | Settled physical success | DSL pass | Finished / planned |",
        "| --- | --- | --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for b in batches:
        rows, n = b["rows"], b["planned"]
        seeds = sorted(s for s in b["seeds"].values() if s is not None)
        span = f"{seeds[0]}–{seeds[-1]}" if seeds else "-"
        lines.append(
            f"| [{b['label']}]({b['label']}/report.md) | {b['run']['task']} | {b['model']['model']} | {b['run']['num_blocks']} | {span} | "
            f"{'on' if b['run'].get('recovery', {}).get('enabled') else 'off'} | "
            f"{_count(rows, 'first_attempt_task_spec_success')}/{n} | {_count(rows, 'task_spec_success')}/{n} | "
            f"{sum(r.get('first_attempt_success', r.get('success')) is True for r in rows)}/{n} | {_count(rows, 'success')}/{n} | "
            f"{_count(rows, 'settled_environment_success')}/{n} | {_count(rows, 'dsl_compliance', 'pass')}/{n} | {len(rows)}/{n} |"
        )
    lines += [
        "",
        "| Batch | Trial | Seed | Status | Task-spec success | Success | DSL compliance | # of tokens used | Demo block tokens | Provider total tokens | Calls | Recovery rounds | Active s | Queue s | Failure / stop reason |",
        "| --- | --- | ---: | --- | --- | --- | --- | --- | ---: | ---: | --- | --- | --- | --- | --- |",
    ]
    for b in batches:
        for r in sorted(b["rows"], key=lambda r: r["trial"]):
            token = str(r.get("known_tokens", 0)) + (
                "*" if r.get("unknown_usage_calls", 0) else ""
            )
            if r.get("demonstration_tokens"):
                token += f" (+{r['demonstration_tokens']} demo)"
            link = (
                f"[trial {r['trial']}]({b['label']}/trials/{r['job_id']}/report.txt)"
                if r.get("run_dir")
                else str(r["trial"])
            )
            lines.append(
                f"| {b['label']} | {link} | {b['seeds'].get(r.get('job_id'), '-')} | {r['status']} | {r.get('task_spec_success')} | {r.get('success')} | "
                f"{r.get('dsl_compliance', 'not_available')} | {token} | {r.get('demonstration_block_tokens', 0):,} | "
                f"{r.get('provider_total_tokens', r.get('known_tokens', 0)):,} | {r.get('model_calls', 0)} | {r.get('recovery_rounds', 0)} | "
                f"{r.get('active_wall_seconds', 0):.2f} | {r.get('queue_wait_seconds', 0):.2f} | {_cell(r.get('stop_reason', r.get('reason', '')))} |"
            )
    lines += [
        "",
        "## Attempts",
        "",
        "| Batch | Trial | Round | Status | Task-spec success | Success | Reason |",
        "| --- | --- | ---: | --- | --- | --- | --- |",
    ]
    for b in batches:
        for r in sorted(b["rows"], key=lambda r: r["trial"]):
            for i, a in enumerate(r.get("attempts", [])):
                lines.append(
                    f"| {b['label']} | {r['trial']} | {i} | {a.get('status')} | {a.get('task_spec_success')} | {a.get('success')} | {_cell(a.get('reason', ''))[:160]} |"
                )
    lines += [
        "",
        "Tokens sum returned input/output usage across policy/helper/repair/transport attempts; reasoning tokens are not added twice. `*` marks incomplete provider usage. `# of tokens used` is the effort count charged to the budgets; `(+N demo)` is the demonstration block summed over requests, and the provider total is their sum. Static compliance does not guarantee task success.",
        "",
    ]
    if notes:
        lines += ["Notes:", ""] + [f"- {n}" for n in notes] + [""]
    return "\n".join(lines)


def write(root, *, title=None, notes=()):
    root = Path(root).resolve()
    if (root / "schedule.json").exists():
        raise ValueError(
            f"{root} is a single batch; batch.write_report owns its report"
        )
    batches = [load_batch(root, d) for d in find_batches(root)]
    if not batches:
        raise ValueError(f"No CaP batches under {root}")
    text = render(root, batches, title=title, notes=notes)
    temporary = root / "report.md.tmp"
    temporary.write_text(text)
    temporary.replace(root / "report.md")
    return root / "report.md"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", type=Path)
    parser.add_argument("--title")
    parser.add_argument("--note", action="append", default=[])
    args = parser.parse_args()
    print(write(args.root, title=args.title, notes=args.note))


if __name__ == "__main__":
    main()
