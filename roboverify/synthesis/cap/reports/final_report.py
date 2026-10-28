"""Per-task tables and figures for a final CaP experiment (one model, six tasks).

Reads every batch below each ROOT (``revision6_experiment`` layout; batches
of one task are merged, e.g. a supplement root), and writes
``<prefix>_data.json``, Markdown tables (``<prefix>_tables.md``) and figures
``<plots>/<prefix>_*.png``::

    python -m synthesis.cap.reports.final_report ROOT [ROOT ...] \\
        --output synthesis/cap/docs --plots synthesis/cap/docs/plots --prefix final_v1

Success is ``task_spec_success`` (the frozen criterion, including the final
release check). A trial rejected before any request because its scene
matches a demonstration (``demonstration_rejected``) is excluded from the
denominator. Provider usage (cache reads/writes, cost) is read from the logged
``response.json`` of every model call.
"""

import argparse
import json
import math
import os
from collections import Counter
from pathlib import Path

from .batch_set_report import TASK_ORDER, find_batches, load_batch

ROUNDS = 5  # round 0 plus four recovery rounds
STATUS_ORDER = (
    "success",
    "primitive_failed",
    "failed",
    "completed_postcondition_false",
    "goal_failed",
    "unreleased_block",
    "constraint_failed",
    "budget_exhausted",
    "other",
)
EXCLUDED = "demonstration_rejected"
LABELS = {
    "stack": "Stack",
    "unstack": "Unstack",
    "reverse": "Reverse",
    "restack": "ReStack",
    "grid": "Grid",
    "pyramid": "Pyramid",
}


def wilson(successes, total, z=1.96):
    if not total:
        return float("nan"), float("nan")
    p = successes / total
    centre = (p + z * z / (2 * total)) / (1 + z * z / total)
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    half /= 1 + z * z / total
    return max(0.0, centre - half), min(1.0, centre + half)


def attempt_status(attempt):
    if attempt.get("task_spec_success"):
        return "success"
    status = attempt.get("status")
    if status == "completed":
        # The policy returned normally but the settled postcondition is false.
        return "completed_postcondition_false"
    return status if status in STATUS_ORDER else "other"


def provider_usage(batch_directory, part="trials"):
    """Totals over the logged responses below ``part`` of one batch (each call
    counted once); ``replaced`` holds trials requeued after infrastructure
    failures."""
    files = {
        os.path.realpath(p) for p in Path(batch_directory, part).rglob("response.json")
    }
    total = Counter()
    providers = Counter()
    for path in files:
        raw = json.loads(Path(path).read_text())
        usage = raw.get("usage") or {}
        details = usage.get("prompt_tokens_details") or {}
        total["calls"] += 1
        total["prompt"] += usage.get("prompt_tokens") or 0
        total["completion"] += usage.get("completion_tokens") or 0
        total["cached"] += details.get("cached_tokens") or 0
        total["written"] += details.get("cache_write_tokens") or 0
        total["cost"] += usage.get("cost") or 0.0
        providers[raw.get("provider") or "unknown"] += 1
    return dict(total), dict(providers)


def summarize(batches):
    """One task's results over its batches."""
    batch = batches[0]
    every = [
        r | {"seed": b["seeds"].get(r.get("job_id"))}
        for b in batches
        for r in b["rows"]
    ]
    excluded = [r for r in every if r.get("status") == EXCLUDED]
    rows = [r for r in every if r.get("status") != EXCLUDED]
    task = batch["run"]["task"]
    n = len(rows)
    final = sum(bool(r.get("task_spec_success")) for r in rows)
    first = sum(bool(r.get("first_attempt_task_spec_success")) for r in rows)
    by_round = [0] * ROUNDS
    for r in rows:
        k = r.get("task_spec_success_round")
        if r.get("task_spec_success") and k is not None:
            by_round[k] += 1
    cumulative = [sum(by_round[: k + 1]) for k in range(ROUNDS)]
    round0 = Counter(
        attempt_status(r["attempts"][0]) for r in rows if r.get("attempts")
    )
    attempts = [a for r in rows for a in r.get("attempts", [])]
    failures = Counter(attempt_status(a) for a in attempts) - Counter(
        {"success": len(attempts)}
    )
    stops = Counter(
        r.get("stop_reason") for r in rows if not r.get("task_spec_success")
    )
    usage, providers, replaced_usage, requeued = Counter(), Counter(), Counter(), []
    for b in batches:
        used, served = provider_usage(b["path"])
        usage.update(used)
        providers.update(served)
        replaced_usage.update(provider_usage(b["path"], "replaced")[0])
        log = Path(b["path"], "requeued.json")
        requeued += json.loads(log.read_text()) if log.exists() else []
    costs = [
        (r.get("prefix_cache") or {}).get("provider_cost_usd") or 0.0 for r in rows
    ]
    return {
        "task": task,
        "blocks": batch["run"].get("num_blocks"),
        "trials": n,
        "planned": sum(b["planned"] for b in batches),
        "excluded": [
            {"seed": r.get("seed"), "reason": (r.get("reason") or "")[:200]}
            for r in excluded
        ],
        "round0_success": first,
        "final_success": final,
        "round0_ci": wilson(first, n),
        "final_ci": wilson(final, n),
        "success_by_round": by_round,
        "cumulative_by_round": cumulative,
        "round0_status": dict(round0),
        "failed_attempt_status": dict(failures),
        "attempts": len(attempts),
        "unreleased_attempts": sum(
            (a.get("final_release") or {}).get("released") is False for a in attempts
        ),
        "failure_stop_reasons": dict(stops),
        "mean_rounds": sum(r.get("recovery_rounds") or 0 for r in rows) / max(n, 1),
        "mean_calls": sum(r.get("model_calls") or 0 for r in rows) / max(n, 1),
        "mean_effort_tokens": sum(r.get("known_tokens") or 0 for r in rows) / max(n, 1),
        "mean_wall_seconds": sum(r.get("active_wall_seconds") or 0 for r in rows)
        / max(n, 1),
        "trial_costs": costs,
        "usage": dict(usage),
        "providers": dict(providers),
        "replaced_usage": dict(replaced_usage),
        "requeued": Counter(e["reason"] for e in requeued),
        "seeds": sorted(r.get("seed") for r in rows if r.get("seed") is not None),
    }


def pct(k, n):
    return f"{k}/{n} ({100 * k / n:.0f}%)" if n else "–"


def ci(bounds):
    lo, hi = bounds
    return "–" if math.isnan(lo) else f"[{100 * lo:.0f}, {100 * hi:.0f}]"


def tables(results):
    out = []
    out.append(
        "| Task | Blocks | Trials | Excluded | Round-0 success | 95% CI | Final success"
        " | 95% CI | Mean rounds | Mean calls |"
    )
    out.append("| --- | ---: | ---: | ---: | ---: | --- | ---: | --- | ---: | ---: |")
    for r in results:
        out.append(
            f"| {LABELS[r['task']]} | {r['blocks']} | {r['trials']} |"
            f" {len(r['excluded'])} |"
            f" {pct(r['round0_success'], r['trials'])} | {ci(r['round0_ci'])} |"
            f" {pct(r['final_success'], r['trials'])} | {ci(r['final_ci'])} |"
            f" {r['mean_rounds']:.2f} | {r['mean_calls']:.2f} |"
        )
    out.append("")
    out.append("Successes by round (cumulative):")
    out.append("")
    out.append("| Task | " + " | ".join(f"≤ round {k}" for k in range(ROUNDS)) + " |")
    out.append("| --- |" + " ---: |" * ROUNDS)
    for r in results:
        out.append(
            f"| {LABELS[r['task']]} | "
            + " | ".join(str(c) for c in r["cumulative_by_round"])
            + " |"
        )
    out.append("")
    out.append("Round-0 outcome:")
    out.append("")
    out.append("| Task | " + " | ".join(STATUS_ORDER) + " |")
    out.append("| --- |" + " ---: |" * len(STATUS_ORDER))
    for r in results:
        out.append(
            f"| {LABELS[r['task']]} | "
            + " | ".join(str(r["round0_status"].get(s, 0)) for s in STATUS_ORDER)
            + " |"
        )
    out.append("")
    out.append("Failed attempts, all rounds:")
    out.append("")
    statuses = STATUS_ORDER[1:]
    out.append(
        "| Task | Attempts | " + " | ".join(statuses) + " | Ended holding a block |"
    )
    out.append("| --- | ---: |" + " ---: |" * (len(statuses) + 1))
    for r in results:
        out.append(
            f"| {LABELS[r['task']]} | {r['attempts']} | "
            + " | ".join(str(r["failed_attempt_status"].get(s, 0)) for s in statuses)
            + f" | {r['unreleased_attempts']} |"
        )
    out.append("")
    out.append("Cost and caching (provider-reported):")
    out.append("")
    out.append(
        "| Task | Calls | Prompt tokens | Cache read | Cache write | Output tokens"
        " | Cost | Cost per trial | Providers |"
    )
    out.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    total = Counter()
    for r in results:
        u = r["usage"]
        total.update(u)
        share = u.get("cached", 0) / max(u.get("prompt", 0), 1)
        out.append(
            f"| {LABELS[r['task']]} | {u.get('calls', 0)} | {u.get('prompt', 0):,} |"
            f" {u.get('cached', 0):,} ({100 * share:.0f}%) | {u.get('written', 0):,} |"
            f" {u.get('completion', 0):,} | ${u.get('cost', 0):.2f} |"
            f" ${u.get('cost', 0) / max(r['trials'], 1):.3f} |"
            f" {', '.join(f'{k} {v}' for k, v in sorted(r['providers'].items()))} |"
        )
    share = total.get("cached", 0) / max(total.get("prompt", 0), 1)
    out.append(
        f"| Scored trials | {total.get('calls', 0)} | {total.get('prompt', 0):,} |"
        f" {total.get('cached', 0):,} ({100 * share:.0f}%) | {total.get('written', 0):,} |"
        f" {total.get('completion', 0):,} | ${total.get('cost', 0):.2f} | | |"
    )
    out.append("")
    out.append(
        "Trials requeued after infrastructure failures (rerun with the same seed):"
    )
    out.append("")
    out.append("| Task | Interrupted at stop | HTTP 429 | Calls | Cost |")
    out.append("| --- | ---: | ---: | ---: | ---: |")
    for r in results:
        q, u = r["requeued"], r["replaced_usage"]
        out.append(
            f"| {LABELS[r['task']]} |"
            f" {sum(v for k, v in q.items() if k.startswith('interrupted'))} |"
            f" {sum(v for k, v in q.items() if '429' in k)} |"
            f" {u.get('calls', 0)} | ${u.get('cost', 0):.2f} |"
        )
    return "\n".join(out) + "\n"


def plots(results, directory, prefix):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    names = [LABELS[r["task"]] for r in results]
    x = np.arange(len(results))
    written = []

    def save(fig, name):
        path = directory / f"{prefix}_{name}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)

    # 1. Round-0 and final success with Wilson intervals.
    fig, ax = plt.subplots(figsize=(8, 4))
    for offset, key, label, colour in (
        (-0.2, "round0", "Round 0", "#9ecae1"),
        (0.2, "final", "Final (≤ 4 recovery rounds)", "#3182bd"),
    ):
        rates = np.array([r[f"{key}_success"] / max(r["trials"], 1) for r in results])
        lo = np.array([r[f"{key}_ci"][0] for r in results])
        hi = np.array([r[f"{key}_ci"][1] for r in results])
        ax.bar(x + offset, 100 * rates, 0.4, label=label, color=colour)
        ax.errorbar(
            x + offset,
            100 * rates,
            yerr=[100 * (rates - lo), 100 * (hi - rates)],
            fmt="none",
            ecolor="black",
            capsize=3,
            linewidth=1,
        )
    ax.set_xticks(x, names)
    ax.set_ylabel("Task success (%)")
    ax.set_ylim(0, 105)
    ax.legend(loc="lower left")
    ax.set_title("Task success per task (95% Wilson intervals)")
    save(fig, "success")

    # 2. Cumulative success over recovery rounds.
    fig, ax = plt.subplots(figsize=(7, 4))
    for r in results:
        n = max(r["trials"], 1)
        ax.plot(
            range(ROUNDS),
            [100 * c / n for c in r["cumulative_by_round"]],
            marker="o",
            label=LABELS[r["task"]],
        )
    ax.set_xticks(range(ROUNDS), ["0"] + [str(k) for k in range(1, ROUNDS)])
    ax.set_xlabel("Recovery round")
    ax.set_ylabel("Cumulative success (%)")
    ax.set_ylim(0, 105)
    ax.legend(ncol=2, fontsize=8)
    ax.set_title("Cumulative task success by recovery round")
    save(fig, "recovery")

    # 3. Round-0 outcome breakdown.
    fig, ax = plt.subplots(figsize=(8, 4))
    bottom = np.zeros(len(results))
    colours = plt.get_cmap("tab10")
    for i, status in enumerate(STATUS_ORDER):
        values = np.array(
            [
                100 * r["round0_status"].get(status, 0) / max(r["trials"], 1)
                for r in results
            ]
        )
        if values.any():
            ax.bar(x, values, 0.6, bottom=bottom, label=status, color=colours(i))
            bottom += values
    ax.set_xticks(x, names)
    ax.set_ylabel("Round-0 attempts (%)")
    ax.legend(fontsize=8, ncol=2, loc="upper right", bbox_to_anchor=(1.0, -0.12))
    ax.set_title("Round-0 outcome per task")
    save(fig, "round0_outcomes")

    # 4. Cost per trial.
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.boxplot([r["trial_costs"] for r in results], tick_labels=names, showmeans=True)
    ax.set_ylabel("Provider cost per trial (USD)")
    ax.set_title("Cost per trial (all rounds, including recovery)")
    save(fig, "cost")

    # 5. Prompt tokens: cache read, cache write, uncached.
    fig, ax = plt.subplots(figsize=(8, 4))
    read = np.array([r["usage"].get("cached", 0) for r in results]) / 1e6
    write = np.array([r["usage"].get("written", 0) for r in results]) / 1e6
    prompt = np.array([r["usage"].get("prompt", 0) for r in results]) / 1e6
    ax.bar(x, read, 0.6, label="cache read", color="#31a354")
    ax.bar(x, write, 0.6, bottom=read, label="cache write", color="#fdae6b")
    ax.bar(
        x,
        prompt - read - write,
        0.6,
        bottom=read + write,
        label="uncached",
        color="#bdbdbd",
    )
    ax.set_xticks(x, names)
    ax.set_ylabel("Prompt tokens (millions)")
    ax.legend()
    ax.set_title("Prompt tokens by cache use")
    save(fig, "cache")
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("roots", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plots", type=Path, required=True)
    parser.add_argument("--prefix", default="final_v1")
    args = parser.parse_args()
    by_task = {}
    for root in args.roots:
        for directory in find_batches(root):
            batch = load_batch(root, directory)
            if batch["rows"]:
                by_task.setdefault(batch["run"]["task"], []).append(batch)
    results = [summarize(by_task[t]) for t in TASK_ORDER if t in by_task]
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"{args.prefix}_data.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )
    (args.output / f"{args.prefix}_tables.md").write_text(tables(results))
    for path in plots(results, args.plots, args.prefix):
        print(path)


if __name__ == "__main__":
    main()
