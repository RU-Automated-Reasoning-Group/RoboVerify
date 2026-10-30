"""Tables and figures for the Revision 9 block-count generalization run.

Reads a run directory written by `experiments/revision9_generalization.py`
(`programs.json`, `results/<kind>/<task>/*.jsonl`, and the RoboVerify
synthesis summaries under `roboverify_synthesis/`), and writes
`<output>/<prefix>_data.json`, `<output>/<prefix>_tables.md` and figures
`<plots>/<prefix>_*.png`::

    python -m synthesis.cap.reports.generalization_report RUN \\
        --output synthesis/cap/docs --plots synthesis/cap/docs/plots --prefix revision9

Every trial of a CaP pool contributes its program's results (trial-weighted);
distinct programs are also counted once each. Success is the driver's
`success` (`task-spec-v2+final-release-v1` for CaP, `task-spec-v2` for
RoboVerify). Intervals are 95% bootstrap intervals over trials (programs are
the unit that varies; cases are shared), or Wilson intervals for a single
program.
"""

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

TASKS = ("stack", "unstack", "reverse", "restack", "grid", "pyramid")
LABELS = {
    "stack": "Stack",
    "unstack": "Unstack",
    "reverse": "Reverse",
    "restack": "ReStack",
    "grid": "Grid",
    "pyramid": "Pyramid",
}
TRAINED = {
    "stack": 4,
    "unstack": 4,
    "reverse": 4,
    "restack": 5,
    "grid": 6,
    "pyramid": 6,
}
SYNTHESIS_GROUPS = ("stack", "stack_big", "unstack")
POOLS = (("final_v2", "final_v2"), ("final_v2_wo_demo", "final_v2 w/o demo"))
COLOURS = {
    "final_v2": "#3182bd",
    "final_v2_wo_demo": "#e6550d",
    "rv_synth": "#31a354",
    "rv_supplied": "#636363",
}
FAILURE_ORDER = (
    "success",
    "move_not_converged",
    "pick_not_converged",
    "release_not_converged",
    "get_failed",
    "postcondition_false",
    "toppled_while_settling",
    "unreleased_block",
    "loop_limit",
    "budget_exhausted",
    "unsupported_count",
    "other",
    "simulator_failure",
)
FAILURE_COLOURS = {
    "success": "#31a354",
    "move_not_converged": "#fd8d3c",
    "pick_not_converged": "#fdae6b",
    "release_not_converged": "#fdd0a2",
    "get_failed": "#3182bd",
    "postcondition_false": "#de2d26",
    "toppled_while_settling": "#fc9272",
    "unreleased_block": "#9e9ac8",
    "loop_limit": "#bcbddc",
    "budget_exhausted": "#756bb1",
    "unsupported_count": "#bdbdbd",
    "other": "#636363",
    "simulator_failure": "#000000",
}


def failure_kind(row):
    """A finer label than the outcome, for the outcome figures."""
    if row.get("success"):
        return "success"
    if row.get("outcome") == "simulator_failure":
        return "simulator_failure"
    status = row.get("status") or ""
    reason = row.get("reason") or ""
    if status == "unsupported_count":
        return "unsupported_count"
    if row.get("outcome") == "budget_exhausted" or "budget" in status:
        return "budget_exhausted"
    for primitive in ("Move", "Pick", "Release"):
        if f"{primitive} did not converge" in reason:
            return f"{primitive.lower()}_not_converged"
    if "Controller step limit" in reason:
        # RoboVerify's executor: "... {'converged': False, 'phase': 'move', ...}".
        phase = re.search(r"'phase': '(\w+)'", reason)
        kind = phase.group(1) if phase else "move"
        kind = {
            "approach": "pick",
            "descend": "pick",
            "close": "pick",
            "retreat": "release",
            "open": "release",
        }.get(kind, kind)
        return (
            f"{kind}_not_converged" if kind in ("move", "pick", "release") else "other"
        )
    if "still has a guard witness" in reason:
        return "loop_limit"
    if "No object satisfies get condition" in reason:
        return "get_failed"
    if status == "unreleased_block" or "unreleased" in reason:
        return "unreleased_block"
    if (
        row.get("task_spec_postcondition")
        and row.get("task_spec_settled_postcondition") is False
    ):
        return "toppled_while_settling"
    if status in ("goal_failed", "completed", "invalid") or "Settled task" in reason:
        return "postcondition_false"
    return "other"


def wilson(k, n, z=1.96):
    if not n:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def bootstrap(values, reps=2000, seed=0):
    import numpy as np

    values = np.asarray(values, float)
    if len(values) <= 1:
        return (float(values.mean()) if len(values) else 0.0,) * 2
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (reps, len(values)))].mean(axis=1)
    return (float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)))


def read_jsonl(paths):
    rows = []
    for path in paths:
        for line in Path(path).read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load(run):
    run = Path(run)
    from synthesis.cap.generalization import testset as ts

    _, cases = ts.load_manifest(ts.TESTSET_DIR / "relaxed_v2.json")
    case_n = {c.case_id: c.layout["num_blocks"] for c in cases}
    task_cases = defaultdict(list)
    for c in cases:
        task_cases[c.task].append(c.case_id)
    trials = json.loads((run / "programs.json").read_text())
    cap = {}
    for row in read_jsonl(sorted((run / "results" / "cap").glob("*/*.jsonl"))):
        cap[(row["task"], row["program"], row["case"])] = row
    rv = defaultdict(dict)
    for path in sorted((run / "results" / "roboverify").glob("*/*.jsonl")):
        group = path.parent.name
        for row in read_jsonl([path]):
            label = row["program"] if group in TASKS else f"{group}:{row['program']}"
            rv[(row["task"], label)][row["case"]] = row
    attempts = defaultdict(list)
    for path in sorted(
        (run / "roboverify_synthesis").glob("*/synth_[0-9][0-9]/synthesis.json")
    ):
        summary = json.loads(path.read_text())
        group = path.parents[1].name
        if group not in SYNTHESIS_GROUPS:
            continue
        summary["group"] = group
        summary["task"] = "unstack" if group == "unstack" else "stack"
        attempts[group].append(summary)
    return dict(
        case_n=case_n,
        task_cases=task_cases,
        trials=trials,
        cap=cap,
        rv=rv,
        attempts=attempts,
    )


def policies(data):
    """{(arm, task): [{"id", "rows": {case: row}}]} for every evaluated policy.

    CaP arms list one entry per trial; RoboVerify arms one per program
    (supplied) or per synthesis attempt (a failed attempt has no rows)."""
    out = defaultdict(list)
    for trial in data["trials"]:
        task, program = trial["task"], trial["alpha_sha256"][:16]
        rows = {
            cid: data["cap"][(task, program, cid)]
            for cid in data["task_cases"][task]
            if (task, program, cid) in data["cap"]
        }
        out[(trial["pool"], task)].append(
            dict(
                id=f"{trial['pool']}/{task}/seed{trial['seed']}",
                program=program,
                rows=rows,
            )
        )
    for (task, label), rows in data["rv"].items():
        arm = "rv_supplied" if label == "supplied" else "rv_synth"
        if label.startswith("stack_big:"):
            arm = "rv_synth_big"
        out[(arm, task)].append(dict(id=label, program=label, rows=rows))
    return out


def per_count(policy_list, case_n, task_cases, key="success"):
    """{N: [per-policy mean of `key` over that policy's cases at N]}."""
    counts = sorted({case_n[c] for c in task_cases})
    table = {}
    for n in counts:
        ids = [c for c in task_cases if case_n[c] == n]
        values = []
        for policy in policy_list:
            got = [policy["rows"][c].get(key) for c in ids if c in policy["rows"]]
            got = [float(v) for v in got if v is not None]
            if got:
                values.append(sum(got) / len(got))
        table[n] = values
    return table


def largest_general_count(policy, case_n, task_cases):
    """Largest N such that every case at every count <= N succeeds (0: none)."""
    best = 0
    for n in sorted({case_n[c] for c in task_cases}):
        ids = [c for c in task_cases if case_n[c] == n]
        if all(policy["rows"].get(c, {}).get("success") for c in ids):
            best = n
        else:
            break
    return best


def summarize(data):
    pols = policies(data)
    case_n, task_cases = data["case_n"], data["task_cases"]
    summary = dict(tasks={}, attempts={})
    for task in TASKS:
        cases = task_cases[task]
        entry = {}
        for arm in (
            "final_v2",
            "final_v2_wo_demo",
            "rv_supplied",
            "rv_synth",
            "rv_synth_big",
        ):
            plist = pols.get((arm, task), [])
            if not plist:
                continue
            complete = [p for p in plist if is_complete(p, cases)]
            rates = [
                sum(bool(r.get("success")) for r in p["rows"].values()) / len(p["rows"])
                for p in complete
            ]
            trained = [c for c in cases if case_n[c] == TRAINED[task]]
            trained_rates = [
                sum(bool(p["rows"][c].get("success")) for c in trained) / len(trained)
                for p in complete
                if all(c in p["rows"] for c in trained)
            ]
            failures = Counter(
                failure_kind(r) for p in complete for r in p["rows"].values()
            )
            distinct = {p["program"]: p for p in complete}
            entry[arm] = dict(
                policies=len(plist),
                complete=len(complete),
                distinct=len(distinct),
                success=sum(rates) / len(rates) if rates else None,
                success_ci=bootstrap(rates) if rates else None,
                distinct_success=(
                    sum(
                        sum(bool(r.get("success")) for r in p["rows"].values())
                        / len(p["rows"])
                        for p in distinct.values()
                    )
                    / len(distinct)
                    if distinct
                    else None
                ),
                trained_success=(
                    sum(trained_rates) / len(trained_rates) if trained_rates else None
                ),
                fully_general=sum(r == 1.0 for r in rates),
                trained_all=sum(r == 1.0 for r in trained_rates),
                largest=[largest_general_count(p, case_n, cases) for p in complete],
                by_count={
                    n: dict(
                        mean=sum(v) / len(v) if v else None,
                        ci=bootstrap(v) if v else None,
                        policies=len(v),
                        cases=sum(1 for c in cases if case_n[c] == n),
                    )
                    for n, v in per_count(complete, case_n, cases).items()
                },
                reward_by_count={
                    n: (sum(v) / len(v) if v else None)
                    for n, v in per_count(
                        complete, case_n, cases, "partial_reward"
                    ).items()
                },
                failures=dict(failures),
            )
        supplied = pols.get(("rv_supplied", task))
        for arm in ("final_v2", "final_v2_wo_demo"):
            if arm not in entry or not supplied:
                continue
            reference = supplied[0]["rows"]
            same = total = identical = 0
            for p in pols[(arm, task)]:
                matches = [
                    bool(p["rows"][c].get("success"))
                    == bool(reference[c].get("success"))
                    for c in cases
                    if c in p["rows"] and c in reference
                ]
                same += sum(matches)
                total += len(matches)
                identical += bool(matches) and all(matches)
            entry[arm]["supplied_agreement"] = same / total if total else None
            entry[arm]["supplied_identical"] = identical
        summary["tasks"][task] = entry
    for group, rows in data["attempts"].items():
        summary["attempts"][group] = dict(
            attempts=len(rows),
            status=dict(Counter(str(r.get("status")) for r in rows)),
            with_program=sum(bool(r.get("has_program")) for r in rows),
            verified=sum(r.get("status") == "verified_model" for r in rows),
            synth_wall_median=(
                sorted(r["synth_wall"] for r in rows)[len(rows) // 2] if rows else None
            ),
        )
    return summary


PARTIAL = (
    False  # --partial: also use policies with some cases missing (progress checks)
)


def is_complete(policy, cases):
    return bool(policy["rows"]) if PARTIAL else len(policy["rows"]) == len(cases)


def pct(x):
    return "—" if x is None else f"{100 * x:.1f}%"


def tables(summary, case_counts):
    lines = []
    lines += [
        "| Task | Cases | Arm | Policies (distinct) | Success, all cases [95% CI] | Distinct programs | Trained count | All cases solved | Trained count solved |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    names = {
        "final_v2": "final_v2",
        "final_v2_wo_demo": "final_v2 w/o demo",
        "rv_supplied": "RoboVerify supplied",
        "rv_synth": "RoboVerify synthesized",
        "rv_synth_big": "RoboVerify synthesized (4× budget)",
    }
    for task in TASKS:
        for arm, e in summary["tasks"][task].items():
            ci = e["success_ci"]
            lines.append(
                f"| {LABELS[task]} | {case_counts[task]} | {names[arm]} | {e['complete']} ({e['distinct']}) |"
                f" {pct(e['success'])} [{pct(ci[0]) if ci else '—'}, {pct(ci[1]) if ci else '—'}] |"
                f" {pct(e['distinct_success'])} | {pct(e['trained_success'])} |"
                f" {e['fully_general']}/{e['complete']} | {e['trained_all']}/{e['complete']} |"
            )
    lines.append("")
    lines += [
        "**Agreement with RoboVerify's supplied program** (per-case success equal; trials whose every case agrees)",
        "",
        "| Task | final_v2 agreement | final_v2 identical | w/o demo agreement | w/o demo identical |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for task in TASKS:
        e = summary["tasks"][task]
        cells = []
        for arm in ("final_v2", "final_v2_wo_demo"):
            a = e.get(arm, {})
            cells += [
                pct(a.get("supplied_agreement")),
                f"{a.get('supplied_identical', 0)}/{a.get('complete', 0)}",
            ]
        lines.append(f"| {LABELS[task]} | " + " | ".join(cells) + " |")
    lines.append("")
    lines += [
        "**Largest generalizing count** (median over trials; share of trials reaching at least the trained count)",
        "",
        "| Task | Trained | final_v2 median | final_v2 ≥ trained | w/o demo median | w/o demo ≥ trained | RoboVerify supplied |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for task in TASKS:
        e = summary["tasks"][task]
        cells = []
        for arm in ("final_v2", "final_v2_wo_demo"):
            values = sorted(e.get(arm, {}).get("largest", []))
            cells.append(str(values[len(values) // 2]) if values else "—")
            cells.append(
                f"{sum(v >= TRAINED[task] for v in values)}/{len(values)}"
                if values
                else "—"
            )
        supplied = e.get("rv_supplied", {}).get("largest", [])
        cells.append(str(supplied[0]) if supplied else "—")
        lines.append(
            f"| {LABELS[task]} | {TRAINED[task]} | " + " | ".join(cells) + " |"
        )
    lines.append("")
    for task in TASKS:
        entry = summary["tasks"][task]
        arms = [a for a in entry if entry[a]["complete"]]
        if not arms:
            continue
        counts = sorted({int(n) for a in arms for n in entry[a]["by_count"]})
        lines += [
            f"**{LABELS[task]}: success by block count** (mean over policies; cases per count in parentheses)",
            "",
            "| Arm | "
            + " | ".join(
                f"{n} ({entry[arms[0]]['by_count'][n]['cases']})" for n in counts
            )
            + " |",
            "| --- |" + " ---: |" * len(counts),
        ]
        for arm in arms:
            cells = [pct(entry[arm]["by_count"][n]["mean"]) for n in counts]
            lines.append(f"| {names[arm]} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += [
        "**Failure kinds** (share of all executions of the arm)",
        "",
        "| Task | Arm | " + " | ".join(FAILURE_ORDER) + " |",
        "| --- | --- |" + " ---: |" * len(FAILURE_ORDER),
    ]
    for task in TASKS:
        for arm, e in summary["tasks"][task].items():
            total = sum(e["failures"].values())
            if not total:
                continue
            cells = [pct(e["failures"].get(k, 0) / total) for k in FAILURE_ORDER]
            lines.append(
                f"| {LABELS[task]} | {names[arm]} | " + " | ".join(cells) + " |"
            )
    lines.append("")
    if summary["attempts"]:
        lines += [
            "**RoboVerify synthesis attempts**",
            "",
            "| Group | Attempts | Statuses | With a program | verified_model | Median synthesis wall (s) |",
            "| --- | ---: | --- | ---: | ---: | ---: |",
        ]
        for group, a in summary["attempts"].items():
            status = ", ".join(f"{k} {v}" for k, v in sorted(a["status"].items()))
            wall = a["synth_wall_median"]
            lines.append(
                f"| {group} | {a['attempts']} | {status} | {a['with_program']} | {a['verified']} |"
                f" {wall:.0f} |"
                if wall is not None
                else f"| {group} | 0 | | | | |"
            )
        lines.append("")
    return "\n".join(lines)


# ------------------------------------------------------------------ figures


def figures(data, summary, directory, prefix):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    written = []

    def save(fig, name):
        path = directory / f"{prefix}_{name}.png"
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)

    arms = (
        ("final_v2", "CaP final_v2", "-", "o"),
        ("final_v2_wo_demo", "CaP final_v2 w/o demo", "-", "s"),
        ("rv_synth", "RoboVerify synthesized", "-", "^"),
        ("rv_supplied", "RoboVerify supplied program", "--", "x"),
    )

    def curves(key, ylabel, name, title, bands=True):
        fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=True)
        for ax, task in zip(axes.flat, TASKS):
            entry = summary["tasks"][task]
            for arm, label, style, marker in arms:
                if arm not in entry or not entry[arm]["complete"]:
                    continue
                table = entry[arm][key]
                ns = sorted(
                    int(n)
                    for n in table
                    if (table[n]["mean"] if key == "by_count" else table[n]) is not None
                )
                if key == "by_count":
                    ys = [table[n]["mean"] for n in ns]
                    lo = [table[n]["ci"][0] for n in ns]
                    hi = [table[n]["ci"][1] for n in ns]
                    if bands and entry[arm]["complete"] > 1:
                        ax.fill_between(
                            ns,
                            np.multiply(lo, 100),
                            np.multiply(hi, 100),
                            color=COLOURS[arm],
                            alpha=0.15,
                            linewidth=0,
                        )
                else:
                    ys = [table[n] for n in ns]
                ys = [np.nan if y is None else 100 * y for y in ys]
                ax.plot(
                    ns,
                    ys,
                    style,
                    marker=marker,
                    ms=4,
                    color=COLOURS[arm],
                    label=f"{label} ({entry[arm]['complete']})",
                )
            ax.axvline(TRAINED[task], color="#999999", lw=0.8, ls=":")
            ax.set_title(LABELS[task])
            ax.set_xlabel("blocks")
            ax.set_ylim(-3, 103)
            ax.grid(alpha=0.3)
        for ax in axes[:, 0]:
            ax.set_ylabel(ylabel)
        handles, labels = [], []
        for ax in axes.flat:
            for h, l in zip(*ax.get_legend_handles_labels()):
                key_label = l.split(" (")[0]
                if key_label not in [x.split(" (")[0] for x in labels]:
                    handles.append(h)
                    labels.append(key_label)
        fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
        fig.suptitle(title)
        fig.tight_layout(rect=(0, 0.05, 1, 0.97))
        path = directory / f"{prefix}_{name}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)

    curves(
        "by_count",
        "success (%)",
        "success",
        "Success against block count (dotted line: trained count; band: 95% bootstrap CI over trials)",
    )
    curves(
        "reward_by_count",
        "mean partial reward (%)",
        "partial_reward",
        "Partial reward against block count",
    )
    curves(
        "by_count",
        "success rate (%)",
        "success_rate",
        "Success rate against block count (task-spec-v2)",
        bands=False,
    )

    # Per-trial success heatmaps: rows are trials, columns block counts.
    pols = policies(data)
    fig, axes = plt.subplots(2, 6, figsize=(16, 7.5))
    for col, task in enumerate(TASKS):
        cases = data["task_cases"][task]
        ns = sorted({data["case_n"][c] for c in cases})
        for row, (pool, pool_label) in enumerate(POOLS):
            ax = axes[row, col]
            plist = [p for p in pols.get((pool, task), []) if is_complete(p, cases)]
            if not plist:
                ax.axis("off")
                continue
            matrix = np.full((len(plist), len(ns)), np.nan)
            for i, p in enumerate(plist):
                for j, n in enumerate(ns):
                    got = [
                        bool(r.get("success"))
                        for c, r in p["rows"].items()
                        if data["case_n"][c] == n
                    ]
                    if got:
                        matrix[i, j] = sum(got) / len(got)
            order = np.argsort(-np.nan_to_num(matrix).sum(axis=1), kind="stable")
            ax.imshow(
                matrix[order],
                aspect="auto",
                cmap="RdYlGn",
                vmin=0,
                vmax=1,
                interpolation="nearest",
            )
            ax.set_xticks(range(len(ns)))
            ax.set_xticklabels(ns, fontsize=6, rotation=90)
            ax.set_yticks([])
            if TRAINED[task] in ns:
                ax.axvline(ns.index(TRAINED[task]), color="black", lw=0.6, ls=":")
            ax.set_title(f"{LABELS[task]}: {pool_label}", fontsize=8)
            if col == 0:
                ax.set_ylabel(f"trials ({len(plist)})")
    fig.suptitle(
        "Per-trial success rate by block count (green 100%, red 0%; rows sorted by overall success)"
    )
    save(fig, "trial_heatmap")

    # Largest generalizing count: ECDF per task and pool.
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, task in zip(axes.flat, TASKS):
        entry = summary["tasks"][task]
        ns = sorted({data["case_n"][c] for c in data["task_cases"][task]})
        for pool, label in POOLS:
            if pool not in entry:
                continue
            values = sorted(entry[pool]["largest"])
            if not values:
                continue
            xs = [0] + ns
            ys = [100 * sum(v >= x for v in values) / len(values) for x in xs]
            ax.step(xs, ys, where="post", color=COLOURS[pool], label=label)
        for arm in ("rv_supplied",):
            if arm in entry and entry[arm]["largest"]:
                ax.axvline(
                    entry[arm]["largest"][0],
                    color=COLOURS[arm],
                    ls="--",
                    lw=1,
                    label="RoboVerify supplied",
                )
        ax.axvline(TRAINED[task], color="#999999", lw=0.8, ls=":")
        ax.set_title(LABELS[task])
        ax.set_xlabel("N")
        ax.set_ylabel("trials solving every case up to N (%)")
        ax.set_ylim(-3, 103)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle(
        "Largest generalizing count (every case at every count up to N succeeds)"
    )
    save(fig, "largest_count")

    # Failure kinds per task and arm.
    fig, axes = plt.subplots(1, 6, figsize=(16, 4.8), sharey=True)
    arm_names = (
        ("final_v2", "v2"),
        ("final_v2_wo_demo", "w/o demo"),
        ("rv_supplied", "RV supplied"),
        ("rv_synth", "RV synth"),
    )
    for ax, task in zip(axes, TASKS):
        entry = summary["tasks"][task]
        present = [
            (a, l)
            for a, l in arm_names
            if a in entry and sum(entry[a]["failures"].values())
        ]
        bottom = np.zeros(len(present))
        for kind in FAILURE_ORDER:
            shares = np.array(
                [
                    100
                    * entry[a]["failures"].get(kind, 0)
                    / sum(entry[a]["failures"].values())
                    for a, _ in present
                ]
            )
            if shares.any():
                ax.bar(
                    range(len(present)),
                    shares,
                    bottom=bottom,
                    color=FAILURE_COLOURS[kind],
                    label=kind if task == TASKS[0] else None,
                )
            bottom += shares
        ax.set_xticks(range(len(present)))
        ax.set_xticklabels([l for _, l in present], rotation=30, fontsize=8)
        ax.set_title(LABELS[task])
    axes[0].set_ylabel("share of executions (%)")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=FAILURE_COLOURS[k]) for k in FAILURE_ORDER
    ]
    fig.legend(
        handles, FAILURE_ORDER, loc="lower center", ncol=6, frameon=False, fontsize=8
    )
    fig.suptitle("Outcome of every execution")
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    path = directory / f"{prefix}_outcomes.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(path)

    # Failure kinds against block count, CaP pools combined.
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=True)
    for ax, task in zip(axes.flat, TASKS):
        cases = data["task_cases"][task]
        ns = sorted({data["case_n"][c] for c in cases})
        counts = {n: Counter() for n in ns}
        for pool, _ in POOLS:
            for p in pols.get((pool, task), []):
                for cid, row in p["rows"].items():
                    counts[data["case_n"][cid]][failure_kind(row)] += 1
        bottom = np.zeros(len(ns))
        for kind in FAILURE_ORDER:
            shares = np.array(
                [100 * counts[n][kind] / max(1, sum(counts[n].values())) for n in ns]
            )
            if shares.any():
                ax.bar(
                    range(len(ns)), shares, bottom=bottom, color=FAILURE_COLOURS[kind]
                )
            bottom += shares
        ax.set_xticks(range(len(ns)))
        ax.set_xticklabels(ns, fontsize=7)
        ax.set_title(LABELS[task])
        ax.set_xlabel("blocks")
    for ax in axes[:, 0]:
        ax.set_ylabel("share of CaP executions (%)")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=FAILURE_COLOURS[k]) for k in FAILURE_ORDER
    ]
    fig.legend(
        handles, FAILURE_ORDER, loc="lower center", ncol=6, frameon=False, fontsize=8
    )
    fig.suptitle("CaP outcomes by block count (both pools)")
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    path = directory / f"{prefix}_outcomes_by_count.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(path)
    return written


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run")
    parser.add_argument("--output", default="synthesis/cap/docs")
    parser.add_argument("--plots", default="synthesis/cap/docs/plots")
    parser.add_argument("--prefix", default="revision9")
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args(argv)
    global PARTIAL
    PARTIAL = args.partial
    data = load(args.run)
    summary = summarize(data)
    case_counts = {t: len(data["task_cases"][t]) for t in TASKS}
    out = Path(args.output)
    (out / f"{args.prefix}_data.json").write_text(
        json.dumps(summary, indent=1, default=str)
    )
    (out / f"{args.prefix}_tables.md").write_text(tables(summary, case_counts))
    for path in figures(data, summary, args.plots, args.prefix):
        print(path)


if __name__ == "__main__":
    main()
