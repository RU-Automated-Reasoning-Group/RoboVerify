"""Figures for the Revision 4.3 prompt factorial.

Reads trial summaries from the main batch and its infrastructure rerun; trials
listed in the rerun's ``rerun_job_ids`` are taken from the rerun only.
"""

import argparse
import json
import math
from pathlib import Path

from ..experiments.revision43_prompt_experiment import SWITCHES

METHOD_NAMES = {
    "cap_google_baseline": ("baseline", "Baseline"),
    "cap_method_1": ("method1", "Method 1"),
    "cap_method_2": ("method2", "Method 2"),
}
SHORT = {
    "explain_target_offsets": "T",
    "explain_convergence_criteria": "C",
    "explain_partial_execution": "P",
    "explain_motion_planning_limits": "M",
}


def trials(main, rerun=None):
    """Summaries of the analysed trials: main batch, rerun replacing its jobs."""
    replaced = set()
    rows = []
    if rerun:
        plan = json.loads((Path(rerun) / "comparison_plan.json").read_text())
        replaced = set(plan["rerun_job_ids"])
        rows += [
            json.loads(p.read_text())
            for p in sorted(Path(rerun).glob("trials/*/summary.json"))
        ]
    rows += [
        row
        for p in sorted(Path(main).glob("trials/*/summary.json"))
        if (row := json.loads(p.read_text()))["job_id"] not in replaced
    ]
    return rows


def wilson(successes, total, z=1.96):
    if not total:
        return float("nan"), float("nan")
    p = successes / total
    centre = (p + z * z / (2 * total)) / (1 + z * z / total)
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    half /= 1 + z * z / total
    return max(0.0, centre - half), min(1.0, centre + half)


def cells(rows, method=None):
    """Per-cell counts of round-0 and final task success, optionally one method."""
    table = {}
    for row in rows:
        if method and row["method"] != method:
            continue
        cell = table.setdefault(row["variant"], {"n": 0, "round0": 0, "final": 0})
        cell["n"] += 1
        cell["round0"] += bool(row["first_attempt_success"])
        cell["final"] += bool(row["task_success"])
    return table


def cell_order(table):
    """Cells grouped by the motion-planning block, then by round-0 success."""
    # Cell names are "p" + one bit per switch in SWITCHES order; M is last.
    return sorted(table, key=lambda c: (c[-1], table[c]["round0"] / table[c]["n"]))


def plot_cells(table, path, *, order=None, scope="3 methods pooled"):
    """Horizontal bars per cell, grouped by the motion-planning block."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Patch

    order = order or cell_order(table)
    labels = [
        " ".join(SHORT[s] if bit == "1" else "·" for s, bit in zip(SWITCHES, cell[1:]))
        + f"   {cell}"
        for cell in order
    ]
    measures = [
        ("round0", "Round-0 success", "#de8f05"),
        ("final", "Final success", "#2077b4"),
    ]
    y = np.arange(len(order))
    height = 0.38
    fig, ax = plt.subplots(figsize=(8.5, 7.2))
    split = sum(cell[-1] == "0" for cell in order)
    ax.axhspan(split - 0.5, len(order) - 0.5, color="#eef5ea", zorder=0)
    for k, (field, name, colour) in enumerate(measures):
        rates = [table[c][field] / table[c]["n"] for c in order]
        bounds = [wilson(table[c][field], table[c]["n"]) for c in order]
        err = np.array(
            [
                [r - lo for r, (lo, _) in zip(rates, bounds)],
                [hi - r for r, (_, hi) in zip(rates, bounds)],
            ]
        )
        offset = (k - 0.5) * height
        ax.barh(y + offset, rates, height, color=colour, label=name, zorder=2)
        ax.errorbar(
            rates,
            y + offset,
            xerr=err,
            fmt="none",
            ecolor="black",
            elinewidth=0.8,
            capsize=2,
            zorder=3,
        )
        for yy, r in zip(y + offset, rates):
            ax.text(
                0.01,
                yy,
                f"{r:.0%}",
                va="center",
                ha="left",
                fontsize=7,
                color="white",
                fontweight="bold",
                zorder=4,
            )
    ax.set_yticks(y, labels, fontfamily="monospace", fontsize=8)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Task success (95% Wilson interval)")
    n = {table[c]["n"] for c in order}
    ax.set_title(
        "Prompt factorial, Qwen3.8 27B: success by cell "
        f"({'/'.join(map(str, sorted(n)))} trials per cell, {scope})",
        fontsize=10,
    )
    handles = [Patch(facecolor=c, label=l) for _, l, c in measures] + [
        Patch(facecolor="#eef5ea", label="Motion-planning limits on (M)")
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=3,
        fontsize=8,
        frameon=False,
    )
    ax.grid(axis="x", alpha=0.3, zorder=1)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main", type=Path, required=True)
    parser.add_argument("--rerun", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = trials(args.main, args.rerun)
    table = cells(rows)
    order = cell_order(table)
    plot_cells(table, args.output, order=order)
    report = {"pooled": table}
    # One figure per method, in the pooled cell order so the three compare row by row.
    for method, (suffix, name) in METHOD_NAMES.items():
        per_method = cells(rows, method)
        output = args.output.with_name(
            f"{args.output.stem}_{suffix}{args.output.suffix}"
        )
        plot_cells(per_method, output, order=order, scope=f"{name}; pooled cell order")
        report[method] = per_method
    print(json.dumps(report, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
