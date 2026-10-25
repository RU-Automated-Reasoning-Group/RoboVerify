"""Figures for the Revision 4.3 feedback ablation (Stage B).

Reads every branch of a Stage B batch (one recovery chain per seed x profile)
and draws paired effects against ``full``, recovery by round, recovery by
method and failure class, and token cost.
"""

import argparse
import json
import math
from pathlib import Path

from .revision43_feedback_experiment import CLASSES, PROFILES, _branches
from .revision43_prompt_report import wilson

METHODS = {
    "cap_google_baseline": "Baseline",
    "cap_method_1": "Method 1",
    "cap_method_2": "Method 2",
}
METHOD_COLOURS = {
    "cap_google_baseline": "#4c72b0",
    "cap_method_1": "#dd8452",
    "cap_method_2": "#55a868",
}
CLASS_NAMES = {
    "move_nonconvergence": "Move",
    "pick_nonconvergence": "Pick",
    "release_nonconvergence": "Release",
    "execution_budget": "Budget",
    "no_witness": "No witness",
    "goal_failure": "Goal",
    "other_runtime": "Other runtime",
}
OUTCOMES = {
    "recovered": "Recovered within 4 rounds (primary)",
    "round1": "Recovered in round 1 (secondary)",
}
MARGIN = 0.10


def branches(directory):
    """One row per finished branch: seed, profile, method, class and outcomes."""
    directory = Path(directory)
    plan = json.loads((directory / "comparison_plan.json").read_text())
    seeds = json.loads(Path(plan["seed_manifest"]).read_text())["seeds"]
    seed_class = {s["seed_id"]: s["class"] for s in seeds}
    rows = []
    for _, variant, run_dir in _branches(directory):
        path = run_dir / "artifacts/trial_001/recovery_result.json" if run_dir else None
        if path is None or not path.is_file():
            continue
        result = json.loads(path.read_text())
        recovered = bool(result.get("success"))
        rows.append(
            {
                "seed": variant["seed_id"],
                "profile": variant["profile"],
                "method": result["method"],
                "class": seed_class[variant["seed_id"]],
                "recovered": recovered,
                "round1": recovered and result.get("recovery_rounds") == 1,
                "rounds": result.get("recovery_rounds") if recovered else None,
                "tokens": result.get("known_tokens") or 0,
            }
        )
    return rows


def paired(rows, outcome, reference="full"):
    """Per profile: rate and paired difference from ``reference`` on shared seeds.

    The difference has Wald 95% and 90% intervals for paired proportions; the
    90% interval inside +-MARGIN is the TOST equivalence criterion.
    """
    value = {(r["seed"], r["profile"]): r[outcome] for r in rows}
    table = {}
    for profile in [p for p in PROFILES if any(r["profile"] == p for r in rows)]:
        seeds = [s for s, p in value if p == profile and (s, reference) in value]
        n = len(seeds)
        b = sum(value[(s, reference)] and not value[(s, profile)] for s in seeds)
        c = sum(value[(s, profile)] and not value[(s, reference)] for s in seeds)
        k = sum(value[(s, profile)] for s in seeds)
        diff = (c - b) / n
        se = math.sqrt(max((b + c) - (c - b) ** 2 / n, 0.0)) / n
        table[profile] = {
            "n": n,
            "k": k,
            "b": b,
            "c": c,
            "diff": diff,
            "ci95": (diff - 1.96 * se, diff + 1.96 * se),
            "ci90": (diff - 1.645 * se, diff + 1.645 * se),
        }
    return table


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")


def plot_effects(rows, path):
    """Forest plot of profile - full for both outcomes, with the TOST margin."""
    plt = _pyplot()
    profiles = [p for p in PROFILES if p != "full"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 6.2), sharey=True)
    for ax, (outcome, title) in zip(axes, OUTCOMES.items()):
        table = paired(rows, outcome)
        ax.axvspan(-MARGIN * 100, MARGIN * 100, color="#eef5ea", zorder=0)
        ax.axvline(0, color="black", lw=0.8, zorder=1)
        for y, profile in enumerate(profiles):
            t = table[profile]
            lo95, hi95 = (100 * v for v in t["ci95"])
            lo90, hi90 = (100 * v for v in t["ci90"])
            equivalent = -MARGIN * 100 < lo90 and hi90 < MARGIN * 100
            colour = "#2077b4" if equivalent else "#c44e52"
            ax.plot([lo95, hi95], [y, y], color=colour, lw=1.2, zorder=2)
            ax.plot([lo90, hi90], [y, y], color=colour, lw=4, alpha=0.6, zorder=2)
            ax.plot(100 * t["diff"], y, "o", color=colour, ms=5, zorder=3)
            ax.text(
                19.5,
                y,
                f"{t['b']}/{t['c']}",
                va="center",
                ha="right",
                fontsize=7,
                color="#555555",
            )
        ref = table["full"]
        ax.set_title(f"{title}\nfull: {ref['k']}/{ref['n']}", fontsize=9)
        ax.set_xlim(-20, 20)
        ax.set_xlabel("Profile − full (percentage points)")
        ax.grid(axis="x", alpha=0.3)
    axes[0].set_yticks(range(len(profiles)), profiles, fontfamily="monospace")
    axes[0].invert_yaxis()
    fig.suptitle(
        "Feedback ablation, Qwen3.8 27B: paired difference from full "
        "(dot; thin 95%, thick 90% Wald CI; numbers b/c)",
        fontsize=10,
    )
    fig.text(
        0.5,
        -0.01,
        "Green band: ±10-point equivalence margin. Blue: 90% CI inside the margin "
        "(equivalent by TOST); red: not shown equivalent. No difference is "
        "significant after Holm correction.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout()
    _save(fig, path)
    plt.close(fig)


def plot_rounds(rows, path):
    """Stacked bars per profile: recovered in round 1..4, or not recovered."""
    plt = _pyplot()
    colours = ["#1a5e9a", "#4a90c8", "#86b9e0", "#c3dcf0", "#d9d9d9"]
    labels = ["Round 1", "Round 2", "Round 3", "Round 4", "Not recovered"]
    fig, ax = plt.subplots(figsize=(8.5, 6))
    for y, profile in enumerate(PROFILES):
        own = [r for r in rows if r["profile"] == profile]
        counts = [sum(r["rounds"] == k for r in own) for k in (1, 2, 3, 4)]
        counts.append(len(own) - sum(counts))
        left = 0
        for k, count in enumerate(counts):
            share = count / len(own)
            ax.barh(
                y,
                share,
                left=left,
                color=colours[k],
                label=labels[k] if y == 0 else None,
            )
            if share >= 0.05:
                ax.text(
                    left + share / 2,
                    y,
                    str(count),
                    va="center",
                    ha="center",
                    fontsize=7,
                    color="white" if k < 2 else "black",
                )
            left += share
    ax.set_yticks(range(len(PROFILES)), PROFILES, fontfamily="monospace")
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of 210 seeds")
    ax.set_title(
        "Feedback ablation, Qwen3.8 27B: round of recovery per profile", fontsize=10
    )
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.09),
        ncol=5,
        fontsize=8,
        frameon=False,
    )
    fig.tight_layout()
    _save(fig, path)
    plt.close(fig)


def plot_groups(rows, path):
    """Heat maps of the primary outcome by profile x method and profile x class."""
    plt = _pyplot()
    import numpy as np

    classes = [c for c in CLASSES if any(r["class"] == c for r in rows)]
    groups = [
        ("method", list(METHODS), [METHODS[m] for m in METHODS]),
        ("class", classes, [CLASS_NAMES[c] for c in classes]),
    ]
    fig, axes = plt.subplots(
        1, 2, figsize=(12, 6.4), gridspec_kw={"width_ratios": [3, len(classes)]}
    )
    for ax, (field, keys, names) in zip(axes, groups):
        grid = np.zeros((len(PROFILES), len(keys)))
        for i, profile in enumerate(PROFILES):
            for j, key in enumerate(keys):
                own = [r for r in rows if r["profile"] == profile and r[field] == key]
                k = sum(r["recovered"] for r in own)
                grid[i, j] = k / len(own)
                ax.text(
                    j,
                    i,
                    f"{k}/{len(own)}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if grid[i, j] < 0.8 else "black",
                )
        image = ax.imshow(grid, cmap="viridis", vmin=0.6, vmax=1.0, aspect="auto")
        ax.set_xticks(range(len(keys)), names, fontsize=8)
        ax.set_yticks(
            range(len(PROFILES)), PROFILES, fontfamily="monospace", fontsize=8
        )
        ax.set_title(f"By {field}", fontsize=9)
    axes[1].set_yticklabels([])
    fig.colorbar(image, ax=axes, shrink=0.8, label="Recovered within 4 rounds")
    fig.suptitle(
        "Feedback ablation, Qwen3.8 27B: recovery by method and failure class",
        fontsize=10,
    )
    _save(fig, path)
    plt.close(fig)


def _cost_panel(ax, rows, title):
    """Mean tokens per chain against the primary outcome, one point per profile."""
    points, uppers = [], []
    for profile in PROFILES:
        own = [r for r in rows if r["profile"] == profile]
        k = sum(r["recovered"] for r in own)
        rate = k / len(own)
        lo, hi = wilson(k, len(own))
        tokens = sum(r["tokens"] for r in own) / len(own) / 1000
        anchor = profile in ("full", "minimal", "legacy_v1")
        colour = (
            "#c44e52"
            if profile == "no_current_source"
            else ("#de8f05" if anchor else "#2077b4")
        )
        ax.errorbar(
            tokens,
            rate,
            yerr=[[rate - lo], [hi - rate]],
            fmt="o",
            color=colour,
            ms=5,
            elinewidth=0.8,
            capsize=2,
        )
        points.append((rate, tokens, profile, colour))
        uppers.append(hi)
    # Labels in a column right of the points, spread vertically, with leader lines.
    column = max(t for _, t, _, _ in points) + 2.5
    gap = 0.0075 * (max(len(rows) // len(PROFILES), 1) / 210) ** -0.5
    placed = []
    for rate, tokens, profile, colour in sorted(points):
        y = max(rate, placed[-1] + gap) if placed else rate
        placed.append(y)
        ax.annotate(
            profile,
            (tokens, rate),
            xytext=(column, y),
            textcoords="data",
            va="center",
            fontsize=7,
            fontfamily="monospace",
            color=colour,
            arrowprops={"arrowstyle": "-", "color": "#bbbbbb", "lw": 0.6},
        )
    ax.set_xlim(right=column + 9)
    ax.set_ylim(top=max(max(placed), max(uppers)) + gap)
    ax.set_xlabel("Mean tokens per chain (thousands; includes the shared round 0)")
    ax.set_ylabel("Recovered within 4 rounds (95% Wilson interval)")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)


def plot_cost(rows, path):
    """Mean tokens per chain against the primary outcome, per profile."""
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    _cost_panel(
        ax,
        rows,
        "Feedback ablation, Qwen3.8 27B: recovery against token cost (210 seeds each)",
    )
    fig.tight_layout()
    _save(fig, path)
    plt.close(fig)


def _by_method(rows):
    return [(m, [r for r in rows if r["method"] == m]) for m in METHODS]


def plot_effects_by_method(rows, path):
    """Forest plot of profile - full per method, dodged within each profile row."""
    plt = _pyplot()
    from matplotlib.lines import Line2D

    profiles = [p for p in PROFILES if p != "full"]
    groups = _by_method(rows)
    offsets = dict(zip(METHODS, (-0.25, 0.0, 0.25)))
    fig, axes = plt.subplots(1, 2, figsize=(12, 8.5), sharey=True)
    limit = 0.0
    for ax, (outcome, title) in zip(axes, OUTCOMES.items()):
        ax.axvspan(-MARGIN * 100, MARGIN * 100, color="#eef5ea", zorder=0)
        ax.axvline(0, color="black", lw=0.8, zorder=1)
        for y in range(0, len(profiles), 2):
            ax.axhspan(y - 0.5, y + 0.5, color="#f4f4f4", zorder=0)
        references = []
        for method, own in groups:
            table = paired(own, outcome)
            colour = METHOD_COLOURS[method]
            for y, profile in enumerate(profiles):
                t = table[profile]
                yy = y + offsets[method]
                lo95, hi95 = (100 * v for v in t["ci95"])
                lo90, hi90 = (100 * v for v in t["ci90"])
                limit = max(limit, abs(lo95), abs(hi95))
                ax.plot([lo95, hi95], [yy, yy], color=colour, lw=1.0, zorder=2)
                ax.plot(
                    [lo90, hi90], [yy, yy], color=colour, lw=3.5, alpha=0.55, zorder=2
                )
                ax.plot(100 * t["diff"], yy, "o", color=colour, ms=4, zorder=3)
            ref = table["full"]
            references.append(f"{METHODS[method]} {ref['k']}/{ref['n']}")
        ax.set_title(f"{title}\nfull: " + ", ".join(references), fontsize=9)
        ax.set_xlabel("Profile − full (percentage points)")
        ax.grid(axis="x", alpha=0.3)
    bound = min(50, 5 * math.ceil(limit / 5 + 0.2))
    for ax in axes:
        ax.set_xlim(-bound, bound)
    axes[0].set_yticks(range(len(profiles)), profiles, fontfamily="monospace")
    axes[0].set_ylim(len(profiles) - 0.5, -0.5)
    handles = [
        Line2D([], [], color=METHOD_COLOURS[m], marker="o", lw=2, label=METHODS[m])
        for m in METHODS
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(
        "Feedback ablation by method (70 seeds each): paired difference from full "
        "(dot; thin 95%, thick 90% Wald CI)",
        fontsize=10,
    )
    fig.text(
        0.5,
        -0.05,
        "Green band: ±10-point equivalence margin of the pooled analysis. Per-method "
        "intervals are about √3 times wider than pooled ones; descriptive only.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _save(fig, path)
    plt.close(fig)


def plot_rounds_by_method(rows, path):
    """Stacked round-of-recovery bars per profile, one panel per method."""
    plt = _pyplot()
    colours = ["#1a5e9a", "#4a90c8", "#86b9e0", "#c3dcf0", "#d9d9d9"]
    labels = ["Round 1", "Round 2", "Round 3", "Round 4", "Not recovered"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=True)
    for ax, (method, own_rows) in zip(axes, _by_method(rows)):
        for y, profile in enumerate(PROFILES):
            own = [r for r in own_rows if r["profile"] == profile]
            counts = [sum(r["rounds"] == k for r in own) for k in (1, 2, 3, 4)]
            counts.append(len(own) - sum(counts))
            left = 0
            for k, count in enumerate(counts):
                share = count / len(own)
                ax.barh(
                    y,
                    share,
                    left=left,
                    color=colours[k],
                    edgecolor="white",
                    linewidth=0.4,
                    label=labels[k] if y == 0 else None,
                )
                if share >= 0.07:
                    ax.text(
                        left + share / 2,
                        y,
                        str(count),
                        va="center",
                        ha="center",
                        fontsize=6.5,
                        color="white" if k < 2 else "black",
                    )
                left += share
        n = len({r["seed"] for r in own_rows})
        ax.set_title(
            f"{METHODS[method]} ({n} seeds)",
            fontsize=10,
            color=METHOD_COLOURS[method],
            fontweight="bold",
        )
        ax.set_xlim(0, 1)
        ax.set_xlabel(f"Share of {n} seeds")
    axes[0].set_yticks(range(len(PROFILES)), PROFILES, fontfamily="monospace")
    axes[0].invert_yaxis()
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        names,
        loc="lower center",
        ncol=5,
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.suptitle(
        "Feedback ablation by method, Qwen3.8 27B: round of recovery per profile",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _save(fig, path)
    plt.close(fig)


def plot_groups_by_method(rows, path):
    """Heat maps of the primary outcome by profile x failure class, one per method."""
    plt = _pyplot()
    import numpy as np

    classes = [c for c in CLASSES if any(r["class"] == c for r in rows)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 6.4), sharey=True)
    for ax, (method, own_rows) in zip(axes, _by_method(rows)):
        grid = np.full((len(PROFILES), len(classes)), np.nan)
        for i, profile in enumerate(PROFILES):
            for j, cls in enumerate(classes):
                own = [
                    r for r in own_rows if r["profile"] == profile and r["class"] == cls
                ]
                if not own:
                    ax.text(
                        j, i, "—", ha="center", va="center", fontsize=7, color="#888888"
                    )
                    continue
                k = sum(r["recovered"] for r in own)
                grid[i, j] = k / len(own)
                ax.text(
                    j,
                    i,
                    f"{k}/{len(own)}",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="white" if grid[i, j] < 0.8 else "black",
                )
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad("#eeeeee")
        image = ax.imshow(
            np.ma.masked_invalid(grid), cmap=cmap, vmin=0.6, vmax=1.0, aspect="auto"
        )
        ax.set_xticks(
            range(len(classes)),
            [CLASS_NAMES[c] for c in classes],
            fontsize=7.5,
            rotation=30,
            ha="right",
        )
        ax.set_title(
            METHODS[method],
            fontsize=10,
            color=METHOD_COLOURS[method],
            fontweight="bold",
        )
    axes[0].set_yticks(
        range(len(PROFILES)), PROFILES, fontfamily="monospace", fontsize=8
    )
    fig.colorbar(image, ax=axes, shrink=0.8, label="Recovered within 4 rounds")
    fig.suptitle(
        "Feedback ablation by method: recovery by failure class (— no seeds of "
        "that class)",
        fontsize=10,
    )
    _save(fig, path)
    plt.close(fig)


def plot_cost_by_method(rows, path):
    """Recovery and mean tokens per profile, the three methods dodged per row."""
    plt = _pyplot()
    from matplotlib.lines import Line2D

    offsets = dict(zip(METHODS, (-0.25, 0.0, 0.25)))
    fig, (rate_ax, cost_ax) = plt.subplots(
        1, 2, figsize=(12, 8.5), sharey=True, gridspec_kw={"width_ratios": [3, 2]}
    )
    for ax in (rate_ax, cost_ax):
        for y in range(0, len(PROFILES), 2):
            ax.axhspan(y - 0.5, y + 0.5, color="#f4f4f4", zorder=0)
    for method, own_rows in _by_method(rows):
        colour = METHOD_COLOURS[method]
        for y, profile in enumerate(PROFILES):
            own = [r for r in own_rows if r["profile"] == profile]
            k = sum(r["recovered"] for r in own)
            rate = k / len(own)
            lo, hi = wilson(k, len(own))
            yy = y + offsets[method]
            rate_ax.plot([100 * lo, 100 * hi], [yy, yy], color=colour, lw=1.2)
            rate_ax.plot(100 * rate, yy, "o", color=colour, ms=4.5)
            tokens = sum(r["tokens"] for r in own) / len(own) / 1000
            cost_ax.plot(tokens, yy, "o", color=colour, ms=4.5)
            cost_ax.plot([0, tokens], [yy, yy], color=colour, lw=0.8, alpha=0.35)
    rate_ax.set_xlabel("Recovered within 4 rounds, % (95% Wilson interval; 70 seeds)")
    rate_ax.set_xlim(55, 100)
    cost_ax.set_xlabel("Mean tokens per chain (thousands; includes round 0)")
    cost_ax.set_xlim(left=0)
    for ax in (rate_ax, cost_ax):
        ax.grid(axis="x", alpha=0.3)
    rate_ax.set_yticks(range(len(PROFILES)), PROFILES, fontfamily="monospace")
    rate_ax.set_ylim(len(PROFILES) - 0.5, -0.5)
    handles = [
        Line2D([], [], color=METHOD_COLOURS[m], marker="o", lw=1.5, label=METHODS[m])
        for m in METHODS
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.suptitle(
        "Feedback ablation by method, Qwen3.8 27B: recovery and token cost per profile",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _save(fig, path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="figures are written here"
    )
    args = parser.parse_args()
    rows = branches(args.directory)
    stem = "revision43_feedback"
    plot_effects(rows, args.output_dir / f"{stem}_effects.png")
    plot_rounds(rows, args.output_dir / f"{stem}_rounds.png")
    plot_groups(rows, args.output_dir / f"{stem}_groups.png")
    plot_cost(rows, args.output_dir / f"{stem}_cost.png")
    plot_effects_by_method(rows, args.output_dir / f"{stem}_effects_by_method.png")
    plot_rounds_by_method(rows, args.output_dir / f"{stem}_rounds_by_method.png")
    plot_groups_by_method(rows, args.output_dir / f"{stem}_groups_by_method.png")
    plot_cost_by_method(rows, args.output_dir / f"{stem}_cost_by_method.png")
    report = {
        outcome: {
            p: {k: v for k, v in t.items()} for p, t in paired(rows, outcome).items()
        }
        for outcome in OUTCOMES
    }
    print(json.dumps({"branches": len(rows)} | report, indent=1))


if __name__ == "__main__":
    main()
