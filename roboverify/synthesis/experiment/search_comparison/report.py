"""Bounded summaries of frozen search trials and deduplicated loop evaluations."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from statistics import median


def read_json(path, *, limit=8_000_000):
    path = Path(path)
    if not path.exists():
        return None
    if path.stat().st_size > limit:
        raise ValueError(f"Summary exceeds {limit} bytes: {path}")
    with path.open() as stream:
        return json.load(stream)


def wilson(successes, total):
    if not total:
        return None
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    radius = (
        z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    )
    return [centre - radius, centre + radius]


def accepted_trial(record, process):
    """An interrupted/error process cannot promote even a retained diagnostic vector."""
    return bool(
        record.get("training_accepted")
        and record.get("status")
        in ("shortlist_exhausted", "budget_exhausted_with_candidate")
        and not record.get("error")
        and process is not None
        and process.get("returncode") == 0
        and process.get("wall_seconds", float("inf")) < 60
    )


def summarize(root):
    root = Path(root)
    protocol = read_json(root / "protocol.json")
    if protocol is None:
        raise ValueError("A frozen protocol.json is required")
    groups, trials = {}, []
    for task in protocol["tasks"]:
        for method in protocol["methods"]:
            records = []
            statuses = Counter()
            for seed in protocol["seeds"]:
                directory = root / "main" / task / method / f"seed{seed}"
                record = read_json(directory / "result.json")
                process = read_json(directory / "process.json")
                if record is None:
                    statuses["process_failed" if process else "pending_or_missing"] += 1
                    trials.append(
                        dict(
                            task=task,
                            method=method,
                            seed=seed,
                            selected=None,
                            training_accepted=False,
                            completed=False,
                        )
                    )
                    continue
                if record.get("pilot"):
                    raise ValueError("Pilot result found in main experiment directory")
                config = record["config"]
                if (config["task"], config["method"], config["seed"]) != (
                    task,
                    method,
                    seed,
                ):
                    raise ValueError("Trial identity differs from its protocol slot")
                selected = record.get("selected")
                item = dict(
                    task=task,
                    method=method,
                    seed=seed,
                    status=record["status"],
                    training_accepted=accepted_trial(record, process),
                    selected=None if selected is None else selected["program_sha256"],
                    completed=True,
                    seconds=record.get("elapsed_seconds"),
                    discovery_seconds=(
                        None if selected is None else selected.get("discovery_seconds")
                    ),
                    arc_rms_m=None if selected is None else selected.get("arc_rms_m"),
                    rollouts=record.get("evaluation_counts", {}).get(
                        "simulator_rollouts", 0
                    ),
                )
                trials.append(item)
                records.append(item)
                statuses[record["status"]] += 1
            accepted = sum(r["training_accepted"] for r in records)
            discoveries = [
                r["discovery_seconds"]
                for r in records
                if r["training_accepted"] and r["discovery_seconds"] is not None
            ]
            groups[(task, method)] = dict(
                task=task,
                method=method,
                planned_trials=len(protocol["seeds"]),
                reported_trials=len(records),
                statuses=dict(statuses),
                training_accepted_trials=accepted,
                search_success_wilson95=wilson(accepted, len(protocol["seeds"])),
                unique_selected_programs=len(
                    {r["selected"] for r in records if r["selected"]}
                ),
                median_discovery_seconds=median(discoveries) if discoveries else None,
                simulator_rollouts=sum(r["rollouts"] for r in records),
            )
    manifest = read_json(root / "frozen-candidates.json")
    actual_cases, unique_programs, missing_cases = 0, 0, 0
    block_counts = protocol["heldout"]["block_counts"]
    heldout_seeds = protocol["heldout"]["seeds"]
    for group in groups.values():
        group["loops"] = {
            str(n): dict(
                selected_successes=0,
                selected_cases=0,
                accepted_search_successes=0,
                end_to_end_denominator=len(protocol["seeds"]) * len(heldout_seeds),
            )
            for n in block_counts
        }
    for candidate in (manifest or {}).get("candidates", []):
        unique_programs += 1
        task, key = candidate["task"], candidate["program_sha256"]
        aliases = candidate["aliases"]
        if len({(a["method"], a["seed"]) for a in aliases}) != len(aliases):
            raise ValueError("Duplicate aliases would overcount evaluation results")
        for n in block_counts:
            for seed in heldout_seeds:
                case = read_json(
                    root / "heldout" / task / key / f"n{n}" / f"seed{seed}.json"
                )
                if case is None:
                    missing_cases += 1
                    continue
                if case["candidate_sha256"] != candidate["file_sha256"]:
                    raise ValueError(
                        "Held-out result has the wrong frozen candidate hash"
                    )
                actual_cases += 1
                for alias in aliases:
                    if (task, alias["method"]) not in groups:
                        continue  # Separate legacy reference is not a matched arm.
                    row = groups[(task, alias["method"])]["loops"][str(n)]
                    row["selected_cases"] += 1
                    row["selected_successes"] += int(case["full_program_success"])
                    row["accepted_search_successes"] += int(
                        case["full_program_success"] and alias["training_accepted"]
                    )
    return dict(
        schema="rv-matched-search-summary-v1",
        root=str(root.resolve()),
        groups=list(groups.values()),
        trials=trials,
        unique_frozen_programs=unique_programs,
        actual_loop_executions=actual_cases,
        missing_loop_cases=missing_cases,
        scope="Fixed training pool and supplied loop scaffold. Search seeds are not independent datasets; aliases share physical evaluations. No formal proof.",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--save-json")
    args = parser.parse_args(argv)
    summary = summarize(args.run)
    if args.save_json:
        Path(args.save_json).write_text(json.dumps(summary, indent=2) + "\n")
    print("task method accepted/planned unique rollouts discovery_s status")
    for row in summary["groups"]:
        elapsed = row["median_discovery_seconds"]
        print(
            row["task"],
            row["method"],
            f'{row["training_accepted_trials"]}/{row["planned_trials"]}',
            row["unique_selected_programs"],
            row["simulator_rollouts"],
            "-" if elapsed is None else round(elapsed, 2),
            row["statuses"],
        )
        for n, counts in row["loops"].items():
            if counts["selected_cases"]:
                print(
                    f'  {n} blocks: selected {counts["selected_successes"]}/{counts["selected_cases"]}; '
                    f'accepted-search + loop {counts["accepted_search_successes"]}/{counts["end_to_end_denominator"]}'
                )
    print(
        f'Unique frozen programs: {summary["unique_frozen_programs"]}; actual loop executions: '
        f'{summary["actual_loop_executions"]}; missing cases: {summary["missing_loop_cases"]}'
    )


if __name__ == "__main__":
    main()
