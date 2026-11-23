"""Bounded complete-program selection among locally accepted block alternatives."""

import heapq
import math
from copy import deepcopy

from synthesis.api.program import Program
from synthesis.cfg.program_source import program_fingerprint
from synthesis.cfg.region import BlockRegion, LoopRegion


def _blocks(cfg):
    for name in cfg.order:
        node = cfg.nodes[name]
        if isinstance(node.region, LoopRegion):
            yield from _blocks(node.region.body_cfg)
        elif isinstance(node.region, BlockRegion):
            yield node


def _sync_loops(cfg):
    for node in cfg.nodes.values():
        if isinstance(node.region, LoopRegion):
            body = node.region.body_cfg
            _sync_loops(body)
            node.region.body = tuple(body.nodes[name].region for name in body.order)


def _complete_frame_score(report):
    """Only a complete, accepted batch supplies a comparable frame score."""
    rows = report.get("demos")
    total = report.get("total")
    if (
        report.get("accepted") is not True
        or ("complete" in report and report["complete"] is not True)
        or report.get("selection_budget_exhausted")
        or report.get("budget_exhausted")
        or type(total) is not int
        or total < 1
        or type(report.get("passed")) is not int
        or type(report.get("evaluated")) is not int
        or report.get("passed") != total
        or report.get("evaluated") != total
        or report.get("unevaluated")
        or not isinstance(rows, (tuple, list))
        or len(rows) != total
    ):
        return None
    values = []
    for row in rows:
        if not isinstance(row, dict) or row.get("accepted") is not True:
            return None
        value = row.get("max_nonmover_drift_m")
        try:
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                return None
        except (TypeError, ValueError):
            return None
        values.append(float(value))
    return max(values)


def select_validated_candidate(cfg, validate, *, limit=8, policy="first"):
    """Try complete executions; never promote a locally failing alternative.

    Nodes carry only alternatives already accepted by the same local endpoint
    and imitation gates. Combinations are visited in increasing sum of ordinal
    ranks, rather than adding distances measured in potentially different units.
    ``validate(cfg, index)`` returns the complete-program validation dictionary
    and owns the shared wall-clock/action budgets. ``first`` stops at the first
    accepted candidate. ``frame-margin`` examines the bounded pool and minimizes
    the worst observed nonmover drift among complete accepted batches; it adds
    no new acceptance threshold. Exact ties retain the earlier candidate.

    Each callback receives fresh block regions, including singleton blocks that
    numeric fitting may change. A complete incumbent survives later rejection or
    reported budget exhaustion. External exceptions propagate and restore the
    entry executable. No incomplete report is promoted by frame ranking.
    """
    if type(limit) is not int or limit < 1:
        raise ValueError("Candidate validation limit must be a positive integer")
    if policy not in ("first", "frame-margin"):
        raise ValueError("Candidate selection policy must be first or frame-margin")
    all_nodes = tuple(_blocks(cfg))
    originals = tuple(node.region for node in all_nodes)
    nodes, choices = [], []
    for position, node in enumerate(all_nodes):
        options, seen = [], set()
        for region in (node.region, *getattr(node, "_contract_alternatives", ())):
            # The surrounding graph was scoped for these exports. A different
            # binding interface requires CFG refinement, not body substitution.
            if region.bindings != node.region.bindings:
                continue
            fingerprint = program_fingerprint(
                Program(len(region.physical), list(region.physical))
            )
            if fingerprint not in seen:
                seen.add(fingerprint)
                options.append(region)
        if len(options) > 1:
            nodes.append(position)
            choices.append(options)

    def install(regions):
        for node, region in zip(all_nodes, regions):
            node.region = region
        _sync_loops(cfg)

    initial = (0,) * len(nodes)
    queue, visited, history = [(0, initial)], {initial}, []
    first_report, selected_report, selected = None, None, None
    selected_regions, selected_score = None, None
    completed, stop_reason = False, "space_exhausted"
    try:
        while queue and len(history) < limit:
            _, indices = heapq.heappop(queue)
            regions = list(originals)
            for position, options, index in zip(nodes, choices, indices):
                regions[position] = options[index]
            install(deepcopy(regions))
            report = validate(cfg, len(history))
            if first_report is None:
                first_report = deepcopy(report)
            score = _complete_frame_score(report)
            exhausted = bool(
                report.get("selection_budget_exhausted")
                or report.get("budget_exhausted")
            )
            accepted = bool(report["accepted"]) and not exhausted
            if policy == "frame-margin":
                accepted = accepted and score is not None
            history.append(
                {
                    "index": len(history),
                    "ranks": list(indices),
                    "accepted": accepted,
                    "reported_accepted": bool(report["accepted"]),
                    "max_nonmover_drift_m": score,
                    "selection_rejection": (
                        "incomplete_or_missing_frame_evidence"
                        if policy == "frame-margin"
                        and report["accepted"]
                        and score is None
                        else None
                    ),
                    "passed": report.get("passed"),
                    "total": report.get("total"),
                    "evaluated": report.get("evaluated", report.get("total")),
                    "seconds": report.get("seconds"),
                    "program_fingerprint": report.get("program_fingerprint"),
                }
            )
            if accepted and (
                selected is None or policy == "first" or score < selected_score
            ):
                selected, selected_report = indices, deepcopy(report)
                selected_regions = deepcopy([node.region for node in all_nodes])
                selected_score = score
            if accepted and policy == "first":
                stop_reason = "first_accepted"
                break
            if exhausted:
                stop_reason = "budget_exhausted"
                break
            for dimension, options in enumerate(choices):
                if indices[dimension] + 1 >= len(options):
                    continue
                successor = list(indices)
                successor[dimension] += 1
                successor = tuple(successor)
                if successor not in visited:
                    visited.add(successor)
                    heapq.heappush(queue, (sum(successor), successor))
        else:
            stop_reason = "candidate_limit" if queue else "space_exhausted"
        completed = True
    finally:
        install(selected_regions if completed and selected is not None else originals)
    result = dict(selected_report if selected_report is not None else first_report)
    if selected is None:
        result["accepted"] = False
    result["candidate_selection"] = {
        "policy": policy,
        "limit": limit,
        "attempted": len(history),
        "selected_ranks": None if selected is None else list(selected),
        "selected_max_nonmover_drift_m": selected_score,
        "stop_reason": stop_reason,
        "block_option_counts": [len(options) for options in choices],
        "trials": history,
    }
    return result
