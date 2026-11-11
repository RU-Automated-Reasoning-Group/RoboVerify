"""Bounded complete-program selection among locally accepted block alternatives."""

import heapq

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


def select_validated_candidate(cfg, validate, *, limit=8):
    """Try complete executions; never promote a locally failing alternative.

    Nodes carry only alternatives already accepted by the same local endpoint
    and imitation gates. Combinations are visited in increasing sum of ordinal
    ranks, rather than adding distances measured in potentially different units.
    ``validate(cfg, index)`` returns the complete-program validation dictionary
    and owns the shared wall-clock budget. Only an accepted graph is committed;
    rejection or an exception restores the original executable.
    """
    if type(limit) is not int or limit < 1:
        raise ValueError("Candidate validation limit must be a positive integer")
    nodes, choices = [], []
    for node in _blocks(cfg):
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
            nodes.append(node)
            choices.append(options)
    originals = [node.region for node in nodes]
    initial = (0,) * len(nodes)
    queue, visited, history = [(0, initial)], {initial}, []
    first_report, selected_report, selected = None, None, None
    try:
        while queue and len(history) < limit:
            _, indices = heapq.heappop(queue)
            for node, options, index in zip(nodes, choices, indices):
                node.region = options[index]
            _sync_loops(cfg)
            report = validate(cfg, len(history))
            if first_report is None:
                first_report = report
            history.append(
                {
                    "index": len(history),
                    "ranks": list(indices),
                    "accepted": bool(report["accepted"]),
                    "passed": report.get("passed"),
                    "total": report.get("total"),
                    "evaluated": report.get("evaluated", report.get("total")),
                    "seconds": report.get("seconds"),
                    "program_fingerprint": report.get("program_fingerprint"),
                }
            )
            if report["accepted"]:
                selected, selected_report = indices, report
                break
            if report.get("selection_budget_exhausted"):
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
    finally:
        if selected is None:
            for node, region in zip(nodes, originals):
                node.region = region
            _sync_loops(cfg)
    result = dict(selected_report if selected_report is not None else first_report)
    result["candidate_selection"] = {
        "limit": limit,
        "attempted": len(history),
        "selected_ranks": None if selected is None else list(selected),
        "block_option_counts": [len(options) for options in choices],
        "trials": history,
    }
    return result
