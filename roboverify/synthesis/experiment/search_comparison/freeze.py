"""Freeze matched inputs, budgets and runtime sources before comparison runs."""

from __future__ import annotations

import json
import time
from dataclasses import fields
from pathlib import Path

from .data import load_training
from .protocol import TrialConfig
from .run import source_hashes


def freeze_protocol(path, data_roots, *, seeds=tuple(range(10)), config=None):
    config = config or TrialConfig("stack", "beam")
    if set(data_roots) != {"stack", "unstack"}:
        raise ValueError("The paired protocol requires both Stack and Unstack inputs")
    if not seeds or len(set(seeds)) != len(seeds) or min(seeds) < 0:
        raise ValueError("Distinct nonnegative run seeds required")
    inputs = {}
    for task, root in data_roots.items():
        examples = load_training(root, task)
        if len(examples) != 15 or [e["seed"] for e in examples] != list(range(15)):
            raise ValueError(
                "The primary protocol uses exactly the declared 15 training seeds"
            )
        inputs[task] = {e["id"]: e["public_sha256"] for e in examples}
    record = {
        "schema": "rv-matched-search-protocol-v1",
        "frozen_unix": time.time(),
        "tasks": ["stack", "unstack"],
        "methods": ["beam", "matched_mcmc"],
        "seeds": list(seeds),
        "config": {
            f.name: getattr(config, f.name)
            for f in fields(config)
            if f.name not in {"task", "method", "seed"}
        },
        "training_roots": {k: str(Path(v).resolve()) for k, v in data_roots.items()},
        "training_inputs": inputs,
        "source_hashes": source_hashes(),
        "initial_program": [],
        "body_template_supplied": False,
        "paired_conditions": [
            "action corpus",
            "nominal contracts",
            "nominal rank",
            "terminal goal",
            "BMC screen",
            "physical acceptance",
            "CEM optimizer and candidate-keyed noise",
            "wall budget",
            "rollout cap",
            "transition cap",
            "initial empty program",
        ],
        "clock_scope": "CLI module startup, imports, input loading, proposal construction, search, BMC, simulation and CEM count toward the internal deadline. The external process cap includes interpreter/uv startup.",
        "external_cap_seconds": 55,
        "kill_grace_seconds": 2,
        "main_timing_schedule": "One fit invocation at a time; paired arm order alternates by run seed. Untimed tests and heldout execution are separate.",
        "numeric_rng": "SHA256 of run seed plus canonical full proposed action rows; independent of search evaluation order",
        "transition_cap_scope": "Structural search transition calls only; numeric nominal replays are counted separately and bounded by total wall time. Simulator rollout cap counts attempted calls, including interrupted ones.",
        "matched_proposals": "Uniform primitive family, then uniform member; insert/delete/replace/swap edits over the full shared alphabet. Beam stops expanding terminal-goal nodes; local MCMC mutations may leave or extend a terminal program.",
        "source_fingerprint_scope": "All synthesis Python sources except tests, fitting/evaluation/report entry points, static Fetch robot/shared XML and STL physics assets, and dependency manifests. External library installations are supplied by the recorded runtime environment.",
        "primary_success": "A complete shared parameter vector passes physical acceptance on every training example before deadline, then its frozen program passes fresh full-loop evaluation. No formal verification is claimed.",
        "failure_policy": "No nominal candidate, BMC failure, incomplete vector, error or outer process timeout remains a failed attempt. A fully evaluated candidate found before budget expiry is retained and labelled separately from termination.",
        "heldout": {
            "seeds": list(range(60, 75)),
            "block_counts": [3, 4, 5],
            "mode": "complete supplied loops, no intermediate resets or settling",
            "load_after_all_candidates_frozen": True,
        },
        "independence": "Beam structure is deterministic; run seeds vary numeric proposals and timing. Exact duplicate frozen bodies are evaluated once per state with aliases retained. Paired starts and duplicate outcomes are not independent trials.",
        "scope": "Controlled matched Metropolis comparison, not unchanged legacy MCMC. Existing CFG scaffolding and independent-controller simulation demonstrations are supplied.",
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
    return record
