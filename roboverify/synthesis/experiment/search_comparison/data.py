"""Strict public observation inputs for paired search experiments.

The fitting loader can read only the declared training seeds. It never returns
teacher commands, targets, controller labels, or collection diagnostics.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def positions(observation, count):
    observation = np.asarray(observation)
    return {i: observation[10 + 12 * i : 13 + 12 * i].copy() for i in range(count)}


def load_training(root, task, *, seeds=tuple(range(15))):
    """Read public training NPZ files, rejecting incomplete or mixed inputs."""
    from synthesis.cfg.reset import Snapshot

    root = Path(root).resolve()
    selected = set(int(s) for s in seeds)
    if not selected or min(selected) < 0:
        raise ValueError("Nonempty nonnegative training seeds required")
    manifest = json.loads((root / "manifest.json").read_text())
    examples = []
    for row in manifest["attempts"]:
        if (
            row["task"] != task
            or row["split"] != "train"
            or row["seed"] not in selected
        ):
            continue
        if not row.get("success") or not row.get("public_file"):
            raise ValueError(f"Declared training example is unavailable: {row['id']}")
        path = (root / row["public_file"]).resolve()
        if path.parent != root / "public" or path.suffix != ".npz":
            raise ValueError(
                "Training inputs must be public NPZ files under the data root"
            )
        with np.load(path, allow_pickle=False) as archive:
            meta = json.loads(str(archive["metadata"]))
            obs = archive["observations"].copy()
            bindings = dict(meta["bindings"])
            snapshot = Snapshot(
                archive["initial_gt_state"].copy(),
                {
                    name: archive["initial_" + name].copy()
                    for name in meta["snapshot_arrays"]
                },
                bindings,
            )
        n = int(row["num_blocks"])
        if (
            obs.ndim != 2
            or len(obs) < 52
            or obs.shape[1] < 10 + 12 * n
            or not np.isfinite(obs).all()
        ):
            raise ValueError("Invalid recorded observation matrix")
        # This archive format explicitly records 50 final settling control steps.
        end = int(
            meta.get(
                "execution_end_index", row.get("execution_end_index", len(obs) - 51)
            )
        )
        if end != len(obs) - 51:
            raise ValueError("Expected the declared 50-step settling tail")
        example = {
            "id": row["id"],
            "task": task,
            "seed": int(row["seed"]),
            "num_blocks": n,
            "transfer_index": int(row["transfer_index"]),
            "split": "train",
            "observations": obs,
            "states": obs,
            "bindings": bindings,
            "initial_snapshot": snapshot,
            "initial_positions": positions(obs[0], n),
            "final_positions": positions(obs[-1], n),
            "execution_end_index": end,
            "final_settle_steps": 50,
            "public_file": str(path),
            "public_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        examples.append(example)
    examples.sort(key=lambda e: e["seed"])
    if [e["seed"] for e in examples] != sorted(selected):
        raise ValueError("Training seeds must occur exactly once")
    if len({e["num_blocks"] for e in examples}) != 1:
        raise ValueError("A trial must have a common physical block count")
    return examples


def make_problem(examples, *, config=None, deadline=None):
    """Task-specific benchmark goals around the generic maintained backend."""
    from synthesis.search.contract import ContractProblem, SearchExample

    shared = set.intersection(*(set(e["bindings"]) for e in examples))
    roles = sorted(
        r for r in shared if all(type(e["bindings"][r]) is int for e in examples)
    )
    adapted = []
    sources = np.asarray([e["bindings"]["b_prime"] for e in examples], int)
    targets = np.asarray(
        [e["bindings"].get("b", e["bindings"]["b0"]) for e in examples], int
    )
    table_centres = np.asarray(
        [min(p[2] for p in e["initial_positions"].values()) for e in examples]
    )
    for e in examples:
        obs = e["observations"][: e["execution_end_index"] + 1]
        p = np.stack(
            [obs[:, 10 + 12 * i : 13 + 12 * i] for i in range(e["num_blocks"])], axis=1
        )
        adapted.append(
            SearchExample(
                positions=p[0],
                gripper=obs[0, :3],
                bindings=e["bindings"],
                gripper_path=obs[:, :3],
                positions_path=p,
                initial_holding=None,
            )
        )
    task = examples[0]["task"]
    if task not in ("stack", "unstack") or any(e["task"] != task for e in examples):
        raise ValueError(
            "This benchmark adapter supports homogeneous Stack/Unstack examples"
        )
    ix = np.arange(len(examples))

    def batch_goal(state):
        p = state.positions
        source = p[ix, sources]
        released = np.asarray(state.holding) == -1
        if task == "stack":
            difference = source - p[ix, targets]
            return (
                released
                & (np.max(np.abs(difference[:, :2]), axis=1) < 0.025)
                & (np.abs(difference[:, 2] - 0.05) < 0.001)
            )
        separated = np.max(np.abs(p[:, :, :2] - source[:, None, :2]), axis=2) >= 0.1
        separated[ix, sources] = True
        return (
            released
            & (np.abs(source[:, 2] - table_centres) <= 0.001)
            & np.all(separated, axis=1)
        )

    return ContractProblem(
        adapted,
        goal=lambda state, i: bool(batch_goal(state)[i]),
        batch_goal=batch_goal,
        allowed_operands=roles,
        config=config,
        deadline=deadline,
    )
