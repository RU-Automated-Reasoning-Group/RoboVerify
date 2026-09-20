"""Lossless NPZ demonstration archives; no pickle or simulator objects on disk."""

import json

import numpy as np

from synthesis.cfg.demos import DemoTrace
from synthesis.cfg.reset import Snapshot


def save_traces(path, traces):
    arrays, metadata = {}, []
    for i, trace in enumerate(traces):
        key = f"d{i}"
        arrays[key + "_states"] = np.asarray(trace.states)
        row = {
            "seed": trace.seed,
            "task": trace.task,
            "num_blocks": trace.num_blocks,
            "snapshots": bool(trace.snapshots),
        }
        if trace.snapshots:
            arrays[key + "_gt"] = np.stack([s.gt_state for s in trace.snapshots])
            row["bindings"] = [s.bindings for s in trace.snapshots]
            row["arrays"] = list(trace.snapshots[0].arrays)
            for name in row["arrays"]:
                arrays[key + "_" + name] = np.stack(
                    [s.arrays[name] for s in trace.snapshots]
                )
            actions, indices = trace.actions
            arrays[key + "_actions"] = np.asarray(actions)
            arrays[key + "_indices"] = np.asarray(indices)
        metadata.append(row)
    arrays["metadata"] = np.asarray(json.dumps({"version": 1, "traces": metadata}))
    np.savez_compressed(path, **arrays)


def load_traces(path):
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["metadata"]))
        if meta["version"] != 1:
            raise ValueError("Unsupported demo archive")
        traces = []
        for i, row in enumerate(meta["traces"]):
            key = f"d{i}"
            states = tuple(data[key + "_states"].copy())
            snapshots, actions = (), ()
            if row["snapshots"]:
                snapshots = tuple(
                    Snapshot(
                        gt.copy(),
                        {
                            name: data[key + "_" + name][t].copy()
                            for name in row["arrays"]
                        },
                        row["bindings"][t],
                    )
                    for t, gt in enumerate(data[key + "_gt"])
                )
                actions = (
                    tuple(data[key + "_actions"].copy()),
                    tuple(data[key + "_indices"].tolist()),
                )
            traces.append(
                DemoTrace(
                    states,
                    snapshots,
                    actions,
                    row["seed"],
                    row["task"],
                    row["num_blocks"],
                )
            )
        return traces
