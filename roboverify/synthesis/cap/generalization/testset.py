"""Versioned test-case sets for the block-count generalization checker.

A test case is one (task, layout, seed) scene in the relaxed environment
(`relax_boundaries_for_generalization_test`) that a physical-feasibility probe
found feasible: the reset scene and a teleported goal are built and stable,
satisfy their conditions, and every block centre is within the arm's measured
reach (Revision 9, Sections 3.3 and 3.5). Feasibility never depends on a
policy.

Two layers:
- the manifest (committed, `testsets/<name>.json`) lists every case with its
  run-config overrides, the probe evidence, and the SHA-256 of its settled
  initial state;
- the snapshots (`<out>/snapshots/<case_id>.npz`, under `runs/`, not
  committed) are rebuilt from the manifest, because the reset depends only on
  the case. A hash mismatch means the environment changed.

CLI (from roboverify/, simulator environment configured):
    python -m synthesis.cap.generalization.testset build NAME SCORED.json [...]
        --source LABEL=SCORED.json --tasks LABEL=task,task --out DIR
    python -m synthesis.cap.generalization.testset verify MANIFEST --out DIR
"""

import argparse
import hashlib
import io
import json
import multiprocessing as mp
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

import numpy as np

TESTSET_DIR = Path(__file__).with_name("testsets")
FORMAT = "generalization-testset-v1"
LAYOUT_FIELDS = (
    "num_blocks",
    "initial_tower_blocks",
    "grid_rows",
    "grid_cols",
    "bottom_row_blocks",
)
EVIDENCE_FIELDS = (
    "initial_drift",
    "goal_drift",
    "goal_settle_shift",
    "reach_margin_initial",
    "reach_margin_goal",
)


@dataclass
class GeneralizationCase:
    task: str
    layout: dict
    seed: int
    initial_state_sha256: str = ""
    evidence: dict = field(default_factory=dict, compare=False)

    @property
    def case_id(self):
        parts = [self.task, f"n{self.layout['num_blocks']}"]
        if "initial_tower_blocks" in self.layout:
            parts.append(f"t{self.layout['initial_tower_blocks']}")
        if "grid_rows" in self.layout:
            parts.append(f"{self.layout['grid_rows']}x{self.layout['grid_cols']}")
        if "bottom_row_blocks" in self.layout:
            parts.append(f"k{self.layout['bottom_row_blocks']}")
        parts.append(f"s{self.seed}")
        return "-".join(parts)

    def run_config_overrides(self):
        """RunConfig fields that build this scene (relaxed environment on)."""
        return dict(
            task=self.task,
            seed=self.seed,
            relax_boundaries_for_generalization_test=True,
            **self.layout,
        )


def run_config(case):
    from synthesis.cap.config import RunConfig

    return RunConfig(**case.run_config_overrides())


# ------------------------------------------------------------------ snapshots


def snapshot_bytes(snapshot):
    """Deterministic NPZ encoding of a cfg.reset.Snapshot."""
    meta = dict(
        bindings=snapshot.bindings,
        goal_layout=(
            None if snapshot.goal_layout is None else snapshot.goal_layout.as_dict()
        ),
        goal_bindings=snapshot.goal_bindings,
    )
    arrays = {"gt_state": np.asarray(snapshot.gt_state)}
    arrays.update(
        {f"array_{name}": np.asarray(v) for name, v in sorted(snapshot.arrays.items())}
    )
    arrays["meta"] = np.frombuffer(
        json.dumps(meta, sort_keys=True).encode(), dtype=np.uint8
    )
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    return buffer.getvalue()


def state_sha256(snapshot):
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(snapshot.gt_state, dtype=np.float64).tobytes())
    for name, value in sorted(snapshot.arrays.items()):
        digest.update(name.encode())
        digest.update(np.ascontiguousarray(value, dtype=np.float64).tobytes())
    meta = dict(
        bindings=snapshot.bindings,
        goal_layout=(
            None if snapshot.goal_layout is None else snapshot.goal_layout.as_dict()
        ),
        goal_bindings=snapshot.goal_bindings,
    )
    digest.update(json.dumps(meta, sort_keys=True).encode())
    return digest.hexdigest()


def load_snapshot(path):
    from synthesis.cfg.reset import Snapshot
    from synthesis.environment.goal_layout import GoalLayout

    with np.load(path) as data:
        meta = json.loads(bytes(data["meta"]).decode())
        arrays = {
            key[len("array_") :]: data[key].copy()
            for key in data.files
            if key.startswith("array_")
        }
        bindings = {k: int(v) for k, v in meta["bindings"].items()}
        return Snapshot(
            data["gt_state"].copy(),
            arrays,
            bindings,
            (
                None
                if meta["goal_layout"] is None
                else GoalLayout.from_dict(meta["goal_layout"])
            ),
            meta["goal_bindings"],
        )


def initial_snapshot(case):
    """The case's settled initial state through CaP's own reset, and the
    top block centre of every column."""
    from synthesis.cap.tasks import prepare_environment
    from synthesis.cfg.reset import capture, inner_env

    env = prepare_environment(run_config(case))
    try:
        inner = inner_env(env)
        obs = inner.flatten_observation(inner._get_obs())
        blocks = [obs[10 + 12 * i : 13 + 12 * i] for i in range(inner.num_blocks)]
        return capture(env), column_tops(blocks)
    finally:
        env.close()


def column_tops(blocks):
    tops = {}
    for p in blocks:
        key = (round(float(p[0]), 2), round(float(p[1]), 2))
        if key not in tops or p[2] > tops[key][2]:
            tops[key] = [float(v) for v in p]
    return sorted(tops.values())


TOPS_TOLERANCE = 1e-4


# ------------------------------------------------------------------- manifest


def cases_from_scored(rows, tasks=None):
    """Feasible probe rows (physical_summary output) as unhashed cases, with
    the probe's initial column tops for the rebuild check."""
    cases = []
    for row in rows:
        if not row.get("feasible") or (tasks and row["task"] not in tasks):
            continue
        layout = {k: row["layout"][k] for k in LAYOUT_FIELDS if k in row["layout"]}
        evidence = {
            k: (None if row.get(k) is None else round(float(row[k]), 6))
            for k in EVIDENCE_FIELDS
        }
        case = GeneralizationCase(row["task"], layout, int(row["seed"]), "", evidence)
        cases.append((case, sorted(row["initial_tops"])))
    return cases


def write_manifest(path, name, cases, *, environment, feasibility, git_commit):
    payload = dict(
        format=FORMAT,
        name=name,
        created=date.today().isoformat(),
        git_commit=git_commit,
        environment=environment,
        feasibility=feasibility,
        counts=_counts(cases),
        cases=[asdict(c) | {"case_id": c.case_id} for c in cases],
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # Readable header, one compact line per case (diffs stay per case).
    head = json.dumps({k: v for k, v in payload.items() if k != "cases"}, indent=1)
    lines = ",\n".join(
        "  " + json.dumps(case, separators=(",", ":")) for case in payload["cases"]
    )
    Path(path).write_text(f'{head[:-2]},\n "cases": [\n{lines}\n ]\n}}\n')
    return payload


def load_manifest(path):
    payload = json.loads(Path(path).read_text())
    if payload.get("format") != FORMAT:
        raise ValueError(f"Not a {FORMAT} manifest: {path}")
    cases = []
    for item in payload["cases"]:
        case = GeneralizationCase(
            item["task"],
            item["layout"],
            item["seed"],
            item["initial_state_sha256"],
            item.get("evidence", {}),
        )
        if case.case_id != item["case_id"]:
            raise ValueError(f"Case id mismatch: {item['case_id']}")
        cases.append(case)
    if len({c.case_id for c in cases}) != len(cases):
        raise ValueError("Duplicate test cases")
    return payload, cases


def _counts(cases):
    counts = {}
    for case in cases:
        key = f"{case.task}"
        counts.setdefault(key, {})
        n = str(case.layout["num_blocks"])
        counts[key][n] = counts[key].get(n, 0) + 1
    return counts


# --------------------------------------------------------------- build/verify


def _materialize(job):
    case, out, expected_tops = job
    snapshot, tops = initial_snapshot(case)
    if expected_tops is not None and (
        len(tops) != len(expected_tops)
        or np.abs(np.asarray(tops) - np.asarray(expected_tops)).max() > TOPS_TOLERANCE
    ):
        raise ValueError(f"{case.case_id}: rebuilt scene differs from the probe")
    if out is not None:
        path = Path(out) / "snapshots" / f"{case.case_id}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(snapshot_bytes(snapshot))
    return case.case_id, state_sha256(snapshot)


def materialize(cases, out, workers, expected=None):
    expected = expected or {}
    jobs = [(c, out, expected.get(c.case_id)) for c in cases]
    if workers <= 1:
        return dict(map(_materialize, jobs))
    with mp.get_context("spawn").Pool(workers, maxtasksperchild=20) as pool:
        return dict(pool.imap_unordered(_materialize, jobs, chunksize=4))


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def build(args):
    from synthesis.environment.relaxed_workspace import RELAXED_CONTACT_BUFFER

    cases, sources, expected = [], [], {}
    for spec in args.source:
        label, path = spec.split("=", 1)
        tasks = None
        for item in args.tasks or ():
            if item.startswith(f"{label}="):
                tasks = set(item.split("=", 1)[1].split(","))
        rows = json.loads(Path(path).read_text())
        for case, tops in cases_from_scored(rows, tasks):
            case.evidence["source"] = label
            cases.append(case)
            expected[case.case_id] = tops
        sources.append(
            dict(label=label, scored=str(path), tasks=sorted(tasks) if tasks else None)
        )
    order = {
        t: i
        for i, t in enumerate(
            ("stack", "unstack", "reverse", "restack", "grid", "pyramid")
        )
    }
    cases.sort(key=lambda c: (order[c.task], c.layout["num_blocks"], c.case_id))
    if len({c.case_id for c in cases}) != len(cases):
        raise ValueError("Sources overlap: duplicate test cases")
    hashes = materialize(cases, args.out, args.workers, expected)
    cases = [
        GeneralizationCase(c.task, c.layout, c.seed, hashes[c.case_id], c.evidence)
        for c in cases
    ]
    payload = write_manifest(
        args.manifest or TESTSET_DIR / f"{args.name}.json",
        args.name,
        cases,
        environment=dict(
            relax_boundaries_for_generalization_test=True,
            contact_buffer=list(RELAXED_CONTACT_BUFFER),
            initial_settling_steps=50,
        ),
        feasibility=dict(
            criterion=(
                "reset passes validate_start and drifts < 5 mm over 100 steps; a "
                "teleported goal drifts < 5 mm over 100 steps and satisfies the "
                "postcondition; every raised top centre is below the path-aware "
                "reach above its column; every table-level centre is reached"
            ),
            sources=sources,
        ),
        git_commit=_git_commit(),
    )
    print(
        f"{len(cases)} cases -> {args.manifest or TESTSET_DIR / (args.name + '.json')}"
    )
    print(json.dumps(payload["counts"]))


def verify(args):
    _, cases = load_manifest(args.manifest)
    hashes = materialize(cases, args.out, args.workers)
    bad = [c.case_id for c in cases if hashes[c.case_id] != c.initial_state_sha256]
    print(f"{len(cases) - len(bad)}/{len(cases)} cases reproduce their initial state")
    for case_id in bad[:20]:
        print("mismatch", case_id)
    return 1 if bad else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("name")
    b.add_argument("--source", action="append", required=True, help="LABEL=scored.json")
    b.add_argument("--tasks", action="append", help="LABEL=task,task (default: all)")
    b.add_argument("--manifest", help="default: testsets/NAME.json")
    b.add_argument("--out", help="snapshot directory (under runs/)")
    b.add_argument("--workers", type=int, default=8)
    v = sub.add_parser("verify")
    v.add_argument("manifest")
    v.add_argument("--out")
    v.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    return build(args) if args.command == "build" else verify(args)


if __name__ == "__main__":
    sys.exit(main())
