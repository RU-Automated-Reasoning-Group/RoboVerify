"""Revision 9 driver: replay final programs on a generalization test-case set.

Every distinct program runs, unchanged, on every case of its task in the
test-case set (`relaxed_v2` by default). Programs come from two sources:
- CaP: the final programs of a CaP experiment (`generalization.programs`),
  replayed through the CaP runtime with a scripted response, recovery and
  demonstrations off, video off;
- RoboVerify: DSL program factories (`module:factory` or `path.py:factory`),
  executed with `cfg.collection.record_execution` from the case snapshot.

Both are judged after 50 unrecorded settling steps; CaP also applies its
final-release check. Budgets scale with the block count (Revision 9,
Section 5.4). The driver makes no model calls.

From roboverify/, simulator environment configured:
    python -m synthesis.cap.experiments.revision9_generalization run \\
        --out DIR --jobs DIR/jobs.json --workers 96
Each job (one program on a chunk of cases) runs in its own process and
appends one row per case to `DIR/results/<kind>/<task>/<program>.jsonl`;
finished cases are skipped on restart.
"""

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

TESTSET = "relaxed_v2"
SNAPSHOT_ROOT = Path("runs/cap-generalization/testsets")
TRAINED = {
    "stack": 4,
    "unstack": 4,
    "reverse": 4,
    "restack": 5,
    "grid": 6,
    "pyramid": 6,
}
BASE_MAX_STEPS = 10_000
BASE_MAX_SIM_STEPS = 10_000
CASE_TIMEOUT = 1_200
GOAL_TASKS = {"grid", "pyramid"}
SIMULATOR_ERROR = re.compile(
    r"MujocoException|contact buffer|constraint buffer|Pre-allocated|"
    r"non-finite|mj_stackAlloc|unstable|NaN",
    re.IGNORECASE,
)
# Large per-run files of successful CaP executions; failures keep everything.
PRUNE_ON_SUCCESS = (
    "stdout.log",
    "artifacts/trial_001/timeline.jsonl",
    "artifacts/trial_001/execution/rollout_001/trajectory.npz",
    "artifacts/trial_001/execution/rollout_001/steps.jsonl",
)


def budgets(task, num_blocks):
    """Scaled per-case budgets: ticks grow with N^3, simulation steps with N."""
    scale = max(1.0, num_blocks / TRAINED[task])
    return dict(
        max_steps=int(math.ceil(BASE_MAX_STEPS * scale**3)),
        max_sim_steps=int(math.ceil(BASE_MAX_SIM_STEPS * scale)),
        timeout=CASE_TIMEOUT,
    )


def classify(success, status, reason, budget_exhausted_by=None):
    """Revision 9 outcome (Section 4); every case is physically feasible."""
    if success:
        return "success"
    if SIMULATOR_ERROR.search(f"{status} {reason}"):
        return "simulator_failure"
    if budget_exhausted_by:
        return "budget_exhausted"
    return "policy_failure"


# ------------------------------------------------------------------ executors


def _cap_case(task, source, case, out):
    from synthesis.cap.configs import config_final_v2_wo_demo as cfg
    from synthesis.cap.llm import ScriptedModel
    from synthesis.cap.reports.partial_reward import attempt_reward
    from synthesis.cap.runner import run

    n = case.layout["num_blocks"]
    overrides = {k: v for k, v in case.run_config_overrides().items() if k != "task"}
    # R9_LOCAL_RUN_ROOT (a local disk such as tmpfs) keeps the per-step run
    # files off a network file system: only result.json is copied to `out`.
    local = os.environ.get("R9_LOCAL_RUN_ROOT")
    keep = Path(out)
    if local:
        out = Path(local) / Path(out).relative_to(Path(out).anchor)
    config, model, instruction = cfg.configuration(
        task,
        **overrides,
        recovery={"enabled": False},
        video_mode="off",
        **budgets(task, n),
        output_root=str(out),
    )
    start = time.time()
    result = run(
        config,
        model,
        instruction=instruction,
        backend=ScriptedModel([f"```python\n{source}\n```"]),
    )
    wall = time.time() - start
    success = result.get("success") is True
    reward = None
    try:
        reward = attempt_reward(
            task, result | {"artifacts_path": "artifacts/trial_001"}
        )
    except Exception as exc:  # noqa: BLE001 - reward is a secondary metric
        reward = None
        result.setdefault("reward_error", f"{type(exc).__name__}: {exc}")
    run_dir = result.get("run_dir")
    if (
        local
        and run_dir
        and Path(run_dir).resolve().is_relative_to(Path(local).resolve())
    ):
        keep.mkdir(parents=True, exist_ok=True)
        summary = keep / f"{case.case_id}.result.json"
        shutil.copyfile(Path(run_dir, "result.json"), summary)
        shutil.rmtree(run_dir)
        run_dir = str(summary)
    elif success and run_dir:
        for name in PRUNE_ON_SUCCESS:
            path = Path(run_dir, name)
            if path.is_file():
                path.unlink()
    return dict(
        wall=wall,
        success=success,
        status=result.get("status"),
        reason=result.get("reason"),
        budget_exhausted_by=result.get("budget_exhausted_by"),
        task_spec_postcondition=result.get("task_spec_postcondition"),
        task_spec_settled_postcondition=result.get("task_spec_settled_postcondition"),
        final_release_ok=(result.get("final_release") or {}).get("released"),
        dsl_compliance=result.get("dsl_compliance"),
        unconverged_primitives=result.get("unconverged_primitives"),
        sim_steps=result.get("simulation_steps"),
        ticks=result.get("execution_ticks"),
        partial_reward=reward,
        run_dir=run_dir,
    )


def _roboverify_env_factory(task, num_blocks, options):
    from synthesis.cfg.goal_collection import make_goal_env
    from synthesis.mcmc.synthesis import make_roboverify_env

    if task in GOAL_TASKS:
        return lambda: make_goal_env(task, options)
    return lambda: make_roboverify_env(task, num_blocks=num_blocks, **options)


def case_snapshot(case):
    """The case's settled initial state; its hash must match the manifest."""
    from synthesis.cap.generalization import testset as ts

    snapshot = ts.load_snapshot(
        SNAPSHOT_ROOT / TESTSET / "snapshots" / f"{case.case_id}.npz"
    )
    if ts.state_sha256(snapshot) != case.initial_state_sha256:
        raise ValueError(f"{case.case_id}: snapshot hash differs from the manifest")
    return snapshot


def _roboverify_case(task, source, case, out):
    from synthesis.cfg.program_source import load_program
    from synthesis.verification_lib.highlevel_verification_lib import HighLevelContext

    context = HighLevelContext(
        mode="declare", use_tbl=task in ("unstack", "reverse", "restack")
    )
    try:
        definition = load_program(
            source,
            context,
            case.layout["num_blocks"],
            allow_goal_program=task in GOAL_TASKS,
            allow_observation_references=True,
        )
    except ValueError as exc:
        # The program's own documented scope (e.g. supplied Reverse: 2-4 blocks).
        return dict(
            wall=0.0,
            success=False,
            status="unsupported_count",
            reason=f"{type(exc).__name__}: {exc}",
            sim_steps=0,
            partial_reward=0.0,
        )
    return run_definition(task, definition, case)


def run_definition(task, definition, case):
    """Execute a RoboVerify ProgramDefinition from the case snapshot."""
    from synthesis.cap.generalization import testset as ts
    from synthesis.cap.tasks import environment_options
    from synthesis.cfg.collection import record_execution, validate_trace

    n = case.layout["num_blocks"]
    snapshot = case_snapshot(case)
    options = environment_options(ts.run_config(case))
    start = time.time()
    trace = record_execution(
        definition,
        seed=case.seed,
        task=task,
        num_blocks=n,
        env_factory=_roboverify_env_factory(task, n, options),
        initial_snapshot=snapshot,
        max_loop_iterations=200,
        timeout_seconds=CASE_TIMEOUT,
        environment_config=options,
        post_settling_steps=50,
    )
    wall = time.time() - start
    valid = validate_trace(trace)
    settled = (trace.metadata.get("validation") or {}).get("settled") or {}
    unconverged = sum(
        1
        for e in trace.events
        if e.get("kind") == "instruction_end"
        and not (e.get("control") or {}).get("converged", True)
    )
    reward = None
    try:
        reward = _roboverify_reward(task, trace, settled)
    except Exception:  # noqa: BLE001 - reward is a secondary metric
        reward = None
    return dict(
        wall=wall,
        success=bool(valid and settled.get("success")),
        status=trace.metadata.get("status"),
        reason=trace.metadata.get("reason") or "; ".join(settled.get("issues") or []),
        task_spec_settled_postcondition=settled.get("success"),
        unconverged_primitives=unconverged,
        sim_steps=len(trace.actions[0]) if trace.actions else 0,
        partial_reward=reward,
    )


def _roboverify_reward(task, trace, settled):
    from synthesis.cap.reports.partial_reward import block_positions, rollout_reward

    if task in GOAL_TASKS:
        occupied = settled.get("goal_occupied")
        return (
            None
            if occupied is None
            else rollout_reward(task, None, None, goal_occupied=occupied)
        )
    n = trace.num_blocks
    positions = settled.get("positions")
    if positions is None:
        # No settled check (the run stopped early): score the last recorded state.
        positions = block_positions(trace.states[-1], n)
    entry = block_positions(trace.states[0], n)
    states = block_positions(list(trace.states), n) if task == "restack" else ()
    b0 = (trace.metadata.get("initial_bindings") or {}).get("b0", 0)
    return rollout_reward(task, entry, positions, b0, states)


EXECUTORS = {"cap": _cap_case, "roboverify": _roboverify_case}


def worker(job_path):
    """Run one job: a program on a chunk of cases, appending one row each."""
    from synthesis.cap.generalization import testset as ts

    job = json.loads(Path(job_path).read_text())
    _, cases = ts.load_manifest(ts.TESTSET_DIR / f"{job['testset']}.json")
    by_id = {c.case_id: c for c in cases}
    results = Path(job["results"])
    results.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if results.exists():
        done = {
            json.loads(line)["case"]
            for line in results.read_text().splitlines()
            if line
        }
    source = job["source"]
    if job["kind"] == "cap":
        source = Path(source).read_text()
    execute = EXECUTORS[job["kind"]]
    for cid in job["cases"]:
        if cid in done:
            continue
        case = by_id[cid]
        row = dict(
            kind=job["kind"],
            program=job["program"],
            task=job["task"],
            case=cid,
            n=case.layout["num_blocks"],
            layout=case.layout,
            seed=case.seed,
        )
        try:
            row |= execute(job["task"], source, case, Path(job["run_root"]))
        except Exception as exc:  # noqa: BLE001 - recorded per case
            row |= dict(
                success=False,
                status="driver_error",
                reason=f"{type(exc).__name__}: {exc}",
                traceback=traceback.format_exc(limit=8),
            )
        row["outcome"] = classify(
            row.get("success"),
            row.get("status"),
            row.get("reason"),
            row.get("budget_exhausted_by"),
        )
        if row.get("status") == "unsupported_count":
            row["outcome"] = "policy_failure"
        with results.open("a") as fh:
            fh.write(json.dumps(row, default=str) + "\n")


# ------------------------------------------------------------------ launcher


def pending(job):
    path = Path(job["results"])
    if not path.exists():
        return len(job["cases"])
    done = {json.loads(line)["case"] for line in path.read_text().splitlines() if line}
    return len(set(job["cases"]) - done)


def live_worker_jobs():
    """Job files of worker processes already running (from any launcher)."""
    out = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
    marker = f"{__spec__.name} worker "
    return {
        line.split(marker, 1)[1].strip() for line in out.splitlines() if marker in line
    }


def launch(args):
    import threading

    out = Path(args.out)
    jobs = json.loads(Path(args.jobs).read_text())
    if args.kind:
        jobs = [j for j in jobs if j["kind"] in args.kind]
    job_dir = out / "jobs"
    job_dir.mkdir(parents=True, exist_ok=True)
    log_dir = out / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    # Jobs already running under an earlier launcher are left to it.
    running = {Path(p).stem for p in live_worker_jobs()}
    todo = [j for j in jobs if j["id"] not in running and pending(j)]
    # Longest first: Grid and the largest counts dominate.
    todo.sort(key=lambda j: -j.get("cost", len(j["cases"])))
    gpus = args.gpus.split(",")
    print(
        f"{len(todo)} of {len(jobs)} jobs pending ({len(running)} running elsewhere),"
        f" {args.workers} workers",
        flush=True,
    )
    gate = threading.Lock()

    def one(index_job):
        index, job = index_job
        path = job_dir / f"{job['id']}.json"
        path.write_text(json.dumps(job))
        env = os.environ | {"CUDA_VISIBLE_DEVICES": gpus[index % len(gpus)]}
        with open(log_dir / f"{job['id']}.log", "a") as log:
            with gate:
                # The worker limit counts every live worker, including those of
                # an earlier launcher that are still finishing.
                while len(live_worker_jobs()) >= args.workers:
                    time.sleep(5)
                process = subprocess.Popen(
                    [sys.executable, "-m", __spec__.name, "worker", str(path)],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=env,
                )
            code = process.wait()
        return job["id"], code

    finished = 0
    with ThreadPoolExecutor(args.workers) as pool:
        for job_id, code in pool.map(one, enumerate(todo)):
            finished += 1
            if code:
                print(f"job {job_id} exited {code}", flush=True)
            if finished % 20 == 0 or finished == len(todo):
                print(
                    f"{time.strftime('%H:%M:%S')} {finished}/{len(todo)} jobs",
                    flush=True,
                )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--out", required=True)
    run_parser.add_argument("--jobs", required=True)
    run_parser.add_argument("--workers", type=int, default=48)
    run_parser.add_argument("--gpus", default="6,7")
    run_parser.add_argument("--kind", nargs="*")
    work = sub.add_parser("worker")
    work.add_argument("job")
    args = parser.parse_args(argv)
    if args.command == "worker":
        worker(args.job)
    else:
        launch(args)


if __name__ == "__main__":
    main()
