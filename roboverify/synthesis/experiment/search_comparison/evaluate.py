"""Evaluate frozen learned bodies in supplied loops on fresh simulation starts.

This module performs no fitting or candidate selection. Loop guards, initialization
and pointer assignments come from the task scaffold; only physical bodies change.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from copy import deepcopy
from pathlib import Path

import numpy as np

from synthesis.api.instructions import Assign, While
from synthesis.cfg.collection import execution_deadline
from synthesis.cfg.reset import capture, collect_recording, inner_env, restore
from synthesis.cfg.tasks import task_spec
from synthesis.predicates.scene import Scene, evaluate, scene_from_obs
from synthesis.util.on import on, scattered_implementation

from .data import positions
from .safety import ContactObserver, decode, settle


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot_digest(snapshot):
    """Hash every restored simulator array and binding, not just qpos/qvel."""
    value = hashlib.sha256()
    for name, array in sorted(
        {"gt_state": snapshot.gt_state, **snapshot.arrays}.items()
    ):
        array = np.asarray(array)
        value.update(json.dumps([name, array.shape, array.dtype.str]).encode())
        value.update(array.tobytes())
    value.update(json.dumps(snapshot.bindings, sort_keys=True).encode())
    return value.hexdigest()


def build_loop_program(task, rows, num_blocks):
    from synthesis.verification_lib.highlevel_verification_lib import HighLevelContext

    if task == "stack":
        from synthesis.examples.stack import build_program
    elif task == "unstack":
        from synthesis.examples.unstack import build_program
    else:
        raise ValueError("Full-loop comparison supports Stack and Unstack")
    if not rows or any(row["kind"] not in ("Pick", "Move", "Release") for row in rows):
        raise ValueError("A nonempty straight-line primitive body is required")
    program = build_program(
        HighLevelContext(use_tbl=task == "unstack"), num_blocks=num_blocks
    )
    loops = [
        (i, item)
        for i, item in enumerate(program.instructions)
        if isinstance(item, While)
    ]
    if len(loops) != 1:
        raise ValueError("Expected one supplied top-level loop")
    loop_index, loop = loops[0]
    suffix = [deepcopy(i) for i in loop.body if isinstance(i, Assign)]
    loop.body = decode(rows) + suffix
    return program, str(loop_index)


def canonical_goal(task, observation, num_blocks, table_top):
    p = positions(observation, num_blocks)
    symbolic = bool(evaluate(task_spec(task)[1], Scene(p, {"b0": 0})))
    if task == "stack":
        return symbolic
    return bool(
        symbolic
        and all(abs(v[2] - table_top - 0.025) <= 0.001 for v in p.values())
        and all(scattered_implementation(p[a], p[b]) for a in p for b in p if a != b)
    )


def local_goal(observation, observer):
    p = positions(observation, observer.n)
    if observer.task == "stack":
        return bool(on(p[observer.mover], p[observer.target]))
    return bool(
        abs(p[observer.mover][2] - observer.table_top - 0.025) <= 0.001
        and all(
            scattered_implementation(p[observer.mover], q)
            for i, q in p.items()
            if i != observer.mover
        )
    )


def release_state(observation, observer):
    contacts = sum(
        1
        for c in observer.data.contact[: observer.data.ncon]
        if c.dist <= observer.distance_tol
        and (
            (
                int(c.geom1) in observer.fingers
                and observer.objects.get(int(c.geom2)) == observer.mover
            )
            or (
                int(c.geom2) in observer.fingers
                and observer.objects.get(int(c.geom1)) == observer.mover
            )
        )
    )
    return {
        "released": bool(float(np.sum(observation[3:5])) > 0.053 and contacts == 0),
        "finger_contacts": contacts,
        "local_goal": local_goal(observation, observer),
    }


class LoopMonitor:
    """Read-only substep audits, with frames measured from each actual loop head."""

    def __init__(self, env, task, num_blocks, rows, loop_path, *, snapshots=False):
        self.env, self.task, self.n = env, task, num_blocks
        self.rows, self.loop_path = rows, loop_path
        self.save_snapshots = snapshots
        self.states, self.snapshots, self.actions, self.events = [], [], [], []
        self.transfers = []
        self.active = self.info = None

    def record(self, observation):
        self.states.append(np.array(observation, copy=True))
        if self.save_snapshots:
            self.snapshots.append(capture(self.env))

    def finish_transfer(self):
        if self.active is None:
            return
        self.active.__exit__(None, None, None)
        diagnostics = self.active.report()
        info = self.info
        info.update(end_index=len(self.states) - 1, diagnostics=diagnostics)
        info["controls_converged"] = bool(
            len(info["controls"]) == len(self.rows)
            and all(c.get("converged", False) for c in info["controls"])
        )
        release = info.get("body_end", {})
        info["physical_pass"] = bool(
            info["controls_converged"]
            and release.get("released", False)
            and release.get("local_goal", False)
            and local_goal(self.states[-1], self.active)
            and diagnostics["no_unintended_contacts"]
            and diagnostics["frame_within_tolerance"]
            and diagnostics["saw_mover_finger_contact"]
            and diagnostics["lift_observed"]
        )
        info["strict_1mm_pass"] = bool(
            info["physical_pass"]
            and diagnostics["max_nonmover_displacement_m"] <= 0.001
        )
        self.transfers.append(info)
        self.active = self.info = None

    def event(self, event):
        self.events.append(deepcopy(event))
        if event["kind"] == "loop_head" and event["path"] == self.loop_path:
            self.finish_transfer()
            example = dict(
                task=self.task, num_blocks=self.n, bindings=event["bindings"]
            )
            self.active = ContactObserver(self.env, example, drift_tolerance_m=0.005)
            self.active.__enter__()
            self.info = dict(
                iteration=event["iteration"],
                start_index=event["index"],
                bindings=dict(event["bindings"]),
                controls=[],
            )
        elif event["kind"] == "instruction_end" and self.active is not None:
            if event["path"].startswith(self.loop_path + ".") and "control" in event:
                self.info["controls"].append(dict(event["control"]))
            if event["path"] == f"{self.loop_path}.{len(self.rows) - 1}":
                self.info["body_end"] = release_state(self.states[-1], self.active)


def save_trace(path, monitor, report):
    if not monitor.snapshots:
        return
    arrays = sorted(monitor.snapshots[0].arrays)
    np.savez_compressed(
        path,
        observations=np.asarray(monitor.states),
        gt_state=np.stack([s.gt_state for s in monitor.snapshots]),
        actions=np.asarray(monitor.actions),
        **{
            "snap_" + name: np.stack([s.arrays[name] for s in monitor.snapshots])
            for name in arrays
        },
    )
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                **report,
                "arrays": arrays,
                "events": monitor.events,
                "bindings": [s.bindings for s in monitor.snapshots],
            },
            indent=2,
        )
        + "\n"
    )


def evaluate_case(task, rows, num_blocks, seed, *, save=False, parity=False):
    """Run one whole program; return outcome and optional exact video capture."""
    from synthesis.api.control import get_move_action
    from synthesis.mcmc.synthesis import (
        make_roboverify_env,
        preserved_global_rng,
        set_np_seed,
    )

    started = time.perf_counter()
    report = dict(
        task=task,
        num_blocks=num_blocks,
        seed=seed,
        used_for_fitting=False,
        inter_iteration_resets=0,
        inter_iteration_settling_steps=0,
        initial_settling_steps=50 if task == "stack" else 0,
        requested_final_settling_steps=50,
        final_settling_steps=0,
        max_loop_iterations=num_blocks + 2,
        full_program_success=False,
        scope="Finite physical execution in supplied loop scaffold; no loop discovery or formal proof.",
    )
    env = monitor = None
    try:
        with preserved_global_rng():
            set_np_seed(seed)
            env = make_roboverify_env(task, num_blocks=num_blocks)
            first = np.asarray(env.reset()[0])
            inner = inner_env(env)
            if task == "stack":
                anchor = first[:3].copy()
                for _ in range(50):
                    first = np.asarray(
                        env.step(get_move_action(first, anchor, close_gripper=False))[0]
                    )
            bindings = {"b0": 0, **({"tbl": "tbl"} if task == "unstack" else {})}
            inner.symbolic_name_to_box_id.update(bindings)
            initial = capture(env)
            report["initial_snapshot_sha256"] = snapshot_digest(initial)
            report["initial_precondition"] = bool(
                evaluate(
                    task_spec(task)[0],
                    scene_from_obs(
                        first, num_blocks, bindings, include_table=task == "unstack"
                    ),
                )
            )
            program, loop_path = build_loop_program(task, rows, num_blocks)
            monitor = LoopMonitor(
                env, task, num_blocks, rows, loop_path, snapshots=save
            )
            original_step = env.step

            def step(action):
                result = original_step(action)
                monitor.actions.append(np.array(action, copy=True))
                return result

            env.step = step
            try:
                program.eval(
                    env,
                    on_state=monitor.record,
                    on_event=monitor.event,
                    initial_observation=first,
                    max_loop_iterations=num_blocks + 2,
                )
                report["execution_end_index"] = len(monitor.states) - 1
                report["loop_exited"] = any(
                    e["kind"] == "loop_exit" and e["path"] == loop_path
                    for e in monitor.events
                )
                # Preserve the final active observer through every settling step.
                obs = monitor.states[-1]
                anchor = obs[:3].copy()
                for _ in range(50):
                    delta = 14 * (anchor - obs[:3])
                    norm = np.linalg.norm(delta)
                    if norm > 0.55:
                        delta *= 0.55 / norm
                    obs = np.asarray(env.step(np.r_[delta, 0.2])[0])
                    monitor.record(obs)
                    report["final_settling_steps"] += 1
            finally:
                monitor.finish_transfer()
                env.step = original_step
            table_id = inner.sim.model.geom_name2id("table0")
            table_top = float(inner.sim.data.geom_xpos[table_id, 2])
            if int(inner.sim.model.geom_type[table_id]) != 0:
                table_top += float(
                    np.sum(
                        np.abs(inner.sim.data.geom_xmat[table_id].reshape(3, 3)[2])
                        * inner.sim.model.geom_size[table_id]
                    )
                )
            transfers = monitor.transfers
            report.update(
                transfers=transfers,
                iterations=len(transfers),
                states=len(monitor.states),
                actions=len(monitor.actions),
                dt=float(inner.dt),
                canonical_final_goal=canonical_goal(
                    task, monitor.states[-1], num_blocks, table_top
                ),
                stable_final_goal=all(
                    canonical_goal(task, s, num_blocks, table_top)
                    for s in monitor.states[-50:]
                ),
                simulator_final_success=bool(inner._is_success(monitor.states[-1])),
                all_controls_converged=bool(transfers)
                and all(t["controls_converged"] for t in transfers),
                all_transfers_physical=bool(transfers)
                and all(t["physical_pass"] for t in transfers),
                no_unintended_contacts=all(
                    t["diagnostics"]["no_unintended_contacts"] for t in transfers
                ),
                max_nonmover_displacement_m=max(
                    (
                        t["diagnostics"]["max_nonmover_displacement_m"]
                        for t in transfers
                    ),
                    default=0.0,
                ),
                error=None,
            )
            report["full_program_success"] = bool(
                report["initial_precondition"]
                and report["loop_exited"]
                and report["iterations"] == num_blocks - 1
                and report["final_settling_steps"] == 50
                and report["canonical_final_goal"]
                and report["stable_final_goal"]
                and report["simulator_final_success"]
                and report["all_transfers_physical"]
            )
            if parity:
                plain, _ = collect_recording(
                    build_loop_program(task, rows, num_blocks)[0],
                    env,
                    initial_observation=restore(env, initial),
                    max_loop_iterations=num_blocks + 2,
                )
                plain = np.asarray(list(plain) + settle(env, plain[-1], 50))
                report["observer_parity_max_error"] = (
                    float(np.max(np.abs(plain - np.asarray(monitor.states))))
                    if plain.shape == np.asarray(monitor.states).shape
                    else None
                )
                if report["observer_parity_max_error"] != 0:
                    raise AssertionError("Observer changed recorded program execution")
    except Exception as exc:
        report.update(error=f"{type(exc).__name__}: {exc}", full_program_success=False)
        if monitor is not None:
            report.update(
                transfers=monitor.transfers,
                states=len(monitor.states),
                iterations=len(monitor.transfers),
            )
    finally:
        if env is not None:
            env.close()
    report["elapsed_seconds"] = time.perf_counter() - started
    return report, monitor


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--task", choices=("stack", "unstack"), required=True)
    parser.add_argument("--num-blocks", type=int, choices=(3, 4, 5), required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seconds", type=float, default=48.0)
    parser.add_argument("--parity", action="store_true")
    args = parser.parse_args(argv)
    if digest(args.candidate) != args.expected_sha256:
        parser.error("Frozen candidate hash mismatch")
    if not 0 < args.seconds <= 49:
        parser.error("Batch runner budget must be in (0,49] seconds")
    if len(args.seeds) != len(set(args.seeds)) or any(s < 0 for s in args.seeds):
        parser.error("Seeds must be distinct and nonnegative")
    candidate = json.loads(Path(args.candidate).read_text())
    if candidate.get("task") != args.task:
        parser.error("Candidate task mismatch")
    rows = candidate["instructions"]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    reports = []
    with execution_deadline(args.seconds):
        for seed in args.seeds:
            if time.perf_counter() - started >= args.seconds - 0.5:
                break
            report, monitor = evaluate_case(
                args.task,
                rows,
                args.num_blocks,
                seed,
                save=seed == 60,
                parity=args.parity,
            )
            report.update(
                candidate_sha256=args.expected_sha256,
                candidate=str(Path(args.candidate).resolve()),
            )
            (output / f"seed{seed}.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )
            if seed == 60 and monitor is not None:
                save_trace(output / "seed60.npz", monitor, report)
            reports.append(report)
            print(
                json.dumps(
                    {
                        k: report.get(k)
                        for k in [
                            "seed",
                            "full_program_success",
                            "iterations",
                            "error",
                            "elapsed_seconds",
                        ]
                    }
                ),
                flush=True,
            )
            if report.get("error", "") and report["error"].startswith("TimeoutError"):
                break
    summary = dict(
        requested_seeds=args.seeds,
        completed_seeds=[r["seed"] for r in reports],
        completed=len(reports) == len(args.seeds),
        success_count=sum(r["full_program_success"] for r in reports),
        cases=len(reports),
        candidate_sha256=args.expected_sha256,
        seconds=time.perf_counter() - started,
    )
    (output / f"batch-{min(args.seeds)}-{max(args.seeds)}.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary), flush=True)
    return 0 if summary["completed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
