"""Bounded workspace survey and paired Reverse placement diagnostics.

These experiments leave production resets and controllers unchanged. Coordinates
passed to --x/--y are relative to the robot base. Run each Unstack batch under a
55-second process timeout; a timed-out batch is not a successful experiment.
"""

import argparse
import hashlib
import json
from contextlib import contextmanager, nullcontext
from itertools import product
from pathlib import Path
from unittest.mock import patch

import numpy as np

from synthesis.api.control import ControlConfig, PrimitiveController, get_move_action
from synthesis.api.instructions import PickByName
from synthesis.cfg.collection import record_execution, validate_trace
from synthesis.cfg.program_source import load_program
from synthesis.cfg.recordings import load_traces, save_traces
from synthesis.cfg.reset import inner_env
from synthesis.experiment.run_logger import RunLogger
from synthesis.experiment.unstack_lateral import (
    build_program,
    dynamic_targets,
    positions,
)
from synthesis.mcmc.synthesis import make_roboverify_env, preserved_global_rng
from synthesis.verification_lib.highlevel_verification_lib import HighLevelContext


def placement_measurements(trace):
    """Measure block/gripper positions at physical instruction boundaries."""
    rows = []
    for event in trace.events:
        if event["kind"] != "instruction_end" or "control" not in event:
            continue
        obs = trace.states[event["index"]]
        selected = event["bindings"]["b_prime"]
        xyz = positions(obs, trace.num_blocks)
        rows.append(
            dict(
                path=event["path"],
                index=event["index"],
                selected=selected,
                support=event["bindings"].get("b"),
                gripper=obs[:3].tolist(),
                blocks=xyz.tolist(),
                payload_minus_gripper_mm=((xyz[selected] - obs[:3]) * 1000).tolist(),
                control=event["control"],
            )
        )
    return rows


def initial_geometry(env):
    inner = inner_env(env)
    sim = inner.sim
    contacts = []
    for contact in sim.data.contact[: sim.data.ncon]:
        names = [
            sim.model.body_id2name(sim.model.geom_bodyid[g]) or ""
            for g in (contact.geom1, contact.geom2)
        ]
        if any(n.startswith("robot0:") for n in names) and any(
            n.startswith("object") for n in names
        ):
            contacts.append(dict(bodies=names, distance=float(contact.dist)))
    obs = inner.flatten_observation(inner._get_obs())
    return dict(
        robot_base_xy=inner.robot_base_xy.tolist(),
        gripper=obs[:3].tolist(),
        table=inner.table_position.tolist(),
        robot_block_contacts=contacts,
    )


@contextmanager
def payload_xy_feedback(*, include_z=False):
    """Track payload XY (optionally XYZ) during table-directed Moves.

    Retains the commanded target, gains, tolerances and instruction budgets.
    The default keeps Z feedback on the gripper; include_z tests payload Z too.
    Uses the simulator's measured block position; not a production DSL change.
    """
    original = PrimitiveController.move_relative

    def move(controller, references, offsets):
        if tuple(references[:2]) != ("tbl", "tbl"):
            return original(controller, references, offsets)
        selected = inner_env(controller.env).symbolic_name_to_box_id["b_prime"]
        target = np.array(
            [
                controller.reference_position(ref)[i] + offsets[i]
                for i, ref in enumerate(references)
            ]
        )

        def delta():
            actual = controller.observation[:3].copy()
            axes = slice(None) if include_z else slice(0, 2)
            actual[axes] = controller.box_position(selected)[axes]
            return target - actual

        return controller._until(
            "payload_xyz_diagnostic" if include_z else "payload_xy_diagnostic",
            lambda: np.linalg.norm(delta()) <= controller.control.position_tolerance,
            lambda: get_move_action(
                controller.observation,
                controller.observation[:3] + delta(),
                gain=controller.control.gain,
                close_gripper=True,
            ),
            lambda: np.linalg.norm(delta()),
        )

    with patch.object(PrimitiveController, "move_relative", move):
        yield


def reset_probe(args, point, seed):
    with preserved_global_rng(), patch(
        "synthesis.environment.cee_us_env.fpp_construction_env.sample_tower_xy",
        side_effect=lambda n, base, **kw: np.asarray(base) + point,
    ):
        np.random.seed(seed)
        env = make_roboverify_env(args.task, args.num_blocks)
        try:
            env.reset()
            return dict(seed=seed, point=point, geometry=initial_geometry(env))
        finally:
            env.close()


def run_trial(args, *, seed, point=None, snapshot=None, variant="baseline", video=None):
    context = HighLevelContext(use_tbl=True)
    definition = (
        build_program(context, args.num_blocks)
        if args.task == "unstack"
        else load_program(
            "synthesis.examples.reverse:build_program", context, args.num_blocks
        )
    )
    if variant == "tight_pick":
        for instruction in definition.program.instructions[1].body:
            if isinstance(instruction, PickByName):
                instruction.control = ControlConfig(position_tolerance=0.0005)
    geometry = {}

    def factory():
        env = make_roboverify_env(args.task, args.num_blocks)
        reset = env.reset

        def audited_reset(*a, **kw):
            result = reset(*a, **kw)
            geometry.update(initial_geometry(env))
            return result

        env.reset = audited_reset
        return env

    reset_patch = (
        patch(
            "synthesis.environment.cee_us_env.fpp_construction_env.sample_tower_xy",
            side_effect=lambda n, base, **kw: np.asarray(base) + point,
        )
        if point is not None
        else nullcontext()
    )
    with reset_patch, (
        dynamic_targets(args.num_blocks, [])
        if args.task == "unstack"
        else nullcontext()
    ), (
        payload_xy_feedback(include_z=variant == "payload_xyz")
        if variant.startswith("payload_")
        else nullcontext()
    ):
        trace = record_execution(
            definition,
            seed=seed,
            num_blocks=args.num_blocks,
            task=args.task,
            env_factory=factory,
            initial_snapshot=snapshot,
            max_loop_iterations=args.num_blocks + 1,
            timeout_seconds=12 if video is None else 45,
            video_path=video,
        )
    valid = validate_trace(trace)
    trace.metadata["experiment"] = dict(
        module=__name__,
        variant=variant,
        source_offset=point,
        production_controller=variant == "baseline",
    )
    if variant.startswith("payload_"):
        # This scoped controller intervention is absent from describe_program.
        # Give its diagnostic archives a distinct executable identity.
        trace.metadata["experimental_controller"] = f"measured-{variant}-feedback-v1"
        trace.metadata["fingerprint"] = hashlib.sha256(
            (
                trace.metadata["fingerprint"]
                + "/"
                + trace.metadata["experimental_controller"]
            ).encode()
        ).hexdigest()
    final = positions(trace.states[-1], args.num_blocks) if trace.states else None
    ordered = final[np.argsort(final[:, 2])] if final is not None else None
    controls = [e["control"] for e in trace.events if "control" in e]
    row = dict(
        seed=seed,
        point=point,
        variant=variant,
        valid=valid,
        status=trace.metadata["status"],
        reason=trace.metadata["reason"],
        elapsed_seconds=trace.metadata["elapsed_seconds"],
        geometry=geometry,
        initial_positions=(
            positions(trace.states[0], args.num_blocks).tolist()
            if trace.states
            else None
        ),
        final_positions=final.tolist() if final is not None else None,
        adjacent_xy_mm=(
            (np.linalg.norm(np.diff(ordered[:, :2], axis=0), axis=1) * 1000).tolist()
            if ordered is not None
            else []
        ),
        max_control_steps=max((c["steps"] for c in controls), default=0),
        failed_controls=[c for c in controls if not c["converged"]],
        iterations=sum(e["kind"] == "loop_head" for e in trace.events),
        normal_exit=any(e["kind"] == "loop_exit" for e in trace.events),
        measurements=placement_measurements(trace),
    )
    return row, trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("unstack", "reverse"), required=True)
    parser.add_argument("--num-blocks", type=int, choices=(2, 3, 4), default=4)
    parser.add_argument("--x", type=float, nargs="+")
    parser.add_argument("--y", type=float, nargs="+")
    parser.add_argument(
        "--samples",
        type=int,
        default=0,
        help="Sample this many points within two-value X/Y bounds instead of a grid",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument(
        "--archive", help="Exact starts for paired Reverse interventions"
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=("baseline", "tight_pick", "payload_xy", "payload_xyz"),
        default=["baseline"],
    )
    parser.add_argument(
        "--initial-only",
        action="store_true",
        help="Inspect reset contacts without executing a program",
    )
    parser.add_argument("--save-traces", action="store_true")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--run-name", default="survey")
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or any(
        not 0 <= seed < 2**32 for seed in args.seeds
    ):
        parser.error("Supply distinct seeds in [0, 2**32)")
    if bool(args.x) != bool(args.y) or (args.archive and args.x):
        parser.error("Use paired --x/--y, or --archive, or neither")
    if args.task != "reverse" and (args.archive or args.variants != ["baseline"]):
        parser.error("Placement interventions support Reverse only")
    if args.initial_only and (
        not args.x
        or args.archive
        or args.variants != ["baseline"]
        or args.save_video
        or args.save_traces
    ):
        parser.error(
            "Initial-only probes require X/Y coordinates and no execution options"
        )
    if any(not np.isfinite(v) for v in (args.x or []) + (args.y or [])):
        parser.error("Coordinates must be finite")
    if args.samples < 0 or (
        args.samples
        and (
            not args.x
            or len(args.x) != 2
            or len(args.y) != 2
            or args.x[0] >= args.x[1]
            or args.y[0] >= args.y[1]
            or len(args.seeds) != 1
        )
    ):
        parser.error(
            "Random sampling needs increasing two-value X/Y bounds and one seed"
        )
    starts = (
        {t.seed: t for t in load_traces(args.archive, require_valid=True)}
        if args.archive
        else {}
    )
    if starts and any(
        s not in starts
        or starts[s].num_blocks != args.num_blocks
        or starts[s].task != args.task
        for s in args.seeds
    ):
        parser.error("Requested seeds, task and block count must match the archive")
    points = list(product(args.x, args.y)) if args.x else [None]
    if args.samples:
        points = (
            np.random.default_rng(args.seeds[0])
            .uniform(
                [args.x[0], args.y[0]], [args.x[1], args.y[1]], size=(args.samples, 2)
            )
            .tolist()
        )
    rows = []
    with RunLogger(
        "runs",
        "tower-workspace",
        vars(args),
        slug=f"{args.task}-{args.num_blocks}-{args.run_name}",
    ) as logger:
        logger.write_artifact("tower_workspace.py", Path(__file__).read_text())
        for point, seed, variant in product(points, args.seeds, args.variants):
            number = len(rows)
            logger.set_status(
                phase="execution", trial=number, seed=seed, point=point, variant=variant
            )
            if args.initial_only:
                rows.append(reset_probe(args, point, seed))
                logger.write_artifact("summary.json", json.dumps(rows, indent=2))
                continue
            snapshot = starts[seed].snapshots[0] if starts else None
            row, trace = run_trial(
                args,
                seed=seed,
                point=point,
                snapshot=snapshot,
                variant=variant,
                video=(
                    logger.artifact_dir("videos") / f"trial-{number:03d}-{variant}.mp4"
                    if args.save_video
                    else None
                ),
            )
            if snapshot is not None and trace.snapshots:
                actual = trace.snapshots[0]
                row["exact_initial_physics"] = bool(
                    np.array_equal(snapshot.gt_state, actual.gt_state)
                    and all(
                        np.array_equal(v, actual.arrays[k])
                        for k, v in snapshot.arrays.items()
                    )
                )
            if starts and variant == "baseline":
                original = starts[seed]
                row["baseline_replay_equal"] = bool(
                    len(trace.states) == len(original.states)
                    and np.allclose(trace.states, original.states, rtol=0, atol=1e-8)
                    and len(trace.actions[0]) == len(original.actions[0])
                    and np.allclose(
                        trace.actions[0], original.actions[0], rtol=0, atol=1e-8
                    )
                )
            if args.save_traces and trace.states:
                save_traces(
                    logger.artifact_dir("traces") / f"trial-{number:03d}.npz", [trace]
                )
            logger.write_artifact(f"trial-{number:03d}.json", json.dumps(row, indent=2))
            rows.append({k: v for k, v in row.items() if k != "measurements"})
            logger.write_artifact("summary.json", json.dumps(rows, indent=2))
        valid = None if args.initial_only else sum(r["valid"] for r in rows)
        logger.finish(
            "completed", trials=len(rows), valid=valid, formal_verification="not_run"
        )
    print(json.dumps(dict(run=str(logger.run_dir), trials=len(rows), valid=valid)))


if __name__ == "__main__":
    main()
