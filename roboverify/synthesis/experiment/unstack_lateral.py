"""Test the supplied Unstack program in the standard right-to-left environment.

Higher Y is right and lower Y is left in the existing Grid/Pyramid convention.
This isolated experiment resolves current minimum-Y, maximum-Z and gripper-Z
operands at execution time; these are not implemented in the symbolic/motion DSL.
The sideways transfer freezes the gripper height reached by the preceding lift.
The supplied invariant is ignored. The environment uses the production Unstack reset.
Run from roboverify with the MuJoCo environment and an external 55-second timeout.
"""

import argparse
import json
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
import z3

from synthesis.api.instructions import (
    MoveByName,
    PickByName,
    ReleaseByName,
    While,
    _eval_primitive,
)
from synthesis.api.program import Program
from synthesis.cfg.collection import record_execution, validate_trace
from synthesis.cfg.program_source import prepare_program
from synthesis.cfg.recordings import save_traces
from synthesis.cfg.tasks import task_identity, task_spec
from synthesis.experiment.run_logger import RunLogger
from synthesis.mcmc.synthesis import make_roboverify_unstack_env as lateral_env
from synthesis.predicates.scene import evaluate, scene_from_obs
from synthesis.util.on import get_block_pos
from synthesis.verification_lib.highlevel_verification_lib import HighLevelContext


def build_program(context, count):
    selected, other, base, table = z3.Consts("b_prime n b0 tbl", context.BoxSort)
    guard = z3.ForAll(
        [other],
        z3.Implies(
            z3.And(
                other != table,
                z3.Or(context.ON_star(base, selected), context.ON_star(other, base)),
            ),
            context.ON_star(selected, other),
        ),
    )
    body = [
        MoveByName("b_prime", "b_prime", "b_prime", target_offset=[0, 0, 0.15]),
        PickByName("b_prime"),
        MoveByName("b_prime", "b_prime", "b_prime", target_offset=[0, 0, 0.15]),
        # The scoped execution wrapper resolves gripper Z once at Move entry.
        MoveByName("b0", "Y_min", "gripper", target_offset=[0, -0.15, 0]),
        MoveByName("b0", "Y_min", "b0", target_offset=[0, 0, 0]),
        ReleaseByName("Z_m", target_z=0.15),
    ]
    definition = prepare_program(
        Program(1, [While(guard, [selected], body, invariant=None)]),
        count,
        source="user-supplied-unstack/right-to-left-horizontal-transfer",
    )
    definition.initial_bindings["tbl"] = "tbl"
    return definition


def positions(obs, count):
    return np.array([get_block_pos(obs, i) for i in range(count)])


@contextmanager
def dynamic_targets(count, audit):
    """Resolve experiment-only references; keep each motion target fixed."""
    originals = {cls: cls.eval for cls in (MoveByName, PickByName, ReleaseByName)}

    def wrap(cls):
        def run(instruction, env, traj, return_image=False):
            observation = np.asarray(env.flatten_observation(env.env._get_obs())).copy()
            xyz = positions(observation, count)
            aliases = env.symbolic_name_to_box_id
            aliases["Y_min"] = int(np.argmin(xyz[:, 1]))
            aliases["Z_m"] = int(np.argmax(xyz[:, 2]))
            hold_gripper_height = (
                cls is MoveByName and instruction.target_box_name_z == "gripper"
            )
            if cls is MoveByName:
                names = (
                    instruction.target_box_name_x,
                    instruction.target_box_name_y,
                    instruction.target_box_name_z,
                )
                target = [
                    float(
                        (
                            observation[2]
                            if axis == 2 and hold_gripper_height
                            else xyz[aliases[name], axis]
                        )
                        + instruction.target_offset[axis].numeric_val()
                    )
                    for axis, name in enumerate(names)
                ]
            elif cls is ReleaseByName:
                target = [
                    *map(float, observation[:2]),
                    float(
                        xyz[aliases["Z_m"], 2]
                        + instruction.target_z_offset.numeric_val()
                    ),
                ]
            else:
                target = xyz[aliases[instruction.grab_box_name]].tolist()
            row = dict(
                instruction=cls.__name__,
                bindings=dict(aliases),
                target=target,
                holds_gripper_height=hold_gripper_height,
                before=xyz.tolist(),
                gripper_before=observation[:3].tolist(),
                index_before=len(traj) - 1,
            )
            if hold_gripper_height:
                # Use the same bounded controller with an absolute waypoint.
                # A zero block-relative offset would still change gripper Z.
                result = _eval_primitive(
                    instruction, env, traj, return_image, "move", target
                )
            else:
                result = originals[cls](instruction, env, traj, return_image)
            row.update(
                control=asdict(instruction.last_control_result),
                after=positions(traj[-1], count).tolist(),
                gripper_after=np.asarray(traj[-1])[:3].tolist(),
                index_after=len(traj) - 1,
            )
            audit.append(row)
            return result

        return run

    with patch.object(MoveByName, "eval", wrap(MoveByName)), patch.object(
        PickByName, "eval", wrap(PickByName)
    ), patch.object(ReleaseByName, "eval", wrap(ReleaseByName)):
        yield


def run(args, logger):
    context = HighLevelContext(use_tbl=True)
    rows = []
    for seed in args.seeds:
        audit = []
        definition = build_program(context, args.num_blocks)
        logger.set_status(phase="execution", seed=seed)
        with dynamic_targets(args.num_blocks, audit):
            trace = record_execution(
                definition,
                seed=seed,
                num_blocks=args.num_blocks,
                task="unstack",
                env_factory=lambda: lateral_env(args.num_blocks),
                max_loop_iterations=args.num_blocks + 1,
                timeout_seconds=12,
            )
        trace.metadata["task_spec"] = task_identity("unstack")
        trace.metadata["extrema_semantics"] = (
            "minimum Y and maximum Z among all physical blocks, recomputed before each instruction"
        )
        trace.metadata["transport_height_semantics"] = (
            "gripper Z captured once at transfer entry and held throughout the move"
        )
        valid = validate_trace(trace)
        pre, post = task_spec("unstack")
        initial_ok = final_ok = all_on_table = None
        if trace.states:
            initial_ok = evaluate(
                pre,
                scene_from_obs(
                    trace.states[0],
                    args.num_blocks,
                    definition.initial_bindings,
                    include_table=True,
                ),
            )
            final_scene = scene_from_obs(
                trace.states[-1],
                args.num_blocks,
                definition.initial_bindings,
                include_table=True,
            )
            final_ok = evaluate(post, final_scene)
            xyz = positions(trace.states[-1], args.num_blocks)
            all_on_table = bool(np.all(np.abs(xyz[:, 2] - 0.425) <= 0.002))
            save_traces(logger.artifact_dir("traces") / f"seed-{seed}.npz", [trace])
        failures = [
            dict(step=i, **record)
            for i, record in enumerate(audit)
            if not record["control"]["converged"]
        ]
        row = dict(
            seed=seed,
            num_blocks=args.num_blocks,
            valid=valid,
            status=trace.metadata["status"],
            reason=trace.metadata["reason"],
            precondition=initial_ok,
            postcondition=final_ok,
            all_on_table_2mm=all_on_table,
            iterations=sum(e["kind"] == "loop_head" for e in trace.events),
            normal_exit=any(e["kind"] == "loop_exit" for e in trace.events),
            elapsed_seconds=trace.metadata["elapsed_seconds"],
            failed_controls=len(failures),
            first_failure=failures[0] if failures else None,
            initial_positions=(
                positions(trace.states[0], args.num_blocks).tolist()
                if trace.states
                else None
            ),
            final_positions=(
                positions(trace.states[-1], args.num_blocks).tolist()
                if trace.states
                else None
            ),
        )
        rows.append(row)
        logger.write_artifact(
            f"seed-{seed}-instructions.json", json.dumps(audit, indent=2)
        )
        logger.write_artifact(f"seed-{seed}-summary.json", json.dumps(row, indent=2))
        logger.log_event(
            "trial",
            f'seed {seed}: {row["status"]}, post={final_ok}, failed_controls={len(failures)}',
            force=True,
        )
    summary = dict(
        num_blocks=args.num_blocks,
        trials=len(rows),
        valid=sum(r["valid"] for r in rows),
        postcondition=sum(r["postcondition"] is True for r in rows),
        trials_summary=rows,
        formal_verification="not_run",
        invariant="ignored",
    )
    logger.write_artifact("summary.json", json.dumps(summary, indent=2))
    logger.finish(
        "completed",
        valid=summary["valid"],
        trials=summary["trials"],
        formal_verification="not_run",
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-blocks", type=int, choices=(2, 3, 4), default=4)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or any(
        not 0 <= seed < 2**32 for seed in args.seeds
    ):
        parser.error("Supply distinct seeds in [0, 2**32)")
    with RunLogger(
        "runs",
        "unstack-lateral",
        dict(
            vars(args),
            layout="right-to-left; X in base+[.54,.70]; Y in base+[.15,.25]; spacing .15",
            transport_height="hold gripper Z at transfer entry",
            transport_z_offset=0.0,
        ),
        slug=f"{args.num_blocks}-blocks",
    ) as logger:
        logger.progress_line(f"Unstack diagnostic: {logger.run_dir}")
        logger.write_artifact("unstack_lateral.py", Path(__file__).read_text())
        summary = run(args, logger)
    print(
        json.dumps(
            dict(
                run=str(logger.run_dir),
                num_blocks=summary["num_blocks"],
                valid=summary["valid"],
                trials=summary["trials"],
                postcondition=summary["postcondition"],
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
