"""Revalidate saved demonstrations, replay actions, and check settled task success.

With ``--post-settling-steps 0`` (the default) the replayed final state must
pass the physical evaluator before and after ``--settle-steps`` settling steps.
A positive ``--post-settling-steps`` instead scores the last snapshot with the
shared settled checker (``task-spec-v2``); the physical evaluator is then a
diagnostic, and a stored settled result must agree with the recomputed one.
This is empirical archive auditing, not synthesis or formal verification.
Run Unstack invocations under a 55-second process timeout.
"""

import argparse
import copy
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from synthesis.cfg.collection import validate_trace
from synthesis.cfg.demos import DemoSegment
from synthesis.cfg.goal_collection import GOAL_TASKS, make_goal_env
from synthesis.cfg.program_source import load_program
from synthesis.cfg.recordings import load_traces
from synthesis.cfg.reset import inner_env, reset_segment, restore
from synthesis.cfg.task_success import (
    check_recorded_task_success,
    criterion,
    settle,
)
from synthesis.cfg.tasks import task_identity
from synthesis.util.on import get_block_pos, using_higher_tolerance
from synthesis.verification_lib.highlevel_verification_lib import HighLevelContext


def audit_archive(path, *, seeds, settle_steps=50, post_settling_steps=0):
    from synthesis.mcmc.synthesis import make_roboverify_env

    traces = load_traces(path, require_valid=True)
    if sorted(t.seed for t in traces) != sorted(seeds):
        raise ValueError(
            "Archive seeds do not exactly match the requested distinct seeds"
        )
    if len(set(seeds)) != len(seeds):
        raise ValueError("Audit seeds must be distinct")
    rows = []
    for trace in traces:
        row = dict(seed=trace.seed, task=trace.task, num_blocks=trace.num_blocks)
        env = None
        try:
            if trace.metadata.get("task_spec") != task_identity(trace.task):
                raise ValueError(
                    "Recorded task identity does not match the current task"
                )
            with using_higher_tolerance(trace.metadata["higher_tolerance"]):
                # Validation writes metadata; leave the original archive data intact.
                candidate = replace(trace, metadata=copy.deepcopy(trace.metadata))
                candidate.metadata["status"] = "completed"
                if not validate_trace(candidate):
                    raise ValueError(
                        candidate.metadata.get("reason", "Trace validation failed")
                    )
                context = HighLevelContext(
                    use_tbl=trace.task in ("unstack", "reverse", "restack")
                )
                current = load_program(
                    trace.metadata["source"],
                    context,
                    trace.num_blocks,
                    allow_goal_program=True,
                    allow_observation_references=True,
                )
                if current.metadata["fingerprint"] != trace.metadata["fingerprint"]:
                    raise ValueError(
                        "Archive program does not match its current factory"
                    )
                config = trace.metadata.get("environment", {})
                env = (
                    make_goal_env(trace.task, config)
                    if trace.task in GOAL_TASKS
                    else make_roboverify_env(trace.task, trace.num_blocks, **config)
                )
                start = restore(env, trace.snapshots[0])
                np.testing.assert_allclose(start, trace.states[0], atol=1e-8, rtol=0)
                final_index = len(trace.states) - 1
                segment = DemoSegment(
                    0,
                    final_index,
                    final_index,
                    trace,
                    trace.snapshots[-1].bindings,
                    goal_bindings=trace.snapshots[-1].goal_bindings,
                )
                final = reset_segment(env, segment, mode="replay")
                replay_error = float(np.max(np.abs(final - trace.states[-1])))
                np.testing.assert_allclose(final, trace.states[-1], atol=1e-8, rtol=0)
                inner = inner_env(env)
                if post_settling_steps:
                    settled = check_recorded_task_success(
                        trace, post_settling_steps, environment=env
                    )
                    final = settle(env, 0)
                    stored = (trace.metadata.get("validation") or {}).get("settled")
                    row.update(
                        settled_task_success=settled.as_dict(),
                        final_physical_success=settled.environment_success_at_end,
                        settled_physical_success=settled.environment_success_settled,
                    )
                    if (
                        stored is not None
                        and stored["post_settling_steps"] == post_settling_steps
                        and stored != settled.as_dict()
                    ):
                        raise ValueError(
                            "Stored settled result differs from the recomputed one"
                        )
                    if not settled.success:
                        raise ValueError(f"Settled task check failed: {settled.issues}")
                else:
                    if not bool(inner._is_success(final)):
                        raise ValueError("Final simulator task-success check is false")
                    final = settle(env, settle_steps, final)
                    if not bool(inner._is_success(final)):
                        raise ValueError(
                            "Task success was lost while settling the final state"
                        )
                    row.update(
                        final_physical_success=True, settled_physical_success=True
                    )
                controls = [e["control"] for e in trace.events if "control" in e]
                row.update(
                    valid=True,
                    replay_max_error=replay_error,
                    primitive_count=len(controls),
                    max_primitive_steps=max((c["steps"] for c in controls), default=0),
                    loop_exits=[
                        dict(path=e["path"], iterations=e["iteration"])
                        for e in trace.events
                        if e["kind"] == "loop_exit"
                    ],
                    final_positions=[
                        get_block_pos(final, i).tolist()
                        for i in range(trace.num_blocks)
                    ],
                )
        except (ValueError, AssertionError, KeyError, TypeError) as exc:
            row.update(valid=False, reason=str(exc))
        finally:
            if env is not None:
                env.close()
        rows.append(row)
    return dict(
        archive=str(path),
        valid=all(row["valid"] for row in rows),
        settle_steps=settle_steps,
        **(
            dict(
                post_settling_steps=post_settling_steps,
                success_criterion=criterion(post_settling_steps),
            )
            if post_settling_steps
            else {}
        ),
        trajectories=rows,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demos", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(15)))
    parser.add_argument(
        "--settle-steps",
        type=int,
        default=50,
        help="Physical-evaluator settling when --post-settling-steps is 0",
    )
    parser.add_argument(
        "--post-settling-steps",
        type=int,
        default=0,
        help="Score the last snapshot with the shared settled checker (task-spec-v2)",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.settle_steps < 0 or args.post_settling_steps < 0:
        parser.error("--settle-steps and --post-settling-steps must be nonnegative")
    result = audit_archive(
        args.demos,
        seeds=args.seeds,
        settle_steps=args.settle_steps,
        post_settling_steps=args.post_settling_steps,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    valid = sum(r["valid"] for r in result["trajectories"])
    print(
        f"{args.demos}: {valid}/{len(result['trajectories'])} passed; report: {args.output}"
    )
    return 0 if result["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
