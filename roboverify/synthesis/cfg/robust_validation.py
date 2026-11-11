"""Finite full-program acceptance on exact training starts, including settling.

Successful local body search is insufficient for this gate: the executable must
finish the whole task, retain the logical postcondition and simulator goal at
every settling step, and pass observed primitive/contact/frame checks. These
finite tests do not establish an unbounded motion proof or generalization.
"""

import json
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from synthesis.api.instructions import (
    LoopBudgetExceeded,
    Move,
    MoveByName,
    Pick,
    PickByName,
    Release,
    ReleaseByName,
    While,
)
from synthesis.cfg.demos import DemoSegment, DemoTrace
from synthesis.cfg.endpoint_contract import EndpointRecorder, check_endpoint_contract
from synthesis.cfg.execute import execute_current
from synthesis.cfg.program_io import save_program_artifact
from synthesis.cfg.program_source import program_fingerprint
from synthesis.cfg.recordings import save_traces
from synthesis.cfg.reset import capture, inner_env, reset_segment
from synthesis.cfg.tasks import task_spec
from synthesis.predicates.scene import evaluate, scene_from_obs
from synthesis.util import on


class ValidationBudgetExceeded(TimeoutError):
    """This validation's cooperative wall/action limit, not an enclosing alarm."""


class InvalidValidationStart(ValueError):
    """An exact restored start violates the current task precondition."""


@dataclass(frozen=True)
class ValidationBudget:
    settling_steps: int = 50
    max_actions: int = 5000
    max_loop_iterations: int = 64
    seconds: float = 120.0
    frame_tolerance_m: float = 0.005

    def __post_init__(self):
        for name in ("max_actions", "max_loop_iterations"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.settling_steps) is not int or self.settling_steps < 0:
            raise ValueError("settling_steps must be a nonnegative integer")
        if not np.isfinite(self.seconds) or self.seconds <= 0:
            raise ValueError("Validation seconds must be positive and finite")
        if not np.isfinite(self.frame_tolerance_m) or self.frame_tolerance_m < 0:
            raise ValueError("Frame tolerance must be finite and nonnegative")


def _bounded_program(program, max_iterations):
    result = deepcopy(program)

    def visit(instructions):
        for instruction in instructions:
            if isinstance(instruction, While):
                instruction.max_iters = min(
                    value
                    for value in (instruction.max_iters, max_iterations)
                    if value is not None
                )
                visit(instruction.body)

    visit(result.instructions)
    return result


class _TransferAudit:
    """Keep the last transfer frame alive after Release and through settling."""

    def __init__(self, recorder, first, count, tolerance):
        self.recorder = recorder
        self.baseline = np.array(first, copy=True)
        self.mover = None
        self.count, self.tolerance = count, tolerance
        self.maximum_drift = 0.0
        self.failures = []

    def observe(self, observation, action_index):
        self.recorder.observe(observation)
        for obj in range(self.count):
            if obj == self.mover:
                continue
            drift = float(
                np.linalg.norm(
                    on.get_block_pos(observation, obj)
                    - on.get_block_pos(self.baseline, obj)
                )
            )
            self.maximum_drift = max(self.maximum_drift, drift)
            if drift > self.tolerance:
                # Bounded detail: retain each object's first observed violation
                # in the current transfer, alongside the global maximum.
                if not any(
                    row["object"] == obj and row["mover"] == self.mover
                    for row in self.failures
                ):
                    self.failures.append(
                        {
                            "action_index": action_index,
                            "object": obj,
                            "mover": self.mover,
                            "drift_m": drift,
                        }
                    )

    def instruction(self, instruction, phase, observation, action_index):
        if phase == "start":
            self.observe(observation, action_index)
        self.recorder.instruction(instruction, phase, observation)
        if phase == "start" and self.recorder.transfer_start is not None:
            if isinstance(instruction, (Pick, PickByName)) or self.mover is None:
                self.baseline = np.array(self.recorder.transfer_start, copy=True)
                self.mover = self.recorder.transfer_mover


class _BlockStates(list):
    pass


def _completed_block(cache, observation, env):
    """Reuse endpoint acceptance with this actual invocation's measured states."""
    recorder = cache["recorder"]
    recorder.observe(observation)
    audit = recorder.finish()
    segment = cache["segment"]
    segment.trace.states = (*segment.trace.states[:2], np.array(observation, copy=True))
    # For whole-program replay, expected hand occupancy is a measured phase,
    # not a teacher-event annotation. Explicit ON still independently requires
    # release and its designated support through check_endpoint_contract.
    if audit["endpoint"]["known"]:
        segment.trace.metadata["holding_by_index"][2] = audit["endpoint"]["holding"]
    bindings = audit["contract_bindings"]
    scene = scene_from_obs(
        observation,
        segment.trace.num_blocks,
        bindings,
        include_table="tbl" in bindings,
        entry_obs=segment.trace.states[0],
    )
    states = _BlockStates([scene])
    states.contract_audit = audit
    return check_endpoint_contract(None, segment, states, cache["post"])


def validate_training_program(
    definition,
    traces,
    env_factory,
    *,
    budget=None,
    post=None,
    artifact_dir=None,
    block_contracts=None,
):
    """Replay a frozen candidate on every complete demonstration's exact start.

    The original executable is saved before any evaluation, even if rejected.
    Replay archives contain candidate events, full snapshots, exact actions and
    observations; teacher events and aliases are never used as candidate events.
    Loop caps affect only a private execution copy. The total wall budget is
    checked between actions/callbacks; any outer CLI alarm propagates unchanged.
    ``env_factory(trace)`` must create the corresponding simulator environment.
    """
    from synthesis.api.guard_eval import NoGuardWitness
    from synthesis.mcmc.synthesis import preserved_global_rng, set_np_seed

    budget = budget or ValidationBudget()
    traces = tuple(traces)
    if not traces:
        raise ValueError("Full-program validation needs training traces")
    started = perf_counter()
    deadline = started + budget.seconds
    fingerprint = program_fingerprint(definition.program)
    block_contracts = dict(block_contracts or {})
    directory = None if artifact_dir is None else Path(artifact_dir)
    if directory is not None:
        directory.mkdir(parents=True, exist_ok=True)
        save_program_artifact(
            directory / "program.json",
            definition,
            metadata={
                "accepted": False,
                "purpose": "frozen training validation candidate",
            },
        )

    def check_deadline():
        if perf_counter() >= deadline:
            raise ValidationBudgetExceeded(
                "Full-program validation wall budget exhausted"
            )

    rows = []
    for index, trace in enumerate(traces):
        row = {
            "demo": index,
            "seed": trace.seed,
            "task": trace.task,
            "num_blocks": trace.num_blocks,
            "accepted": False,
            "reasons": [],
            "settling_requested": budget.settling_steps,
            "settling_completed": 0,
        }
        rows.append(row)
        env = None
        original_step = None
        observations, snapshots, action_indices, actions, events = [], [], [], [], []
        recorder = transfer = None
        began = perf_counter()
        try:
            check_deadline()
            if not trace.snapshots or not trace.states:
                raise ValueError(
                    "Exact archived full starts are required for validation"
                )
            if trace.task not in ("stack", "unstack", "reverse", "restack"):
                raise ValueError("Full-program validation supports tower tasks only")
            task_pre, default_post = task_spec(trace.task)
            task_post = default_post if post is None else post
            with preserved_global_rng():
                set_np_seed(trace.seed)
                env = env_factory(trace)
                check_deadline()
                segment = DemoSegment(
                    index,
                    0,
                    len(trace.states) - 1,
                    trace,
                    dict(definition.initial_bindings),
                    entry_index=0,
                )
                first = reset_segment(env, segment, mode="reset")
                inner = inner_env(env)
                # reset_segment restores physical state and archived aliases;
                # replace all aliases before executing this candidate.
                aliases = dict(definition.initial_bindings)
                inner.symbolic_name_to_box_id = aliases
                if env is not inner and "symbolic_name_to_box_id" in getattr(
                    env, "__dict__", {}
                ):
                    env.symbolic_name_to_box_id = aliases
                row["initial_bindings"] = dict(aliases)
                row["restore_max_error"] = float(
                    np.max(np.abs(first - np.asarray(trace.states[0])))
                )
                observations.append(np.array(first, copy=True))
                snapshots.append(capture(env))
                action_indices.append(0)
                entry_scene = scene_from_obs(
                    first,
                    trace.num_blocks,
                    aliases,
                    include_table="tbl" in aliases,
                    entry_obs=first,
                )
                row["initial_precondition"] = bool(evaluate(task_pre, entry_scene))
                if not row["initial_precondition"]:
                    raise InvalidValidationStart(
                        "Exact start violates the task precondition"
                    )
                recorder = EndpointRecorder(
                    env, segment, first, frame_tolerance_m=budget.frame_tolerance_m
                )
                transfer = _TransferAudit(
                    recorder, first, trace.num_blocks, budget.frame_tolerance_m
                )
                paths = {}
                candidate = _bounded_program(
                    definition.program, budget.max_loop_iterations
                )

                def collect_paths(instructions, prefix=""):
                    for number, instruction in enumerate(instructions):
                        path = f"{prefix}{number}"
                        paths[id(instruction)] = path
                        if isinstance(instruction, While):
                            collect_paths(instruction.body, path + ".")

                collect_paths(candidate.instructions)
                instructions_by_path = {path: key for key, path in paths.items()}
                starts, ends = {}, {}
                for name, contract in block_contracts.items():
                    locations = tuple(contract["paths"])
                    if not locations or any(
                        path not in instructions_by_path for path in locations
                    ):
                        raise ValueError(
                            f"Block {name} has absent executable locations"
                        )
                    starts.setdefault(locations[0], []).append(name)
                    ends.setdefault(locations[-1], []).append(name)
                active_blocks, loop_entries = {}, {}
                row["block_contracts"] = []

                def record_observation(observation, bindings):
                    observation = np.asarray(observation)
                    if not np.isfinite(observation).all():
                        raise ValueError("Nonfinite candidate observation")
                    actual = inner.flatten_observation(inner._get_obs())
                    if observation.shape != np.asarray(actual).shape or not np.allclose(
                        observation, actual, atol=1e-8, rtol=0
                    ):
                        raise ValueError(
                            "Candidate observation disagrees with its simulator snapshot"
                        )
                    transfer.observe(observation, len(actions))
                    for block in active_blocks.values():
                        block["recorder"].observe(observation)
                    if (
                        not observations
                        or action_indices[-1] != len(actions)
                        or snapshots[-1].bindings != bindings
                        or not np.array_equal(observations[-1], observation)
                    ):
                        observations.append(np.array(observation, copy=True))
                        snapshots.append(capture(env))
                        action_indices.append(len(actions))

                def observe(observation, bindings):
                    check_deadline()
                    record_observation(observation, bindings)

                def instruction_event(instruction, phase, observation):
                    check_deadline()
                    path = paths[id(instruction)]
                    if (
                        block_contracts
                        and phase == "start"
                        and isinstance(
                            instruction,
                            (
                                Pick,
                                PickByName,
                                Move,
                                MoveByName,
                                Release,
                                ReleaseByName,
                            ),
                        )
                        and not any(
                            path in contract["paths"]
                            for contract in block_contracts.values()
                        )
                    ):
                        raise ValueError(
                            f"Physical instruction {path} has no learned block contract"
                        )
                    if phase == "start" and isinstance(instruction, While):
                        loop_entries[path] = np.array(observation, copy=True)
                    if phase == "start":
                        for name in starts.get(path, ()):
                            if name in active_blocks:
                                raise ValueError(
                                    f"Block {name} reentered before its endpoint"
                                )
                            enclosing = [
                                loop
                                for loop in loop_entries
                                if path.startswith(loop + ".")
                            ]
                            frozen = (
                                loop_entries[
                                    max(
                                        enclosing,
                                        key=lambda value: len(value.split(".")),
                                    )
                                ]
                                if enclosing
                                else first
                            )
                            block_trace = DemoTrace(
                                (
                                    np.array(frozen, copy=True),
                                    np.array(observation, copy=True),
                                    np.array(observation, copy=True),
                                ),
                                num_blocks=trace.num_blocks,
                                metadata={"holding_by_index": {}},
                            )
                            block_segment = DemoSegment(
                                index,
                                1,
                                2,
                                block_trace,
                                dict(inner.symbolic_name_to_box_id),
                                entry_index=0,
                            )
                            active_blocks[name] = {
                                "segment": block_segment,
                                "recorder": EndpointRecorder(
                                    env,
                                    block_segment,
                                    observation,
                                    frame_tolerance_m=budget.frame_tolerance_m,
                                ),
                                "post": block_contracts[name]["post"],
                            }
                    # Preserve alias-only boundaries too, without simulating.
                    observe(observation, dict(inner.symbolic_name_to_box_id))
                    events.append(
                        {
                            "kind": "instruction_" + phase,
                            "path": path,
                            "index": len(observations) - 1,
                            "bindings": dict(inner.symbolic_name_to_box_id),
                        }
                    )
                    transfer.instruction(instruction, phase, observation, len(actions))
                    for name, block in active_blocks.items():
                        if path in block_contracts[name]["paths"]:
                            block["recorder"].instruction(
                                instruction, phase, observation
                            )
                    if phase == "end":
                        for name in ends.get(path, ()):
                            block = active_blocks.pop(name)
                            result = _completed_block(block, observation, env)
                            row["block_contracts"].append(
                                {
                                    "name": name,
                                    "post": str(block["post"]),
                                    "action_index": len(actions),
                                    "bindings": block["recorder"].contract_bindings
                                    or block["segment"].bindings,
                                    **asdict(result),
                                }
                            )
                            row["reasons"].extend(
                                f"block:{name}:{reason}" for reason in result.reasons
                            )

                original_step = env.step

                def step(action):
                    check_deadline()
                    if len(actions) >= budget.max_actions:
                        raise ValidationBudgetExceeded(
                            "Full-program validation action budget exhausted"
                        )
                    outcome = original_step(action)
                    actions.append(np.array(action, copy=True))
                    # Keep the final completed action even if the next callback
                    # notices an exhausted wall budget.
                    record_observation(
                        inner.flatten_observation(inner._get_obs()),
                        dict(inner.symbolic_name_to_box_id),
                    )
                    return outcome

                env.step = step
                execute_current(
                    candidate,
                    env,
                    first,
                    on_state=observe,
                    on_instruction=instruction_event,
                )
                if active_blocks:
                    raise ValueError(
                        "Execution ended before an active block's contract endpoint"
                    )
                row["program_actions"] = len(actions)

                def checkpoint(observation, number):
                    scene = scene_from_obs(
                        observation,
                        trace.num_blocks,
                        dict(inner.symbolic_name_to_box_id),
                        include_table="tbl" in inner.symbolic_name_to_box_id,
                        entry_obs=first,
                    )
                    logical = bool(evaluate(task_post, scene))
                    physical = bool(inner._is_success(observation))
                    if not logical:
                        row["reasons"].append("logical_postcondition_lost")
                    if not physical:
                        row["reasons"].append("simulator_task_goal_lost")
                    return {
                        "settling_step": number,
                        "logical_post": logical,
                        "simulator_goal": physical,
                    }

                final = observations[-1]
                row["checkpoints"] = [checkpoint(final, 0)]
                for number in range(1, budget.settling_steps + 1):
                    shape = getattr(getattr(env, "action_space", None), "shape", (4,))
                    env.step(np.zeros(shape))
                    final = inner.flatten_observation(inner._get_obs())
                    observe(final, dict(inner.symbolic_name_to_box_id))
                    row["checkpoints"].append(checkpoint(final, number))
                    row["settling_completed"] = number
                row["physical_audit"] = recorder.finish()
                row["reasons"].extend(row["physical_audit"]["reasons"])
                contact = row["physical_audit"]["endpoint"]
                if not contact["known"] or contact["holding"] is not None:
                    row["reasons"].append("final_hand_not_observed_empty")
                if transfer.failures:
                    row["reasons"].append("sampled_transfer_frame")
                check_deadline()
                row["accepted"] = not row["reasons"]
        except (
            ValidationBudgetExceeded,
            LoopBudgetExceeded,
            NoGuardWitness,
            ValueError,
            KeyError,
            TypeError,
            IndexError,
            AttributeError,
            RuntimeError,
        ) as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            row["reasons"].append(
                "initial_task_precondition_failed"
                if isinstance(exc, InvalidValidationStart)
                else (
                    "budget_exhausted"
                    if isinstance(exc, (ValidationBudgetExceeded, LoopBudgetExceeded))
                    else "execution_failed"
                )
            )
        finally:
            if env is not None:
                if recorder is not None and "physical_audit" not in row:
                    row["physical_audit"] = recorder.finish()
                    row["reasons"].extend(row["physical_audit"]["reasons"])
                if original_step is not None:
                    env.step = original_step
                env.close()
            row["actions"] = len(actions)
            row["seconds"] = perf_counter() - began
            if transfer is not None:
                row["max_nonmover_drift_m"] = transfer.maximum_drift
                row["frame_failures"] = transfer.failures
            row["reasons"] = sorted(set(row["reasons"]))
            if directory is not None:
                stem = f"demo-{index}-seed-{trace.seed}"
                if observations:
                    replay = DemoTrace(
                        tuple(observations),
                        tuple(snapshots),
                        (tuple(actions), tuple(action_indices)),
                        seed=trace.seed,
                        task=trace.task,
                        num_blocks=trace.num_blocks,
                        events=tuple(events),
                        metadata={
                            "source": "candidate-training-validation",
                            "fingerprint": fingerprint,
                            "initial_bindings": dict(definition.initial_bindings),
                            "accepted": row["accepted"],
                            "reasons": row["reasons"],
                            "program_actions": row.get("program_actions"),
                            "settling_steps": row["settling_completed"],
                        },
                    )
                    archive = directory / (stem + ".npz")
                    save_traces(archive, [replay])
                    row["replay_archive"] = str(archive)
                (directory / (stem + ".json")).write_text(
                    json.dumps(row, indent=2, allow_nan=False) + "\n"
                )
    result = {
        "accepted": all(row["accepted"] for row in rows),
        "passed": sum(row["accepted"] for row in rows),
        "total": len(rows),
        "program_fingerprint": fingerprint,
        "budget": asdict(budget),
        "block_contracts": {
            name: {"paths": list(contract["paths"]), "post": str(contract["post"])}
            for name, contract in block_contracts.items()
        },
        "seconds": perf_counter() - started,
        "demos": rows,
        "scope": "finite training replay: task/logical goal at every final settling step; primitive contacts/convergence and controller-action sampled transfer frame; no swept collision, unbounded motion, or generalization proof",
    }
    if directory is not None:
        (directory / "summary.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n"
        )
    return result
