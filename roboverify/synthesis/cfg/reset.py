"""Faithful segment starts: full simulator snapshots and deterministic replay."""

from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np

from synthesis.environment.goal_layout import (
    GoalLayout,
    decode_goal_bindings,
    encode_goal_bindings,
)


@dataclass
class Snapshot:
    gt_state: np.ndarray
    arrays: dict
    bindings: dict
    goal_layout: GoalLayout | None = None
    goal_bindings: dict | None = None


def inner_env(env):
    return env.unwrapped if hasattr(env, "unwrapped") else env


def capture(env):
    inner = inner_env(env)
    data = inner.sim.data
    arrays = {
        name: np.array(getattr(data, name), copy=True)
        for name in (
            "ctrl",
            "mocap_pos",
            "mocap_quat",
            "qacc_warmstart",
            "qfrc_applied",
            "xfrc_applied",
            "userdata",
        )
        if getattr(data, name, None) is not None
    }
    return Snapshot(
        inner.get_GT_state().copy(),
        arrays,
        deepcopy(getattr(inner, "symbolic_name_to_box_id", {})),
        getattr(inner, "goal_layout", None),
        (
            encode_goal_bindings(inner.symbolic_name_to_goal)
            if hasattr(inner, "goal_layout")
            else None
        ),
    )


def restore_bindings(env, snapshot):
    inner = inner_env(env)
    inner.symbolic_name_to_box_id = deepcopy(snapshot.bindings)
    if snapshot.goal_layout is not None:
        inner.symbolic_name_to_goal = decode_goal_bindings(
            snapshot.goal_bindings, snapshot.goal_layout
        )


def restore(env, snapshot):
    inner = inner_env(env)
    case = getattr(inner, "case", None)
    goal_env = case in ("RoboVerifyGrid", "RoboVerifyPyramid")
    if goal_env != (snapshot.goal_layout is not None):
        raise ValueError("Snapshot goal layout is incompatible with this environment")
    if goal_env:
        rows = (
            (inner.grid_cols,) * inner.grid_rows
            if case == "RoboVerifyGrid"
            else inner.pyramid_layer_sizes
        )
        if rows != snapshot.goal_layout.row_lengths:
            raise ValueError("Snapshot goal row layout does not match this environment")
        decode_goal_bindings(snapshot.goal_bindings, snapshot.goal_layout)
    inner.set_GT_state(snapshot.gt_state.copy())
    if goal_env and inner.goal_layout != snapshot.goal_layout:
        raise ValueError("Snapshot goal geometry disagrees with its simulator state")
    for name, value in snapshot.arrays.items():
        getattr(inner.sim.data, name)[:] = value
    inner.sim.forward()
    # Forward recomputes derived state; preserve the solver's recorded warm start.
    # It also renormalizes mocap quaternions, which can move a recorded unit
    # quaternion by one ulp; restore the recorded value so snapshots round-trip.
    for name in ("qacc_warmstart", "mocap_quat"):
        if name in snapshot.arrays:
            getattr(inner.sim.data, name)[:] = snapshot.arrays[name]
    restore_bindings(env, snapshot)
    return inner.flatten_observation(inner._get_obs())


@dataclass
class Recording:
    snapshots: list = field(default_factory=list)
    action_indices: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    states: list = field(default_factory=list)
    events: list = field(default_factory=list)


def collect_recording(
    program,
    env,
    *,
    recording=None,
    initial_observation=None,
    max_loop_iterations=None,
    on_frame=None
):
    """Record the same execution used for learning and optional video rendering.

    A caller-supplied Recording retains the partial execution if the program raises.
    Frames are captured initially and after actions; rendering never adds actions.
    """
    recording = Recording() if recording is None else recording
    original_step = env.step
    invocations = {}

    def step(action):
        result = original_step(action)
        recording.actions.append(np.array(action, copy=True))
        if on_frame is not None:
            on_frame(env)
        return result

    def on_state(obs):
        actual = inner_env(env).flatten_observation(inner_env(env)._get_obs())
        if not np.allclose(obs, actual, atol=1e-8, rtol=0):
            raise ValueError(
                "Recorded observation does not match its simulator snapshot"
            )
        recording.states.append(np.array(obs, copy=True))
        recording.snapshots.append(capture(env))
        recording.action_indices.append(len(recording.actions))
        if len(recording.states) == 1 and on_frame is not None:
            on_frame(env)

    def on_event(event):
        if event["kind"] == "loop_enter":
            invocations[event["path"]] = invocations.get(event["path"], -1) + 1
        event["invocation"] = invocations.get(event["path"])
        recording.events.append(deepcopy(event))

    env.step = step
    try:
        program.eval(
            env,
            on_state=on_state,
            on_event=on_event,
            initial_observation=initial_observation,
            max_loop_iterations=max_loop_iterations,
        )
    finally:
        env.step = original_step
    return tuple(recording.states), recording


def reset_segment(env, segment, *, mode="replay"):
    """Default to replay; callers may opt into measured exact reset."""
    trace = segment.trace
    if mode not in ("reset", "replay"):
        raise ValueError("Unknown segment reset mode")
    if mode == "reset" and trace.snapshots:
        observation = restore(env, trace.snapshots[segment.t_start])
    elif trace.snapshots and trace.actions:
        observation = restore(env, trace.snapshots[0])
        actions, indices = trace.actions
        for action in actions[: indices[segment.t_start]]:
            observation = env.step(np.array(action, copy=True))[0]
        # Symbolic assignments are not actions; reinstate recorded aliases.
        restore_bindings(env, trace.snapshots[segment.t_start])
    else:
        raise ValueError(
            "Full simulator snapshots and action recordings are required; recollect demonstrations"
        )
    inner_env(env).symbolic_name_to_box_id.update(segment.bindings)
    if segment.goal_bindings is not None:
        inner = inner_env(env)
        bindings = encode_goal_bindings(inner.symbolic_name_to_goal)
        bindings.update(segment.goal_bindings)
        inner.symbolic_name_to_goal = decode_goal_bindings(bindings, inner.goal_layout)
    if not np.allclose(observation, trace.states[segment.t_start], atol=1e-8, rtol=0):
        raise ValueError(
            "Segment replay/reset did not reproduce its recorded observation"
        )
    return observation
