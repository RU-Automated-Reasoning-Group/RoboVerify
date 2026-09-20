"""Faithful segment starts: full simulator snapshots and deterministic replay."""

from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Snapshot:
    gt_state: np.ndarray
    arrays: dict
    bindings: dict


def inner_env(env):
    return env.unwrapped if hasattr(env, "unwrapped") else env


def capture(env):
    inner = inner_env(env)
    data = inner.sim.data
    arrays = {name: np.array(getattr(data,name),copy=True) for name in
              ("ctrl","mocap_pos","mocap_quat","qacc_warmstart","qfrc_applied","xfrc_applied","userdata")
              if getattr(data,name,None) is not None}
    return Snapshot(inner.get_GT_state().copy(), arrays, deepcopy(getattr(inner,"symbolic_name_to_box_id",{})))


def restore(env, snapshot):
    inner = inner_env(env)
    inner.set_GT_state(snapshot.gt_state.copy())
    for name, value in snapshot.arrays.items():
        getattr(inner.sim.data,name)[:] = value
    inner.sim.forward()
    # Forward recomputes derived state; preserve the solver's recorded warm start.
    if "qacc_warmstart" in snapshot.arrays:
        inner.sim.data.qacc_warmstart[:] = snapshot.arrays["qacc_warmstart"]
    inner.symbolic_name_to_box_id = deepcopy(snapshot.bindings)
    return inner.flatten_observation(inner._get_obs())


@dataclass
class Recording:
    snapshots: list = field(default_factory=list)
    action_indices: list = field(default_factory=list)
    actions: list = field(default_factory=list)


def collect_recording(program, env):
    """Capture at every observation append, including controller-internal steps."""
    recording = Recording()
    original_step = env.step
    def step(action):
        recording.actions.append(np.array(action,copy=True))
        return original_step(action)
    def on_state(obs):
        recording.snapshots.append(capture(env))
        recording.action_indices.append(len(recording.actions))
    env.step = step
    try:
        states = program.eval(env, on_state=on_state)
    finally:
        env.step = original_step
    return tuple(states), recording


def reset_segment(env, segment, *, mode="replay"):
    """Default to replay; callers may opt into measured exact reset."""
    trace = segment.trace
    if mode not in ("reset","replay"):
        raise ValueError("Unknown segment reset mode")
    if mode == "reset" and trace.snapshots:
        observation = restore(env, trace.snapshots[segment.t_start])
    elif trace.snapshots and trace.actions:
        observation = restore(env, trace.snapshots[0])
        actions, indices = trace.actions
        for action in actions[:indices[segment.t_start]]:
            observation = env.step(np.array(action,copy=True))[0]
        # Symbolic assignments are not actions; reinstate recorded aliases.
        inner_env(env).symbolic_name_to_box_id = dict(trace.snapshots[segment.t_start].bindings)
    elif trace.replay is not None:
        observation = trace.replay(env, segment.t_start)
    else:
        raise ValueError("Legacy observation-only demos require their deterministic replay source")
    inner_env(env).symbolic_name_to_box_id.update(segment.bindings)
    if not np.allclose(observation, trace.states[segment.t_start], atol=1e-8, rtol=0):
        raise ValueError("Segment replay/reset did not reproduce its recorded observation")
    return observation


def legacy_replay(program, seed):
    """Replay an observation-only demo when its generating program/seed are known."""
    def replay(env, target):
        from synthesis.mcmc.synthesis import preserved_global_rng, set_np_seed
        class Reached(Exception): pass
        seen, result = 0, None
        def observe(obs):
            nonlocal seen, result
            if seen == target:
                result = np.array(obs,copy=True)
                raise Reached()
            seen += 1
        with preserved_global_rng():
            set_np_seed(seed)
            try: program.eval(env,on_state=observe)
            except Reached: return result
        raise ValueError("Replay ended before segment start")
    return replay
