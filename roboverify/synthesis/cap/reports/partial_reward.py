"""Partial task reward of one CaP rollout, for Fig. 11-style progress curves.

The paper (Section 6.1, below Fig. 11) assigns the reward uniformly per block:
stacked on the base block b0 (Stack), placed on the table (Unstack), reversed
(Reverse), 0.5 to each of the unstack and stack subtasks (ReStack), and at a
goal location (Grid, Pyramid). The RoboVerify environments only report binary
success, so ``partial-reward-v1`` makes these sentences concrete with the
shared geometric predicates (``util/on.py``) and goal occupancy
(``environment/goal_layout.py``):

- Stack: blocks x != b0 with ON*(x, b0), over N - 1.
- Unstack: blocks x != b0 resting on the table (no block below x and the
  centre within half a block of the resting height), over N - 1. b0 starts
  on the table and earns nothing.
- Reverse: with the entry tower t1 (top), ..., tN = b0 (bottom), the largest
  k such that t1 rests on the table and the blocks below each ti (i <= k) are
  exactly t1, ..., t(i-1); reward k / N.
- ReStack: 0.5 x (unstacked share) + 0.5 x (stacked share). The unstack
  subtask covers U, b0 and the blocks above it in the entry tower (what the
  supplied program parks); x in U counts once some recorded or settled state
  has x above none of the blocks it was above at entry. The stack subtask
  counts, when b0 rests on the table, the blocks x != b0 with ON*(x, b0),
  over N - 1.
- Grid, Pyramid: occupied goals over goals (the checker's ``goal_occupied``).

A rollout's reward is read on the state that decides success: after the 50
unrecorded settling steps (``task_success.positions``). It is a reporting
metric only; no reward reaches the model or the recovery loop.
"""

import json
from pathlib import Path

import numpy as np

from synthesis.util.on import BLOCK_LENGTH, on_star_implementation

VERSION = "partial-reward-v1"


def block_positions(observations, num_blocks):
    """(..., N, 3) block centres from flattened observations (``10 + 12 i``)."""
    observations = np.asarray(observations, dtype=float)
    return np.stack(
        [observations[..., 10 + 12 * i : 13 + 12 * i] for i in range(num_blocks)],
        axis=-2,
    )


def below(positions, x):
    """Blocks y != x with ON*(x, y)."""
    return {
        y
        for y in range(len(positions))
        if y != x and on_star_implementation(positions[x], positions[y])
    }


def resting_on_table(positions, x, rest_height):
    return not below(positions, x) and positions[x][2] <= rest_height + BLOCK_LENGTH / 2


def stacked_on(positions, b0):
    return sum(
        x != b0 and on_star_implementation(positions[x], positions[b0])
        for x in range(len(positions))
    )


def stack(entry, final, b0, rest_height, states=()):
    return stacked_on(final, b0) / (len(final) - 1)


def unstack(entry, final, b0, rest_height, states=()):
    placed = sum(
        x != b0 and resting_on_table(final, x, rest_height) for x in range(len(final))
    )
    return placed / (len(final) - 1)


def reverse(entry, final, b0, rest_height, states=()):
    tower = sorted(range(len(entry)), key=lambda x: -entry[x][2])
    k = 0
    for i, x in enumerate(tower):
        if below(final, x) != set(tower[:i]):
            break
        if i == 0 and not resting_on_table(final, x, rest_height):
            break
        k += 1
    return k / len(entry)


def restack(entry, final, b0, rest_height, states=()):
    column = [
        x for x in range(len(entry)) if on_star_implementation(entry[x], entry[b0])
    ]
    supports = {x: below(entry, x) for x in column}
    visited = list(states) + [final]
    unstacked = sum(
        any(not (below(state, x) & supports[x]) for state in visited) for x in column
    )
    stacked = (
        stacked_on(final, b0) / (len(final) - 1)
        if resting_on_table(final, b0, rest_height)
        else 0.0
    )
    return 0.5 * unstacked / len(column) + 0.5 * stacked


TOWER_REWARDS = {
    "stack": stack,
    "unstack": unstack,
    "reverse": reverse,
    "restack": restack,
}


def rollout_reward(task, entry, final, b0=None, states=(), goal_occupied=None):
    """Partial reward of one rollout from block positions ((N, 3) arrays)."""
    if task in ("grid", "pyramid"):
        return float(np.mean(goal_occupied))
    entry, final = np.asarray(entry, float), np.asarray(final, float)
    # Every task's entry scene has a block resting on the table.
    rest_height = float(entry[:, 2].min())
    return float(TOWER_REWARDS[task](entry, final, b0, rest_height, states))


def _load(path):
    with np.load(path, allow_pickle=True) as data:
        metadata = json.loads(str(data["metadata"]))
        return data["d0_states"], metadata["traces"][0]


def attempt_reward(task, attempt):
    """Partial reward of a logged attempt (one recovery round), or None when the
    round produced no settled rollout (e.g. the program never passed the check).

    Batch summaries keep only booleans; the checker's positions are in the
    round's own ``result.json``."""
    outcome = attempt.get("task_success")
    if not isinstance(outcome, dict):
        result = Path(attempt["run_dir"], "result.json")
        outcome = json.loads(result.read_text()).get("task_success")
    if not isinstance(outcome, dict) or not outcome.get("positions"):
        return None
    final = np.asarray(outcome["positions"], float)
    if task in ("grid", "pyramid"):
        return rollout_reward(task, None, final, goal_occupied=outcome["goal_occupied"])
    trial = Path(attempt["run_dir"], attempt.get("artifacts_path") or "")
    states, trace = _load(trial / "initial_state.npz")
    entry = block_positions(states[0], len(final))
    b0 = trace["bindings"][0]["b0"]
    recorded = ()
    if task == "restack":
        rollouts = sorted(trial.glob("execution/rollout_*/trajectory.npz"))
        if rollouts:
            recorded = block_positions(_load(rollouts[-1])[0], len(final))
    return rollout_reward(task, entry, final, b0, recorded)
