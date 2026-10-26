"""One task-success definition for RoboVerify and CaP, with optional final settling.

``post_settling_steps`` selects the criterion. With 0 (the default) the
postcondition is evaluated on the state where the execution ended
(``task-spec-v1``). A positive value holds the gripper still for that many
unrecorded steps after the execution ends and decides success on the settled
state only (``task-spec-v2``); the end-of-execution postcondition and the
environment's physical evaluator remain diagnostics.

Tower tasks evaluate the shared ``task_spec`` formulas. Grid and Pyramid
evaluate goal occupancy (20 mm, 3D): goals empty and blocks pairwise Scattered at
entry, every goal occupied at the end. The Release-event checks of
``validate_goal_trace`` are a demonstration-quality filter, not part of this
criterion.

Settling steps never enter a recording: callers settle after recording has
stopped, or on a fresh environment restored from a trace's last snapshot.
"""

from dataclasses import asdict, dataclass, field

import numpy as np

SETTLING_STEPS = 50
CRITERIA = {0: "task-spec-v1", "settled": "task-spec-v2"}
GOAL_TASKS = ("grid", "pyramid")
TABLE_TASKS = ("unstack", "reverse", "restack")


def criterion(post_settling_steps):
    return CRITERIA[0] if post_settling_steps == 0 else CRITERIA["settled"]


def check_steps(post_settling_steps):
    if type(post_settling_steps) is not int or post_settling_steps < 0:
        raise ValueError("post_settling_steps must be a nonnegative integer")
    return post_settling_steps


def settle(env, steps=SETTLING_STEPS, observation=None):
    """Hold the gripper at its current position for ``steps`` steps.

    Each step commands the arm toward the gripper position at the start of
    settling, with gripper command 0.0. The finger actuators are position
    actuators whose target the environment sets to the current finger opening
    plus the command, so 0.0 holds the opening at the start of every step.
    Returns the observation after the last step.
    """
    from synthesis.api.control import get_move_action
    from synthesis.cfg.reset import inner_env

    inner = inner_env(env)
    if observation is None:
        observation = inner.flatten_observation(inner._get_obs())
    target = np.asarray(observation[:3]).copy()
    for _ in range(steps):
        env.step(get_move_action(observation, target, close_gripper=False))
        observation = inner.flatten_observation(inner._get_obs())
    return observation


@dataclass
class TaskSuccess:
    criterion: str
    post_settling_steps: int
    completed: bool
    unconverged: int
    pre: bool | None = None
    post_at_end: bool | None = None
    post_settled: bool | None = None
    success: bool = False
    environment_success_at_end: bool | None = None
    environment_success_settled: bool | None = None
    positions: list = field(default_factory=list)
    goal_occupied: list | None = None
    issues: list = field(default_factory=list)

    def __bool__(self):
        return self.success

    def as_dict(self):
        return asdict(self)


def _positions(observation, num_blocks):
    return np.array(
        [observation[10 + 12 * i : 13 + 12 * i] for i in range(num_blocks)],
        dtype=float,
    )


def task_conditions(inner, task, observation, entry_observation, bindings):
    """(pre, post) of ``task`` with ``observation`` as the final state."""
    from synthesis.cfg.tasks import task_spec
    from synthesis.environment.goal_layout import occupancy
    from synthesis.predicates.scene import evaluate, scene_from_obs
    from synthesis.util.on import scattered_implementation

    if task in GOAL_TASKS:
        goals = inner.goal_layout.positions
        entry = _positions(entry_observation, inner.num_blocks)
        pre = not occupancy(entry, goals).any() and all(
            scattered_implementation(entry[i], entry[j])
            for i in range(len(entry))
            for j in range(i)
        )
        final = occupancy(_positions(observation, inner.num_blocks), goals)
        return bool(pre), bool(final.any(axis=0).all())
    pre, post = task_spec(task)
    bindings = dict(bindings)
    include_table = task in TABLE_TASKS

    def scene(current):
        return scene_from_obs(
            current,
            inner.num_blocks,
            bindings,
            include_table=include_table,
            entry_obs=entry_observation,
        )

    return bool(evaluate(pre, scene(entry_observation))), bool(
        evaluate(post, scene(observation))
    )


def check_task_success(
    env,
    task,
    *,
    entry_observation,
    bindings,
    completed,
    unconverged,
    post_settling_steps=0,
):
    """Score an execution that has just ended in ``env``.

    ``bindings`` are the input handles the program or policy was given (b0 and,
    for table tasks, tbl); ``entry_observation`` is the execution's first
    state and supplies the frozen entry geometry. With a positive
    ``post_settling_steps`` the environment is settled in place.
    """
    from synthesis.cfg.reset import inner_env

    steps = check_steps(post_settling_steps)
    inner = inner_env(env)
    result = TaskSuccess(criterion(steps), steps, bool(completed), int(unconverged))
    observation = inner.flatten_observation(inner._get_obs())
    try:
        result.pre, result.post_at_end = task_conditions(
            inner, task, observation, entry_observation, bindings
        )
        result.environment_success_at_end = bool(inner._is_success(observation))
        if steps:
            observation = settle(env, steps, observation)
            _, result.post_settled = task_conditions(
                inner, task, observation, entry_observation, bindings
            )
            result.environment_success_settled = bool(inner._is_success(observation))
    except (KeyError, ValueError, TypeError) as exc:
        result.issues.append(f"{type(exc).__name__}: {exc}")
    result.positions = _positions(observation, inner.num_blocks).tolist()
    if task in GOAL_TASKS:
        from synthesis.environment.goal_layout import occupancy

        result.goal_occupied = (
            occupancy(np.array(result.positions), inner.goal_layout.positions)
            .any(axis=0)
            .tolist()
        )
    post = result.post_settled if steps else result.post_at_end
    if not result.completed:
        result.issues.append("Execution did not complete")
    if result.unconverged:
        result.issues.append(f"{result.unconverged} primitive(s) did not converge")
    if result.pre is False:
        result.issues.append("Initial task condition is false")
    if post is False:
        result.issues.append(
            "Settled task condition is false"
            if steps
            else "Final task condition is false"
        )
    result.success = (
        bool(result.completed and not result.unconverged and result.pre and post)
        and not result.issues
    )
    return result


def trace_environment(trace):
    """A fresh environment matching a recorded trace's task and layout."""
    from synthesis.cfg.goal_collection import make_goal_env
    from synthesis.mcmc.synthesis import make_roboverify_env

    config = dict(trace.metadata.get("environment") or {})
    if trace.task in GOAL_TASKS:
        return make_goal_env(trace.task, config)
    return make_roboverify_env(trace.task, num_blocks=trace.num_blocks, **config)


def unconverged_primitives(events):
    return sum(
        1
        for event in events
        if event.get("kind") == "instruction_end"
        and not event.get("control", {}).get("converged", True)
    )


def check_recorded_task_success(
    trace,
    post_settling_steps=SETTLING_STEPS,
    *,
    environment=None,
    bindings=None,
    completed=None,
):
    """The same check for a recorded trace, from its last full snapshot.

    The snapshot is restored into ``environment`` (or a fresh one, closed
    afterward); the trace itself is not modified. Bindings default to the
    trace's ``initial_bindings`` and completion to its recorded status.
    """
    from synthesis.cfg.reset import restore
    from synthesis.util.on import using_higher_tolerance

    if not trace.states or not trace.snapshots:
        raise ValueError("Settled revalidation requires full simulator snapshots")
    if completed is None:
        completed = trace.metadata.get("status") in ("completed", "valid", "invalid")
    if bindings is None:
        bindings = trace.metadata["initial_bindings"]
    env = trace_environment(trace) if environment is None else environment
    try:
        with using_higher_tolerance(trace.metadata.get("higher_tolerance", 0.001)):
            final = restore(env, trace.snapshots[-1])
            if not np.allclose(final, trace.states[-1], atol=1e-8, rtol=0):
                raise ValueError("Last snapshot does not reproduce the last state")
            return check_task_success(
                env,
                trace.task,
                entry_observation=np.asarray(trace.states[0]),
                bindings=bindings,
                completed=completed,
                unconverged=unconverged_primitives(trace.events),
                post_settling_steps=post_settling_steps,
            )
    finally:
        if environment is None:
            env.close()
