"""Task construction and independent host-only scoring, imported lazily."""

from .contracts import (
    GOAL_TASKS,
    TABLE_TASKS,
    BudgetExceeded,
    CapError,
    EnvironmentUnavailable,
)

TASK_INSTRUCTIONS = {
    "stack": "Stack all blocks into a single tower rooted at the input block b0.",
    "unstack": "Dismantle the initial tower so that every block rests on the table.",
    "reverse": "Reverse the complete order of the initial tower, so its original base b0 becomes its top.",
    "restack": "Rearrange all blocks from their initial towers into one tower rooted at b0.",
    "grid": "Place all blocks in the specified grid goal layout.",
    "pyramid": "Place all blocks in the specified pyramid goal layout.",
}


INPUT_CONTEXT = "b0 is the input block."
TABLE_CONTEXT = (
    " tbl is the table. Use it only as a Move/Release reference: X(tbl) and Y(tbl)"
    " are the table frame origin, and Z(tbl) is half a block below the table"
    " surface, so Z(tbl) + 0.05 is the centre height of a block resting on the"
    " table. tbl cannot be picked, has no get_block_position, and is never"
    " returned by queries. In predicates tbl relates only to itself: ON_star,"
    " Higher and eq hold for (tbl, tbl); Scattered never holds for tbl."
)


GOAL_CONTEXT = (
    "h0 is the head goal of the goal layout and null is the empty goal; neither"
    " can be reassigned. Goals form rows: MoveRight(g) advances goal variable g"
    " to the next goal in its row (null after the last), and MoveDown(g) advances"
    " a row head to the next row's head (null after the last row). There is no"
    " input block b0 and no tbl handle."
)


def task_context(task):
    """Model-facing description of the bound input handles."""
    if task in GOAL_TASKS:
        return GOAL_CONTEXT
    return INPUT_CONTEXT + (TABLE_CONTEXT if task in TABLE_TASKS else "")


def check_task(task):
    if task not in TASK_INSTRUCTIONS:
        raise ValueError(f"Unknown task: {task}")


# Upstream collection settles these tasks after a fresh reset; Unstack and
# Reverse record from the reset state. Snapshots are never settled again.
SETTLING_TASKS = frozenset({"stack", "restack", "grid", "pyramid"})


def environment_options(config):
    """Resolved layout options for the task, as upstream collection records them."""
    if config.task == "restack":
        from synthesis.environment.restack_reset import resolve_tower_blocks

        return {
            "initial_tower_blocks": resolve_tower_blocks(
                config.num_blocks, config.initial_tower_blocks
            )
        }
    if config.task in {"grid", "pyramid"}:
        from synthesis.cfg.goal_collection import resolve_goal_configuration

        options = {
            name: getattr(config, name)
            for name in ("grid_rows", "grid_cols", "bottom_row_blocks")
            if getattr(config, name) is not None
        }
        return resolve_goal_configuration(
            config.task, num_blocks=config.num_blocks, **options
        )[1]
    return {}


def settling_steps(config, initial_snapshot=None):
    from synthesis.cfg.collection import STACK_SETTLING_STEPS

    if initial_snapshot is not None or config.task not in SETTLING_TASKS:
        return 0
    return STACK_SETTLING_STEPS


def prepare_environment(config, initial_snapshot=None):
    check_task(config.task)
    try:
        import numpy as np

        from synthesis.api.control import get_move_action
        from synthesis.cfg.reset import restore
        from synthesis.mcmc.synthesis import make_roboverify_env
    except BudgetExceeded:
        raise
    except Exception as exc:
        raise EnvironmentUnavailable(
            f"Simulator prerequisites are unavailable: {type(exc).__name__}: {exc}"
        ) from exc

    options = environment_options(config)
    previous_rng = np.random.get_state()
    env = None
    try:
        np.random.seed(config.seed)
        if config.task in GOAL_TASKS:
            from synthesis.cfg.goal_collection import make_goal_env

            env = make_goal_env(config.task, options)
        else:
            env = make_roboverify_env(
                config.task, num_blocks=config.num_blocks, **options
            )
        if initial_snapshot is not None:
            restore(env, initial_snapshot)
        else:
            obs = env.reset()[0]
            target = obs[:3].copy()
            for _ in range(settling_steps(config)):
                obs = env.step(get_move_action(obs, target, close_gripper=False))[0]
        return env
    except BaseException:
        if env is not None:
            env.close()
        raise
    finally:
        np.random.set_state(previous_rng)


# Shared task specifications exist for these tasks; the formulas of the last
# three quantify over the relational table, as in upstream collection.
TASK_SPEC_TASKS = frozenset({"stack", "unstack", "reverse", "restack"})
SUCCESS_CRITERION = "task-spec-v1"


def _task_scene(inner, task, obs, entry_obs):
    from synthesis.predicates.scene import scene_from_obs

    return scene_from_obs(
        obs,
        inner.num_blocks,
        inner.symbolic_name_to_box_id,
        include_table=task in TABLE_TASKS,
        entry_obs=entry_obs,
    )


def validate_start(env, task, entry_obs):
    """Reject a start that violates the shared precondition before any model call.

    Reverse must also be the evaluator's canonical tower, and Restack's b0 must
    be inside its source tower; both are environment invariants beyond the spec.
    """
    from synthesis.cfg.reset import inner_env
    from synthesis.cfg.tasks import task_spec
    from synthesis.predicates.scene import evaluate
    from synthesis.predicates.term import (
        atom,
        conjunction,
        disjunction,
        exists,
        negate,
        ref,
    )

    inner = inner_env(env)
    if task in GOAL_TASKS:
        if not goal_start_holds(inner, entry_obs):
            raise CapError(
                "Initial state violates the goal-task precondition: goals must be"
                " empty and blocks pairwise Scattered"
            )
        return
    if task not in TASK_SPEC_TASKS:
        return
    scene = _task_scene(inner, task, entry_obs, entry_obs)
    pre, _ = task_spec(task)
    if not evaluate(pre, scene):
        raise CapError("Initial state violates the shared task precondition")
    if task == "reverse":
        positions = [
            entry_obs[10 + 12 * i : 13 + 12 * i] for i in range(inner.num_blocks)
        ]
        if (
            inner._roboverify_tower_order_from_positions(positions)
            != inner._roboverify_canonical_tower_order()
        ):
            raise CapError(
                "Reverse snapshot is not the evaluator's canonical initial tower"
            )
    elif task == "restack":
        x, b0 = ref("x"), ref("b0")
        in_tower = exists(
            ["x"],
            conjunction(
                negate(atom("eq", x, b0)),
                negate(atom("eq", x, ref("tbl"))),
                disjunction(atom("ON_star", x, b0), atom("ON_star", b0, x)),
            ),
        )
        if not evaluate(in_tower, scene):
            raise CapError("Restack b0 is not inside a source tower")


def _goal_occupancy(inner, obs):
    import numpy as np

    from synthesis.environment.goal_layout import occupancy

    blocks = np.array([obs[10 + 12 * i : 13 + 12 * i] for i in range(inner.num_blocks)])
    return blocks, occupancy(blocks, inner.goal_layout.positions)


def goal_start_holds(inner, obs):
    """goal-occupancy-v1 precondition, as in validate_goal_trace."""
    from synthesis.util.on import scattered_implementation

    blocks, occupied = _goal_occupancy(inner, obs)
    return not occupied.any() and all(
        scattered_implementation(blocks[i], blocks[j])
        for i in range(len(blocks))
        for j in range(i)
    )


def goal_terminal_holds(inner, obs):
    """goal-occupancy-v1 postcondition: every goal occupied within 20 mm (3D)."""
    _, occupied = _goal_occupancy(inner, obs)
    return bool(occupied.any(axis=0).all())


def evaluate_terminal(env, task, entry_obs):
    """Score the final state with the physical evaluator and the shared task_spec.

    ``dsl_postcondition`` keeps its Stack/Unstack-only definition for comparison
    with earlier results; ``task_spec_*`` covers every task with a shared spec.
    """
    from synthesis.cfg.reset import inner_env
    from synthesis.cfg.tasks import task_spec
    from synthesis.predicates.scene import evaluate

    inner = inner_env(env)
    observation = inner.flatten_observation(inner._get_obs())
    result = {
        "environment_success": bool(inner._is_success(observation)),
        "dsl_postcondition": None,
        "task_spec_precondition": None,
        "task_spec_postcondition": None,
    }
    if task in GOAL_TASKS:
        # No task_spec formula exists; goal-occupancy-v1 replaces it. The
        # event-level checks of validate_goal_trace are not applied (Revision 5).
        result["task_spec_precondition"] = goal_start_holds(inner, entry_obs)
        result["task_spec_postcondition"] = goal_terminal_holds(inner, observation)
    elif task in TASK_SPEC_TASKS:
        pre, post = task_spec(task)
        result["task_spec_precondition"] = bool(
            evaluate(pre, _task_scene(inner, task, entry_obs, entry_obs))
        )
        result["task_spec_postcondition"] = bool(
            evaluate(post, _task_scene(inner, task, observation, entry_obs))
        )
        if task in {"stack", "unstack"}:
            result["dsl_postcondition"] = result["task_spec_postcondition"]
    return result


def settled_environment_success(env, steps=None):
    """Re-check physical success after holding the gripper still, as audit_demos does.

    Call only after recording has stopped: these steps change the simulator
    state and are not part of the policy rollout.
    """
    from synthesis.api.control import get_move_action
    from synthesis.cfg.collection import STACK_SETTLING_STEPS
    from synthesis.cfg.reset import inner_env

    inner = inner_env(env)
    observation = inner.flatten_observation(inner._get_obs())
    target = observation[:3].copy()
    for _ in range(STACK_SETTLING_STEPS if steps is None else steps):
        observation = env.step(
            get_move_action(observation, target, close_gripper=False)
        )[0]
    return bool(inner._is_success(observation))
