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


# Upstream collection settles every task after a fresh reset (50 steps).
# Snapshots are never settled again.
SETTLING_TASKS = frozenset(TASK_INSTRUCTIONS)


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
    if initial_snapshot is not None or config.task not in SETTLING_TASKS:
        return 0
    return config.initial_settling_steps


def prepare_environment(config, initial_snapshot=None):
    check_task(config.task)
    try:
        import numpy as np

        from synthesis.cfg.reset import restore
        from synthesis.cfg.task_success import settle
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
            settle(env, settling_steps(config), env.reset()[0])
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
SUCCESS_CRITERION = "task-spec-v1"  # post_settling_steps 0


FINAL_RELEASE_VERSION = "final-release-v1"
GRIPPER_GEOMS = (
    "robot0:gripper_link",
    "robot0:l_gripper_finger_link",
    "robot0:r_gripper_finger_link",
)
HOLD_DISTANCE = 0.03  # demonstrations.HOLD_DISTANCE


def success_criterion(config):
    """task-spec-v1 without final settling, task-spec-v2 with it, plus the
    CaP-only release check of Revision 7.4 when enabled."""
    from synthesis.cfg.task_success import criterion

    name = criterion(config.post_settling_steps)
    if getattr(config, "final_release_check", False):
        name += "+" + FINAL_RELEASE_VERSION
    return name


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
    from synthesis.cfg.reset import inner_env
    from synthesis.cfg.task_success import SETTLING_STEPS, settle

    observation = settle(env, SETTLING_STEPS if steps is None else steps)
    return bool(inner_env(env)._is_success(observation))


def settled_task_outcome(env, config, entry_obs, adapter, completed):
    """task-spec-v2 fields from the shared checker (cfg/task_success.py).

    The checker settles ``env`` for ``config.post_settling_steps`` steps; call
    it only after recording has stopped. Bindings are the inputs the adapter
    gave the policy. ``task_spec_success`` is the settled result; the
    end-of-rollout postcondition and both physical evaluations are diagnostics.
    """
    from synthesis.cfg.task_success import check_task_success
    from synthesis.util.on import using_higher_tolerance

    with using_higher_tolerance(config.higher_tolerance):
        result = check_task_success(
            env,
            config.task,
            entry_observation=entry_obs,
            bindings=adapter.input_bindings,
            completed=completed,
            unconverged=adapter.unconverged_primitives,
            post_settling_steps=config.post_settling_steps,
        )
    return {
        "task_spec_success": result.success,
        "task_spec_precondition": result.pre,
        "task_spec_postcondition": result.post_at_end,
        "task_spec_settled_postcondition": result.post_settled,
        "settled_environment_success": result.environment_success_settled,
        "task_success": result.as_dict(),
    }


def gripper_block_contacts(inner):
    """{physical block id: sorted gripper geom names} over active contacts."""
    import re

    model, data = inner.sim.model, inner.sim.data
    touching = {}
    for contact in data.contact[: data.ncon]:
        names = (model.geom_id2name(contact.geom1), model.geom_id2name(contact.geom2))
        for gripper, other in (names, names[::-1]):
            block = re.fullmatch(r"object(\d+)", other or "")
            if gripper in GRIPPER_GEOMS and block:
                touching.setdefault(int(block.group(1)), set()).add(gripper)
    return {block: sorted(geoms) for block, geoms in sorted(touching.items())}


def final_release_outcome(env, adapter):
    """final-release-v1: no block may touch the gripper when the rollout ends.

    Evaluate after recording has stopped and before final settling; no
    simulation step is taken. A block in contact with the gripper link or a
    finger is unreleased. Closed fingers, blocks near a closed gripper and the
    adapter's Pick/Release ledger are diagnostics only.
    """
    import numpy as np

    from synthesis.api.control import DEFAULT_CONTROL
    from synthesis.cfg.reset import inner_env

    inner = inner_env(env)
    observation = inner.flatten_observation(inner._get_obs())
    contacts = gripper_block_contacts(inner)
    closed = float(np.sum(observation[3:5])) <= (
        DEFAULT_CONTROL.gripper_threshold + DEFAULT_CONTROL.gripper_tolerance
    )
    near = []
    if closed:
        for i in range(inner.num_blocks):
            block = observation[10 + 12 * i : 13 + 12 * i]
            if np.linalg.norm(block - observation[:3]) <= HOLD_DISTANCE:
                near.append(i)
    ledger = getattr(adapter, "ledger_held", None)
    return {
        "version": FINAL_RELEASE_VERSION,
        "released": not contacts,
        "unreleased_blocks": list(contacts),
        "contacts": [
            {"block": block, "geoms": geoms} for block, geoms in contacts.items()
        ],
        "fingers_closed": closed,
        "near_closed_gripper": near,
        "ledger_held": None if ledger is None else str(ledger),
        "issues": [
            f"block_{block} is still in contact with the gripper at the end of"
            " the rollout"
            for block in contacts
        ],
    }
