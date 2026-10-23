"""Task construction and independent host-only scoring, imported lazily."""

from .contracts import AdapterGap, BudgetExceeded, CapError, EnvironmentUnavailable

TASK_INSTRUCTIONS = {
    "stack": "Stack all blocks into a single tower rooted at the input block b0.",
    "unstack": "Dismantle the initial tower so that every block rests on the table.",
    "reverse": "Reverse the complete order of the initial tower, so its original base b0 becomes its top.",
    "restack": "Rearrange all blocks from their initial towers into one tower rooted at b0.",
    "grid": "Place all blocks in the specified grid goal layout.",
    "pyramid": "Place all blocks in the specified pyramid goal layout.",
}


def check_task(task):
    if task not in TASK_INSTRUCTIONS:
        raise ValueError(f"Unknown task: {task}")
    if task in {"grid", "pyramid"}:
        raise AdapterGap(
            f"{task} needs the shared Goal traversal/occupancy and physical reference interface"
        )


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

    # Upstream replaced PartialStack with a Restack environment (2026-10-23).
    task = config.task
    previous_rng = np.random.get_state()
    env = None
    try:
        np.random.seed(config.seed)
        env = make_roboverify_env(task, num_blocks=config.num_blocks)
        if initial_snapshot is not None:
            restore(env, initial_snapshot)
        else:
            obs = env.reset()[0]
            if config.task == "stack":
                target = obs[:3].copy()
                for _ in range(50):
                    obs = env.step(get_move_action(obs, target, close_gripper=False))[0]
        return env
    except BaseException:
        if env is not None:
            env.close()
        raise
    finally:
        np.random.set_state(previous_rng)


def validate_start(env, task, entry_obs):
    from synthesis.cfg.reset import inner_env
    from synthesis.cfg.tasks import task_spec
    from synthesis.predicates.scene import evaluate, scene_from_obs

    inner = inner_env(env)
    if task in {"stack", "unstack"}:
        pre, _ = task_spec(task)
        scene = scene_from_obs(
            entry_obs,
            inner.num_blocks,
            inner.symbolic_name_to_box_id,
            include_table=task == "unstack",
            entry_obs=entry_obs,
        )
        if not evaluate(pre, scene):
            raise CapError("Initial state violates the shared task precondition")
    elif task == "reverse":
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


def evaluate_terminal(env, task, entry_obs):
    from synthesis.cfg.reset import inner_env
    from synthesis.cfg.tasks import task_spec
    from synthesis.predicates.scene import evaluate, scene_from_obs

    inner = inner_env(env)
    observation = inner.flatten_observation(inner._get_obs())
    result = {
        "environment_success": bool(inner._is_success(observation)),
        "dsl_postcondition": None,
    }
    if task in {"stack", "unstack"}:
        _, post = task_spec(task)
        scene = scene_from_obs(
            observation,
            inner.num_blocks,
            inner.symbolic_name_to_box_id,
            include_table=task == "unstack",
            entry_obs=entry_obs,
        )
        result["dsl_postcondition"] = bool(evaluate(post, scene))
    return result
