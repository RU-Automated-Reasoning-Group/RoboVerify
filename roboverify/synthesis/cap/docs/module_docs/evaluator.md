# Evaluator

Scores an attempt with RoboVerify's criterion, so CaP and RoboVerify are judged the same way.

## `task_spec_success`

Set in [`runner.run_once`](../../runner.py#L467-L482). An attempt succeeds when all of these hold:

- the program finished normally (no exception);
- every primitive converged ([`adapter.unconverged_primitives`](../../adapter.py#L327-L328) is 0);
- the task precondition held at the start;
- the task postcondition holds at the final state.

[`evaluate_terminal`](../../tasks.py#L236-L269) computes the pre/postcondition values:

- **Tower tasks** (Stack, Unstack, Reverse, ReStack): the shared `task_spec` formulas from `synthesis.cfg.tasks`, evaluated with the same predicate semantics as RoboVerify.
- **Grid and Pyramid:** goal occupancy at the start and end (`goal-occupancy-v1`).

## Diagnostics (not used for success)

- `environment_success`: the environment's physical evaluator at the final frame.
- `settled_environment_success`: the same after 50 extra steps with the gripper held ([`settled_environment_success`](../../tasks.py#L272)).

Recovery stops when the configured `success_field` is true; the frozen configurations use `task_spec_success`.
