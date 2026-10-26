# Rollout

Runs a checked program in MuJoCo from the trial's initial state. This is Loop 2: the program's own control flow alternates between reading the environment and acting on it. No LLM is involved.

## Components

| Component | Role | Code |
| --- | --- | --- |
| Environment setup | Build the task environment and restore the saved initial snapshot (or create a fresh scene from the seed and settle it); check the restore is exact and the task precondition holds | [`prepare_environment`](../../tasks.py#L90), [`verify_restoration`](../../recovery.py#L131), [`validate_start`](../../tasks.py#L152) |
| Interpreter | Executes the program AST statement by statement (`while`, `if`, assignments, helper calls); evaluates conditions with the DSL semantics | [`Runtime.execute`](../../runtime.py#L46-L75), [`Runtime.block`](../../runtime.py#L217) |
| Adapter | The DSL API the program calls: observation getters, predicates and queries read the current state; `Pick`/`Move`/`Release` drive the shared RoboVerify controllers | [`RoboVerifyAdapter.call`](../../adapter.py#L141-L332) |
| Recorder | Wraps `env.step`: records every action and full state, counts simulation steps, saves the trajectory and queues the video | [`Recorder`](../../recording.py#L12), [`Recorder.step`](../../recording.py#L116) |

## One primitive call

[`adapter.py`](../../adapter.py#L253-L332):

1. Resolve the object, table or goal handles; reject invalid uses (e.g. picking `tbl`).
2. Build the RoboVerify instruction (`PickByName`, `MoveByName` with three offsets, `ReleaseByName` with a height) and run its controller: 50-step budget, 2 mm tolerance.
3. Log `primitive_start`, controller phases and `primitive_end` (converged, steps, position error).
4. If the controller did not converge, raise `PrimitiveFailure`. The rollout ends with status `primitive_failed`.

## How a rollout ends

- **Normal return:** the program finishes; then the [Evaluator](evaluator.md) scores the final state.
- **Exception:** primitive failure, a `get` with no matching object, an API misuse, or an exhausted execution/simulation budget. The final state is still evaluated and recorded for feedback.

Every round restores the same initial state; a failed trajectory is never continued.
