# CFG synthesis, collection and task success

Follow the repository-wide [AGENTS.md](../../../AGENTS.md) for environment setup,
project decisions, implementation conventions, and validation. This file adds
configuration rules for the task-success criterion and settling, shared by
RoboVerify and the CaP baseline. The design is in
[docs/settled_postcondition.md](docs/settled_postcondition.md), and the results
of the 2026-10-26 change are in
[docs/settled_post_report.md](docs/settled_post_report.md).

## Success criterion

- Decide task success only through [`task_success.py`](task_success.py):
  `check_task_success` for a live environment that has just finished an
  execution, and `check_recorded_task_success` for a recorded trace (restores
  its last snapshot). Do not reimplement the pre/postcondition check in a caller.
- `post_settling_steps` selects the criterion:
  - `0` (the default everywhere) is `task-spec-v1`: the postcondition on the last
    recorded state. It exists to reproduce earlier results.
  - `N > 0` is `task-spec-v2`: the postcondition after N unrecorded settling
    steps decides success. Use **50** for new experiments and for every
    RoboVerify–CaP comparison, and score both systems with the same value.
- Under `task-spec-v2`, the end-of-execution postcondition (`post_at_end`) and the
  physical evaluator `_is_success` (end and settled) are diagnostics only.
- Tower tasks evaluate `tasks.task_spec`. Grid and Pyramid evaluate 20 mm goal
  occupancy on the same settled state. The Release-event checks of
  `goal_collection.validate_goal_trace` are a demonstration-quality filter, not
  part of the shared criterion.
- Pass bindings explicitly: the input handle (`b0`/`h0`) and the frozen entry
  geometry for `ON_star_zero`. RoboVerify uses the trace's `initial_bindings`;
  CaP uses the bindings its adapter gave the policy.
- Every result records `success_criterion` and `post_settling_steps`. Never
  compare a v1 number with a v2 number without saying so.

## Settling

- Settle only through `task_success.settle(env, steps)`: it holds the gripper
  position with gripper command 0.0 (the finger position actuators keep the
  opening at the start of each step). Do not add inlined settling loops.
- **Initial settling** is a configuration option that is **on by default**:
  every fresh reset of all six tasks settles for `initial_settling_steps` steps
  (default 50, `SETTLING_STEPS`) before recording or before the policy starts.
  Keep the default for all new work. Set 0 only to reproduce runs made before
  2026-10-26, whose Unstack/Reverse starts were unsettled, and record that
  setting with the results. A supplied or archived snapshot is restored without
  settling, whatever the option.
- Invariant learning and `execute_goal_program` always settle for 50 steps; they
  have no option.
- **Final settling** never enters a recording. Settle after recording stops, or
  on a fresh environment restored from the last snapshot. Do not append settling
  steps to `states`, `actions`, `indices`, snapshots or video.
- Search heuristics (PostScore, edge labels, the MCMC reward) and `Program.eval`
  without an initial observation do not settle; they are not the success
  criterion.

## Options by entry point

| Entry point | Option | Effect |
| --- | --- | --- |
| `collect_demos`, `record_execution` | `--initial-settling-steps N` / `initial_settling_steps=N` (default 50) | settles each fresh reset for N steps before recording; 0 disables it; recorded in `metadata.initialization.settling_steps` |
| `collect_demos` | `--post-settling-steps N` | stores the settled check in `metadata.validation.settled`; `validate_trace` accepts only settled successes |
| `audit_demos` | `--post-settling-steps N` | rescores each last snapshot and compares it with the stored result; with 0, `--settle-steps` keeps the physical-evaluator check |
| `revalidate_settled` | `--demos ... --post-settling-steps N` (default 50) | writes `settled_validation.json` next to each archive; the NPZ is never modified |
| `synthesize_cfg` | `--post-settling-steps N` | validates demonstrations from stored settled results (trace metadata or a sidecar naming the archive SHA-256), without the simulator; fails with an explicit error when neither exists |
| `learn_invariant` | `--post-settling-steps N` | settles generated counterexample executions before validation |
| `run_program_rollouts` | `initial_settling_steps=N` (default 50), `task=..., post_settling_steps=N` (function arguments; no CLI) | reports `task_spec_success` from the shared checker |
| CaP `RunConfig` | `initial_settling_steps` (default 50) | settles each fresh reset before the policy starts; 0 disables it; recorded per rollout as `settling_steps` |
| CaP `RunConfig` | `post_settling_steps` | runs the checker after recording, outside the trial deadline and uncharged to the policy budget; `task_spec_success` is the settled result |

## Archives and frozen configurations

- Keep existing archives unchanged so their SHA-256 pins stay valid. Revalidate
  them under v2 into sidecars with `revalidate_settled`; report an archive that
  fails, and replace it only by a separate, recorded collection step.
- Collect new demonstrations with `--post-settling-steps 50`, and audit them with
  `audit_demos --post-settling-steps 50`.
- CaP `config_frozen_v1`–`v3` keep `post_settling_steps: 0`. `config_frozen_v4`
  is v3 plus `post_settling_steps: 50` and the Unstack/Reverse archives
  re-collected from settled starts (`demos/cap-revision6-v4/`). Never edit a
  frozen version; record a change as a new `config_frozen_vN.py`.
- Unstack and Reverse runs made before 2026-10-26 started from unsettled scenes.
  Recomputing them under v2 settles only the final state; say so when reporting
  them. To rerun one of them exactly, set `initial_settling_steps: 0`; all
  frozen CaP versions, v1–v4, otherwise use the default 50.
