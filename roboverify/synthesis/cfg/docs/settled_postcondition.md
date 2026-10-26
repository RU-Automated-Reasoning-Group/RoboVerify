# Plan: one settled task-success checker for RoboVerify and CaP

- **Status:** implemented on `feature/settled-success` (2026-10-26) with the decisions in Section 3, revised the same day so that final settling is an option (`post_settling_steps`, default 0). Implementation notes, results and deviations are in Section 5.
- **Agreement (user and Yuning, 2026-10-26):**
  - every execution of all six tasks starts from a scene settled for 50 steps after the environment is initialized;
  - a task's postcondition is checked only after 50 settling steps at the end of an execution, and that check is the only success criterion;
  - RoboVerify and the CaP baseline call the same checker in `synthesis/cfg/`, so both are scored by one definition;
  - final settling is a configurable option, `post_settling_steps`. The default 0 keeps the original logic (`task-spec-v1`, postcondition on the last recorded state); a positive value selects `task-spec-v2` with that many settling steps. Comparisons use 50.
- **Why:** the shared definition today (`task-spec-v1`, [`collection.py:validate_trace`](../collection.py)) evaluates the postcondition on the last recorded state. ON* is a column relation, so a block that is still falling, or that rests on a tower it will slide off, can satisfy it. In the CaP Revision 4.3 results, 547 attempts satisfied the postcondition symbolically but failed the physical evaluator; 78 of them did not settle into towers ([Revision 4.3 report, §4](../../cap/docs/revision_4_3_report.md)).

## 1. Current behavior

No code path evaluates the task postcondition (the `task_spec` formula, or goal occupancy for Grid/Pyramid) after final settling. The only post-execution settling checks the environment's physical evaluator `_is_success`, not the task specification.

| Site | Check | State evaluated | Final settling |
| --- | --- | --- | --- |
| [`collection.py:298`](../collection.py) `validate_trace` | completed, all primitives converged; tower tasks: `task_spec` through `validate_demonstrations`; goal tasks: `validate_goal_trace` | last recorded | no |
| [`demo_validation.py:30`](../demo_validation.py) `validate_demonstrations` | pre at `t_start`, post at `t_end`; `post_reached` (post at any step) | last recorded | no |
| [`goal_collection.py:59`](../goal_collection.py) `validate_goal_trace` | every goal occupied at the end, plus checks at each `ReleaseByName` event (displacement, payload on a goal) | last recorded, event states | no |
| [`entry/collect_demos.py:255`](../../entry/collect_demos.py) | `validate_trace` decides acceptance | last recorded | no |
| [`entry/audit_demos.py:89`](../../entry/audit_demos.py) | replays actions, `_is_success`, settles `--settle-steps` (default 50), `_is_success` again | replayed final, then settled | **yes, `_is_success` only** |
| [`entry/synthesize_cfg.py:91`](../../entry/synthesize_cfg.py), [`verified_synthesis.py:312`](../verified_synthesis.py) | `validate_demonstrations` on archive segments | last recorded | no |
| [`experiment/invariant_learning/tasks.py:370`](../../experiment/invariant_learning/tasks.py) | `validate_trace` | last recorded | no |
| [`entry/run_rollouts.py:69`](../../entry/run_rollouts.py), [`mcmc/synthesis.py:1031`](../../mcmc/synthesis.py) | `_is_success` | last | no |
| [`cap/tasks.py:236`](../../cap/tasks.py) `evaluate_terminal` | `_is_success`; tower tasks: `task_spec` on the live state; goal tasks: its own occupancy check, without the Release-event checks | live final state | no |
| [`cap/tasks.py:272`](../../cap/tasks.py) `settled_environment_success` | `_is_success` after 50 settling steps (diagnostic) | settled | **yes, `_is_success` only** |

Search heuristics evaluate edge labels, not the task's success criterion: PostScore in [`straightline.py`](../straightline.py), edge labels in [`refine.py`](../refine.py) and [`quotient.py`](../quotient.py), and the MCMC reward. They are out of scope.

**CaP reimplements the check.** It reuses `task_spec` but builds its own scene from the live environment's bindings. It has no `status == "completed"` gate, only the exception path, and its goal check omits the Release-event checks. `task_spec_success` (`cap/runner.py:478`) is `unconverged == 0 ∧ pre ∧ post`.

**Settling exists only as inlined copies** of one loop. It holds the gripper at its position at the start of settling, with gripper command 0.0:

```python
target = obs[:3].copy()
for _ in range(STACK_SETTLING_STEPS):          # 50, cfg/collection.py:30
    obs = env.step(get_move_action(obs, target, close_gripper=False))[0]
```

**Initial settling skips Unstack and Reverse.** `collection.py:239` settles a fresh reset only for `("stack", "restack", *GOAL_TASKS)`, and CaP's `SETTLING_TASKS` copies that list. Unstack and Reverse record the reset state directly; the `collect_demos` docstring states this without a reason, and no review entry records one. History:

- settling was added for Stack only, to remove the robot's initial transient (c3e0b43, [discrepancy 24](../../../../PAPER-DISCREPANCIES.md#24-the-first-stack-placement-inherits-a-transient-robot-state-and-a-grasp-offset));
- ReStack, Grid and Pyramid added themselves when they were introduced (33131b4, b286d62);
- the lateral Unstack/Reverse reset (a0d825b) did not.

The transient in discrepancy 24 belongs to the robot, not to the scene, so it applies to Unstack and Reverse as well.

Copies:

- `collection.py:238`: initial settling, before recording; this is the "settles for 50 steps" of Grid/Pyramid collection;
- `experiment/invariant_learning/tasks.py:304`, `entry/execute_goal_program.py:46`, `cap/tasks.py:121`: initial settling;
- `entry/audit_demos.py:91`, `cap/tasks.py:272`: final settling.

## 2. Proposed design

### 2.1 One checker in `cfg/task_success.py`

```python
SETTLING_STEPS = 50                      # = STACK_SETTLING_STEPS
CRITERIA = {0: "task-spec-v1",           # post_settling_steps == 0: last recorded state
            "settled": "task-spec-v2"}   # post_settling_steps > 0: after settling

def settle(env, steps=SETTLING_STEPS):
    """Hold the gripper still for `steps` steps; return the final observation.
    The single replacement for the inlined loops (initial and final)."""

def check_task_success(env, task, *, entry_observation, bindings,
                       completed, unconverged, post_settling_steps=0,
                       task_options=None) -> TaskSuccess:
    """Score an execution that has just ended in `env` (state not yet changed).
    1. pre on the entry observation;
    2. post on the current (unsettled) state;
    3. if post_settling_steps > 0: settle(env, post_settling_steps) and
       evaluate post on the settled state;
    4. physical evaluator at the end (and after settling) -> diagnostic.
    success = completed and unconverged == 0 and pre and post, where post is
    the settled post when post_settling_steps > 0 and the end post otherwise."""

def check_recorded_task_success(trace, environment_factory, post_settling_steps):
    """The same check for a recorded trace: restore the last snapshot into a
    fresh environment, then call check_task_success."""
```

`TaskSuccess` records:

- `criterion`, `settling_steps`;
- `completed`, `unconverged`;
- `pre`, `post_at_end`, `post_settled`, `success`;
- `environment_success_at_end`, `environment_success_settled`;
- the settled block positions (and goal occupancy for goal tasks);
- `issues` (as in `DemoValidation`).

Tower tasks evaluate `task_spec(task)`. Grid and Pyramid evaluate `goal-occupancy` (20 mm, 3D) on the same settled state.

**Bindings.** Bindings are passed explicitly: the input handle `b0`/`h0` and the frozen entry geometry for `ON_star_zero`. RoboVerify passes the trace's `initial_bindings`. CaP passes the bindings it gave the policy. Neither side reads bindings from an unrelated source.

### 2.2 Recordings stay unchanged

The settling steps are never appended to `states`, `actions`, `indices`, snapshots or video. Appending them would change:

- `t_end` and the final segment's imitation distance;
- quotient exit segments;
- audit replay equality;
- CaP's compacted demonstrations and their token counts;
- every archive's comparability.

The checker settles after recording has stopped, or on a restored copy of the last snapshot. Collection stores only the result in trace metadata: `metadata.validation.settled` with the fields above, and the settled block positions.

### 2.3 Callers

Every caller takes `post_settling_steps` (default 0). With 0, its behavior is unchanged, so existing RoboVerify code, tests and archives keep working without modification.

| Caller | Change |
| --- | --- |
| `record_execution` / `validate_trace` / `collect_demos` | `record_execution(..., post_settling_steps=0)`: with a positive value, after recording ends, settle the live environment and store the result in `metadata.validation.settled`. `validate_trace` applies `task-spec-v2` when that result is present and `task-spec-v1` otherwise. `collect_demos --post-settling-steps N`. |
| `audit_demos` | Replace its `_is_success` settling loop with `check_recorded_task_success`. Report a mismatch with the stored `metadata.validation.settled`. |
| `synthesize_cfg`, `verified_synthesis`, `invariant_learning` | Archive validation reads the stored settled result; the step stays simulator-free. A `--revalidate-settled` option, or the audit, recomputes it from snapshots. |
| RoboVerify evaluation of synthesized programs on held-out scenes (`run_rollouts`, and the scenes shared with CaP) | Score with `check_task_success`, so RoboVerify and CaP numbers come from one function. |
| CaP `runner` | Replace `evaluate_terminal`'s task-spec part and `settled_environment_success` with one `check_task_success` call. Settling stays outside the trial deadline, after recording, and is not charged to the policy budget, as `settled_environment_success` is today. `task_spec_success` becomes the checker's `success`. |
| CaP recovery | Unchanged: it keys on `success_field`. Its failure feedback should report whether the post failed at the end or only after settling (Section 3, decision 6). |
| Search heuristics, MCMC reward | Unchanged (out of scope). |

### 2.4 Versioning and records

- The success field name `task_spec_success` is kept. Its criterion is `task-spec-v1` with `post_settling_steps` 0 and `task-spec-v2` otherwise, and every result records `success_criterion` and `post_settling_steps`.
- Archives collected under v1 keep their arrays. A one-off revalidation writes a sidecar report (`settled_validation.json`) next to each archive rather than rewriting the NPZ, so the SHA-256 pins in `config_frozen_v3` remain valid.
- Archives that fail under v2 are reported, not silently dropped. Replacing them is a separate collection step.
- CaP's frozen configurations are not edited. Results computed with v2 state the criterion. Earlier reports (Revision 4.3, Revision 5) remain v1 unless they are recomputed (Section 3, decision 5).
- `PAPER-DISCREPANCIES.md` and the README status get an entry for the success-criterion change, since it affects acceptance of demonstrations and of both systems' executions.

### 2.5 Tests

**New tests** (`cfg/test_task_success.py`, synthetic fixtures; one MuJoCo case):

- a falling block satisfies the post at the end but not after settling;
- a transiently unsettled tower that settles correctly passes;
- recordings have the same length before and after the check;
- the recorded-trace check equals the live check on the same final snapshot;
- goal tasks use occupancy on the settled state;
- settling steps are not charged to a CaP budget.

**Tests pinning today's behavior** that will change or need extending:

- `cfg/test_collection.py:82,99`
- `cfg/test_demo_validation.py:23,40`
- `cfg/test_goal_collection.py:110,196`
- `cfg/test_settled_demos.py:58,90,113`
- `cfg/test_restack.py:111,190`
- `cfg/test_audit_demos.py:28`
- `experiment/invariant_learning/test_simulator.py:97`
- `cap/tests/test_runner.py:210-250,276-284,337-343`
- `cap/tests/test_revision2.py:112`
- `cap/tests/test_budget_timing.py:155` (asserts the settling step count)
- `cap/tests/test_recovery.py:87,172`
- `cap/configs/test_config_frozen_v1.py:25`

## 3. Decisions (2026-10-26)

| # | Decision | Outcome |
| --- | --- | --- |
| 1 | Success needs the post after settling only, or at the end and after settling? | **After settling only.** `post_at_end` is kept as a diagnostic. |
| 2 | Grid/Pyramid Release-event checks | **Not part of the shared criterion.** They stay in RoboVerify's demonstration acceptance as a demonstration-quality filter. The shared criterion is settled occupancy for both systems. |
| 3 | Physical evaluator `_is_success` | **Diagnostic only**, before and after settling, for both systems. |
| 4 | CaP recovery feedback | **Unchanged.** The settled postcondition is the only success criterion; recovery feedback gets no separate "post held at the end but not after settling" location, so recovery prompts do not change. |
| 5 | Settling action | **One `settle()` helper:** hold the gripper's position with gripper command 0.0 for 50 steps. It replaces all six inlined copies, initial and final. Confirm in the environment source that 0.0 holds the finger opening before relying on it. |
| 6 | Recompute earlier results under v2 | **Yes:** Revision 5 and Revision 4.3, from the saved final snapshots. CaP's `rollout_*/trajectory.npz` stores the full simulator snapshot of every state, so every attempt's final state can be restored and settled. Report v1 and v2 side by side. |
| 7 | Initial settling | **All six tasks.** Unstack and Reverse gain the 50-step initial settling, in collection, CaP, CFG execution and invariant learning (fresh resets only; supplied snapshots are restored without settling, as today). |
| 8 | Existing demonstration archives | Stack, ReStack, Grid and Pyramid archives are kept unchanged and revalidated under `task-spec-v2` (50 steps) into sidecar reports. Unstack and Reverse archives were recorded from unsettled starts and are **re-collected** with `post_settling_steps` 50, including the CaP Revision 6 archives (into new directories; `demos/cap-revision6/{unstack,reverse}` stay as they are). |
| 9 | CaP configuration | `RunConfig.post_settling_steps` (default 0, the original logic; v1–v3 are not edited and keep it). A new `config_frozen_v4`: v3 plus `post_settling_steps` 50 and the re-collected Unstack/Reverse archives, pinned by SHA-256. Unstack/Reverse evaluation scenes change because their starts are now settled. |
| 10 | Order of changes | 1. `cfg/task_success.py` (`settle`, the checkers) and initial settling for all tasks. 2. The `post_settling_steps` option (default 0) for collection, audit, CFG, invariant-learning and rollout callers. 3. CaP runner and `config_frozen_v4`. 4. Archive revalidation and Unstack/Reverse re-collection. 5. Recomputation of Revision 5 and 4.3. One commit each; a `PAPER-DISCREPANCIES.md` entry and README status update record the criterion and initial-settling changes. |
| 11 | Final settling is optional (2026-10-26) | **`post_settling_steps`, default 0.** 0 keeps the original last-state logic everywhere; comparisons between RoboVerify and CaP use 50. Initial settling is not optional: it applies to all six tasks. |

## 4. Not changed

- The task specifications themselves (`cfg/tasks.py`) and the predicate semantics (`util/on.py`, Higher tolerance).
- Symbolic and motion verification: they reason about programs, not about a simulated final state.
- Initial settling before recording, apart from moving it into `settle()`.

## 5. Implementation (2026-10-26)

### 5.1 What was built

- **`cfg/task_success.py`.** `settle()`, `TaskSuccess`, `check_task_success` (live environment), `check_recorded_task_success` (restores a trace's last snapshot), `criterion()` (`task-spec-v1` for 0, `task-spec-v2` otherwise), and the archive helpers `revalidate_archive`, `write_sidecar` and `stored_settled_results`.
- **Gripper command 0.0.** The finger actuators are MuJoCo position actuators, and gym's `ctrl_set_action` sets their target to the current finger position plus the command. With 0.0, each step targets the finger opening at the start of that step, held by the actuator over the step's 20 substeps. The target is re-anchored every step, so slow drift is not corrected: in a Revision 5 final state with the open gripper above a tower, the openings went from 42.9/43.9 mm to 40.6/41.9 mm over 50 steps. The arm holds the XYZ position it had when settling began. The command neither opens nor closes the gripper.
- **One settling loop.** `settle()` replaces the inlined loops in collection, CaP setup, CaP's settled diagnostic, `audit_demos`, `execute_goal_program` and invariant learning. A MuJoCo test confirms that it reproduces both earlier loop forms bit for bit (observation and full simulator state), so existing Stack, ReStack, Grid and Pyramid starts are unchanged.
- **Initial settling for all six tasks.** `record_execution`, CaP's `prepare_environment` and `run_program_rollouts` settle every fresh reset for 50 steps; snapshots are restored without settling.
- **`post_settling_steps`** (default 0, the original logic):
  - `record_execution` stores the settled result in `metadata.validation.settled`; `validate_trace` applies `task-spec-v2` when it is present. For goal tasks the Release-event checks stay; only the final-coverage check moves to the settled state.
  - `collect_demos`, `audit_demos`, `synthesize_cfg` and `learn_invariant` take `--post-settling-steps N`. `synthesize_cfg` reads stored results without the simulator: from trace metadata, or from a `settled_validation.json` sidecar that names the archive's SHA-256. It fails with an explicit error naming the revalidation command when neither exists.
  - `synthesis.entry.revalidate_settled` writes the sidecars; NPZ files are never modified.
  - CaP: `RunConfig.post_settling_steps`. With 0, results are as before. With a positive count, the shared checker runs after recording, outside the trial deadline and uncharged to the budget, with the adapter's input bindings. `task_spec_success` is the settled result; results add `post_settling_steps`, `task_spec_settled_postcondition` and the checker's record (`task_success`).
- **`config_frozen_v4`:** v3 plus `post_settling_steps: 50` and the re-collected Unstack/Reverse archives.

### 5.2 Results

- **Archives** ([Revision 6, §6.9](../../cap/docs/revision_6_add_demonstration.md#69-settled-criterion-and-config_frozen_v4-2026-10-26)): the re-collected Unstack and Reverse archives are 15/15 valid under `task-spec-v2` and audited. The Stack, ReStack, Grid and Pyramid archives pass `task-spec-v2` 15/15 each in sidecars. The superseded v3 Unstack and Reverse archives would also pass (15/15 each). No archive fails under v2.
- **Revision 5** ([report §2.4](../../cap/docs/revision_5_report.md#24-settled-criterion-task-spec-v2-recomputed-2026-10-26)): only Stack changes, final 19 → 17 of 20 (two leaning towers, both censored).
- **Revision 4.3** ([report §3.7](../../cap/docs/revision_4_3_report.md#37-settled-criterion-task-spec-v2-recomputed-2026-10-26)): 5,998 → 5,839 successful attempts (159 lost, none gained). The selections hold on point estimates.

### 5.3 Deviations from Sections 2–3

1. **v1–v3 CaP Unstack/Reverse scenes change too.** Initial settling is not optional, so it applies to every CaP configuration, not only v4. Fresh-seed Unstack and Reverse trials on this branch start from settled scenes, unlike the recorded Revision 5 and Revision 6 runs. v1–v3 prompts and the `post_settling_steps: 0` scoring are unchanged.
2. **`run_rollouts` has no CLI.** The option is a function parameter: `run_program_rollouts(..., task=..., post_settling_steps=N)`. The historical verifier entry points that call it were not given new flags.
3. **Search paths are unchanged.** `Program.eval` without an initial observation and the MCMC `_is_success` reward still reset without settling (Section 2.3: search heuristics are out of scope). CFG and standalone MCMC restore archived settled starts.
4. **Revalidation path.** Instead of a `--revalidate-settled` flag, `synthesize_cfg` reads a sidecar written by the new `revalidate_settled` entry point (or stored per-trace results). `audit_demos --post-settling-steps N` recomputes and compares a stored result.
5. **CaP diagnostics with a positive count.** `evaluate_terminal` still supplies the end-of-rollout physical evaluator and `dsl_postcondition`. The checker supplies `task_spec_precondition`, `task_spec_postcondition` (end), the settled postcondition and `settled_environment_success`, so the environment is settled once.
6. **Invariant learning has no archive input.** `learn_invariant --post-settling-steps N` settles its generated counterexample executions before `validate_trace`.
7. **`demos/pipeline-audit-20261023/` is absent** from this checkout and was not revalidated.
