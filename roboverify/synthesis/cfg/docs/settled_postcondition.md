# Plan: one settled task-success checker for RoboVerify and CaP

- **Status:** plan for review (2026-10-26). Nothing in this document is implemented yet.
- **Agreement (user and Yuning, 2026-10-26):** a task's postcondition is checked after 50 settling steps at the end of an execution. RoboVerify and the CaP baseline call the same checker in `synthesis/cfg/`, so both are scored by one definition.
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

Copies:

- `collection.py:238`: initial settling, before recording; this is the "settles for 50 steps" of Grid/Pyramid collection;
- `experiment/invariant_learning/tasks.py:304`, `entry/execute_goal_program.py:46`, `cap/tasks.py:121`: initial settling;
- `entry/audit_demos.py:91`, `cap/tasks.py:272`: final settling.

## 2. Proposed design

### 2.1 One checker in `cfg/task_success.py`

```python
SETTLING_STEPS = 50                      # = STACK_SETTLING_STEPS
CRITERION = "task-spec-v2"               # settled postcondition

def settle(env, steps=SETTLING_STEPS):
    """Hold the gripper still for `steps` steps; return the final observation.
    The single replacement for the inlined loops (initial and final)."""

def check_task_success(env, task, *, entry_observation, bindings,
                       completed, unconverged, steps=SETTLING_STEPS,
                       task_options=None) -> TaskSuccess:
    """Score an execution that has just ended in `env` (state not yet changed).
    1. pre on the entry observation;
    2. post on the current (unsettled) state            -> diagnostic;
    3. settle(env, steps);
    4. post on the settled state                         -> decides success;
    5. physical evaluator before/after settling         -> diagnostic.
    success = completed and unconverged == 0 and pre and settled_post."""

def check_recorded_task_success(trace, environment_factory, steps=SETTLING_STEPS):
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

| Caller | Change |
| --- | --- |
| `collect_demos` / `validate_trace` | After recording ends, call `check_task_success` on the live environment. Accept a demonstration only if `success`. |
| `audit_demos` | Replace its `_is_success` settling loop with `check_recorded_task_success`. Report a mismatch with the stored `metadata.validation.settled`. |
| `synthesize_cfg`, `verified_synthesis`, `invariant_learning` | Archive validation reads the stored settled result; the step stays simulator-free. A `--revalidate-settled` option, or the audit, recomputes it from snapshots. |
| RoboVerify evaluation of synthesized programs on held-out scenes (`run_rollouts`, and the scenes shared with CaP) | Score with `check_task_success`, so RoboVerify and CaP numbers come from one function. |
| CaP `runner` | Replace `evaluate_terminal`'s task-spec part and `settled_environment_success` with one `check_task_success` call. Settling stays outside the trial deadline, after recording, and is not charged to the policy budget, as `settled_environment_success` is today. `task_spec_success` becomes the checker's `success`. |
| CaP recovery | Unchanged: it keys on `success_field`. Its failure feedback should report whether the post failed at the end or only after settling (Section 3, decision 6). |
| Search heuristics, MCMC reward | Unchanged (out of scope). |

### 2.4 Versioning and records

- The success field name `task_spec_success` is kept. Its criterion changes from `task-spec-v1` to `task-spec-v2`, and every result records `success_criterion`.
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

## 3. Decisions for review

| # | Decision | Recommendation |
| --- | --- | --- |
| 1 | Does success need the post **after settling only**, or **at the end and after settling**? | After settling only. Keep `post_at_end` as a diagnostic. A transient violation at the last recorded step (for example, a block still sliding into place) should not fail an execution that ends correctly. |
| 2 | Grid/Pyramid Release-event checks (`validate_goal_trace`: displacement, payload on a goal at each Release) | Keep them in RoboVerify's demonstration acceptance as a demonstration-quality filter, outside the shared success criterion. The shared criterion is settled occupancy for both systems. This matches the Revision 5 decision that CaP has no event-level goal checks. Alternatively, apply them to both; CaP rollouts would then need Release events in the same form. |
| 3 | Physical evaluator `_is_success` | Diagnostic only, before and after settling, for both systems. |
| 4 | Settling action | One `settle()` helper: hold the gripper's position with gripper command 0.0 for 50 steps (today's behavior). Replace all six inlined copies, initial and final. Before relying on it, confirm in the environment source that 0.0 holds the finger opening. |
| 5 | Recompute earlier results under v2? | Recompute Revision 5 (120 trials), and Revision 4.3 if it is still cited, from the saved final states. CaP saves full rollout trajectories; whether every run saved a restorable final snapshot must be checked first. Report v1 and v2 side by side. |
| 6 | CaP recovery feedback | Report "post held at the end but not after settling" as a distinct failure location, since it is a different repair target. This changes recovery prompts, so it would be a new frozen configuration version, `config_frozen_v4`. |
| 7 | Existing demonstration archives | Revalidate into sidecar reports and keep the NPZ files unchanged (Section 2.4). Collect replacements only if some fail. |
| 8 | Scope of the first change | 1. `cfg/task_success.py` with `settle` and the checkers. 2. Collection and audit. 3. CaP runner. 4. Revalidation and recomputation. One commit each. |

## 4. Not changed

- The task specifications themselves (`cfg/tasks.py`) and the predicate semantics (`util/on.py`, Higher tolerance).
- Symbolic and motion verification: they reason about programs, not about a simulated final state.
- Initial settling before recording, apart from moving it into `settle()`.
