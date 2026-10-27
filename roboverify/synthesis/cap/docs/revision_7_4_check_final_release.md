# Revision 7.4: check for unreleased blocks at the end of a CaP rollout

- **Status:** implemented 2026-10-27 (Section 8). No model run under `config_frozen_v8` yet.
- **Decision (2026-10-27):** the fix is CaP-only. RoboVerify's verified programs cannot end with a grasped block (Section 3), so the shared checker [`cfg/task_success.py`](../../cfg/task_success.py) and the relational DSL stay unchanged.
- **Goal:** when a CaP rollout ends, detect any block that is still grasped or touched by the gripper, and count such an attempt as a failure.

## 1. Problem

`task-spec-v2` ([settled-success design](../../cfg/docs/settled_postcondition.md)) decides success on the state after 50 settling steps. `settle()` commands the arm toward the gripper's position at the start of settling and keeps the finger opening fixed, so a block that is still grasped stays in the gripper during settling.

The postconditions use block positions only:

- tower tasks evaluate `task_spec` (ON*, Higher, Scattered, ON*_zero) on block centres;
- Grid and Pyramid evaluate goal occupancy: a block centre within 20 mm (3D) of each goal centre.

Neither looks at the gripper. If an execution ends with a block grasped, aligned over the tower or inside a goal's 20 mm ball, it counts as a success although nothing physical supports the block. An upper Pyramid goal can be "occupied" by a block held in the air.

**When it can happen.** Release opens the gripper and then retreats to `Z(reference) + height_offset`, so an execution that ends with Release leaves no block grasped. The gap needs a policy that does not end with Release. A CaP policy is arbitrary Python over the API, so nothing prevents it from finishing with Pick or Move; Move commands the gripper closed.

**Secondary case.** After a Release with a small or negative `height_offset`, the open fingers can still touch the block. Open fingers are unlikely to hold a block up, but nothing checks for contact between the gripper and a block.

## 2. Evidence (Revisions 7.1–7.3, `task-spec-v2`)

An audit of the 288 successful CaP attempts (Haiku 5.5, `config_frozen_v6`/`v7`) found:

- Replaying each attempt's last snapshot through `check_recorded_task_success` with 50 steps reproduced all 288 stored verdicts.
- In every success, the gripper ends at least 9.9 cm from every block centre after settling. None relied on a held block.
- 12 successes ended with the gripper commanded closed. In 11 the last primitives were `Move, Release, Move`, and in one Unstack they were `Move, Move, Move`. The gripper was empty in all 12, 10–40 cm from the nearest block.

The gap is real but has not occurred. The change is a guard, and existing successes are expected to be unchanged (Section 6.3 verifies this).

## 3. Why CaP only

**No hold predicate in the DSL.** The relational vocabulary has no gripper-state atom:

- tower tasks: `ON_star`, `ON`, `Higher`, `Scattered`, `ON_star_zero`, `eq`;
- goal tasks: `HasBlock`, `AtGoal` and goal equality/null.

Adding `forall x. not Holding(x)` to every postcondition would need:

- a geometric definition of the new atom;
- a new symbol in `HighLevelContext`, with WP rules and inference-vocabulary changes;
- new CaP API and instruction text;
- a separate conjunct for the goal tasks, whose postcondition is occupancy.

That is a large change to guard against a state RoboVerify's abstract model cannot reach.

**RoboVerify already excludes it structurally:**

- [`cfg/verification.py`](../../cfg/verification.py) `propose_summaries` abstracts each `Pick … Move … Release` into `Put(held, target)`. An unfinished placement raises `"Unfinished placement at CFG exit"`. A loop cannot split a placement, `Get` cannot occur inside one, and Release must name the held block.
- [`cfg/program_adapter.py`](../../cfg/program_adapter.py) applies the same check to supplied programs.
- During search, `CFGPenalty` scores such a candidate as failed. During verification, the error ends the run as invalid, never as `verified_model`.
- [`verification_lib/motion_verification.py`](../../verification_lib/motion_verification.py) `check_contract_realization` requires `released` for every discharged placement contract.

Only CaP policies can end with a block grasped, so the check belongs in the CaP runner.

## 4. Design

### 4.1 Definition (`final-release-v1`)

Evaluate the state at the **end of the rollout**: after the policy returns or fails, after recording has stopped, and before the 50 settling steps.

- **Criterion (unreleased block):** a physical block that has an active MuJoCo contact with any gripper geom (`robot0:gripper_link`, `robot0:l_gripper_finger_link`, `robot0:r_gripper_finger_link`). This covers a grasp (contact with both fingers) and the secondary case (an open finger still touching a block).
- **Diagnostics (recorded, not part of the criterion):**
  - `fingers_closed`: the controller's finger test, `obs[3] + obs[4] <= 0.053`;
  - `near_closed_gripper`: block centres within 0.030 m of a closed gripper (the demonstrations' `held_block` rule);
  - `ledger_held`: the block of the last converged Pick that no later Release followed, tracked by the adapter. This is the literal "Pick not released" reading of the problem.
  - Per contact: block, gripper geom, and whether each finger touches it.

The criterion is physical rather than the ledger because the ledger can disagree with the scene in two ways. A Pick can converge with the block slipping out, so the ledger says held but nothing is grasped. And a Release can be followed by a finger still touching the block, so the ledger says released but the gripper is in contact. The ledger is still logged, so any disagreement is visible (Decision 1).

### 4.2 Success and status

With the option on:

```
task_spec_success = task_success.success (shared task-spec-v2 checker, unchanged)
                    and final_release.released
success_criterion = "task-spec-v2+final-release-v1"
```

- `task_success` (the shared checker's result), `success` (environment success) and `settled_environment_success` keep their meanings, so RoboVerify and CaP still share the same settled check.
- With the option off, results are byte-identical to the current runner. No `final_release` field is written and the criterion string stays `task-spec-v2`.
- **Status `unreleased_block`** (Decision 3): when the policy finished normally (status `completed` or `goal_failed`) and a block is unreleased, the status becomes `unreleased_block` and `reason` lists the blocks. An exception status (`primitive_failed`, `failed`, `budget_exhausted`, ...) is kept, because it is the earlier cause; `final_release` is still recorded.

### 4.3 Result fields

```json
"final_release": {
  "version": "final-release-v1",
  "released": true,
  "unreleased_blocks": [],
  "contacts": [{"block": 3, "geoms": ["robot0:l_gripper_finger_link"]}],
  "fingers_closed": false,
  "near_closed_gripper": [],
  "ledger_held": null,
  "issues": []
}
```

When `released` is false, `issues` holds, for example, `"block_3 is still in contact with the gripper at the end of the rollout"`. If the contact data is unavailable (an evaluation exception), `released` is null and `task_spec_success` is false, matching the existing settled-evaluation failure path.

### 4.4 Recovery feedback

The recovery evidence today does not say *why* an attempt failed the task specification. It carries `status`, `environment_success`, `dsl_postcondition` and the terminal goal clauses, but no settled or task-spec fields. An attempt that fails only this check would show `status: completed`, `environment_success: true` and goal clauses that are all true, which is contradictory evidence for the model.

With the option on, add one evidence item, `success_check`, to the `fixed_items` layout:

```json
"success_check": {
  "criterion": "task-spec-v2+final-release-v1",
  "postcondition_at_end": true,
  "postcondition_after_settling": true,
  "released": false,
  "unreleased_blocks": ["block_3"],
  "note": "Success is decided after the robot stops; every block must be released and untouched by the gripper at the end of the policy."
}
```

- Block labels use `block_<i>`, as in the goal clauses.
- `item_delivery.success_check` is recorded like the other items.
- The item states the measured outcome and the success condition only, not a strategy.
- It also covers the existing case where the postcondition holds at the end but fails after settling.

### 4.5 Task instruction (Decision 2)

The instructions describe the goal ("Physical completion requires …"), but not that the gripper must be empty. To keep success conditions stated rather than hidden, add one sentence to every task's instruction, before the final "Generate and invoke …" line:

> When the policy ends, no block may remain grasped or touched by the gripper.

This states a goal condition, not a decomposition, so it follows the prompt rule of [Revision 7](revision_7_update_prompt_and_config.md#3-rules-for-later-changes). It is a new instruction version, `task-instructions-v3`.

## 5. Implementation plan

1. **Config** ([`config.py`](../config.py)): add `RunConfig.final_release_check: bool = False` and validate that it is a Boolean. It requires `post_settling_steps > 0`, because the criterion extends `task-spec-v2`.
2. **Detection** ([`tasks.py`](../tasks.py)): add `final_release_outcome(env, adapter)`. It reads `inner.sim.data.contact[:ncon]`, maps geom ids to the three gripper geoms and the `object<i>` block geoms, and computes the diagnostics of Section 4.1 from the observation. It takes no simulation step.
3. **Ledger** ([`adapter.py`](../adapter.py)): track `ledger_held`. A converged Pick sets it to the physical id; a converged Release clears it; an unconverged primitive leaves it unchanged. This is diagnostic only.
4. **Runner** ([`runner.py`](../runner.py)): in the `finally` block, when the option is on and `env`/`entry` exist, call `final_release_outcome` **before** `settled_task_outcome`, because settling steps physics. Then combine success and status as in Section 4.2. `tasks.success_criterion(config)` returns the combined string when the option is on.
5. **Recovery** ([`recovery.py`](../recovery.py)): add the `success_check` item (Section 4.4) to the `fixed_items` evidence and to `item_delivery`. `summarize` adds `task_spec_success` and `final_release.released` when present. `legacy` and earlier profiles are unchanged.
6. **Instructions** (if Decision 2 is accepted): add a new `INSTRUCTIONS` set (`task-instructions-v3`) in a new frozen file. `config_frozen_v6` and `v7` are not edited.
7. **Frozen configuration** (`configs/config_frozen_v8.py`): v7 plus `final_release_check: True`, plus `task-instructions-v3` if accepted. `configs/test_config_frozen_v8.py` pins the exact v7→v8 diff and checks that rendered prompts differ from v7's only by the instruction sentence.
8. **Experiment driver:** add `v8` to `revision6_experiment.FROZEN`.
9. **Docs:** fill in this document's results, update the CaP `AGENTS.md`/README pointers to v8, and add a short note in [`settled_postcondition.md`](../../cfg/docs/settled_postcondition.md) that CaP v8 adds a CaP-only release check on top of `task-spec-v2`.

## 6. Tests and validation

### 6.1 Unit tests (fixtures, no simulator)

- Defaults: the option is off, results have no `final_release` field, and the criterion string is unchanged. Existing runner and prompt-parity tests must still pass unchanged.
- Validation: a non-Boolean value is rejected, and the option is rejected when `post_settling_steps` is 0.
- `final_release_outcome` on a stub contact list: no contacts → released; one finger → unreleased; both fingers → unreleased, with `near_closed_gripper` set when closed and within 0.030 m.
- Runner combination: a shared-checker success with an unreleased block gives `task_spec_success` false while `task_success.success` stays true. An evaluation exception gives `released` null and `task_spec_success` false.
- Recovery: the `success_check` item appears only with the option on, names `block_<i>`, and is recorded in `item_delivery`.
- Ledger: Pick → held; Release → cleared; a failed Pick leaves the ledger unchanged.

### 6.2 Simulator test (MuJoCo, scripted policy)

Two-block Stack:

- `Pick(x); Move(b0, b0, b0, [0, 0, 0.05])` ends with x held on top of b0. Expect the shared checker's success to be true, `final_release.released` false and `task_spec_success` false.
- The same policy followed by `Release(b0, 0.10)` gives `released` true and `task_spec_success` true.

### 6.3 Re-score archived runs (offline, no model calls)

Restore the last recorded snapshot (the end of the rollout, before settling) of every attempt in `runs/cap-revision7-1`, `cap-revision7-2` and `cap-revision7-3`, and evaluate `final_release_outcome`. Expected from Section 2: 0 of the 288 successes change. Report the counts here. The script is one-off and stays outside the repository.

### 6.4 Smoke

`config_frozen_v8`, Haiku 5.5 on OpenRouter, one trial per task, scene seed 0, about $0.2. Check that `final_release` is present, the criterion string is right, and no prompt differs from v7 except as planned.

Full test suite and `git diff --check` before committing. One commit per step group: config/detection/runner, recovery, frozen v8, docs.

## 7. Decisions

1. **Criterion:** decided 2026-10-27: physical contact (Section 4.1). The ledger is a diagnostic only.
2. **Instruction sentence:** decided 2026-10-27: add it (Section 4.5, `task-instructions-v3`).
3. **Status:** decided 2026-10-27: add the status `unreleased_block` (Section 4.2).

## 8. Implementation and validation (2026-10-27)

Implemented as planned in Section 5, steps 1–9. Deviations and details:

- `success_check` is added whenever the attempt's result has `final_release`, in any evidence layout (v8 uses `fixed_items`). It reads `task_spec_postcondition` and `task_spec_settled_postcondition` from the result, because recovery replaces each attempt's `task_success` dictionary with its legacy Boolean. It also carries `released`.
- `summarize` adds `task_spec_success` and `released` only when the result has `final_release`, so recovery prompts of v1–v7 are unchanged.
- The ledger is cleared by a converged Release (step 3).

**Tests** ([`tests/test_final_release.py`](../tests/test_final_release.py), [`configs/test_config_frozen_v8.py`](../configs/test_config_frozen_v8.py)):

- option default and validation;
- contact classification on stub contacts (none, one finger, a two-finger grasp with closed fingers, near-gripper and ledger diagnostics);
- runner: `unreleased_block` with `task_success.success` still true; the option off writes no release fields; the ledger after `Pick` and after `Pick` + `Release`;
- recovery: round 1's prompt contains `"unreleased_blocks":["block_0"]` and `"status":"unreleased_block"`; `item_delivery.success_check` is `complete`;
- v8 differs from v7 only in `final_release_check`; instructions differ only by the sentence; rendered prompts differ only in the instruction.

**MuJoCo** (two-block Stack, scripted policy, no model):

| Policy | Shared settled check | Environment evaluator | `final_release` | Status |
| --- | --- | --- | --- | --- |
| `Pick(x)`, `Move` to 0.055 m over b0, end | success | success | block 1 touches both fingers, fingers closed | `unreleased_block` |
| same, then `Release(b0, 0.15)` | success | success | released | `completed` |

The first row is the gap of Section 1. Before this change, it counted as a success.

**Re-score of archived runs** (Section 6.3): the end-of-rollout snapshots of all 767 attempts in `runs/cap-revision7-1`, `cap-revision7-2` and `cap-revision7-3` were restored and checked.

- All 288 successes are released, so no verdict changes.
- 176 failed attempts end with a block between closed fingers. All of them stopped with an exception (`primitive_failed` 139, `failed` 33, `budget_exhausted` 4), so their status is kept; none would become `unreleased_block`.

The v8 smoke (Section 6.4) has not been run yet.
