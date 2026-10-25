# Revision 5: extend CaP to the remaining RoboVerify tasks

- **Status:** step 1 (§4.7) implemented on 2026-10-23; the §4.1 shared changes and the §4.2 `tbl` handle on 2026-10-25; the remaining steps are planned. Decided on 2026-10-23: CaP adopts RoboVerify's success criterion (§3.3) as a new `task_spec_success` field (§3.5), `tbl` option A (§4.2), and uniform default time limits (§4.1).
- **Goal:** run Unstack, Reverse, Restack, Grid and Pyramid in CaP with task definitions, scenes and success criteria that match RoboVerify's.
- **References:** [shared task conditions](../../cfg/tasks.py), [goal-task validation](../../cfg/goal_collection.py), [collector](../../entry/collect_demos.py), [collection audit](../../entry/audit_demos.py), [environment evaluators](../../environment/cee_us_env/fpp_construction_env.py), [reference programs](../../examples/), [CaP task adapter](../tasks.py), [CaP DSL adapter](../adapter.py).

## 1. Reuse the Stack adaptation structure

| Layer | Existing Stack approach | Required for each additional task |
| --- | --- | --- |
| Task specification | Shared pre/postconditions and a physical evaluator | Score with RoboVerify's shared specification (§3.3); report the physical evaluator as a diagnostic |
| Concrete interpretation | Relational predicates evaluated on simulator observations | Use the task's domain (`tbl`, goals), frozen entry geometry and tolerances consistently |
| Initial scene | Environment reset or exact archived snapshot; 50 settling steps | Use the same reset and settling policy as upstream collection (§4.1) |
| CaP API | Block handles, `Pick`/`Move`/`Release`, predicates, queries, position reads | Decide on table and goal references (§4.2, §4.7) |
| Validation | Normal completion, primitive convergence, start check, terminal check | Same, using the task's start and terminal definitions |
| Instruction | Hand-written sentence in `TASK_INSTRUCTIONS` | Derive it from the authoritative criterion and record the exact text |

## 2. Task support after the upstream merge (2026-10-23)

Upstream `main` (`67d48dc`) supplies specifications, reference programs and collection for all six tasks. The remaining gap is in the CaP adapter.

| Task | Shared spec ([`cfg/tasks.py`](../../cfg/tasks.py)) | Physical evaluator | Reference program | Upstream collection | CaP today |
| --- | --- | --- | --- | --- | --- |
| Unstack | pre `∀x. x = tbl ∨ ON*(x, b0)`; post `∀x,y. ON*(x, y) → x = y` | `_roboverify_unstack_all_on_ground` | [unstack.py](../../examples/unstack.py) (uses collection-only motion references) | yes | shared pre at start; physical success plus DSL post diagnostic |
| Reverse | pre: physical blocks `ON*(·, b0)` and `ON*_zero = ON*`; post: physical blocks `ON*_zero(x, y) ↔ ON*(y, x)` | `_roboverify_reverse_tower_complete` | [reverse.py](../../examples/reverse.py) (2–4 blocks) | yes | canonical-order start check; physical success only |
| Restack | pre: `b0 ≠ tbl`; every physical pair is tower-related or Scattered; post: physical blocks `ON*(·, b0)` | `_roboverify_stack_tower_complete` (same as Stack) | [restack.py](../../examples/restack.py) (two flat loops) | yes | no start check; physical success only |
| Grid | no formula; `task_identity` declares `goal-occupancy-v1` (pre: goals empty, blocks pairwise Scattered; post: every goal occupied within 20 mm) | `_roboverify_goal_layout_complete` | [goal_layout.py](../../examples/goal_layout.py) | yes | `AdapterGap` before model calls |
| Pyramid | same as Grid | same as Grid | same as Grid | yes | `AdapterGap` before model calls |

- Upstream collection validates tower tasks with `task_spec` (`cfg/collection.py:validate_trace`) and goal tasks with `validate_goal_trace`. The physical `_is_success` is checked by `audit_demos`, including after 50 extra settling steps.
- The 2026-10-23 audit (`demos/pipeline-audit-20261023/`) passed every task. It establishes executable reference programs, not CaP results.
- The standalone `verify_reverse`/`verify_partial` entry points and the PartialStack environment were removed upstream. CaP `restack` already uses the upstream Restack environment ([pre-merge review](pre_merge_upstream_main_20261023.md)).
- Goal-aware synthesis and verification remain unsupported upstream (PAPER-DISCREPANCIES entry 40). CaP needs only execution and evaluation for Grid/Pyramid.

### 2.1 Current CaP task specification

CaP does not take its task definition from one source:

| Stage | Current source | Location |
| --- | --- | --- |
| Model instruction | Hand-written sentence; not derived from `task_spec` or the evaluator | `tasks.py:TASK_INSTRUCTIONS` |
| Initial scene | Environment reset or snapshot; Stack and Restack settle for 50 steps (§4.1, 2026-10-25; before that, Stack only) | `tasks.py:prepare_environment` |
| Start check | Shared pre for all tower tasks; Reverse also canonical order, Restack also b0 inside a source tower (§4.1, 2026-10-25; before that, Stack/Unstack pre and Reverse order only) | `tasks.py:validate_start` |
| Terminal check | Physical `_is_success` for all tasks; `dsl_postcondition` only for Stack/Unstack | `tasks.py:evaluate_terminal` |

The derived success metrics are also inconsistent:

| Field | Definition | Location |
| --- | --- | --- |
| `success` | Physical evaluator only | `runner.py` |
| `first_attempt_success` | `success` and `dsl_postcondition is not False` | `batch.py` |
| `task_success` (recovery) | Stack/Unstack: `success` and `dsl_postcondition is True`; others: `success` | `recovery.py` |

The [README](../README.md#implemented-scope) states that the DSL postcondition is a diagnostic rather than a scoring gate. `first_attempt_success` and `task_success` contradict it for Stack/Unstack.

## 3. Evaluation criterion

### 3.1 The Stack implementation as the template

| Stage | Stack implementation |
| --- | --- |
| Scene | `prepare_environment` seeds NumPy, calls the Stack reset (`sample_stack_xy`: all blocks on the table, pairwise Scattered, inside the bounded workspace, clear of the gripper) and settles for 50 steps. A snapshot is restored without settling. |
| Start | `validate_start` evaluates the shared precondition `∀x≠y. ¬ON*(x, y) ∧ Scattered(x, y)` and `∀x,y. Higher(x, y)` on the entry observation. The domain is physical blocks only, `b0` comes from the environment binding, and Higher uses the configured tolerance. A violation fails the trial before any model call. |
| Terminal | `evaluate_terminal` reports the physical `_roboverify_stack_tower_complete` result and the shared postcondition `∀x. ON*(x, b0)` separately. |

The precondition is a check, not a generator: the reset sampler is written to satisfy it and is narrower than it (workspace bounds, gripper clearance).

The start check has no known defect and already matches RoboVerify. The issues are in terminal scoring:

1. `success` uses the physical evaluator, whereas RoboVerify accepts on the shared postcondition (§3.2).
2. The symbolic and physical postconditions disagree in both directions (§3.4), and the three result fields combine them differently (§3.5).
3. The 50 settling steps are a literal in `tasks.py` and `runner.py` rather than the upstream `STACK_SETTLING_STEPS`, and the settling task list differs from upstream collection.

After fixing these issues, every task follows the same pattern:
- an upstream reset or snapshot, with upstream settling;
- the shared precondition evaluated at entry;
- the shared postcondition as the success criterion;
- the physical evaluator as a diagnostic.

Grid/Pyramid substitute the goal-occupancy start and terminal checks for the formulas (§4.6).

### 3.2 How RoboVerify judges success

CaP is RoboVerify's baseline, so it adopts RoboVerify's criterion. RoboVerify's task definition is the shared `task_spec`, evaluated numerically on simulator states. The physical evaluator is only an independent audit:

| RoboVerify stage | Criterion | Location |
| --- | --- | --- |
| Demonstration acceptance | Normal completion, every primitive converged, shared precondition at the first state, shared postcondition at the final state | `cfg/collection.py:validate_trace` → `cfg/demo_validation.py` |
| Goal-task demonstrations | Same, with goal occupancy (goals empty and blocks Scattered at start; every goal occupied at the end) | `cfg/goal_collection.py:validate_goal_trace` |
| Straight-line synthesis | PostScore: fraction of segments whose candidate rollout reaches the segment postcondition; a candidate is accepted only at 1.0 | `cfg/straightline.py` |
| Final result | `verified_model`: symbolic and motion proofs against the shared pre/postcondition | `entry/synthesize_cfg.py` |
| Physical evaluator | Not part of acceptance. `audit_demos` checks `_is_success` on the final state and after 50 extra settling steps; legacy MCMC/rollout code reports it as a metric | `entry/audit_demos.py`, `mcmc/synthesis.py` |

### 3.3 Criterion for CaP

- **Primary success**, matching `validate_trace`:
  - the trial ends with normal completion;
  - every primitive converged;
  - the start satisfies the shared precondition;
  - the final state satisfies the shared postcondition.
- **Grid/Pyramid:** use the final-state goal-occupancy check from `validate_goal_trace`. Its event-level checks follow the reference program's instruction paths: a release must not displace an occupied goal, and the released block must reach a goal. CaP policies have different paths, so adapt those checks by primitive kind, or drop them, and record the choice.
- **Diagnostics:** report the physical evaluator, final and after 50 settling steps, as `audit_demos` does. It never gates success.

Two definitions of the same task disagree in both directions (§3.4). The diagnostic therefore shows where a symbolic success is physically questionable, without changing the criterion RoboVerify itself is held to.

### 3.4 Where the symbolic and physical criteria differ

Neither implies the other, because they encode different notions (L = 50 mm):

- **Symbolic `ON*`** (`util/on.py:on_star_implementation`) is a column relation. `ON*(x, b0)` holds when x is within L/2 = 25 mm of b0 in X and Y and not lower than b0. It has no contact and no table semantics, because the abstract block algebra does not model support.
- **Physical evaluator** (`_roboverify_stack_tower_complete`) is a contact chain. b0 must be within 20 mm (`threshold`) of table height. Each next block must satisfy `on()` relative to the previous block: within 25 mm in X and Y, and 0 ≤ dz < 1.5L = 75 mm. Each level must have exactly one candidate.

| Task | Symbolic holds, physical fails | Physical holds, symbolic fails |
| --- | --- | --- |
| Stack | b0 off the table; vertical gap of at least 1.5L, e.g. a final block still held above the tower (Release retreats to 0.15 m) | Leaning tower: each adjacent pair within L/2 in XY, but some block 25 mm or more from b0; rare with 2 mm Move/Release tolerances |
| Restack | Same as Stack | Same as Stack |
| Unstack | A block raised above the table but on no other block | none expected: on-table and no `on()` relation imply no off-diagonal `ON*` |
| Reverse | Reversed order without the new bottom on the table | Leaning reversed tower (as for Stack) |

In a settled, released, nearly aligned tower the two agree. The physical Reverse evaluator compares against the canonical order `[b0, others by ascending ID]`, whereas the shared spec uses the frozen entry `ON*_zero`. `_reset_sim_roboverify_tower` builds exactly the canonical order, so the two agree on fresh resets and on snapshots of them.

### 3.5 Result fields

A result field is a value written to each trial's `result.json` and aggregated into reported success rates. Three Boolean fields exist, added at different times:

| Field | Current definition | Origin |
| --- | --- | --- |
| `success` | Physical evaluator | Revision 1 runner |
| `first_attempt_success` | `success` and `dsl_postcondition is not False` | Revision 4 batch |
| `task_success` | Stack/Unstack: `success` and `dsl_postcondition is True`; others: `success` | Revision 4 recovery |

None of them equals RoboVerify's criterion (§3.3).

**Decision (2026-10-23).** Add a field and keep the existing three unchanged:

- **`task_spec_success`** implements the §3.3 criterion:
  - normal completion;
  - primitive convergence;
  - the shared precondition at entry;
  - the shared postcondition at the final state, or goal occupancy for Grid/Pyramid.
- **`success_criterion: "task-spec-v1"`** is recorded alongside it.
- **Primary metric:** `task_spec_success` for new runs; batch, recovery and report summaries aggregate it next to the existing fields. In recovery it is computed per attempt, and the recovery outcome is the first attempt for which it holds.
- **Existing fields:** `success`, `first_attempt_success` and `task_success` keep their definitions, so Revision 4.x and Stage B results stay comparable. They become diagnostics.
- **Missing values:** older results have no `task_spec_success`. Do not backfill it from their saved fields; recompute it from saved final states only when needed and label the result as recomputed.

**Implementation (2026-10-23).**

- `tasks.evaluate_terminal` reports `task_spec_precondition` (entry) and `task_spec_postcondition` (final) for Stack, Unstack, Reverse and Restack. It includes the table in the domain for the last three, as upstream collection does. `dsl_postcondition` remains Stack/Unstack-only.
- The adapter counts `unconverged_primitives` itself, so a policy that catches `PrimitiveFailure` cannot report task-spec success.
- `runner` sets `task_spec_success` only after normal completion, so a raised failure scores false. It also records `success_criterion` and computes `settled_environment_success` after recording ends, outside the trial deadline. The 50 settling steps are not charged to the simulation budget.
- Recovery reports `task_spec_success`, `task_spec_success_round` and `first_attempt_task_spec_success`. The new `recovery.success_field` selects the field that ends recovery. Its default, `task_success`, preserves cap-recovery behavior; `task_spec_success` stops at the first attempt meeting RoboVerify's criterion.
- Batch summaries add first-attempt and final task-spec success columns.
- Validation:
  - all 223 CaP tests pass;
  - real MuJoCo Stack runs of a scripted reference policy (seeds 0 and 1) scored `task_spec_success` and `settled_environment_success` true;
  - live OpenRouter smoke trials with failing outcomes recorded every new field.

**Further consequences.**

- Derive each task instruction from the shared specification and version it in the prompt package. The current Stack sentence matches `∀x. ON*(x, b0)` but does not mention the same-column tolerance.
- Report "symbolic success, physical failure" counts separately.

## 4. Per-task adaptation

### 4.1 Shared changes

- **Environment configuration.** Add task options to the CaP configuration: `initial_tower_blocks` for Restack, `grid_rows`/`grid_cols` for Grid, `bottom_row_blocks` for Pyramid. Validate conflicts with `num_blocks` through the upstream resolvers (`resolve_tower_blocks`, `resolve_goal_configuration`). Record the resolved values in trial metadata.
- **Settling.** Match upstream collection: Stack, Restack, Grid and Pyramid settle for 50 steps; Unstack and Reverse do not. Snapshots bypass reset and settling, as in upstream replay.
- **Start validation.** For tower tasks, evaluate the shared precondition with `include_table=True` wherever the formula mentions `tbl` (Unstack, Reverse, Restack), as upstream collection does. Also check the physical assumptions the evaluator relies on (Reverse canonical order). Reject a violating start before any model call.
- **Terminal evaluation.** Apply §3 uniformly. The symbolic postcondition needs `entry_obs` for frozen geometry (Reverse).
- **Scenes.** Use held-out seeds or archived snapshots that are separate from any demonstration set used in [Revision 6](revision_6_add_demonstration.md). Keep paired scenes across methods.
- **Time limits (user decision, 2026-10-23).** All six tasks, including Unstack, use the same CaP defaults: 600 seconds per trial, and for recovery 600 seconds per round and 3000 seconds total. Rollout length remains bounded by `max_sim_steps` and the 50-step primitive limit.
  - Current code clamps Unstack to 60 seconds in three places: the trial `timeout` (`runner.py`), the recovery total (`recovery.py`) and the batch watchdog (`batch.py`). That 60-second cap is per-trial wall-clock time covering generation, repair, queries, physical execution and recording; deferred video rendering is outside it.
  - Remove the three Unstack special cases. Note in the CaP `AGENTS.md`/README that the repository's Unstack 60-second rule governs RoboVerify's `synthesize_cfg` process alarm, not CaP trials.
  - Record the effective caps in each trial's metadata. Earlier CaP Unstack results ran under the 60-second cap; do not mix them with uncapped results.

**Implementation (2026-10-25).**

- `RunConfig` has `initial_tower_blocks` (Restack) and `grid_rows`/`grid_cols`/`bottom_row_blocks` (goal tasks), with `run_cap` options of the same names. A value for another task is rejected. `tasks.environment_options` resolves them through `resolve_tower_blocks` and `resolve_goal_configuration`; `adapter_config.json` records the resolved values and the settling steps.
- Fresh scenes settle for `STACK_SETTLING_STEPS` (50) for Stack, Restack, Grid and Pyramid; snapshots are not settled. The Stack settling loop is unchanged.
- `validate_start` evaluates the shared precondition for all four tower tasks, with the table in the domain for Unstack, Reverse and Restack. Reverse also keeps its canonical-order check. Restack also requires b0 to share a column (`ON*` in either direction) with another physical block; the configured tower height is not checked, so older snapshots recorded as `partial_stack` still load.
- The Unstack 60-second special cases are removed from `runner.py`, `recovery.py` and `batch.py`, and from the `run_cap` CLI, which no longer skips draining the deferred video queue for Unstack. The effective caps are the configured values, which `config.json` already records.
- Real MuJoCo check: fresh starts for seeds 0–4 pass `validate_start` for Stack (3 blocks), Unstack (3), Reverse (3) and Restack (4 blocks, tower heights 3 and 2).

### 4.2 Table reference (Unstack, Reverse, Restack)

`tbl` is the table as an element of the relational block domain:

- **Predicates:** it is isolated. Only `ON*(tbl, tbl)` and `Higher(tbl, tbl)` hold among pairs involving it, and Scattered is false for every such pair (`util/on.py`, matching the high-level axioms).
- **Formulas:** Unstack, Reverse and Restack quantify over a domain that includes `tbl`. Their formulas therefore use `x = tbl` or `x ≠ tbl` to include or exclude it explicitly.
- **Programs:** a variable may hold `tbl`, for example `b <- tbl` in Reverse, meaning "place the next block on the table".
- **Stack:** `tbl` does not appear. `task_spec("stack")` does not mention it, upstream collection and `synthesize_cfg` build the Stack context without it (`use_tbl` is false), and the reference program starts from `b <- b0`. CaP Stack evaluates its formulas with `include_table=False`, and the CaP predicate API never includes the table for any task.

The upstream controllers now resolve `tbl` in `Move`/`Release` through the physical table frame: XY is `table_position[:2]` and the reference Z is half a block below the table surface. All three reference programs use it. CaP currently has no `tbl` handle, so a policy must place blocks on the table relative to another block resting on it, using coordinate arithmetic on offsets.

**Decision (2026-10-23): option A.**

- Expose `tbl` as a `Move`/`Release` reference only, resolved through the same table frame as the upstream controllers.
- It is not pickable and not part of the query domain.
- Predicates keep the relational semantics: only `ON*(tbl, tbl)` and `Higher(tbl, tbl)` hold, and Scattered is false for any pair involving `tbl`.
- This matches the primitive vocabulary of RoboVerify's reference programs and requires a checker/API contract version bump.

**Implementation (2026-10-25).**

- `contracts.task_inputs` binds `b0` and, for Unstack, Reverse and Restack, `tbl`. The adapter, runtime, both methods' static checks and `DSLContract.for_task` use the same inputs, so `tbl` appears in the recorded contract's `input_types`. A Stack policy that names `tbl` fails the checker as an unknown name.
- The adapter maps `tbl` to the upstream `"tbl"` reference for `Move`/`Release`, which the shared controller resolves with `TABLE_REFERENCE_MODE` (`table-surface-minus-half-block-v1`). `Pick(tbl)` and `get_block_position(tbl)` raise `CapError`; queries range over blocks only. Predicates evaluate `tbl` on a scene with the table included, so only `ON_star`, `Higher` and `eq` hold, and only for `(tbl, tbl)`.
- The model-facing description is the method `context` (`tasks.task_context`), recorded as `input_context`; the three prompt packages are unchanged, so Stack prompts and `prompt_package_sha256` values are identical to Revision 4.x. `adapter_config.json` records `table_handle` (`cap-table-handle-v1` and the reference mode) and the predicate domain.
- **No checker version bump.** The checker's rules are unchanged; the new input is visible in the contract's `input_types` and its digest. Bumping `CHECKER_VERSION` would invalidate existing checkpoints and recorded contracts, including the Stack round-0 seeds that recovery experiments reuse, without any change in what Stack checks.
- Validation: unit tests for binding, table-frame targets, refused operations, relational predicates and the contract for all three methods. In real MuJoCo, scripted Method 1 policies modelled on the upstream Unstack and Reverse reference programs, placing with `tbl`, scored `task_spec_success`, `success` and `settled_environment_success` true for seeds 0 and 1 (3 blocks), with no unconverged primitive.

The rejected option B kept the block-only API and documented a table-placement idiom. Collection-only motion references (`Y_min`, `Z_m`, `gripper`) used by the Unstack reference program stay unavailable: upstream synthesis/verification loading rejects them too.

### 4.3 Unstack

- Environment and start: lateral tower with b0 at the bottom; the shared precondition is already checked.
- Terminal: shared postcondition (no off-diagonal `ON*`) as success; `_roboverify_unstack_all_on_ground` (every block at table height, no `on()` relation) as diagnostic, final and settled.
- Free table space is on the robot's lower-Y side of the tower; the instruction must not prescribe a strategy beyond the task statement.
- Work: §3.5, §4.1, §4.2 (`tbl` handle), the instruction revision.

### 4.4 Reverse

- Environment and start: lateral canonical tower. Check both the shared precondition (`ON*_zero = ON*` at entry) and the canonical order.
- Terminal: shared postcondition against frozen entry `ON*_zero` as success (currently `None`); `_roboverify_reverse_tower_complete` as diagnostic, final and settled.
- `ON_star_zero` is already exposed for Reverse only and keeps trial-entry geometry; keep it.
- The reference program supports 2–4 blocks. Use the same range for CaP comparisons unless a larger range is collected and audited.

### 4.5 Restack

- Environment: one lateral source tower containing b0 plus loose blocks; `initial_tower_blocks` defaults to `max(2, num_blocks − 1)`. Settle for 50 steps (currently skipped in CaP).
- Start: shared precondition, plus the environment invariant that b0 is inside the source tower.
- Terminal: shared postcondition as success; `_roboverify_stack_tower_complete` as diagnostic, final and settled.
- Temporary placements need free table space on the lower-Y side; see §4.2.
- Update the instruction: the start contains one source tower and loose blocks, not several towers.

### 4.6 Grid and Pyramid: goal interface

The upstream DSL uses goal-only pointer traversal: `GoalAssign`, `MoveRight`, `MoveDown`, goal equality/null guards, and the occupancy predicates `AtGoal`/`HasBlock`. `Move`/`Release` accept goal-center references; `Pick` accepts physical blocks only. CaP has no goal type.

Required CaP interface (names to be settled with the checker contract):

| Element | Behavior |
| --- | --- |
| Goal handles | Opaque per-trial handles, separate from block handles; head goal `h0`; a null value |
| Traversal | `right(g)` and `down(g)` return a goal or null, following the immutable layout's r/d links |
| Predicates | `AtGoal(b)` and `HasBlock(g)` with the shared 20 mm occupancy semantics |
| Actions | `Move`/`Release` accept goal references; `Pick` rejects them |
| Position read | `get_goal_position(g)`, analogous to `get_block_position` |
| Queries | Goal and block domains stay separate; the shared checker needs a goal type |

- Environment: `make_goal_env` with the configured dimensions; 50 settling steps.
- Start: goals empty and blocks pairwise Scattered (as in `validate_goal_trace`).
- Terminal: final-state goal occupancy, as in `validate_goal_trace`, as success. `_roboverify_goal_layout_complete` uses the same occupancy function and serves as the final and settled diagnostic. There is no symbolic formula.
- Pyramid's upper goals require supporting blocks below; occupancy is checked in 3D.
- Recording: goal layouts and goal bindings in snapshots use the upstream goal recording extension.

### 4.7 Implementation order

1. `task_spec_success` (§3.5) with physical diagnostics, for all tasks. **Implemented.**
2. §4.1 shared changes, including the uniform time limits, and the §4.2 `tbl` handle. **Implemented.**
3. Unstack, then Reverse, then Restack.
4. Goal interface, then Grid, then Pyramid.

## 5. Validation and acceptance

- Unit tests per task: valid and invalid starts; goal and non-goal terminal states; agreement and disagreement cases from §3; handle binding (`b0`, `tbl`, goals).
- Run each task's upstream reference program through the CaP adapter API. It must pass the CaP start and terminal checks. This checks the adapter, not CaP.
- Check snapshot restoration, action/observation alignment, primitive convergence and optional video frame counts.
- Run CaP comparisons on paired held-out scenes once each adapter is ready, after Stage B finishes and only within the GPU and request limits in force.
- This document does not authorize implementation or new experiments yet.
