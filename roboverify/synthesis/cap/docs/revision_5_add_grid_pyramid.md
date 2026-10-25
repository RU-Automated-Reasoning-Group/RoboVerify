# Revision 5: Grid and Pyramid in CaP

- **Status:** implemented and validated 2026-10-25 on branch `feature/cap-goal-interface` (Section 7). The 20-trial Grid/Pyramid runs under `config_frozen_v2` have not been run yet.
- **Goal:** run Grid and Pyramid in CaP with the same task definition, scenes and success criterion as RoboVerify, as steps 4 of [Revision 5](revision_5_add_tasks.md#47-implementation-order).
- **Constraint (user decision, 2026-10-25):** CaP is RoboVerify's baseline and must use RoboVerify's DSL. The goal interface may only expose primitives, predicates and statements that RoboVerify's DSL already has. No new DSL elements are added.

## 1. Why the tasks do not run today

`tasks.check_task` returns `adapter_gap` for Grid and Pyramid before any model call. RoboVerify supplies the goal semantics, but CaP's adapter, checker and prompt know only blocks:

| Needed for the task | RoboVerify (upstream) | CaP today |
| --- | --- | --- |
| Goal identities and layout | `GoalLayout`, `GoalRef`, head `h0`, `null` ([goal_layout.py](../../environment/goal_layout.py)) | No goal handles; goal positions are not readable by a policy |
| Traversal | `GoalAssign`, `MoveRight`, `MoveDown` ([instructions.py](../../api/instructions.py)) | None |
| Occupancy predicates | `AtGoal(b)`, `HasBlock(g)` in guards ([guard_eval.py](../../api/guard_eval.py)) | Predicates are ON, ON_star, Higher, Scattered, ON_star_zero, eq only |
| Placement at a goal | `Move`/`Release` accept a goal reference (`goal-center-xyz-v1`) | Move/Release accept blocks and `tbl` only |
| Types | Separate `Box` and `Goal` sorts | The checker has one handle kind, `object` |
| Task definition and scoring | `goal-occupancy-v1` ([goal_collection.py](../../cfg/goal_collection.py)); no `task_spec` formula | `task_spec_success` covers the four tower tasks only |
| Scenes and recording | `make_goal_env`, goal layout and cursor bindings in snapshots | Not wired into `prepare_environment` or the recorder |

The shared `task_spec` cannot be reused for scoring: the predicate language in `predicates/` has no Goal sort, and upstream marks goal-aware synthesis and verification as unsupported ([discrepancy 40](../../../../PAPER-DISCREPANCIES.md#40-grid-and-pyramid-execution-and-verification-have-separate-support)).

## 2. RoboVerify's goal DSL

The elements CaP may expose, with their upstream semantics:

| Element | Upstream form | Semantics |
| --- | --- | --- |
| Goal constants | `h0`, `null` | `h0` is the layout head (goal 0); `null` is the empty goal. Neither may be reassigned. |
| Goal assignment | `GoalAssign(x, y)` (`x <- y`) | Copies a goal binding. Goal and block variables are separate namespaces. |
| Advance in a row | `MoveRight(x)` | `x := x.r`, the next goal in the row, or `null` after the last. Advancing `null` is an error. |
| Advance to the next row | `MoveDown(x)` | `x := x.d`, the next row's head (the next layer up in a Pyramid), or `null`. Only a row head has a `d` link. |
| Block at a goal | `AtGoal(b)` | Block b's centre is within 20 mm (3D) of *some* goal centre. |
| Goal occupied | `HasBlock(g)` | *Some* block's centre is within 20 mm (3D) of goal g; false for `null`. |
| Goal comparison | `x == y`, `x != null` in guards | Identity of goal references. |
| Placement | `Move(r1, r2, r3, δ)`, `Release(r, z)` with goal references | A goal contributes its centre coordinate on that axis. |
| Pick | `Pick(b)` | Physical blocks only. |

The paper's program for both tasks (Figs. 17–18; [examples/goal_layout.py](../../examples/goal_layout.py)):

```text
d <- h0
while d != null:
    r <- d
    while r != null and ∃b'. ¬AtGoal(b'):
        Move(b', b', r, [0, 0, 0.15]); Pick(b'); Move(b', b', r, [0, 0, 0.15])
        Move(r, r, r, [0, 0, 0.15]); Move(r, r, r, [0, 0, 0]); Release(r, 0.15)
        MoveRight(r)
    MoveDown(d)
```

## 3. Design

### 3.1 Inputs

| Task | Bound input handles |
| --- | --- |
| Stack | `b0` |
| Unstack, Reverse, ReStack | `b0`, `tbl` |
| Grid, Pyramid | `h0`, `null` (goal handles); no `b0` and no `tbl` |

Goal tasks have no designated base block, so the adapter must not require a `b0` binding for them.

### 3.2 Goal handles

- **Goal handles are opaque per-trial values of a new kind, `goal`.** They are distinct from block handles; physical goal indices stay in the adapter, as block IDs do.
- **Goal assignment is ordinary Python assignment** of a goal value (`r = d`), which is `GoalAssign`.
- **`h0` and `null` are protected inputs** and cannot be reassigned, as upstream forbids.

### 3.3 Traversal: `MoveRight(x)`, `MoveDown(x)`

- **Statements with upstream semantics,** not functions that return a goal: `MoveRight(r)` rebinds the variable `r` in the current scope to `r.r`.
  - The argument must be a plain goal variable name, not `h0`, `null`, an expression or a call.
  - Inside a helper function, the rebinding affects that function's local variable, as any assignment does.
- **They neither move the robot nor step physics.** They are not physical actions, but they are state updates: they are not allowed in conditions, query lambdas or condition helpers.
- **Advancing `null` is a runtime error** (`CapError`), as upstream raises.

### 3.4 Predicates

- **`AtGoal(b)`** takes one block and **`HasBlock(g)`** one goal. Both use upstream's `occupancy` with the 20 mm tolerance on the current observation; the table is excluded.
- **Goal identity:** `==` and `!=` between goal values, including `null`, as for blocks. `eq` stays a block predicate.
- **The existing block predicates remain available** in goal tasks (ON, ON_star, Higher, Scattered, eq). `ON_star_zero` remains Reverse-only.

### 3.5 Actions

| Primitive | Reference operands |
| --- | --- |
| `Pick(b)` | Block only |
| `Move(x_ref, y_ref, z_ref, [dx, dy, dz])` | Each reference: block, or goal in goal tasks (or `tbl` in table tasks); mixing is allowed, as in `Move(b, b, r, …)` |
| `Release(ref, height_offset)` | Block or goal (or `tbl`) |

A `null` reference is a runtime error. The adapter maps a goal handle to an upstream goal alias, so the shared controller resolves it with `GOAL_REFERENCE_MODE` (`goal-center-xyz-v1`).

### 3.6 Queries

- **`exists`, `forall` and `get` keep ranging over blocks only.** This covers the paper's guard `∃b'. ¬AtGoal(b')`.
- **Goal-sorted quantification is not exposed.** Upstream guards can quantify over goals, but CaP lambdas carry no type annotation. Goal traversal is done with the cursor statements instead. This restricts the DSL; it adds nothing.

### 3.7 Observation API: goal positions

- **`get_goal_position(g)` is added to CaP's observation API**, next to `get_gripper_position()` and `get_block_position(b)` (user decision, Section 5, decision 3).
  - These readers are not RoboVerify DSL elements. They are the observation space from which a policy parameterizes primitives.
  - It returns the goal centre as an immutable `Position3`; `null` is a runtime error.
- **Condition rule unchanged:** in `dsl_only` mode, loop, branch and selection conditions must not depend on any position reader. Positions may only parameterize primitives. In `coordinate_predicates` mode they may be compared, as today.
- **Version bump:** the observation API becomes `positions-v2`. It equals `positions-v1` plus goal positions in goal tasks.

### 3.8 Not added

- `right(g)` / `down(g)` as value-returning functions (the DSL has statements).
- Goal binders in query lambdas (Section 3.6).

### 3.9 Scenes and scoring

- **Scenes:** `make_goal_env` with the frozen layout (Grid 3 rows × 2 columns; Pyramid `bottom_row_blocks: 3`), 50 settling steps after reset (already in `SETTLING_TASKS`), snapshots restored without settling.
- **Start check:** every goal empty and physical blocks pairwise Scattered, as in `validate_goal_trace`. A violating start fails before any model call.
- **`task_spec_success` for goal tasks** keeps the same structure as for the tower tasks:
  - normal completion;
  - every primitive converged;
  - the start check holds;
  - every goal is occupied in the final state (the terminal check of `validate_goal_trace`).
- **Event-level checks are dropped.** `validate_goal_trace` also checks per-release events (a release must not displace an occupied goal; the released block must reach a goal). These checks follow the reference program's instruction paths, which CaP policies do not share. Revision 5 §3.3 allows dropping them if the choice is recorded; dropped by user decision (Section 5, decision 2).
- **Diagnostics:** `_roboverify_goal_layout_complete` (same occupancy function), final and after 50 settling steps.

### 3.10 Recording and video

- **Snapshots:** capture the goal layout and the adapter's goal cursor bindings, using the upstream goal recording extension (recording version 2 with layout and bindings).
- **Video and replay:** deferred video and restoration reuse `cfg/reset.restore`, which already restores goal layouts. Goal marker mocaps stay fixed.

### 3.11 Prompt, instruction and configuration version

- **Prompt package `typed_v4`** = `typed_v3` plus API entries for `h0`/`null`, `MoveRight`, `MoveDown`, `AtGoal`, `HasBlock`, `get_goal_position` and goal references in Move/Release. Entries are rendered only when a task's API set contains them. The rendered prompt for the four tower tasks is therefore byte-identical to `typed_v3`; implementation will verify this by hashing the rendered prompts.
- **Task instructions** for Grid and Pyramid follow the Stack template, derived from `goal-occupancy-v1`:
  - the inputs;
  - initial condition: goals empty, blocks pairwise Scattered;
  - goal: every goal occupied within 20 mm;
  - block size;
  - the physical-completion sentence;
  - the closing line.
- **`config_frozen_v2`** = v1 with `prompt_version: typed_v4`, observation API `positions-v2`, the two instructions and the goal layouts. This follows the AGENTS.md rule: an interface-driven change becomes a new version used for all six tasks. Because the tower-task prompts are unchanged, v1 tower-task results remain valid under v2, provided the rendered-prompt hashes match.

## 4. Checker for this version (`shared-contract-v4`)

### 4.1 Value kinds

| Kind | Values |
| --- | --- |
| `object` | Block handles: `b0`, query results, lambda binders |
| `goal` | **New.** `h0`, `null`, and variables assigned from them |
| `number`, `bool`, `position`, `vector`, `none` | Unchanged |
| `tbl` | Remains an `object` input in table tasks (Revision 5 §4.2); the adapter enforces its Move/Release-only use at runtime |

### 4.2 Predicates

| Predicate | Signature | Tasks |
| --- | --- | --- |
| `ON(a, b)`, `ON_star(a, b)`, `Higher(a, b)`, `Scattered(a, b)`, `eq(a, b)` | object × object → bool | All six |
| `ON_star_zero(a, b)` | object × object → bool | Reverse |
| `AtGoal(b)` | object → bool | **New for** Grid, Pyramid |
| `HasBlock(g)` | goal → bool | **New for** Grid, Pyramid |

### 4.3 Primitives and statements

| Name | Signature | Tasks | Physical action |
| --- | --- | --- | --- |
| `Pick(b)` | object | All | Yes |
| `Move(x, y, z, [dx, dy, dz])` | ref × ref × ref × vector, ref = object or goal | All (goal refs only in Grid, Pyramid) | Yes |
| `Release(r, h)` | ref × number | All (goal ref only in Grid, Pyramid) | Yes |
| `MoveRight(x)`, `MoveDown(x)` | goal variable name (statement) | **New for** Grid, Pyramid | No (cursor update) |

### 4.4 Queries and readers

| Name | Signature | Note |
| --- | --- | --- |
| `exists(λ…)`, `forall(λ…)` | binders range over blocks | Unchanged |
| `get(λx…)` | one block binder → object | Unchanged |
| `get_gripper_position()`, `get_block_position(b)` | → position | CaP observation API, not RoboVerify DSL; unchanged |
| `get_goal_position(g)` | goal → position | **New for** Grid, Pyramid (observation API `positions-v2`); `null` fails at runtime |

### 4.5 New and changed rules

| Rule | Diagnostic | Applies to |
| --- | --- | --- |
| Goal and block values are distinct kinds; every API argument must have its declared kind (`Pick(h0)`, `HasBlock(b)` and `ON(r, b)` fail) | `TYPE` (existing) | Goal tasks |
| `h0` and `null` cannot be assigned or used as function parameters | `PROTECTED_BINDING` (existing, extended) | Goal tasks |
| `MoveRight`/`MoveDown` take exactly one plain goal variable name, not `h0` or `null` | `TRAVERSAL_TARGET` (new) | Goal tasks |
| `MoveRight`/`MoveDown` are statements only, never used as values | `TRAVERSAL_TARGET` (new) | Goal tasks |
| `MoveRight`/`MoveDown` are not allowed in conditions, query lambdas or condition helpers | `ACTION_IN_CONDITION` (existing, extended to cursor updates) | Goal tasks |
| Goals support only `==` and `!=` | `IDENTITY_COMPARISON` (existing, extended) | Goal tasks |
| In `dsl_only` mode, conditions must not depend on any position reader, including `get_goal_position` | `NUMERIC_CONDITION` (existing, extended to goal positions) | All |
| Query binders are blocks | `QUERY` (existing) | All |
| APIs outside the task's set are rejected | `API_UNAVAILABLE` (existing) | All |

Condition modes, loop/helper/dependency rules and the bounded-analysis contract are unchanged. The four tower tasks accept exactly the same programs as under `shared-contract-v3`.

**Version bump.** The accepted language changes (a new kind and new statements), so `CHECKER_VERSION` becomes `shared-contract-v4`. `DSLContract` will continue to accept recorded `v3` contracts for post-hoc audits of earlier runs. New checkpoints record v4, so recovery seeds recorded under v3 are not reused with v2 runs.

## 5. Decisions (user, 2026-10-25)

1. **Grid layout:** the upstream default, 3 rows × 2 columns with 10 cm spacing. Fig. 10e shows six goals without fixing rows versus columns. Pyramid: `bottom_row_blocks: 3` (layers 3, 2, 1).
2. **Event-level goal checks:** dropped. CaP scores only start and final occupancy (Section 3.9).
3. **Observation space:**
   - The position readers stay. In `dsl_only` mode they cannot appear in loop, branch or selection conditions, but the model needs block, gripper and goal positions to parameterize primitives.
   - `get_goal_position(g)` is therefore added to the observation API (Section 3.7).
   - The readers are an observation interface, not RoboVerify DSL elements.

## 6. Validation

- **Unit tests:**
  - goal handle binding;
  - traversal at row ends and the Pyramid layer change;
  - `null` errors;
  - AtGoal/HasBlock against `occupancy`;
  - Move/Release goal targets;
  - Pick rejection;
  - each rule in Section 4.5 (accept and reject cases);
  - tower-task contracts unchanged.
- **Reference program through the CaP API:** the Section 2 program, written as a CaP policy, must score `task_spec_success` in real MuJoCo for Grid 3×2 and Pyramid 3 on several seeds. This checks the adapter, not CaP.
- **Prompts:** rendered tower-task prompts under `typed_v4` hash-equal to `typed_v3`.
- **Smoke run:** one real-model trial per goal task; then 20 trials each under `config_frozen_v2`, reported in `revision_5_report.md`.

## 7. Implementation and validation (2026-10-25)

### 7.1 What was built

| Area | Change |
| --- | --- |
| Contracts | `GoalHandle`; `GOAL_TASKS`, `GOAL_PREDICATES`, `TRAVERSALS`, `get_goal_position`; per-task inputs and input kinds (`task_input_types`); `CHECKER_VERSION = shared-contract-v4`, `OBSERVATION_API_VERSION = positions-v2`, with v3/positions-v1 accepted for audits |
| Checker (`validation.py`) | Kind `goal`; `MoveRight`/`MoveDown` checked as statements (`traversal_flow`, rule `TRAVERSAL_TARGET`); `h0`/`null` protected; goal predicates, goal positions and goal Move/Release references typed. Goal API names are recognised and protected only where the task exposes them, and tower-task diagnostics keep their exact messages. Local helper checks and Method 1's helper discovery receive the task's inputs and API set |
| Runtime | Executes `MoveRight(x)`/`MoveDown(x)` by rebinding `x` in the current scope; goal identity comparisons; inputs typed by handle kind |
| Adapter | Goal handles (`h0` = goal 0, `null`); traversal along the layout's r/d links; `AtGoal`/`HasBlock` from upstream `occupancy`; `get_goal_position`; goal references mapped to upstream goal aliases (`goal-center-xyz-v1`); Pick and block readers reject goals; `null` references fail |
| Baseline | Before `exec`, rewrites the statement `MoveRight(g)` to `g = MoveRight(g)` (exec'd Python cannot rebind a caller's variable); applied only when the task exposes the goal API |
| Tasks and runner | `make_goal_env` with the resolved layout; goal start check and `goal-occupancy-v1` scoring in `evaluate_terminal`; goal input context; `adapter_config.json` records the layout, goal handles, reference mode and tolerance |
| Recording, video, recovery | Snapshots carry the goal layout and cursor bindings (upstream format); deferred video derives the layout options from the snapshot; logs serialise `{"goal_handle": token}`; `goal_clauses` feedback lists `HasBlock(goal_i)`; primitive target derivation resolves goal references to goal centres |
| Prompt and config | `typed_v4` (goal-task header/footer/examples and Move/Release entries as a variant selected by the goal API); `configs/config_frozen_v2.py` (v1 + `typed_v4` + layouts + Grid/Pyramid instructions) |

**Deviation from Section 3.3:** Baseline gets the in-place semantics by rewriting the statement before `exec`, not by interpreting it. Any other use of `MoveRight` in Baseline code (for example `x = MoveRight(g)`) runs unchanged and is reported by the post-hoc audit as `TRAVERSAL_TARGET`.

### 7.2 Validation

- **Unit tests:** 18 new tests in `tests/test_goal_interface.py`, covering the checker rules of Section 4.5, tower parity, the adapter and runtime on a goal fixture, Baseline's rewrite, prompt hashes and v1/v2. Full CaP suite and full repository suite (AGENTS.md): 665 tests, OK (4 skipped).
- **Tower parity on real programs.** The base checker (main worktree, `6dd84a6`) and the new checker evaluated 3,000 executed policies sampled from the Revision 4.3 and Revision 5 runs.
  - Each policy was checked under the Stack, Unstack and Reverse contracts in both condition modes: 18,000 checks.
  - Verdict, rule, message, line and column were identical in all 18,000 (17,745 pass, 255 fail).
- **Prompts:** the rendered API text for the four tower tasks is identical under `typed_v3` and `typed_v4` in both condition modes, and the method templates, baseline prefixes and rules are identical.
- **Reference program (Section 2) in real MuJoCo,** scripted responses under `config_frozen_v2` settings:
  - **12/12 trials** reached `task_spec_success`, physical success and settled physical success, with no unconverged primitive (Grid 3×2 and Pyramid 3; Method 2 seeds 0–2, Method 1 seed 3, Baseline seed 4, two deferred-video trials).
  - The deferred videos rendered.
  - Artifacts: `runs/cap-revision5/20261025-goal-interface-reference/`.
- **Recovery path:** a deliberately unreachable round 0 followed by the reference program recovered in round 1 for both tasks. The feedback contained `HasBlock(goal_i)` clauses and goal-referenced target derivations.
- **Real-model smoke** (`config_frozen_v2`, seed 100, one trial per task; `runs/cap-revision5/20261025-goal-interface-smoke/`):
  - **Grid:** round 0 `goal_failed` (two goals left empty), recovered in round 1; `task_spec_success` and physical success true; 2 model calls, 28.5k tokens.
  - **Pyramid:** round 0 failed because the policy used a cursor that had advanced to `null` as a Move reference (a runtime error by design), recovered in round 1; `task_spec_success` and physical success true; 2 model calls, 21.9k tokens.
  - Every generated program passed the checker; the model used `h0`, cursor statements, `AtGoal` queries and goal references as documented.
