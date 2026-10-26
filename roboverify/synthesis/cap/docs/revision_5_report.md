# Revision 5: six-task results under the frozen CaP configuration

**Status: completed 2026-10-25.** All six tasks ran 20 trials each (scene seeds 0–19) with the frozen CaP baseline, scored with RoboVerify's criterion `task_spec_success`.

| | Stack | Unstack | Reverse | ReStack | Grid | Pyramid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Round 0 | 14/20 | 3/20 | 1/20 | 0/20 | 12/20 | 16/20 |
| Final (within 4 recovery rounds) | **19/20** | **14/20** | **2/20** | **0/20** | **19/20** | **20/20** |

- **Stack, Grid and Pyramid are nearly solved.** Unstack is solved mostly through recovery. Reverse and ReStack almost never succeed.
- **The failures are motion failures, not interface failures.** Most attempts in the three hard tasks end with a `Move` that does not converge, mostly because the computed target is unreachable or wrong (Section 3.1). No attempt misused `tbl` or a goal handle (Section 3.4).

- **Plan:** [Revision 5 task plan](revision_5_add_tasks.md); goal interface: [revision_5_add_grid_pyramid.md](revision_5_add_grid_pyramid.md); block counts and pre/postconditions: [task_specification.md](task_specification.md).
- **Analysis artifacts:** `runs/cap-revision5/analysis/` (`chains.csv`, one row per trial; `attempts.csv`, one row per attempt; `summary.json`; the script `analyze_revision5.py`). Links into `runs/` work only in a checkout that has the run folders.

## 1. Configuration

| Item | Tower tasks | Grid, Pyramid |
| --- | --- | --- |
| Frozen configuration | [`config_frozen_v1`](../configs/config_frozen_v1.py) | [`config_frozen_v2`](../configs/config_frozen_v2.py) |
| Prompt package | `typed_v3` | `typed_v4` (= `typed_v3` plus the goal API; tower-task prompts are byte-identical between the two) |
| Checker / observation API | `shared-contract-v3` / `positions-v1` | `shared-contract-v4` / `positions-v2` |
| Batch | `runs/cap-revision5/20261025-frozen-v1-tower-tasks/` | `runs/cap-revision5/20261025-frozen-v2-goal-tasks/` |
| Run | 2026-10-25 15:13–17:44 EDT | 2026-10-25 21:37–22:07 EDT |
| Code (trial `git_sha`) | `8094f79`; the batch also recorded `6dd84a6` and `3db2ac9`, which differ only in docs | `1248116` |

Shared settings, from the [Revision 4.3 report, §3.5](revision_4_3_report.md#35-frozen-configuration):

- **Model:** Qwen3.8 27B FP8, `reasoning_effort=medium`, reasoning-termination guard on; vLLM on GPUs 6 and 7 behind port 8106.
- **Method:** Method 2, `dsl_only`, primitive prompt `p0101`.
- **Recovery:** on, up to 4 rounds, feedback profile `full`, `success_field: task_spec_success`.
- **Budgets:** 100k tokens initially, 200k per round, 900k per trial.
- **Scenes:** fresh resets at the paper's block counts:
  - Stack, Unstack, Reverse: 4 blocks;
  - ReStack: 5 blocks with a 3-block source tower;
  - Grid: 3 × 2;
  - Pyramid: bottom row of 3.

  Upstream settling applies. Each trial uses its own scene seed (trial *k* uses seed *k* − 1).
- **Instructions:** the `INSTRUCTIONS` entries of each configuration. They are derived from the shared task specification; Unstack/Reverse/ReStack were added in `ae07f2d` before any run used them.
- **Interfaces:**
  - Unstack, Reverse and ReStack bind `tbl` ([Revision 5, §4.2](revision_5_add_tasks.md#42-table-reference-unstack-reverse-restack)).
  - Grid and Pyramid bind `h0` and `null` and use `MoveRight`/`MoveDown`, `AtGoal`/`HasBlock` and goal references.

**Scoring.**

- **Tower tasks:** `task_spec_success` requires normal completion, every primitive converged, the shared precondition at entry and the shared postcondition at the final state.
- **Grid and Pyramid:** the same, with the start and final goal-occupancy checks of `goal-occupancy-v1`, without the per-release event checks.
- **Diagnostics:** the physical evaluator, at the final frame and after 50 settling steps.

## 2. Results per task

Counts are trials out of 20; intervals are 95% Wilson intervals. "Success round" is the round of the first attempt satisfying `task_spec_success` (0 = first attempt).

| Task | Round 0 | Final | Success round (0/1/2/3/4) | Stop reasons | Attempts |
| --- | --- | --- | --- | --- | ---: |
| Stack | 14 (70%) [48, 85] | 19 (95%) [76, 99] | 14 / 1 / 2 / 2 / 0 | 19 success, 1 rounds exhausted | 35 |
| Unstack | 3 (15%) [5, 36] | 14 (70%) [48, 85] | 3 / 2 / 1 / 5 / 3 | 14 success, 6 rounds exhausted | 75 |
| Reverse | 1 (5%) [1, 24] | 2 (10%) [3, 30] | 1 / 0 / 0 / 0 / 1 | 2 success, 16 rounds exhausted, 2 stopped before a rollout | 90 |
| ReStack | 0 (0%) [0, 16] | 0 (0%) [0, 16] | — | 18 rounds exhausted, 2 stopped before a rollout | 92 |
| Grid | 12 (60%) [39, 78] | 19 (95%) [76, 99] | 12 / 5 / 1 / 0 / 1 | 19 success, 1 rounds exhausted | 35 |
| Pyramid | 16 (80%) [58, 92] | 20 (100%) [84, 100] | 16 / 2 / 1 / 1 / 0 | 20 success | 27 |

### 2.1 Attempt outcomes

| Task | completed | primitive_failed | goal_failed | failed | budget_exhausted | generation_failed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Stack | 19 | 15 | 1 | 0 | 0 | 0 |
| Unstack | 13 | 59 | 3 | 0 | 0 | 0 |
| Reverse | 2 | 63 | 18 | 5 | 1 | 1 |
| ReStack | 0 | 81 | 4 | 2 | 4 | 1 |
| Grid | 19 | 3 | 13 | 0 | 0 | 0 |
| Pyramid | 20 | 1 | 6 | 0 | 0 | 0 |

- `failed` attempts are all `get` queries with no matching block (`No object satisfies get condition`).
- ReStack's `budget_exhausted` covers one round-0 token-allowance stop and three execution/query step-budget stops.
- `generation_failed` is a response cut at the 32,768-token output cap.
- An attempt counts as `completed` when it finishes normally and the physical evaluator succeeds. In this batch `completed` attempts are exactly the `task_spec_success` attempts, except for one Unstack attempt (Section 2.3).

### 2.2 Cost

| Task | Tokens, total | Median tokens per trial | Mean calls per trial | Active time, total | Median per trial |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stack | 594k | 11.5k | 1.9 | 4.1 h | 5 min |
| Unstack | 1.89M | 112k | 4.4 | 12.4 h | 45 min |
| Reverse | 2.79M | 144k | 5.5 | 18.4 h | 55 min |
| ReStack | 3.30M | 185k | 6.6 | 21.0 h | 64 min |
| Grid | 532k | 10.2k | 1.8 | 2.5 h | 4 min |
| Pyramid | 378k | 11.1k | 1.4 | 2.1 h | 5 min |

Totals include every recovery round. ReStack includes the infrastructure rerun of trial 009 instead of the original (Section 4).

### 2.3 Symbolic versus physical success

- **Final successes are physical successes as well.** Of the 74 trials that end with `task_spec_success`:
  - 73 also pass the physical evaluator at the final frame;
  - all 74 pass it after 50 settling steps.
- **One symbolic-only success:** Unstack trial 009, round 0.
  - The final frame satisfies the shared postcondition but not the physical evaluator.
  - After settling, both hold, so the last block was still settling when the recording stopped, as in [Revision 4.3 §4](revision_4_3_report.md#4-case-study-symbolic-success-without-physical-success).
  - Links: [attempt](../../../runs/cap-revision5/20261025-frozen-v1-tower-tasks/unstack/trials/cap_method_2-dsl_only-009/cap/20261025-191305-8094f79-cap_method_2-recovery-d0ccc549/attempts/round_000/cap/20261025-191308-8094f79-cap_method_2-unstack-2944a9c0), [policy](../../../runs/cap-revision5/20261025-frozen-v1-tower-tasks/unstack/trials/cap_method_2-dsl_only-009/cap/20261025-191305-8094f79-cap_method_2-recovery-d0ccc549/attempts/round_000/cap/20261025-191308-8094f79-cap_method_2-unstack-2944a9c0/artifacts/trial_001/checkpoints/cp_0002/policy.py).
- **No attempt passes the physical evaluator while failing the shared postcondition.**

### 2.4 Settled criterion `task-spec-v2` (recomputed 2026-10-26)

The tables above use `task-spec-v1`: the postcondition on the state where the rollout ended. Under `task-spec-v2` ([settled postcondition plan](../../cfg/docs/settled_postcondition.md)) the postcondition is evaluated only after 50 settling steps (gripper held, command 0.0). Each attempt's last rollout snapshot was restored and scored by the shared checker (`cfg/task_success.py`), without new model calls. The earlier numbers above are unchanged.

| | Stack | Unstack | Reverse | ReStack | Grid | Pyramid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Round 0, v1 | 14/20 | 3/20 | 1/20 | 0/20 | 12/20 | 16/20 |
| Round 0, v2 | 13/20 | 3/20 | 1/20 | 0/20 | 12/20 | 16/20 |
| Final, v1 | 19/20 | 14/20 | 2/20 | 0/20 | 19/20 | 20/20 |
| Final, v2 | 17/20 | 14/20 | 2/20 | 0/20 | 19/20 | 20/20 |
| Chains censored under v2 | 2 | 0 | 0 | 0 | 0 | 0 |

- **Only Stack changes.** Two Stack attempts that ended recovery as v1 successes fail after settling: trial 011 round 0 and trial 015 round 2. Both are leaning four-block towers: while settling, the top block's X offset from b0 grows from 22 mm to 28 mm and from 24 mm to 28 mm, past the 25 mm ON* column tolerance. The physical evaluator accepts both before and after settling.
- **Censoring.** Recovery stopped at the first v1 success, so these two chains never ran the rounds that v2 would have triggered. Their v2 final result (failure) is a lower bound; the Stack v2 final count is 17 to 19.
- **No attempt gains success under v2**, and every v2 success also passes the physical evaluator after settling.
- **Unstack and Reverse started from unsettled scenes** in this batch; the recomputation settles only the final state. Runs under [`config_frozen_v4`](../configs/config_frozen_v4.py) settle every start.
- **Consistency checks (354 attempts):** every attempt has a rollout; every last snapshot restores its recorded state exactly; the recomputed v1 result equals the recorded `task_spec_success`, and the recomputed postcondition the recorded `task_spec_postcondition`, in all 354; the settled physical evaluator equals the recorded `settled_environment_success` in all 354.
- **Artifacts:** `runs/cap-revision5/analysis/settled-v2/` (`attempts.csv`, `chains.csv`, `summary.json`, scripts `recompute_revision5.py` and `settled_attempt.py`).

## 3. Failure analysis

### 3.1 Move non-convergence dominates the tower tasks

Unconverged primitives over all attempts:

| Task | Move | Pick | Release |
| --- | ---: | ---: | ---: |
| Stack | 10 | 4 | 1 |
| Unstack | 59 | 0 | 0 |
| Reverse | 58 | 3 | 2 |
| ReStack | 74 | 6 | 1 |
| Grid | 2 | 1 | 0 |
| Pyramid | 0 | 1 | 0 |

Final position error of the unconverged Moves:

| Task | < 1 cm | 1–5 cm | ≥ 5 cm |
| --- | ---: | ---: | ---: |
| Stack | 4 | 4 | 2 |
| Unstack | 8 | 26 | 25 |
| Reverse | 7 | 25 | 26 |
| ReStack | 14 | 29 | 31 |

- **Most failed Moves end far from their target.** In Unstack, Reverse and ReStack, 162 of 191 failed Moves (85%) end 1 cm or more away, and 82 (43%) end 5 cm or more away. The controller cannot reach these targets within the 50-step primitive budget: they lie outside the reachable workspace, or behind an obstruction.
- **Typical cause: offset arithmetic.** For example, an Unstack policy passes an absolute coordinate as a relative offset:
  - it calls `Move(top, top, top, [cx - dx - blk[0], cy - dy - blk[1], -dz])`, where `blk` is the block's absolute position;
  - the target therefore lands near the world origin and the Move stops 17.5 cm short;
  - links: [policy](../../../runs/cap-revision5/20261025-frozen-v1-tower-tasks/unstack/trials/cap_method_2-dsl_only-001/cap/20261025-191305-8094f79-cap_method_2-recovery-b17824c5/attempts/round_000/cap/20261025-191308-8094f79-cap_method_2-unstack-3172bb49/artifacts/trial_001/checkpoints/cp_0002/policy.py), [attempt](../../../runs/cap-revision5/20261025-frozen-v1-tower-tasks/unstack/trials/cap_method_2-dsl_only-001/cap/20261025-191305-8094f79-cap_method_2-recovery-b17824c5/attempts/round_000/cap/20261025-191308-8094f79-cap_method_2-unstack-3172bb49).

  The same trial recovers in round 4.
- **Why the tower tasks are harder than Stack.** Unstack, Reverse and ReStack must place blocks on free table space beside a lateral tower, relative to `tbl` or to other blocks. Every such target is computed with offsets. Stack places every block on b0's column, and the goal tasks place every block on a goal reference with a zero XY offset (Section 3.3).

### 3.2 Other tower-task failures

| Failure | Reverse | ReStack |
| --- | ---: | ---: |
| `get` query found no block (`failed`) | 5 | 2 |
| Execution/query step budget exhausted | 0 | 3 |
| Round-0 token allowance exceeded before a rollout | 1 (trial 001) | 1 (trial 018) |
| Output cut at the 32,768-token cap | 1 (trial 007, round 2) | 1 (trial 008, round 0) |

- **Two kinds of stop before any rollout.** The four trials in the last two rows stopped before a rollout: they are the "stopped before a rollout" counts in Section 2.
  - **Token allowance.** After one or two generate/repair calls had used part of the 100k round-0 allowance, the next request no longer fit, and the trial stopped. This is a budget outcome under the frozen configuration, not an infrastructure failure.
  - **Truncated output.** A response cut at the output cap is a model outcome.
- **Reverse goal failures (18 attempts)** complete their programs but do not satisfy the reversed-order postcondition. Their final states were not classified further. Round 0 of trial 002 is an example: [attempt](../../../runs/cap-revision5/20261025-frozen-v1-tower-tasks/reverse/trials/cap_method_2-dsl_only-002/cap/20261025-191305-8094f79-cap_method_2-recovery-0c647cc6/attempts/round_000/cap/20261025-191307-8094f79-cap_method_2-reverse-3cf263e1).

### 3.3 Grid and Pyramid

- **Release from height.** Every Grid and Pyramid goal failure leaves goals empty: 1–6 of the 6 goals per attempt. The typical policy picks a block, moves 15 cm above the goal and calls `Release(g, 0.15)` without descending. The block drops and misses the 20 mm occupancy region, or knocks over an earlier block.
  - Example: Grid trial 016, round 0, all six goals empty after the run: [attempt](../../../runs/cap-revision5/20261025-frozen-v2-goal-tasks/grid/trials/cap_method_2-dsl_only-016/cap/20261026-013740-1248116-cap_method_2-recovery-7e135ebf/attempts/round_000/cap/20261026-013742-1248116-cap_method_2-grid-ef2c00bf).
  - Grid trial 016 left goals empty in rounds 0–2 and succeeded in round 4. Whether later rounds succeed by adding a descent was not checked attempt by attempt.
- **Grid has the only failing goal-task trial.** Grid trial 015 used all four recovery rounds without success: [chain](../../../runs/cap-revision5/20261025-frozen-v2-goal-tasks/grid/trials/cap_method_2-dsl_only-015/cap/20261026-013740-1248116-cap_method_2-recovery-13f4cc7e).
- **Primitive failures are rare:** 3 attempts in Grid and 1 in Pyramid. Placement targets are goal references with a zero XY offset, so no offset arithmetic is needed.
- **All Grid and Pyramid policies traverse the layout with `MoveRight`/`MoveDown`** (Section 3.4). In Pyramid, `MoveDown` moves up one layer, so following the traversal fills the lower layers first.

### 3.4 Interface usage

| Task | Attempts | Policies using `tbl` | Policies using `MoveRight`/`MoveDown` | Interface misuse |
| --- | ---: | ---: | ---: | --- |
| Unstack | 75 | 59 | — | none |
| Reverse | 90 | 66 | — | none |
| ReStack | 92 | 50 | — | none |
| Grid | 35 | — | 35 | none |
| Pyramid | 27 | — | 27 | none |

- **No interface misuse.** No attempt raised a `tbl` error ("tbl cannot be picked" or "not a block"), and none used a `null` goal cursor as a reference.
- **The interfaces are sufficient.** The low success rates in Reverse and ReStack come from motion planning and offset arithmetic (Section 3.1), not from missing or misunderstood interface elements. Under the [Section 3.6 rule](revision_4_3_report.md#36-six-task-evaluation) of the Revision 4.3 report, they are results, not grounds for a new configuration version.
- **No policy called `get_goal_position`.** The goal tasks need only goal references in `Move`/`Release`.

### 3.5 Example successes

| Task | Trial | Links |
| --- | --- | --- |
| Stack | 001, round 0 | [attempt](../../../runs/cap-revision5/20261025-frozen-v1-tower-tasks/stack/trials/cap_method_2-dsl_only-001/cap/20261025-191305-8094f79-cap_method_2-recovery-45d38d79/attempts/round_000/cap/20261025-191308-8094f79-cap_method_2-stack-1be3ffa0) |
| Unstack | 001, round 4 | [attempt](../../../runs/cap-revision5/20261025-frozen-v1-tower-tasks/unstack/trials/cap_method_2-dsl_only-001/cap/20261025-191305-8094f79-cap_method_2-recovery-b17824c5/attempts/round_004/cap/20261025-195957-8094f79-cap_method_2-unstack-54bd03b1) |
| Reverse | 012, round 0 | [attempt](../../../runs/cap-revision5/20261025-frozen-v1-tower-tasks/reverse/trials/cap_method_2-dsl_only-012/cap/20261025-195418-8094f79-cap_method_2-recovery-e7448bbe/attempts/round_000/cap/20261025-195420-8094f79-cap_method_2-reverse-f378a09d) |
| Grid | 002, round 0 | [chain](../../../runs/cap-revision5/20261025-frozen-v2-goal-tasks/grid/trials/cap_method_2-dsl_only-002/cap/20261026-013739-1248116-cap_method_2-recovery-a3407d10) |
| Pyramid | 001, round 0 | [chain](../../../runs/cap-revision5/20261025-frozen-v2-goal-tasks/pyramid/trials/cap_method_2-dsl_only-001/cap/20261026-013739-1248116-cap_method_2-recovery-5532aa70) |

Each attempt directory holds `artifacts/trial_001/checkpoints/*/policy.py` (the executed policy) and `artifacts/trial_001/execution/rollout_001/video.mp4`.

## 4. Infrastructure

- **Restoration bug and rerun (ReStack trial 009).**
  - **What happened.** Recovery round 1 stopped before any rollout with `Initial-state restoration did not reproduce full snapshot`.
  - **Cause.** `cfg/reset.restore` calls `sim.forward()`, which renormalizes the mocap quaternion. For this scene that changes the recorded value by one ulp (1.1e-16), so the exact-equality check failed. Replaying the recorded round-0 actions from either restoration reproduces the trajectory exactly, so physics was unaffected.
  - **Fix.** `a786334` restores the recorded `mocap_quat` after `forward()`, as for `qacc_warmstart`, with a regression test. `cfg/reset.py` is shared with RoboVerify, so the fix applies there too.
  - **Replacement.** Trial 009 was rerun with the same frozen-v1 job definition and scene seed in `runs/cap-revision5/20261025-frozen-v1-tower-tasks-rerun/restack/`, at `a786334`. This is a documented infrastructure replacement, as in [Revision 4.3 §1.4](revision_4_3_report.md#14-server-failure-and-rerun-2026-10-23). The analysis uses the rerun in place of the original. The rerun used all four recovery rounds and ended with `primitive_failed`.
  - **Scope.** It was the only trial ended by infrastructure. There were no transport errors, worker crashes or non-zero worker exits.
- **Goal-task launch.**
  - The first launch was rejected by `create_batch` before any trial ran: it accepted Method 2 only with `typed_v3`, while config v2 uses `typed_v4`.
  - `1248116` accepts any prompt package that has Method 2 templates, with a test.
  - The goal tasks then ran at `1248116`, after the goal interface was merged (`bb639f5`).
- **Code changes during the tower batch.** The main worktree's HEAD moved from `8094f79` to `3db2ac9` while trials were starting. The intervening commits change documentation only (`git diff 8094f79 3db2ac9 -- roboverify/synthesis` touches only `.md` files), so every tower trial ran the same code.

## 5. Interpretation and limits

- **Task difficulty under one frozen configuration.**
  - Stack, Grid and Pyramid reach 95–100% within four recovery rounds; in these tasks every placement target is a block column or a goal reference.
  - Unstack reaches 70%, mostly through recovery.
  - Reverse (10%) and ReStack (0%) need several precise table placements and transfers per trial, and nearly all of their failures are unreachable or mis-computed Move targets.
- **The configuration was tuned on Stack only** ([Revision 4.3 §3](revision_4_3_report.md#3-selected-cap-configuration-for-roboverify-comparisons)) and transferred without re-tuning, by design. The Stack round-0 rate here (70%, 4 blocks) is below the 3-block Revision 4.3 estimate (about 84%) but within its noise.
- **Twenty trials per task** detect only large differences: a 95% interval spans about ±20 points in the middle of the range.
- **Comparison with RoboVerify:**
  - These numbers are CaP's side of the comparison, on fresh scenes with the same block counts and the same `task_spec_success` criterion.
  - RoboVerify's synthesized programs still need to be executed and scored on the same scene seeds under the same criterion.
  - Revision 6 adds demonstration input to CaP ([revision_6_add_demonstration.md](revision_6_add_demonstration.md)).
- **One model, `dsl_only`, Method 2 only.** Other methods and condition modes were not rerun on the new tasks.
