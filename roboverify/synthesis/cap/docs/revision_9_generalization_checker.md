# Revision 9: a block-count generalization checker for CaP policies

- **Status:** plan, 2026-10-28. Nothing in this document is implemented. The
  feasibility limits in Section 3 come from the max-block probe in
  `runs/cap-generalization/20261028-max-blocks/` (Section 3.4).
- **Input:** the final programs of a CaP experiment. First the 300 `final_v2`
  programs (246 distinct after alpha-renamed AST deduplication, see
  [Revision 8, Section 3.1](revision_8_verification.md#31-size-and-duplicates)),
  later the `final_v2_wo_demo` programs.
- **Question:** each policy was generated and scored at one block count per
  task (Stack, Unstack, Reverse 4; ReStack 5 with a 3-block tower; Grid 3 × 2;
  Pyramid 3-2-1). The task instructions say that the number of blocks is not
  given ([Revision 7.1](revision_7_1_general_task_spec.md)), so a correct
  policy must also work at other counts. For every program and task, the
  checker increases the block count until the policy fails or the scene can no
  longer be built or reached by the arm, and reports the largest count at which
  the policy still succeeds.
- **Cost:** no model calls. Programs are replayed through the CaP runtime with
  scripted responses, so the only cost is simulation time.

## 1. Summary

1. **Run the unchanged policy at larger counts.** The source is replayed as a
   scripted response (`create_batch(..., responses=[source])`), with recovery
   and demonstrations off. The success criterion stays
   `task-spec-v2+final-release-v1`, and partial reward
   ([TASK_PARTIAL_REWARD.md](TASK_PARTIAL_REWARD.md)) is a graded secondary
   metric.
2. **The environment stops growing early.** The measured limits (feasible on
   all 10 seeds; Section 3.3) are:
   - Stack: 6 blocks; 7 on 3 of 10 seeds.
   - Unstack: 7; up to 10 on some seeds.
   - Reverse: 7; 8 on 7 of 10 seeds.
   - ReStack: 6 (4-block tower + 2 loose); 7 on 9 of 10 seeds.
   - Grid: 9 (3 × 3).
   - Pyramid: 10 (4-3-2-1).

   Different mechanisms stop each task: sampler capacity, explicit workspace
   rejection, arm reach in Z, controller non-convergence and MuJoCo's contact
   buffer. Section 3 gives the details.
3. **A failure counts against the policy only on a scene shown to be
   feasible.** RoboVerify's supplied programs are a poor oracle beyond the
   trained counts:
   - the Reverse program refuses more than 4 blocks;
   - the Stack program's 7-block tower fails the settled check;
   - the ReStack program exhausts its own loop budget from 7 blocks;
   - the tower programs raise their approach height with N.

   The checker therefore treats a scene as feasible when any execution on it
   succeeds: the supplied program, an adjusted feasibility oracle, or any CaP
   program in the pool. A policy failure on a scene that nothing solves is
   reported as unresolved and kept out of the policy's failure rate.
4. **Recommended order:**
   - **G0:** feasibility tables per task: reset probes on 10 seeds per count,
     plus the oracle.
   - **G1:** scripted replay of all distinct programs over the schedule
     (5 seeds per count).
   - **G2:** the report.
   - **G3 (optional):** wo_demo programs and comparisons.

## 2. How block counts are configured today

| Task | Trained setting (`TASKS` in [config_final_v1.py](../configs/config_final_v1.py), inherited by final_v2) | Run-config fields that change with N | Layout resolver |
| --- | --- | --- | --- |
| Stack | `num_blocks: 4` | `num_blocks` | [stack_reset.py](../../environment/stack_reset.py) `sample_stack_xy` |
| Unstack | `num_blocks: 4` | `num_blocks` | [tower_reset.py](../../environment/tower_reset.py) (one lateral column) |
| Reverse | `num_blocks: 4` | `num_blocks` | same as Unstack |
| ReStack | `num_blocks: 5`, `initial_tower_blocks: 3` | `num_blocks`, `initial_tower_blocks` | [restack_reset.py](../../environment/restack_reset.py) |
| Grid | `num_blocks: 6`, `grid_rows: 3`, `grid_cols: 2` | `grid_rows`, `grid_cols`, `num_blocks = rows × cols` | goal layout in [fpp_construction_env.py](../../environment/cee_us_env/fpp_construction_env.py) |
| Pyramid | `num_blocks: 6`, `bottom_row_blocks: 3` | `bottom_row_blocks = k`, `num_blocks = k(k+1)/2` | same |

- **Validation:** `RunConfig` accepts the layout options only for their own
  task (`TASK_OPTIONS` in [config.py](../config.py)).
  `tasks.environment_options` resolves them through the same resolvers as
  collection (`resolve_tower_blocks`, `resolve_goal_configuration`).
- **The MuJoCo model is generated per count** (`xml_gen.py`, `xml_gen_grid.py`
  under `environment/cee_us_env/fpp_construction/`), so there is no fixed block
  slot. The observation layout (`10 + 12 i`) also grows with N.
- **Budgets that scale with N.** The final configuration's per-trial budgets
  are:
  - `timeout` 3,600 s;
  - `max_steps` 10,000 execution ticks;
  - `max_sim_steps` 10,000;
  - per-primitive budget 50 steps (`ControlConfig`; fixed and not scaled).

  Execution ticks count query evaluations as well as steps (`budget.tick` in
  [adapter.py](../adapter.py) and [runtime.py](../runtime.py)). A pair
  quantifier evaluated in a loop costs O(N²) ticks per test and O(N³) per
  trial. For comparison, a 6-block Pyramid trial used 1,210 ticks and 318 sim
  steps at the trained size. Section 5.4 scales these budgets.

## 3. What limits the block count

### 3.1 Mechanisms

| Mechanism | Where | Effect |
| --- | --- | --- |
| Stack sampler capacity | `sample_stack_xy`: X ∈ [0.54, 0.70], Y ∈ [−0.20, 0.20] from the base, radius 0.70 m, per-axis separation 0.10 m, 0.10 m gripper clearance, 100 × 200 retries | `ValueError: Could not sample N scattered Stack blocks` |
| ReStack loose-block capacity | `RESTACK_X_RANGE` (0.40, 0.70), separation 0.105 m → at most 3 columns: the tower plus 2 loose blocks | `ValueError: Restack source workspace cannot fit this many loose blocks`; N can grow only through the tower height |
| Goal-layout reach | `_install_goal_layout`: every goal within `manipulability_r = 0.8531` m (XY) of the base | `ValueError: Goal layout exceeds the robot's nominal XY workspace` |
| Goal-task source capacity | `_roboverify_random_scattered_layout` plus the deterministic fallback inside the source rectangle | `ValueError: Cannot fit N scattered blocks in the goal-task workspace` |
| Contact buffer | generated XML, `nconmax` 100 | `MujocoException: Pre-allocated contact buffer is full` at about 20 blocks |
| Arm reach in Z | Fetch kinematics; the controller's 2 mm tolerance within a 50-step primitive | Primitive non-convergence (`primitive_failed`) |
| Supplied-program design | `examples/reverse.py` accepts 2–4 blocks; the tower examples set their clearance to `N·L + 0.10` above the table reference; `While` defaults to `max_iters=10` | Oracle failures that say nothing about task feasibility |

### 3.2 Arm reach (measured)

The empty gripper was servoed with the shared controller (`get_move_action`)
to one block above an N-level tower: target Z = resting height + N·L, with
L = 0.05 m and a resting centre height of 0.425 m. The table gives the error
after 50 steps (one primitive budget); converged means at most 2 mm.

| Location (from the base) | 4 | 5 | 6 | 7 | 8 | 10 | 12 levels |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Stack source centre (0.62, 0.00) | ✓ | ✓ | ✓ | ✓ | ✓ | 16 mm | 76 mm |
| Tower region, far corner (0.70, 0.25) | ✓ | ✓ | ✓ | 15 mm | 33 mm | 75 mm | 150 mm |
| Tower region, near corner (0.54, 0.15) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 51 mm |

The limit therefore depends on the sampled tower position.
- **Unstack and Reverse:** 6 levels are reachable everywhere in the source
  region, and 7–10 only in part of it.
- **What the probe ignores:** payload, transport paths and the destination
  side. The Reverse and ReStack destinations, and the CaP policies'
  temporary rows, lie elsewhere.

### 3.3 Feasible counts (measured)

Probe: `runs/cap-generalization/20261028-max-blocks/` (its README has the
details and the commands). Seeds 0–9 per layout. A scene is feasible when any
witness succeeds on it under `task-spec-v2`:
- the supplied program;
- the low-clearance oracles of Section 4 (transport at the highest block plus
  6 cm or 4 cm);
- five sampled final_v2 CaP programs (seeds 0–2 only).

Reset alone builds every Unstack, Reverse and ReStack tower up to 16 blocks.
Those scenes are stable for 100 extra steps and satisfy their preconditions,
so for these tasks the limit comes from the arm, not from scene construction.

| Task | Feasible on all 10 seeds | Feasible on some seeds | Limit |
| --- | --- | --- | --- |
| Stack | 2–6 | 7 (3/10) | Sampler: 7 blocks on 4/10 seeds, 8+ on none. A robot-built 7-tower can fall during settling |
| Unstack | 2–7 | 8 (8/10), 9 (7/10), 10 (3/10) | Arm reach in Z: Move/approach to the tower top does not converge. The supplied program already fails at 6 because of its `Z_m + 0.15` retreat |
| Reverse | 2–7 | 8 (7/10) | Arm reach in Z; the rebuilt 8–9 block tower falls during settling. The supplied program refuses N > 4 |
| ReStack (tower + 2 loose) | 4–6 | 7 (9/10) | At most 2 loose blocks fit. The rebuilt N-block tower falls at 8, after which the stacking loop never ends |
| Grid (rows × cols) | up to 3 × 3 and 4 × 2 | — | 2 × 4, 4 × 3 and 5 × 1 build but no witness converges. 4 × 4, 5 × 2 and 3 × 5 are outside the goal workspace. 20+ blocks overflow the contact buffer |
| Pyramid (k) | k = 2–4 | — | k = 5 cannot be sampled; k ≥ 6 overflows the contact buffer |

Pyramid can therefore be tested at only one count above the trained one,
unless the source workspace is enlarged (decision D5).

### 3.4 Probes

`runs/cap-generalization/20261028-max-blocks/` contains everything:
- `reset/`: reset and precondition probes;
- `oracles/`: `collect_demos` runs of the supplied and low-clearance programs,
  and a video of the falling ReStack tower;
- `cap_replay/`: the CaP preview;
- `witness_table.json`.

The earlier three-seed probes (`scratchpad/r9/`) are superseded.

## 4. Separating "policy failed" from "scene infeasible"

Every (program, task, count, seed) execution gets one outcome:

| Outcome | Meaning |
| --- | --- |
| `success` | `task_spec_success` under `task-spec-v2+final-release-v1` |
| `policy_failure` | The policy fails and the scene is **feasible by witness**: some execution on the same scene and count succeeded |
| `unresolved` | The policy fails and no execution on this scene has succeeded yet. This is neither a success nor a policy failure |
| `scene_infeasible` | The reset raised (sampler, workspace, source capacity, contact buffer). No policy runs |
| `budget_exhausted` | A scaled budget ran out (Section 5.4). Reported separately and, by default, counted as a policy failure on feasible scenes (decision D3) |

**Feasibility witnesses**, for each (task, count, seed). The same seed gives
the same initial state for every program, because the reset depends only on
the seed and the layout options. A scene is feasible when any of these
succeeds on it:
1. RoboVerify's supplied program, where it accepts N;
2. an **adjusted oracle** (decision D2): the supplied program with
   - transport at the highest block plus a 6 cm hover (`Z_m` reference)
     instead of `N·L + 0.10`, with vertical rises before horizontal moves;
   - the Reverse range check lifted;
   - `max_iters` raised to 2N.

   The probe's versions are in
   `runs/cap-generalization/20261028-max-blocks/oracles/programs/`; they
   would become factories under `synthesis/cap/generalization/oracles.py`.
   `synthesis/examples/` stays unchanged;
3. any CaP program of the pool on that scene.

A success is a proof of feasibility. A failure of every candidate is evidence
of infeasibility, but not a proof. Arm-reach tables (Section 3.2) annotate
unresolved scenes, for example "tower top beyond the reachable Z at this XY",
but never turn them into failures.

## 5. The checker

### 5.1 Count schedule per task

The schedule starts at the trained count and extends to one count past the
first scene-infeasible count, so that each curve shows where the environment
stops:

| Task | Schedule |
| --- | --- |
| Stack | 4, 5, 6, 7 |
| Unstack | 4, 5, 6, 7, 8, 9, 10 |
| Reverse | 4, 5, 6, 7, 8, 9 |
| ReStack (tower, loose) | (3, 2) = 5, (4, 2) = 6, (5, 2) = 7, (6, 2) = 8 |
| Grid (rows × cols) | 3 × 2, 4 × 2, 3 × 3, 2 × 4, 4 × 3 |
| Pyramid (k) | 3, 4, 5 |

Smaller counts (2, 3) are optional (decision D6). They test the same
generality claim downward, and the environments support them.

### 5.2 Scenes and stopping

- **Seeds:** 5 per count (seeds 0–4; decision D1), the same for every
  program. At the trained count these scenes overlap the evaluation scenes,
  which [Revision 8 M0](revision_8_verification.md#7-milestones) covers.
- **No early stopping by default:** every program runs the whole schedule,
  because a failure at N does not imply failure at N + 1 (a scene-dependent
  failure can recur). The cost is small (Section 5.5).
- **Largest generalizing count of a program:** the largest N such that, at
  every scheduled count up to N, all feasible-by-witness scenes succeed. A
  count where every scene is `scene_infeasible` or `unresolved` does not
  break the chain, but it is marked.

### 5.3 Execution

- **Batches:** one scripted batch per distinct program:
  - `create_batch(directory, config, model, responses=[source], experiment_variants=[...], seed_per_trial=True, trials=5)`;
  - one variant per count, whose `run_config` override sets `num_blocks` and
    the layout options;
  - `recovery.enabled: false`, demonstrations off, `video_mode: "off"`
    (render only failures, on request).
- **Model:** `ScriptedModel` returns the source for the single round-0 call.
  The shared DSL checker re-checks it. With demonstrations off, the
  information-boundary check does not apply.
- **No model:** no key and no request gate. `transport_control` is unused
  because nothing goes out.
- **Run configuration:** final_v2's run configuration
  (`config_final_v2.run_config(task, demonstrations=False, ...)`) with only
  the counts and budgets overridden. The condition mode, prompt package,
  controller, settling and release check stay as they are.
- **Oracles:** the supplied and adjusted oracles run once per (task, count,
  seed) through the collection entry point, into the same results table.

### 5.4 Budgets

Fixed budgets would confound "does not generalize" with "ran out of budget".
Recommendation (decision D3):
- scale `max_steps` and `max_sim_steps` by `(N / N_trained)³` (ticks) and
  `(N / N_trained)` × 4 (sim steps);
- keep the 50-step primitive budget and `timeout` 3,600 s;
- record the usage of every budget per execution.

A policy that loops without progress is then still caught, as
`budget_exhausted`.

### 5.5 Cost

- **Jobs:** 246 distinct programs × about 5 counts × 5 seeds ≈ 6,200 scripted
  executions, plus about 30 counts × 5 seeds of oracle runs.
- **Time per execution:** scripted CaP trials include process start-up,
  checking, recording and settling. The pool's average is about 20–60 s. The
  collection probes took 3–14 s per trajectory.
- **Wall time:** about 35–100 CPU-hours, or roughly 1–3 hours on 32–48
  workers on GPUs 6 and 7 with the existing scheduler. There is no API cost.

### 5.6 Outputs

Artifacts go under `runs/cap-generalization/<run>/`:
- `programs.json`: hash, task, trials sharing the source, source path;
- one batch per program;
- `oracles/`;
- `scenes.json`: the feasibility witnesses per (task, count, seed);
- `results.jsonl`: one row per execution, with outcome, partial reward, env
  steps, budget usage, failure status and reason.

The report is `docs/revision_9_report.md`, written by
`reports/generalization_report.py`. Per task it contains:
- a feasibility table (scene_infeasible / unresolved / feasible by count);
- success rate against count, over distinct programs and also weighted by
  trial, with Wilson intervals;
- mean partial reward against count;
- the distribution of the largest generalizing count;
- the failure statuses at the first failing count, for example
  `primitive_failed` versus `failed` versus postcondition false;
- examples of the most common failure shape, linked to sources.

Figures:
- success against N per task (one panel per task, as in the final report);
- partial reward against N;
- a stacked bar of outcomes per count.

## 6. Code layout

- `synthesis/cap/generalization/`:
  - `programs.py`: extract the final programs from a run root (successful
    round, active checkpoint) and deduplicate on the alpha-renamed AST hash.
    This is shared with Revision 8 M0.
  - `schedule.py`: per-task count schedules → run-config overrides, and
    scaled budgets.
  - `feasibility.py`: reset probes per (task, count, seed), oracle runs and
    the witness table.
  - `oracles.py`: adjusted oracle factories (Section 4).
  - `outcomes.py`: classification from batch rows and witnesses.
- `synthesis/cap/experiments/revision9_generalization.py`: driver. It runs
  without model access, is resumable, and has GPUs 6 and 7 only in its
  allowlist.
- `synthesis/cap/reports/generalization_report.py`.

**Tests** (`synthesis/cap/tests/test_generalization.py`):
- schedule overrides produce valid `RunConfig`s for every task and count;
- scripted replay of a small fixed Stack program at 3 and 4 blocks in MuJoCo;
- witness logic: one success makes a scene feasible; all failures make it
  unresolved; a reset error makes it infeasible;
- the largest generalizing count over synthetic outcome tables, including
  gaps;
- budget scaling.

## 7. Risks

- **Scene sampling bias.** Fresh resets at large N are biased toward layouts
  that the bounded sampler can still place. At Stack 7 the sampler fails on
  some seeds, so the feasible scenes are the easy ones. Report the
  sampler-failure rate next to every curve.
- **The oracle is a witness, not a judge.** Unresolved scenes can hide real
  infeasibility or merely an oracle limitation. The reach tables annotate them;
  they are never scored.
- **Programs hard-code layout constants**, for example temporary rows at fixed
  offsets from `b0`, and fixed heights. Failing at large N for that reason is a
  genuine generalization failure, not an artefact.
- **Settling at height.** Tall towers can topple during the 50 settling steps;
  the Stack oracle already does at 7. This is part of the criterion, but report
  the settled versus end-of-rollout postcondition separately.
- **ReStack semantics at larger N.** `b0` is at a random level of a taller
  tower, and the unstack phase grows with it. Partial reward's unstack subtask
  uses the entry column above `b0`
  ([TASK_PARTIAL_REWARD.md, Section 3.4](TASK_PARTIAL_REWARD.md#34-restack)).
- **Reverse uses `ON_star_zero`.** CaP freezes the geometry at program entry,
  which is independent of N, so no special handling is needed.
- **Contact buffer.** Above about 20 blocks, MuJoCo's `nconmax` stops
  simulation. Raising it is an environment change (decision D5).
- **Budget scaling** changes per-trial limits. It must be applied identically
  to every program and recorded in the results.

## 8. Open decisions

1. **D1. Seeds per count.** Recommendation: 5, the same for all programs. The
   alternative is 10, which doubles the cost but tightens the per-program
   estimates.
2. **D2. Adjusted oracles.** Recommendation: add them under
   `cap/generalization/oracles.py`, with `Z_m` + hover transport, no Reverse
   range check and `max_iters` = 2N, as feasibility witnesses only. The
   alternative is to use only the supplied programs plus the CaP pool as
   witnesses.
3. **D3. Budgets.** Recommendation: scale `max_steps` and `max_sim_steps` with
   N (Section 5.4), and count `budget_exhausted` on a feasible scene as a
   policy failure. The alternative is final_v2's fixed budgets, which make
   larger counts partly a budget test.
4. **D4. Schedule and stopping.** Recommendation: run the full schedule
   (Section 5.1) for every program, with no early stopping. The alternative is
   to stop each program at its first failing count, which is cheaper but
   loses non-monotone cases.
5. **D5. Environment limits.** Recommendation: keep the environments unchanged
   for this revision and report each ceiling as the end of the test range. The
   alternative is a separately named environment option that enlarges the
   source workspaces (Stack, Pyramid) and `nconmax`. That would be a
   documented environment change with its own validation of the expert
   programs.
6. **D6. Smaller counts.** Recommendation: include 2 and 3 for the tower tasks
   (and 2 × 2 for Grid, k = 2 for Pyramid) as a cheap sanity check. The
   alternative is upward only.
7. **D7. Program pool.** Recommendation: run final_v2 (246 distinct programs)
   first, and wo_demo when that experiment finishes, with identical schedules
   and seeds so the two can be compared.
