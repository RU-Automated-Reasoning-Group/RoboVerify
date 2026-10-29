# Revision 9: a block-count generalization checker for CaP policies

- **Status:** plan, 2026-10-28. Nothing in this document is implemented. The
  feasibility limits in Section 3 come from the max-block probe in
  `runs/cap-generalization/20261028-max-blocks/` (Section 3.4).
  2026-10-29: the opt-in relaxed reset bounds of Section 3.5 are implemented
  for all six tasks and measured; the checker itself is not implemented.
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
2. **Physical limits, independent of any policy** (all 10 seeds;
   Section 3.3):
   - Stack: 6 blocks (7 on 4 of 10 seeds);
   - Unstack and Reverse: 18 (19 on some seeds);
   - ReStack: 18 = 16-block tower + 2 loose (19 on some seeds);
   - Grid: 9 (3 × 3), also 4 × 2, 2 × 4 and 5 × 1;
   - Pyramid: 10 (4-3-2-1).

   The tower tasks reach the arm's height limit at 18–19 blocks, and from 20
   blocks MuJoCo's contact buffer overflows. Stack is limited by the reset
   sampler, Grid by goal cells out of reach, and Pyramid by source capacity.
   With the opt-in relaxed bounds
   (`relax_boundaries_for_generalization_test`, Section 3.5), the limits on
   all 10 seeds become:
   - Stack, Unstack and Reverse: 19;
   - ReStack: 19 in total, with 2, 4 or 8 loose blocks;
   - Grid: 16 (4 × 4);
   - Pyramid: unchanged at k = 4.

   MuJoCo's contact buffer is then the binding limit for every task.
3. **A failure counts against the policy only on a physically feasible
   scene.** Feasibility is decided per (task, count, seed) without running
   any program:
   - the reset scene and one teleported goal scene are built, stable and
     satisfy the pre- and postcondition;
   - every top block centre is within the arm's measured reach.

   Controller budgets, transport heights and program design are therefore
   part of what the checker measures, not part of feasibility. Section 4
   gives the details.
4. **Recommended order:**
   - **G0:** physical feasibility tables per task (done for seeds 0–9,
     Section 3.3).
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
| Arm reach in Z | Fetch kinematics | Physical limit for towers (Section 3.3). Within one 50-step primitive the reachable height is lower still, so that shortfall is `primitive_failed`, a policy/controller failure |
| Supplied-program design | `examples/reverse.py` accepts 2–4 blocks; the tower examples set their clearance to `N·L + 0.10` above the table reference; `While` defaults to `max_iters=10` | Oracle failures that say nothing about task feasibility |

### 3.2 Arm reach within one primitive budget (measured)

This table is about the controller, not about physics; Section 3.3 has the
physical reach. The empty gripper was servoed with the shared controller (`get_move_action`)
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

### 3.3 Physically feasible counts (measured)

Probe: `runs/cap-generalization/20261028-max-blocks/physical/`; its README
has the method, the commands and images of the initial and goal state for
every count. Seeds 0–9 per layout. A seed is feasible when all three hold:
- **initial:** the reset scene (CaP's `prepare_environment` and
  `validate_start`) stays within 5 mm for 100 extra steps;
- **goal:** a teleported goal configuration stays within 5 mm for 100 steps
  and satisfies the postcondition;
- **reach:** every top block centre of both scenes is below the highest
  gripper Z the arm reaches at that column. Reach is measured by servoing the
  empty gripper (1,000 steps per move), bisecting on Z, with 2 mm tolerance.
  The target is reached if either path gets there: straight, or rise, move
  horizontally and descend 6 cm. A straight path alone gets stuck and
  underestimates reach; an earlier version of this table used it and reported
  9 blocks for the towers.

| Task | Feasible on all 10 seeds | Largest N on some seeds | Binding limit |
| --- | --- | --- | --- |
| Stack | 2–6 | 7 (4/10) | Reset sampler: 7 scattered blocks on 4/10 seeds, 8+ on none |
| Unstack | 2–18 | 19 (8/10) | Reach above the tower column: 1.22–1.53 m in the tower region, against a top centre of 1.24 m at N = 18. 20 blocks overflow the contact buffer |
| Reverse | 2–18 | 19 (7/10) | As Unstack, for both the source and the reversed tower |
| ReStack (tower + 2 loose) | 4–18 (tower 2–16) | 19 (8/10) | As Unstack, for the rebuilt N-block tower. Reset allows at most 2 loose blocks |
| Grid (rows × cols) | ≤ 3 × 3; 4 × 2, 2 × 4, 5 × 1 | — | 4 × 3: the far cell (0.85 m from the base) is out of reach even at table height. Larger layouts fail the environment's 0.8531 m goal-workspace check |
| Pyramid (k) | 2–4 | — | k = 5 sources cannot be placed; k ≥ 6 overflows the contact buffer |

Towers of up to 19 blocks are built and stable. At 20 blocks MuJoCo's
contact buffer (`nconmax` 100) overflows.

A separate best-site test (`best_site/` of the probe) asked how tall a tower
can be where the arm reaches highest. Within 0.2–0.4 m of the base, reach
exceeds 1.59 m. Towers were built and their top centre reached by the open
gripper on the staged path, without disturbing the tower:
- 18 blocks at every tested site, including the tower region's near corner;
- 17 blocks at its far corner (0.70, 0.25);
- 19 fails: the arm knocks the tower over while approaching, or the tower
  falls by itself.

Moving b0 to the best site therefore does not raise the ceiling above 18
without changing the simulator.

The earlier policy-witness probe (supplied programs, low-clearance oracles,
sampled CaP programs; `oracles/`, `cap_replay/`) found lower limits:
Unstack 7, Reverse 7, ReStack 6, Grid 2 × 4 unsolved. The difference between
those numbers and this table is controller and program behaviour, which is
what this revision measures. For example:
- Unstack N = 10: within the arm's reach on every seed, but the shared Move
  did not converge within its 50-step budget;
- ReStack N = 8: the 8-block tower built by the robot fell during stacking.

Pyramid can therefore be tested at only one count above the trained one.
The relaxed bounds of Section 3.5 do not change this: there, k = 5 fits the
workspace but overflows the contact buffer (decision D5).

### 3.4 Probes

`runs/cap-generalization/20261028-max-blocks/` contains everything:
- `physical/`: the physical-limit probe of Section 3.3, with images
  (`images/`, `sheet_<task>.png`) and the path-aware reach map
  (`reach_v2.jsonl`);
- `best_site/`: the tallest tower at the best XY, with images;
- `reset/`: reset and precondition probes;
- `oracles/`: `collect_demos` runs of the supplied and low-clearance programs,
  and a video of the falling ReStack tower;
- `cap_replay/`: a preview of five CaP programs per task at larger counts;
- `witness_table.json`.

The earlier three-seed probes (`scratchpad/r9/`) are superseded.

### 3.5 Relaxed reset bounds (implemented)

- **Status:** implemented 2026-10-29 at the user's request, for all six
  tasks. Commits `7d9ec08` (environment) and `3f82cdb` (CaP `RunConfig`).
  Details are in
  [synthesis/docs/relaxed_reset_bounds.md](../../docs/relaxed_reset_bounds.md).
- **Option:** `relax_boundaries_for_generalization_test`, default off; one
  switch for every task. It is a constructor keyword, a `make_roboverify_*` /
  `make_goal_env` option and a `RunConfig` field. `environment_options`
  records it only when it is on.
- **Region:** resets and goals use one shared region measured as reachable:
  base-relative X ≥ 0.35 m and XY radius ≤ 0.80 m.
  - **Towers:** Stack's `b0` and the Unstack/Reverse/ReStack source towers
    are sampled at X 0.40–0.60 m, where reach is highest. The towers stay on
    their original side.
  - **Loose blocks:** ReStack's loose blocks may use the whole higher-Y side.
  - **Goal tasks:** Grid and Pyramid goals are centred at X 0.50 m, and their
    sources may use the whole lower-Y side.
- **Unchanged:** Scattered separation, the gripper clearance and explicit
  failure on exhausted retries. With the switch off, every reset is
  identical, draw for draw.
- **Measured limits:** the Section 3.3 probe was run with the switch on,
  10 seeds per layout. Data is in `runs/cap-generalization/20261028-max-blocks/physical_relaxed/`.

  | Task | Default (Section 3.3) | Relaxed, feasible on all 10 seeds | Relaxed binding limit |
  | --- | --- | --- | --- |
  | Stack | 2–6 | 2–19 | Contact buffer at 20 |
  | Unstack, Reverse | 2–18 | 2–19 | Contact buffer at 20 |
  | ReStack | total 4–18, 2 loose | total 4–19 with 2, 4 or 8 loose (two layouts lose seed 0 to an unreachable edge point) | Contact buffer at 20 |
  | Grid | ≤ 3 × 3, 4 × 2, 2 × 4, 5 × 1 | up to 4 × 4; also 3 × 4, 4 × 3, 5 × 3, 5 × 2, 2 × 5, 6 × 1 | Workspace (3 × 5, 6 × 2 and others have a goal outside it); 20 goals overflow the contact buffer |
  | Pyramid | k = 2–4 | k = 2–4 | k = 5: the 15-block goal overflows the contact buffer while settling |

- **Contact buffer:** `nconmax` is 100, and about 15 contacts are always
  taken by the robot base against the table plane. Raising it is a separate
  environment change (decision D5).
- **Probe adaptations:**
  - the arm is parked away from every goal block before the goal teleport;
    the old park point stood inside the relaxed tower sites;
  - Unstack's probe goal cells stay inside the region;
  - table-level block centres are checked for reach exactly.
- **Baseline:** relaxed scenes differ from the trained layouts even at the
  trained count. The checker therefore runs the relaxed reset from the
  trained count, so that layout failures can be separated from count
  failures (Section 7).

## 4. Separating "policy failed" from "scene infeasible"

Every (program, task, count, seed) execution gets one outcome:

| Outcome | Meaning |
| --- | --- |
| `success` | `task_spec_success` under `task-spec-v2+final-release-v1` |
| `policy_failure` | The policy fails on a **physically feasible** scene (Section 3.3) |
| `scene_infeasible` | The scene at this (task, count, seed) is not physically feasible: the reset raises, the initial or goal scene is unstable or violates its condition, or a top block is out of reach. The policy may still run, but the outcome is not scored |
| `budget_exhausted` | A scaled budget ran out (Section 5.4). Reported separately and, by default, counted as a policy failure on feasible scenes (decision D3) |

**Physical feasibility, not witnesses.** The same seed gives the same initial
state for every program, because the reset depends only on the seed and the
layout options. Feasibility is therefore computed once per (task, count, seed)
by the Section 3.3 probe, before any program runs. A successful execution is
not needed as proof.

Controller and program limits remain policy failures. Examples:
- a Move that does not converge within the shared 50-step primitive budget;
- a tower built by the robot that falls;
- a transport height above the reachable Z.

The low-clearance oracles and the supplied programs stay as diagnostics only.
They show that a feasible scene can also be solved through the DSL.

## 5. The checker

### 5.1 Count schedule per task

The schedule starts at the trained count and extends to the largest count
that is physically feasible on at least one seed (Sections 3.3 and 3.5):

| Task | Default reset | Relaxed reset (Section 3.5) |
| --- | --- | --- |
| Stack | 4, 5, 6, 7 | 4–19 |
| Unstack | 4–19 | 4–19 |
| Reverse | 4–19 | 4–19 |
| ReStack (tower, loose) | (3, 2) = 5 … (17, 2) = 19 | as default, plus more loose blocks, e.g. (11, 8) = 19 |
| Grid (rows × cols) | 3 × 2, 4 × 2, 2 × 4, 3 × 3, 5 × 1 | 3 × 2 … 4 × 4 (12 layouts) |
| Pyramid (k) | 3, 4 | 3, 4 |

Seeds that are physically infeasible at a count are excluded from that
count's denominator and reported separately.

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
  every scheduled count up to N, all physically feasible scenes succeed. A
  count where every scene is `scene_infeasible` does not break the chain, but
  it is marked.

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
- **Feasibility:** the physical probe runs once per (task, count, seed)
  before any program, into `scenes.json`.

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

- **Jobs:** 246 distinct programs × about 10 counts on average (Section 5.1) × 5 seeds
  ≈ 12,000 scripted
  executions, plus the physical probe (about 1–6 s per scene and 18 s per
  reach column).
- **Time per execution:** scripted CaP trials include process start-up,
  checking, recording and settling. The pool's average is about 20–60 s. The
  collection probes took 3–14 s per trajectory.
- **Wall time:** about 70–200 CPU-hours, or roughly 2–6 hours on 32–48
  workers on GPUs 6 and 7 with the existing scheduler. There is no API cost.

### 5.6 Outputs

Artifacts go under `runs/cap-generalization/<run>/`:
- `programs.json`: hash, task, trials sharing the source, source path;
- one batch per program;
- `scenes.json`: physical feasibility per (task, count, seed), with the
  failing condition and the reach margin;
- `results.jsonl`: one row per execution, with outcome, partial reward, env
  steps, budget usage, failure status and reason.

The report is `docs/revision_9_report.md`, written by
`reports/generalization_report.py`. Per task it contains:
- a feasibility table (scene_infeasible / feasible by count, with the
  binding limit);
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
  - `feasibility.py`: the physical probe of Section 3.3 (reset, goal
    teleport, stability, reach map) and images of the initial and goal state.
  - `outcomes.py`: classification from batch rows and the feasibility table.
- `synthesis/cap/experiments/revision9_generalization.py`: driver. It runs
  without model access, is resumable, and has GPUs 6 and 7 only in its
  allowlist.
- `synthesis/cap/reports/generalization_report.py`.
- The relaxed reset bounds (Section 3.5) are already in place:
  `environment/relaxed_workspace.py`, the samplers, the environment
  constructor and factories, and `RunConfig`. The checker only sets the
  switch in its run-config overrides.

**Tests** (`synthesis/cap/tests/test_generalization.py`):
- schedule overrides produce valid `RunConfig`s for every task and count;
- scripted replay of a small fixed Stack program at 3 and 4 blocks in MuJoCo;
- feasibility logic: a reset error, an unstable or false goal, or a top block
  above reach each make a scene infeasible;
- the largest generalizing count over synthetic outcome tables, including
  gaps;
- budget scaling.

## 7. Risks

- **Scene sampling bias.** Fresh resets at large N are biased toward layouts
  that the bounded sampler can still place. At Stack 7 the sampler fails on
  some seeds, so the feasible scenes are the easy ones. Report the
  sampler-failure rate next to every curve.
- **Relaxed layout shift.** The relaxed reset (Section 3.5) changes the
  layout distribution even at the trained count:
  - Stack blocks may lie 0.70 m to the side;
  - towers move up to 0.14 m towards the base;
  - goal layouts move 0.15 m towards the base.

  Report relaxed curves relative to the relaxed baseline at the trained
  count, and report the default-reset schedule separately.
- **Feasibility is a necessary condition, not a constructive proof.** The
  probe checks the start, a goal state and reach, not every intermediate
  state of some solution. For example, carrying a block over a tall tower may
  need a little more height than its top centre. Report the reach margin next
  to each policy failure near the limit.
- **Programs hard-code layout constants**, for example temporary rows at fixed
  offsets from `b0`, and fixed heights. Failing at large N for that reason is a
  genuine generalization failure, not an artefact.
- **Settling at height.** Tall towers can topple during the 50 settling steps;
  robot-built 7–8 block towers did in the witness probe. This is part of the criterion, but report
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
2. **D2. Feasibility criterion.** Recommendation: use the physical
   feasibility of Section 3.3 (build, stability, goal teleport, measured
   reach) and treat the oracles as diagnostics only. The alternative is
   execution witnesses (supplied programs, oracles, the CaP pool). These
   conflate controller limits with physics and give lower ceilings.
3. **D3. Budgets.** Recommendation: scale `max_steps` and `max_sim_steps` with
   N (Section 5.4), and count `budget_exhausted` on a feasible scene as a
   policy failure. The alternative is final_v2's fixed budgets, which make
   larger counts partly a budget test.
4. **D4. Schedule and stopping.** Recommendation: run the full schedule
   (Section 5.1) for every program, with no early stopping. The alternative is
   to stop each program at its first failing count, which is cheaper but
   loses non-monotone cases.
5. **D5. Environment limits.** Workspace bounds are decided: the user chose
   one opt-in switch that relaxes them for every task (2026-10-29;
   Section 3.5). Still open is `nconmax`, now the binding limit of every task
   (19 blocks; Pyramid k = 4).
   - Recommendation: keep it, and report 19 as the end of the test range.
   - The alternative is to raise it under a separately named option. That
     needs a new stability and approach check for towers of 20 or more
     blocks.
6. **D6. Smaller counts.** Recommendation: include 2 and 3 for the tower tasks
   (and 2 × 2 for Grid, k = 2 for Pyramid) as a cheap sanity check. The
   alternative is upward only.
7. **D7. Program pool.** Recommendation: run final_v2 (246 distinct programs)
   first, and wo_demo when that experiment finishes, with identical schedules
   and seeds so the two can be compared.
