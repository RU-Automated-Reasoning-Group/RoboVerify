# Revision 9: a block-count generalization checker for CaP policies

- **Status:** 2026-10-28 plan; updated 2026-10-29 for the test-case set
  `relaxed_v2`.
  - Implemented: the opt-in relaxed environment (Section 3.5), the 50-step
    primitive reach (Section 3.3.1), the test-case set `relaxed_v2`
    (Section 3.6), program extraction (`generalization/programs.py`) and the
    replay driver (`experiments/revision9_generalization.py`, Section 5).
  - Reach was redefined by the 50-step primitive budget on 2026-10-29; this
    withdrew `relaxed_v1` in favour of `relaxed_v2`. Limits measured with the
    earlier long-servo reach (Sections 3.2–3.4) are kept as history and
    overestimate what one Move reaches.
  - Results: [revision_9_report.md](revision_9_report.md).
- **Input:** the final programs of a CaP experiment. First the 300 `final_v2`
  programs (246 distinct after alpha-renamed AST deduplication, see
  [Revision 8, Section 3.1](revision_8_verification.md#31-size-and-duplicates)),
  later the `final_v2_wo_demo` programs.
- **Question:** each policy was generated and scored at one block count per
  task (Stack, Unstack, Reverse 4; ReStack 5 with a 3-block tower; Grid 3 × 2;
  Pyramid 3-2-1). The task instructions say that the number of blocks is not
  given ([Revision 7.1](revision_7_1_general_task_spec.md)), so a correct
  policy must also work at other counts. For every program and task, the
  checker runs the policy at every count up to the largest one the environment
  can physically hold, and reports where the policy succeeds and fails.
- **Design principles** (confirmed by the user, 2026-10-29):
  1. **A post-hoc evaluator of block-count generalization.** It evaluates
     finished policies; nothing is trained or regenerated.
  2. **As many blocks as possible.** The test range is limited only by
     physics, not by what any policy (or the supplied program) can do.
     More counts give more test cases on which a policy can fail. The
     checker's environment is therefore the opt-in relaxed one
     (`relax_boundaries_for_generalization_test`, Section 3.5): the
     reachable workspace and enlarged MuJoCo contact buffers.
  3. **Feasibility only asks whether the initial and target states are
     physically realizable and reachable.** Both must be buildable and
     stable, and every column must be served by Moves within the
     primitive's 50-step budget (decided by the user, 2026-10-29).
     It is decided without running a policy (Sections 3.3 and 4).
  4. **Every failure is attributed** either to the environment or to the
     policy (Section 4):
     - environment: an infeasible scene, or a simulator failure during the
       rollout;
     - policy: anything else on a feasible scene.
- **Cost:** no model calls. Programs are replayed through the CaP runtime with
  scripted responses, so the only cost is simulation time.

## 1. Summary

1. **Run the unchanged policy at larger counts.** The source is replayed as a
   scripted response (`create_batch(..., responses=[source])`), with recovery
   and demonstrations off. The success criterion stays
   `task-spec-v2+final-release-v1`, and partial reward
   ([TASK_PARTIAL_REWARD.md](TASK_PARTIAL_REWARD.md)) is a graded secondary
   metric.
2. **Physical limits, independent of any policy** (relaxed environment,
   reach within one 50-step Move; Sections 3.3.1 and 3.5):
   - Stack: 6 on all 10 seeds, 7–10 on 9, 11 on 8, 12 on 2;
   - Unstack and Reverse: 10 on all seeds, 11 on 7, 12 on 3;
   - ReStack: 10 in total, with 2, 4 or 8 loose blocks (11 on 28/30
     layouts and seeds, 12 on 1);
   - Grid: 23 shapes up to 6 × 4 (24), each on all seeds, with goals on the
     robot's midline and sources in any free area;
   - Pyramid: k = 5 (15 blocks).

   The binding limits are one Move's reach: from the arm's start posture the
   gripper gets no higher than about 1.0 m within 50 steps (towers of 10–11
   blocks), and it cannot get within 2 mm of the table in front of the
   robot's body (base-relative X 0.35 m near the midline). The feasible seeds
   form the test-case set `relaxed_v2` (721 cases; Section 3.6), which is the
   checker's schedule.

   The relaxed environment (`relax_boundaries_for_generalization_test`,
   Section 3.5) adds the reachable workspace and enlarged contact buffers.
   The default environment's limits (Section 3.3: Stack 6, towers 18, Grid
   3 × 3, Pyramid k = 4) used a servo reach of up to about 3,000 steps and
   are no longer used.
3. **A failure counts against the policy only on a physically feasible
   scene.** Feasibility is decided per (task, count, seed) without running
   any program:
   - the reset scene and one teleported goal scene are built, stable and
     satisfy the pre- and postcondition;
   - every initial and goal column is served by Moves within the 50-step
     primitive budget (Section 3.3.1).

   Controller budgets, transport heights and program design are therefore
   part of what the checker measures, not part of feasibility. Section 4
   gives the details.
4. **Order:**
   - **G0:** physical feasibility and the test-case set (done: `relaxed_v2`).
   - **G1:** scripted replay of every distinct final program of `final_v2`
     and `final_v2_wo_demo` on every case of its task (done, Section 5).
   - **G2:** the report ([revision_9_report.md](revision_9_report.md)),
     including RoboVerify's programs on the same cases.

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
| Supplied-program design | `examples/reverse.py` accepted 2–4 blocks until 2026-10-30 (check removed, [report](revision_9_report.md)); the tower examples set their clearance to `N·L + 0.10` above the table reference; `While` defaults to `max_iters=10` | Oracle failures that say nothing about task feasibility |

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

#### 3.3.1 Reach within one Move (correction, 2026-10-29)

The reach above used up to 1,000 servo steps per segment and up to three
segments. A Move has 50 steps. From the arm's start posture the gripper rises
to about 1.0 m within a few dozen steps and then creeps at about 0.2 mm per
step, with forearm/wrist self-contact observed; after hundreds of steps it is
higher and moves freely again. The "1.59 m" ceiling was therefore reachable
only by servoing far beyond one Move. A smoke run showed the effect: a CaP
Reverse policy lifting the gripper above a 12-block tower stopped at 0.994 m
and failed with `Move did not converge`
(`runs/cap-generalization/20261029-policy-timing/diagnostics/`).

On the user's decision, reach is now defined by the primitive budget
(`synthesis/cap/generalization/reach.py`). With an empty gripper in an
otherwise empty relaxed scene, every initial and goal column must allow:
- one Move from the start posture to the hover point, 0.05 m above the top
  centre;
- one Move down to the top centre, where a block is grasped or released;
- one Move back up to the hover point, which a carried block needs to clear
  the column.

Each Move uses Move's own controller, its 50-step budget and its 2 mm
tolerance. A long chain of millimetre Moves could creep higher; the criterion
deliberately excludes that. Measured with this check:

| Column site (from the base) | Tallest tower served |
| --- | ---: |
| X 0.40–0.50, Y 0 or ±0.20 | 11 blocks |
| X 0.60, Y 0.20; X 0.55, Y 0.25 | 10 blocks |
| Table level, X 0.35, Y 0 (in front of the body) | not served: the arm stays 2–3 mm off target even after 1,000 steps |
| Table level elsewhere in the relaxed region | served |

The re-scoring is in `runs/cap-generalization/20261029-budget-feasibility/`
(`scripts/rescore.py`, `columns.jsonl`). It re-scores the existing probe
scenes; stability and conditions were not re-run, because they do not depend
on reach. The default-environment limits in the table above were not
re-measured; they overestimate in the same way.

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
  tasks. Commits `7d9ec08` (environment) and `3f82cdb` (CaP `RunConfig`),
  plus the enlarged contact buffers.
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
  - **Goal tasks:** Grid and Pyramid goals are centred at X 0.50 m and on
    the robot's midline (Y = 0). Their sources may use any part of the region
    at least 0.105 m (Chebyshev) from every goal centre (commit `d7ecfea`).
- **Contact buffers:** `nconmax` 1000 and `njmax` 4000, instead of MuJoCo's
  defaults of 100 and 500.
  - At 100 contacts, about 15 are always taken by the robot base against the
    table plane. A 20-block tower or a settling 15-block pyramid then
    overflowed.
  - The worst case measured, a toppling 40-block tower, used 306 contacts
    and 1,230 constraint rows. The chosen sizes leave a 3× margin.
  - Step time is unchanged. MuJoCo's automatic stack grows to about 640 MB
    per environment, against 2.5 GB for njmax 8000.
- **Unchanged:** Scattered separation, the gripper clearance and explicit
  failure on exhausted retries. With the switch off, every reset is
  identical, draw for draw, and the model XML is unchanged.
- **Measured limits:** the Section 3.3 probe was run with the switch on,
  10 seeds per layout; towers were tested up to 26 blocks and Pyramid up to
  k = 7. The table below uses the 50-step primitive reach of Section 3.3.1;
  the earlier long-servo limits (towers 22, Pyramid k = 7) are withdrawn.
  - Data: `runs/cap-generalization/20261028-max-blocks/physical_relaxed_v2/`.
  - Re-scoring: `runs/cap-generalization/20261029-budget-feasibility/`.

  | Task | Default (Section 3.3, long servo) | Relaxed: all 10 seeds | Relaxed: some seeds | Binding limit |
  | --- | --- | --- | --- | --- |
  | Stack | 2–6 | 2–6 | 7–10 (9/10), 11 (8/10), 12 (2/10) | One Move's height (about 1.0 m); seed 5 starts a block in front of the body |
  | Unstack | 2–18 | 2–10 | 11 (7/10), 12 (3/10) | One Move's height |
  | Reverse | 2–18 | 2–10 | 11 (7/10), 12 (3/10) | One Move's height |
  | ReStack | total 4–18, 2 loose | total 4–10 with 2, 4 or 8 loose | 11 (28/30), 12 (1/29) | One Move's height (rebuilt tower) |
  | Grid | ≤ 3 × 3, 4 × 2, 2 × 4, 5 × 1 | 23 shapes, 4–24 cells, up to 6 × 4 | — | A goal cell at X 0.35 m on the midline (1 × 4, 1 × 5, 3 × 4, 3 × 5, 5 × 4) is not served; 6 or more columns exceed the region; shapes such as 5 × 5 leave too little source area |
  | Pyramid | k = 2–4 | k = 2–5 (15 blocks) | — | k = 6 and 7 put a goal at X 0.35 m on the midline; k = 8 exceeds the region |

  Grid and Pyramid were probed separately (`physical_relaxed_v3/`), with
  every Grid shape from 2 × 2 to 30 cells and Pyramid k up to 8.

  There were no simulator failures, and no robot contacts after a goal
  teleport.
- **Probe adaptations:**
  - the arm is parked away from every goal block before the goal teleport;
    the old park point stood inside the relaxed tower sites;
  - Unstack's probe goal cells stay inside the region;
  - table-level block centres are checked for reach exactly.
- **Baseline:** relaxed scenes differ from the trained layouts even at the
  trained count. The checker therefore runs the relaxed reset from the
  trained count, so that layout failures can be separated from count
  failures (Section 7).

### 3.6 Test-case set `relaxed_v2`

Every seed found feasible in the relaxed probes, with reach decided by the
50-step primitive budget (Section 3.3.1), is a test case. Each case is one
(task, layout, seed) scene, independent of any policy.

- **Manifest** (committed): `synthesis/cap/generalization/testsets/relaxed_v2.json`.
  - One line per case: run-config overrides, probe evidence (drifts and
    reach margins), and the SHA-256 of the settled initial state.
  - The header records the environment (relaxed switch, buffers, 50 settling
    steps), the feasibility criterion, the source probe runs and the git
    commit.
- **Snapshots** (not committed): `runs/cap-generalization/testsets/relaxed_v2/snapshots/<case_id>.npz`.
  The reset depends only on the case, so they are rebuilt from the
  manifest.
- **Builder:** `synthesis/cap/generalization/testset.py`.
  - `build` rebuilds every feasible probe scene through CaP's own
    `prepare_environment`. It checks that the rebuilt column tops match the
    probe (1e-4 m), writes the snapshot and records the hash.
  - `verify` rebuilds every case and compares hashes. A mismatch means the
    environment changed and the set must be rebuilt under a new name.
- **History:** `relaxed_v1` (1,524 cases, commit `f06f1ad`) used the
  long-servo reach and is withdrawn. `relaxed_v2` is the subset of it that
  passes the 50-step reach: all 721 cases keep their `relaxed_v1` hash.
- **Contents:**

  | Task | Cases | Counts |
  | --- | ---: | --- |
  | Stack | 96 | 2–12 |
  | Unstack | 100 | 2–12 |
  | Reverse | 100 | 2–12 |
  | ReStack | 158 | totals 4–12, with 2, 4 or 8 loose blocks |
  | Grid | 227 | 23 shapes, 4–24 cells |
  | Pyramid | 40 | k = 2–5 (3–15 blocks) |
  | **Total** | **721** | |

  Seeds 0–9 per layout. Counts near each limit have fewer cases, because
  only feasible seeds are included.
- **Seeds per block count:** rows are block counts, columns are tasks, and
  each entry is the number of feasible seeds (cases) at that count; "—" means
  no case. ReStack and Grid have several layouts per count (source-tower
  heights, grid shapes); for them the entry sums the seeds over the layouts,
  whose number is in parentheses. Pyramid counts are k(k+1)/2.

  | Blocks | Stack | Unstack | Reverse | ReStack | Grid | Pyramid |
  | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | 2 | 10 | 10 | 10 | — | — | — |
  | 3 | 10 | 10 | 10 | — | — | 10 |
  | 4 | 10 | 10 | 10 | 10 | 20 (2) | — |
  | 5 | 10 | 10 | 10 | 10 | 10 | — |
  | 6 | 10 | 10 | 10 | 20 (2) | 30 (3) | 10 |
  | 7 | 9 | 10 | 10 | 20 (2) | 10 | — |
  | 8 | 9 | 10 | 10 | 20 (2) | 30 (3) | — |
  | 9 | 9 | 10 | 10 | 20 (2) | 20 (2) | — |
  | 10 | 9 | 10 | 10 | 29 (3) | 20 (2) | 10 |
  | 11 | 8 | 7 | 7 | 28 (3) | — | — |
  | 12 | 2 | 3 | 3 | 1 | 20 (2) | — |
  | 14 | — | — | — | — | 10 | — |
  | 15 | — | — | — | — | 10 | 10 |
  | 16 | — | — | — | — | 20 (2) | — |
  | 18 | — | — | — | — | 9 | — |
  | 20 | — | — | — | — | 8 | — |
  | 24 | — | — | — | — | 10 | — |
  | **Total** | **96** | **100** | **100** | **158** | **227** | **40** |

## 4. Separating policy failures from environment limits

Every (program, task, count, seed) execution gets one outcome. The first two
failure outcomes are **environment** failures and are not scored against the
policy; the others are **policy** outcomes.

| Outcome | Attributed to | Meaning |
| --- | --- | --- |
| `success` | policy | `task_spec_success` under `task-spec-v2+final-release-v1` |
| `scene_infeasible` | environment | The scene at this (task, count, seed) is not physically feasible: the reset raises, the initial or goal scene is unstable or violates its condition, or a block centre is out of reach. Decided before any policy runs; the policy may still run, but the outcome is not scored |
| `simulator_failure` | environment | The rollout raised a simulator error rather than a policy-level one: MuJoCo's contact or constraint buffer overflowed, the state became non-finite, or the step raised `MujocoException`. Recorded with the error and the step. A count with any such outcome is reported, and the buffers are checked |
| `policy_failure` | policy | The policy fails on a **physically feasible** scene without a simulator failure |
| `budget_exhausted` | policy (by default) | A scaled budget ran out (Section 5.4). Reported separately and counted as a policy failure on feasible scenes (decision D3) |

**Physical feasibility, not witnesses.** The same seed gives the same initial
state for every program, because the reset depends only on the seed and the
layout options. Feasibility is therefore computed once per (task, count, seed)
by the Section 3.3 probe, before any program runs. A successful execution is
not needed as proof.

Controller and program limits remain policy failures, because the
feasibility check has shown that the environment can hold the start and goal
states and that the arm reaches every block. Examples:
- a Move that does not converge within the shared 50-step primitive budget;
- a tower built by the robot that falls (the teleported goal tower stood);
- a transport height above the reachable Z.

The low-clearance oracles and the supplied programs stay as diagnostics only.
They show that a feasible scene can also be solved through the DSL.

## 5. The checker

### 5.1 Count schedule per task

With the relaxed environment the schedule is the test-case set `relaxed_v2`
(Section 3.6). It covers every count physically feasible on at least one
seed, including the counts below the trained one. The default-reset column
is kept for comparison:

| Task | Default reset (long-servo reach) | Relaxed reset, `relaxed_v2` |
| --- | --- | --- |
| Stack | 4, 5, 6, 7 | 2–12 |
| Unstack | 4–19 | 2–12 |
| Reverse | 4–19 | 2–12 |
| ReStack (tower, loose) | (3, 2) = 5 … (17, 2) = 19 | totals 4–12, with 2, 4 and 8 loose blocks |
| Grid (rows × cols) | 3 × 2, 4 × 2, 2 × 4, 3 × 3, 5 × 1 | 23 shapes from 2 × 2 to 6 × 4 |
| Pyramid (k) | 3, 4 | 2–5 |

Seeds that are physically infeasible at a count are excluded from that
count's denominator and reported separately.

Smaller counts (2, 3) are optional (decision D6). They test the same
generality claim downward, and the environments support them.

### 5.2 Scenes and stopping

- **Seeds:** every case of `relaxed_v2`: all feasible seeds of 0–9 per
  layout, the same for every program (decision D1). At the trained count
  these relaxed scenes differ from the training and evaluation scenes of
  final_v2 (Section 7, relaxed layout shift).
- **No early stopping by default:** every program runs the whole schedule,
  because a failure at N does not imply failure at N + 1 (a scene-dependent
  failure can recur). The cost is small (Section 5.5).
- **Largest generalizing count of a program:** the largest N such that, at
  every scheduled count up to N, all physically feasible scenes succeed. A
  count where every scene is `scene_infeasible` does not break the chain, but
  it is marked.

### 5.3 Execution

- **Jobs:** one process per (distinct program, chunk of cases of its task),
  about 220 blocks of cases per chunk. Every case is one `runner.run` call:
  - `config_final_v2_wo_demo.configuration(task, ...)`, which is final_v2's
    run configuration with demonstrations off, overridden only by the case
    (seed, layout, relaxed switch) and the budgets of Section 5.4;
  - `recovery.enabled: false`, `video_mode: "off"`.
  - Videos, when wanted, must be `deferred`, never `inline`: inline rendering
    changes the simulation (`stack-n22-s0` fails after 1,000 steps with it
    and 1,556 without, reproducibly), while deferred recording gave the same
    execution as no video in all 18 timing cases
    (`runs/cap-generalization/20261029-policy-timing/report.md`). Deferred
    recording makes execution 1.2–2× slower, so it is off for the full set; a
    case can be re-run with video afterwards, because execution is
    deterministic.
- **Model:** `ScriptedModel` returns the program (the executed checkpoint's
  `policy.py`) for the single round-0 call. The shared DSL checker re-checks
  it. With demonstrations off, the information-boundary check does not apply.
- **No model:** no key and no request gate.
- **Run configuration:** the condition mode, prompt package, controller,
  settling (50 steps) and final-release check stay as in final_v2.
- **RoboVerify programs** run through `cfg.collection.record_execution`
  from the case snapshot (hash checked against the manifest), with the same
  50-step primitives and 50 settling steps, scored by the shared task check
  (`task-spec-v2`).
- **Resumable:** each job appends one row per case to
  `results/<kind>/<task>/<program>-<chunk>.jsonl` and skips finished cases.
- **Feasibility** is part of the test-case set (Section 3.6); nothing is
  probed per run.

### 5.4 Budgets

Fixed budgets would confound "does not generalize" with "ran out of budget".
Implemented (decision D3), with N_trained from `TASKS` (4, 4, 4, 5, 6, 6):
- `max_steps` (execution ticks) 10,000 × max(1, N / N_trained)³;
- `max_sim_steps` 10,000 × max(1, N / N_trained);
- the 50-step primitive budget unchanged; `timeout` 1,200 s per case.

Usage of every budget is recorded per execution. A policy that loops without
progress is then still caught, as `budget_exhausted`.

### 5.5 Cost

- **Jobs:** each program runs the test cases of its own task:
  252 distinct final_v2 programs and 300 distinct final_v2_wo_demo programs
  (552 in total; two programs occur in both pools), on 721 cases: 68,086
  scripted executions.

  Revision 8 counted 246 distinct final_v2 programs; `programs.alpha_hash`
  renames only the names a program binds itself, which merges slightly fewer
  Grid and Pyramid programs.
- **Time per execution** (measured, video off,
  `runs/cap-generalization/20261029-policy-timing/report.md`): about 7 s
  fixed per process plus (2.2 + 0.32 N) ms per simulation step, with 45–95
  steps per block for a completed run. Executions that fail early take
  2–7 s.
- **Wall time:** one policy (one program per task) on all 721 cases takes
  1.0–2.0 CPU-hours; all programs with all 10 seeds take 41–82 CPU-hours,
  about 1–2 hours on 48 workers on GPUs 6 and 7. There is no API cost. A
  program that loops without progress runs to its budget; a per-case limit
  of 120–300 s bounds this.
- **Memory:** about 1.1 GB per worker process, about 50 GB on 48 workers.
- **Measured** (2026-10-29 run): the 68,086 CaP executions took 207
  CPU-hours (median 6.7 s per case), 2.2 hours on up to 100 workers. With
  per-step run files on the network file system the run was I/O bound (about
  150 cases per minute); writing them to tmpfs (`R9_LOCAL_RUN_ROOT`) raised
  this to 400–700 per minute.

### 5.6 Outputs

Artifacts go under `runs/cap-generalization/20261029-full-eval/`:
- `programs.json`: every trial's program (pool, task, trial, seed, round,
  source path, text and alpha-renamed hashes);
- `jobs.json`: the jobs (program, chunk of cases);
- `results/<kind>/<task>/*.jsonl`: one row per execution, with outcome,
  partial reward, simulation steps, ticks, budget usage, status and reason;
- `cap_runs/`: the CaP run directories (large per-step files removed for
  successful executions).

The report is [revision_9_report.md](revision_9_report.md), written by
`reports/generalization_report.py`. Per task it contains:
- success rate against count, over distinct programs and weighted by trial,
  with Wilson intervals;
- mean partial reward against count;
- the distribution of the largest generalizing count;
- the failure statuses, for example `primitive_failed` versus
  postcondition false;
- RoboVerify's programs on the same cases.

Figures:
- success against N per task (one panel per task, as in the final report);
- partial reward against N;
- a stacked bar of outcomes per count.

## 6. Code layout

- `synthesis/cap/generalization/`:
  - `testset.py`: versioned test-case sets, with `build` and `verify`
    (Section 3.6); manifests live in `testsets/`. Tests:
    `synthesis/cap/tests/test_generalization_testset.py`.
  - `reach.py`: the 50-step primitive reach check (Section 3.3.1).
  - `programs.py`: extracts the final program of every successful trial
    (successful round, active checkpoint) and deduplicates on the
    alpha-renamed AST hash. Shared with Revision 8 M0.
  - `feasibility.py` (not in the repository): the physical probe of
    Sections 3.3 and 3.5 lives as diagnostic scripts in
    `runs/cap-generalization/20261028-max-blocks/scripts/`. A new test-case
    set needs it in the repository.
- `synthesis/cap/experiments/revision9_generalization.py`: the driver.
  Scaled budgets (Section 5.4), outcome classification (Section 4), the CaP
  and RoboVerify executors, and a resumable process-per-job launcher. It
  makes no model calls; `--gpus` defaults to 6,7. `R9_LOCAL_RUN_ROOT` puts
  the per-step CaP run files on a local disk and keeps only `result.json`.
- `synthesis/cap/reports/generalization_report.py`: tables and figures of
  [revision_9_report.md](revision_9_report.md).
- The relaxed reset bounds (Section 3.5): `environment/relaxed_workspace.py`,
  the samplers, the environment constructor and factories, and `RunConfig`.
  The driver only sets the switch through the case's run-config overrides.

**Tests** (`synthesis/cap/tests/test_generalization.py`):
- budget scaling and outcome classification;
- program extraction and the alpha-renamed hash (renamed locals and dropped
  comments hash equal; a changed constant does not);
- the report's failure kinds and largest generalizing count over synthetic
  rows.

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
- **Contact buffer.** With MuJoCo's default `nconmax`, simulation stops at
  about 20 blocks. The relaxed environment enlarges the buffers (Section 3.5).
  A buffer overflow during a rollout is still recorded as
  `simulator_failure`, not as a policy failure.
- **Budget scaling** changes per-trial limits. It must be applied identically
  to every program and recorded in the results.

## 8. Decisions

1. **D1. Seeds per count.** Decided (2026-10-29): every case of
   `relaxed_v2`, i.e. all feasible seeds of 0–9, the same for all programs.
2. **D2. Feasibility criterion.** Decided: physical feasibility (Section 3.3:
   build, stability, goal teleport) with reach within one 50-step Move
   (Section 3.3.1). The oracles are diagnostics only.
3. **D3. Budgets.** Adopted: ticks scale with (N / N_trained)³ and simulation
   steps with N / N_trained (Section 5.4); `budget_exhausted` on a feasible
   scene is a policy failure.
4. **D4. Schedule and stopping.** Adopted: the full schedule for every
   program, without early stopping.
5. **D5. Environment limits.** Decided by the user (2026-10-29;
   Section 3.5): one opt-in switch relaxes every task's workspace bounds and
   enlarges MuJoCo's contact buffers (1000 contacts, 4000 constraint rows).
   - On the user's request, Grid and Pyramid goals were also centred on the
     midline and their sources allowed into any free area. This raised Grid
     to 6 × 4 and Pyramid to k = 5 under the 50-step reach.
   - The remaining limits are one Move's reach: height for the towers, and
     the table in front of the robot's body for Grid and Pyramid.
6. **D6. Smaller counts.** Adopted: counts 2 and 3 (and 2 × 2, k = 2) are
   in `relaxed_v2`.
7. **D7. Program pool.** Decided: final_v2 and final_v2_wo_demo, with
   identical cases, plus RoboVerify's programs.
