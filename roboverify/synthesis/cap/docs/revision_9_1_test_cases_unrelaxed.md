# Revision 9.1: test-case set `unrelaxed_v1` (default resets)

- **Status:** built and verified 2026-10-31 (git `82b48ae` plus the
  builder change committed with this set).
- **Question:** [Revision 9](revision_9_generalization_checker.md) tests
  block-count generalization in the opt-in relaxed environment, whose resets
  and goals differ from the ones the policies were trained on. This set
  contains the same kind of cases, built with
  `relax_boundaries_for_generalization_test` off. Every scene comes from the
  task's own reset, as in training and scoring:
  - Stack: `synthesis/environment/stack_reset.py`;
  - Unstack and Reverse: `tower_reset.py`;
  - ReStack: `restack_reset.py`;
  - Grid and Pyramid: the default goal layouts of `fpp_construction_env.py`.
- **Method:** unchanged from
  [Revision 9, Sections 3.3–3.6](revision_9_generalization_checker.md#33-physically-feasible-counts-measured):
  the same probe, the same seeds 0–9, the same layouts as `relaxed_v2`, and
  the same 50-step reach criterion.
- **Result:** 388 cases (`relaxed_v2`: 721). The default bounds remove
  Stack above 7 blocks, every ReStack layout with more than 2 loose blocks,
  every Grid shape above 3 × 3 (and 5 × 1), and Pyramid k = 5. Towers stop
  at 11 blocks instead of 12, because the default tower region lies farther
  from the base.

## 1. What a case is

As in `relaxed_v2`: one (task, layout, seed) scene, independent of any
policy.
- **Layout:** the run-config fields that set the block count:
  - `num_blocks`;
  - ReStack: `initial_tower_blocks`;
  - Grid: `grid_rows`, `grid_cols`;
  - Pyramid: `bottom_row_blocks`.
- **Scene:** CaP's `prepare_environment` with that layout and seed and the
  relaxed switch off, after 50 settling steps. Its settled full state is the
  case's initial state.
- **Identity:** the case id (for example `restack-n7-t5-s3`) does not record
  the environment, so 387 of the 388 ids also occur in `relaxed_v2`. None of
  their scenes or hashes is the same. The set name identifies the scene, and
  snapshots are kept per set.

## 2. Feasibility

A seed is a case when all of these hold (the `relaxed_v2` criterion):
- **initial:** the reset scene passes `validate_start` and stays within 5 mm
  for 100 extra steps;
- **goal:** a teleported goal configuration stays within 5 mm for 100 steps
  and satisfies the postcondition;
- **long-servo reach:** every raised top centre is below the path-aware
  reach above its column, and every table-level centre is reached
  ([Revision 9, Section 3.3](revision_9_generalization_checker.md#33-physically-feasible-counts-measured));
- **50-step reach:** every initial and goal column is served by one Move to
  the hover point 0.05 m above the top centre, one Move down to it and one
  Move back up. Each Move has 50 steps and a 2 mm tolerance
  ([Section 3.3.1](revision_9_generalization_checker.md#331-reach-within-one-move-correction-2026-10-29),
  [`reach.py`](../generalization/reach.py)).

The 50-step reach was measured in an empty **default** scene
(`PrimitiveReach(relaxed=False)`). As a cross-check, all 1,521 columns were
also measured in an empty relaxed scene. Every result agreed: relaxation
changes the resets and the contact buffers, not the arm or the table.

## 3. How it was built

Everything is in `runs/cap-generalization/20261031-unrelaxed-testset/`
(scripts in `scripts/`, run from `roboverify/` with
`runs/cap-generalization/20261029-full-eval/scripts/env.sh`):
1. **Reused default probe:** `20261028-max-blocks/physical/`, the Revision 9
   Section 3.3 probe, ran with the switch off. It covers:
   - Stack 2–10;
   - Unstack and Reverse 2–20;
   - ReStack with 2 loose blocks, totals 4–20;
   - 11 Grid shapes;
   - Pyramid k = 2–5.
2. **Supplementary probe** (`probe_extra/`, `scripts/probe_extra.py`): the 25
   `relaxed_v2` layouts that the default probe lacked, 10 seeds each, through
   the unchanged `physical_probe.scene_job`:
   - Stack 11 and 12;
   - the 9 ReStack layouts with 4 or 8 loose blocks;
   - 14 Grid shapes.

   Only Grid 4 × 1 and 2 × 2 could be built. Their columns were already in
   the default reach map, so no new long-servo reach was needed.
3. **Scoring** (`scripts/score.py`): the 2026-10-28 rule (initial, goal and
   long-servo reach), then the 50-step reach of every column.
   - Outputs: `scenes_scored.json`, `columns.jsonl` and `summary.txt`, with
     one line per layout.
   - Counts above the `relaxed_v2` range (towers 13–19) were scored too. None
     passes the 50-step reach.
4. **Build:** the builder rebuilds every feasible scene through
   `prepare_environment`. It checks the column tops against the probe
   (1e-4 m), writes the snapshot and records its hash:

   ```bash
   .venv/bin/python -m synthesis.cap.generalization.testset build unrelaxed_v1 --unrelaxed \
       --source unrelaxed=runs/cap-generalization/20261031-unrelaxed-testset/scenes_scored.json \
       --out runs/cap-generalization/testsets/unrelaxed_v1 --workers 50
   ```

   Every rebuilt scene matched the 2026-10-28 probe. The default resets
   therefore have not changed since that probe.
- **Manifest** (to be committed):
  [`synthesis/cap/generalization/testsets/unrelaxed_v1.json`](../generalization/testsets/unrelaxed_v1.json).
  The header records `relax_boundaries_for_generalization_test: false`,
  MuJoCo's default contact buffers (`nconmax` 100, `njmax` 500) and
  50 settling steps.
- **Snapshots** (not committed):
  `runs/cap-generalization/testsets/unrelaxed_v1/snapshots/<case_id>.npz`,
  388 files, 2 MB.
- **Builder change:** [`testset.py`](../generalization/testset.py) now records
  the relaxed switch in the manifest header.
  - `load_manifest` gives every case that setting, and `run_config` /
    `run_config_overrides` rebuild the scene with it.
  - `build --unrelaxed` selects the default resets.
  - `relaxed_v2` records the switch as on, and it loads and rewrites
    unchanged apart from the creation date. A header without the key is read
    as relaxed.
- **Verification:**
  - `testset verify`: 388/388 `unrelaxed_v1` cases reproduce their initial
    state. 721/721 `relaxed_v2` cases still reproduce theirs after the
    builder change.
  - Conditions (`scripts/check_pre.py`, `conditions.jsonl`): each snapshot
    was restored into the environment that the Revision 9 driver builds from
    the case (`environment_options(run_config(case))`). On it, 388/388 cases
    satisfy the task precondition, and none already satisfies the
    postcondition. `task_conditions` checks Scattered sources and empty goals
    for Grid and Pyramid.

## 4. Contents

| Task | Cases | Counts |
| --- | ---: | --- |
| Stack | 54 | 2–7 |
| Unstack | 84 | 2–11 |
| Reverse | 84 | 2–11 |
| ReStack | 66 | totals 4–11, 2 loose blocks (tower 2–9) |
| Grid | 70 | 7 shapes, 4–9 cells |
| Pyramid | 30 | k = 2–4 (3–10 blocks) |
| **Total** | **388** | |

Seeds per block count follow the conventions of the
[`relaxed_v2` table](revision_9_generalization_checker.md#36-test-case-set-relaxed_v2):
an entry is the number of feasible seeds, and "—" means no case. The number
of layouts is in parentheses where a count has more than one. Every Grid
shape and every Pyramid k has all 10 seeds.
- **Grid:** 4 is 4 × 1 and 2 × 2, 6 is 3 × 2 and 2 × 3, 8 is 4 × 2 and
  2 × 4, and 9 is 3 × 3 (rows × cols).
- **ReStack:** every count has a single layout (tower N − 2, plus 2 loose
  blocks).

| Blocks | Stack | Unstack | Reverse | ReStack | Grid | Pyramid |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 10 | 10 | 10 | — | — | — |
| 3 | 10 | 10 | 10 | — | — | 10 |
| 4 | 10 | 10 | 10 | 10 | 20 (2) | — |
| 5 | 10 | 10 | 10 | 10 | — | — |
| 6 | 10 | 10 | 10 | 10 | 20 (2) | 10 |
| 7 | 4 | 10 | 10 | 10 | — | — |
| 8 | — | 10 | 10 | 10 | 20 (2) | — |
| 9 | — | 7 | 7 | 9 | 10 | — |
| 10 | — | 5 | 5 | 6 | — | 10 |
| 11 | — | 2 | 2 | 1 | — | — |
| **Total** | **54** | **84** | **84** | **66** | **70** | **30** |

Partial counts keep these seeds:
- Stack 7: 0, 2, 5, 7;
- Unstack and Reverse 9: 0, 1, 2, 3, 5, 7, 9;
- Unstack and Reverse 10: 1, 2, 5, 7, 9;
- Unstack and Reverse 11: 7, 9;
- ReStack 9 (7 + 2): all except 3;
- ReStack 10 (8 + 2): 1, 2, 5, 6, 7, 9;
- ReStack 11 (9 + 2): 4.

Every trained count has all 10 seeds:
- Stack, Unstack and Reverse 4;
- ReStack 5 = 3 + 2;
- Grid 3 × 2;
- Pyramid k = 3.

## 5. Dropped counts and seeds

The default bounds are listed in
[relaxed_reset_bounds.md, Section 1](../../docs/relaxed_reset_bounds.md#1-default-bounds-that-limit-block-counts).

| Task | Dropped | Reason |
| --- | --- | --- |
| Stack | 7 on 6/10 seeds; 8–12 on every seed | Reset sampler: X 0.54–0.70, Y −0.20–0.20, radius ≤ 0.70 m, 0.10 m separation. `ValueError: Could not sample N scattered Stack blocks` |
| Unstack | 9–11 on some seeds; 12 on every seed | 50-step reach. The Move to the hover point above the source tower does not converge. The highest served top centre is 0.924 m (11 blocks); tops from 0.815 m (9 blocks) fail at the far sites of the tower region X 0.54–0.70, Y 0.15–0.25 |
| Reverse | As Unstack | As Unstack; from 9 blocks the reversed tower's column fails too |
| ReStack | 9–11 on some seeds and 12 on every seed (2 loose); every layout with 4 or 8 loose blocks | 50-step reach above the rebuilt N-block tower, mostly. `restack_reset.py` fits at most 2 loose blocks (`Restack source workspace cannot fit this many loose blocks`) |
| Grid | 16 of the 23 `relaxed_v2` shapes | 12 fail the 0.8531 m goal-workspace check: 6–9 × 1, 5–8 × 2, 2 × 5, 5–6 × 3 and 4 × 4. 4 × 3 has a far cell (0.85 m) out of reach even at table level. In 5 × 1 the cell at (0.65, 0.50) from the base (radius 0.82 m) is served by the hover Move, but the descent to table level does not converge within 50 steps. 6 × 4 and 4 × 5 overflow the default contact buffer at reset |
| Pyramid | k = 5 (15 blocks) | `Cannot fit 15 scattered blocks in the goal-task workspace` |

There were no other failures. Every built scene was stable and every
teleported goal satisfied its postcondition.

## 6. Comparison with `relaxed_v2`

| Task | `relaxed_v2` | `unrelaxed_v1` | What limits the default |
| --- | --- | --- | --- |
| Stack | 96 cases, 2–12 | 54, 2–7 | The sampler. In relaxed mode the limit is the arm's height |
| Unstack, Reverse | 100 each, 2–12 (10 on all seeds) | 84 each, 2–11 (8 on all seeds) | The default tower region (X 0.54–0.70) lies farther from the base than the relaxed tower sites (X 0.40–0.60), so one Move reaches less high above it |
| ReStack | 158, totals 4–12, 2, 4 or 8 loose | 66, totals 4–11, 2 loose | At most 2 loose blocks; tower reach as above |
| Grid | 227, 23 shapes, 4–24 cells | 70, 7 shapes, 4–9 cells | Goals are centred on the initial gripper and must lie within 0.8531 m. At 20 or more blocks MuJoCo's contact buffer overflows |
| Pyramid | 40, k = 2–5 | 30, k = 2–4 | Source rectangle |
| **Total** | **721** | **388** | |

- **No new counts:** `unrelaxed_v1` has no count above `relaxed_v2`'s range.
  - Its only layout outside `relaxed_v2` is Grid 5 × 1, and it has no case.
  - Seed 5 of Stack 7 is a case here but not in `relaxed_v2`, where a block
    starts in front of the robot's body.
- **What the set measures:**
  - At counts above the trained one, `unrelaxed_v1` tests generalization on
    the trained reset distribution.
  - `relaxed_v2` tests more counts, but it also changes the layouts.
  - A policy that passes `unrelaxed_v1` but fails `relaxed_v2` at the same
    count fails on the layout, not on the count.

## 7. Notes

- **Driver and report:** the Revision 9 driver
  (`experiments/revision9_generalization.py`, `TESTSET = "relaxed_v2"`) and
  `reports/generalization_report.py` still name `relaxed_v2`. Evaluating
  `unrelaxed_v1` needs that name changed or made an option. The scenes
  themselves are rebuilt correctly, because they come from the manifest's
  setting.
- **Default contact buffer:** scenes up to 11 blocks stay far below the
  100-contact limit. A policy that knocks over a large scene could still
  overflow it during a rollout. Revision 9 counts that outcome as a
  simulator failure, an environment failure.
- **Restoring a case:** restore its snapshot; do not call `env.reset(seed=…)`.
  CaP seeds NumPy's global generator before the reset. A Gym-seeded reset
  draws differently, so for Stack 7 it can even fail to sample a scene.
