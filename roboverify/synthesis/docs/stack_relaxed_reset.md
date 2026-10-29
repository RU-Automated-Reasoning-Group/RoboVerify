# Relaxed Stack reset for generalization tests

- **Status:** plan, 2026-10-29. Not implemented.
- **Requested by:** the user, for the block-count generalization checker
  ([CaP Revision 9](../cap/docs/revision_9_generalization_checker.md),
  Section 3.5).
- **Option:** `relax_boundaries_for_generalization_test` (bool, default
  `False`). Stack only.
- **Purpose:** the default Stack reset cannot place more than 6 blocks on every
  seed (7 on 4 of 10 seeds, none at 8). The limit comes from the sampling
  rectangle, not from the arm or the simulator. Tower tasks reach 18 blocks
  on every seed. With the option on, the reset should produce scattered initial
  states from which an 18-block tower is physically feasible.

## 1. What limits the default reset

[`stack_reset.py`](../environment/stack_reset.py) samples every block centre
relative to the robot base:

| Constant | Value | Rule in `sample_stack_xy` |
| --- | --- | --- |
| `STACK_X_RANGE` | (0.54, 0.70) m | uniform X offset |
| `STACK_Y_RANGE` | (−0.20, 0.20) m | uniform Y offset |
| `STACK_MAX_BASE_DISTANCE` | 0.70 m | reject if the XY distance from the base is larger |
| `STACK_GRIPPER_CLEARANCE` | 0.10 m | reject if closer to the initial gripper XY |
| `2 * BLOCK_LENGTH` | 0.10 m | reject if both \|dx\| and \|dy\| to an earlier block are below it (Scattered) |

- **Retries:** each block gets up to 200 candidates. If one cannot be placed,
  the whole layout restarts, up to 100 times; then the sampler raises
  `ValueError`.
- **Capacity:** the rectangle is 0.16 × 0.40 m. A perfect 0.10 m lattice
  would hold at most 2 × 5 = 10 centres, and random sequential placement jams
  earlier, at 6–7.
- **Caller:** `FetchPickAndPlaceConstruction._reset_sim_roboverify_stack`
  ([fpp_construction_env.py](../environment/cee_us_env/fpp_construction_env.py))
  places every block at `height_offset`.
- **Tower base:** the Stack postcondition requires the tower on `b0`
  (`base_block_id` 0, `_roboverify_stack_tower_complete`). The reachable
  height above `b0`'s column therefore bounds the tower.

## 2. Measurements behind the relaxed bounds

All numbers come from `runs/cap-generalization/20261028-max-blocks/`.

- **Path-aware reach** (`physical/reach_v2.jsonl`): the highest gripper Z
  reached at each 5 cm cell, with 2 mm tolerance, on a straight path or a
  rise–traverse–descend path. In base-relative coordinates:
  - X 0.40–0.60, |Y| ≤ 0.20: 1.45 m up to the 1.59 m search ceiling;
  - X 0.40–0.70, |Y| ≤ 0.40, radius ≤ 0.75 m: at least 1.08 m everywhere,
    so every cell can be grasped at table height;
  - X ≤ 0.30 near Y = 0 lies under the robot and is not reachable.
- **Tower height:** an 18-block tower has its top centre at 1.242 m. The
  best-site test (`best_site/`) built 18-block towers and reached their top
  with the open gripper at every tested site, including the default region's
  near corner. 19 failed at every site: the arm knocked the tower over, or the
  tower fell by itself. From 20 blocks, MuJoCo's contact buffer (`nconmax`
  100) overflows.
- **Sampler capacity** (the sampler's rules only, 20 seeds, no simulation),
  as the number of seeds that produced a layout:

  | Region (base-relative) | 7 | 8 | 15 | 16 | 17 | 18 | 19 | 20 |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
  | Default: X 0.54–0.70, \|Y\| ≤ 0.20, r ≤ 0.70 | 9 | 0 | – | – | – | – | – | – |
  | X 0.40–0.70, \|Y\| ≤ 0.30, r ≤ 0.75 | 20 | 20 | 12 | 1 | 0 | – | – | – |
  | X 0.40–0.65, \|Y\| ≤ 0.40, r ≤ 0.75 | 20 | 20 | 20 | 20 | 12 | 2 | 0 | – |
  | **X 0.40–0.70, \|Y\| ≤ 0.40, r ≤ 0.75** | 20 | 20 | 20 | 20 | 20 | 19 | 9 | 1 |

## 3. Planned behaviour

With `relax_boundaries_for_generalization_test=True`:

1. **Tower base first.** `b0` (`base_block_id`) is sampled first, uniformly in
   X 0.40–0.60 and Y −0.20–0.20 from the base. The reach there is at least
   0.2 m above an 18-block top centre. Every seed still gets a different
   tower site.
2. **Other blocks** are sampled in X 0.40–0.70 and Y −0.40–0.40, within
   0.75 m of the base.
3. **Unchanged:**
   - the 0.10 m Scattered separation;
   - the 0.10 m initial-gripper clearance;
   - resting height;
   - the bounded retries. Exhaustion still raises `ValueError` and never
     returns a partial layout. The retry budget may be raised if the
     validation in Section 5 shows that sampler failures, rather than space,
     limit 18 blocks.
4. **Default unchanged.** With the option off, the sampler makes the same
   random draws in the same order as today. Archived seeds, collection, CaP
   runs and the Section 6.2 invariant-learning experiment (which reads
   `STACK_X_RANGE` / `STACK_Y_RANGE` for its scene bounds) are unaffected.

The relaxed bounds are new named constants (for example
`RELAXED_STACK_BASE_X_RANGE`, `RELAXED_STACK_X_RANGE`), next to the existing
ones. They do not replace them.

## 4. Plumbing

| Layer | Change |
| --- | --- |
| `environment/stack_reset.py` | `sample_stack_xy(..., *, relax_boundaries_for_generalization_test=False, base_index=0)` and the relaxed constants |
| `environment/cee_us_env/fpp_construction_env.py` | Constructor keyword, stored and passed to `EzPickle`; `ValueError` if it is true for any case other than `RoboVerifyStack`; `_reset_sim_roboverify_stack` passes it and `base_block_id` |
| `mcmc/synthesis.py` | `make_roboverify_stack_env` and `make_roboverify_env` accept the keyword; other tasks reject it |
| `cap/config.py`, `cap/tasks.py` | `RunConfig` field, allowed for Stack only (`TASK_OPTIONS`). `environment_options` includes it only when true, so existing configurations and fingerprints are unchanged; trial manifests record it |
| `entry/collect_demos.py` | Not planned. Collection keeps the default reset |

## 5. Validation

- **Unit tests** (`environment/test_stack_reset.py`):
  - the default layout for fixed seeds is unchanged (regression on stored
    positions);
  - relaxed layouts satisfy Scattered, the gripper clearance, the relaxed
    rectangle and radius, and keep `b0` inside the tower-site rectangle;
  - relaxed sampling is seeded and relative to the base;
  - the option is rejected for other tasks;
  - one simulator reset at 12 blocks keeps every block at resting height and
    the `b0` binding.
- **Physical probe:** the Section 3.3 probe of Revision 9
  (`physical_probe.py`, seeds 0–9) for Stack N = 4–19 in relaxed mode, with
  images. A seed is feasible when its initial scene is stable, the teleported
  goal tower on `b0` is stable and satisfies the postcondition, and every top
  centre is within reach. The expected limit is 18, the tower limit; the
  probe decides it.
- **Supplied program:** collection of the supplied Stack program
  (`synthesis.examples.stack:build_program`) at 4 blocks in relaxed mode, as a
  sanity check that the wider layout does not break the shared primitives at
  the trained count. Diagnostic only.

## 6. Interpretation limits

- **Two changes at once.** A relaxed scene changes both the block count and
  the layout distribution: blocks may lie up to 0.40 m to the side and
  0.14 m closer to the base than in training. The checker therefore also runs
  the relaxed reset at the trained count (4 blocks). Failures there are layout
  generalization failures, and the count curve should be read relative to
  that baseline.
- **Not a reachability proof** for arbitrary paths. Carrying a block over a
  tall tower, or past other blocks on the wider table, may still fail. Those
  failures count against the policy on a physically feasible scene.
- **The ceiling stays at 18–19** unless `nconmax` and the tall-tower stability
  are changed. That is a separate environment change.
