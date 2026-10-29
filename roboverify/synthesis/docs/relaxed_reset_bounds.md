# Relaxed reset bounds for generalization tests

- **Status:** implemented 2026-10-29 (commits `7d9ec08` environment, `3f82cdb`
  CaP; contact buffers added the same day). Default off.
- **Requested by:** the user, for the block-count generalization checker
  ([CaP Revision 9](../cap/docs/revision_9_generalization_checker.md),
  Section 3.5).
  - A first plan covered Stack only. The user then asked for one switch for
    every task, and for the largest contact buffers that cover every case.
  - The checker evaluates whether policies generalize to arbitrary block
    counts. It should therefore test as many blocks as the environment can
    physically hold, independent of what any policy can do.
- **Option:** `relax_boundaries_for_generalization_test`. It is available
  - on the environment constructor;
  - on the `make_roboverify_*` factories and `make_goal_env`;
  - as a `RunConfig` field in CaP.
- **Purpose:** with the default resets, Stack stops at 6 blocks, Grid at
  3 × 3 and Pyramid at k = 4. Those limits come from sampling rectangles,
  goal placement and MuJoCo's default contact buffer, not from the arm. With
  the switch on, every task uses the region the arm was measured to reach, and
  the buffers are enlarged. The limit then becomes physical: the arm's height,
  the reachable table area, or the stability of the goal.

## 1. Default bounds that limit block counts

| Task | Default bound | Where | Limit it causes |
| --- | --- | --- | --- |
| Stack | X 0.54–0.70, Y −0.20–0.20, radius ≤ 0.70 m | `stack_reset.py` | 6 blocks on all seeds, 7 on 4/10 |
| Unstack, Reverse | Tower at X 0.54–0.70, Y 0.15–0.25 | `tower_reset.py` | Reach at the far corner is 1.22 m. 19 blocks only on some seeds |
| ReStack | Tower and loose blocks in one Y strip (0.15–0.25), loose X 0.40–0.70 | `restack_reset.py` | At most 2 loose blocks |
| Grid, Pyramid | Goals centred on the initial gripper (X 0.65); radius ≤ 0.8531 m; sources in a small rectangle | `fpp_construction_env.py` | Grid 3 × 3 (the far cell of 4 × 3 is out of reach); Pyramid k = 5 sources do not fit |
| All | MuJoCo `nconmax` 100, `njmax` 500 | generated model XML | 20 blocks overflow the contact buffer |

All coordinates are base-relative, in metres.

## 2. Relaxed geometry and simulator capacity

One shared region (`relaxed_workspace.py`): **X ≥ 0.35 m and XY radius ≤
0.80 m.**
- **Reach map:** every 5 cm cell of this region was reached by the empty
  gripper at Z ≥ 0.64 m.
  - Reach was measured path-aware (straight, or rise, traverse and descend)
    with 2 mm tolerance.
  - The maps are the 2026-10-28 probe plus the relaxed run, which added the
    |Y| > 0.60 m cells.
  - Points between cells near the high-|Y| edge are not guaranteed
    (Section 3). The per-seed feasibility check covers them.
- **Excluded near the base:** X ≤ 0.30 m near Y = 0 lies under the robot and
  is unreachable.

| Task | Relaxed layout |
| --- | --- |
| Stack | `b0` (the tower base, from `base_block_id`) first, in X 0.40–0.60, Y −0.20–0.20; the other blocks anywhere in the region |
| Unstack, Reverse | Tower in X 0.40–0.60, Y 0.15–0.25 (still on the higher-Y side) |
| ReStack | Tower as Unstack; loose blocks anywhere in the region with Y ≥ 0.15. The 2-loose-block capacity check is skipped |
| Grid, Pyramid | Goals centred at X 0.50 (shifted forward to X ≥ 0.35), first row at Y 0.10 as before; every goal must lie in the region, else `ValueError`. Sources anywhere in the region with Y ≤ −0.08 |

- **Contact buffers:** `nconmax` 1000 and `njmax` 4000
  (`RELAXED_CONTACT_BUFFER`), written into the generated model XML. Section 4
  explains the choice.
- **Unchanged:**
  - Scattered separation (0.10 m; 0.105 m for ReStack and goal sources);
  - the 0.10 m initial-gripper clearance;
  - resting heights;
  - the side on which each task keeps its sources and goals.
- **Sampling:**
  - random sequential placement, up to 100 restarts × 400 candidates;
  - if every restart jams, a random subset of the separation lattice, which
    is Scattered by construction;
  - a `ValueError` when even the lattice is too small.
- **Default unchanged:** with the switch off, every sampler makes the same
  random draws as before, and the model XML is byte-identical. Unit tests pin
  a pre-change Stack layout and the default buffer sizes. Archived seeds,
  collection, CaP runs and the Section 6.2 invariant-learning experiment are
  unaffected. CaP records the switch in `environment_options` only when it is
  on.

## 3. Measured limits (relaxed physical probe)

- **Probe:** the Revision 9 Section 3.3 probe, run with the switch on,
  contact buffers included.
- **Location:** `runs/cap-generalization/20261028-max-blocks/physical_relaxed_v2/`,
  produced by `physical_probe_relaxed.py` and `physical_summary_relaxed.py`.
  - `physical_relaxed/` is the earlier run with the default buffers.
- **Seeds:** 0–9 per layout. Towers and Stack were tested up to 26 blocks,
  and Pyramid up to k = 7.

A seed is feasible when all of these hold:
- the reset scene passes `validate_start` and stays within 5 mm for 100
  steps;
- a teleported goal stays within 5 mm and satisfies the postcondition;
- every raised top centre is below the reach measured above its column;
- every table-level block centre is reached exactly.

Reach above 1.55 m was measured again with a 2.2 m ceiling. The original
reach measurement stops at 1.6 m. No column of any scene reached above
1.594 m.

| Task | Default: feasible on all 10 seeds | Relaxed: all 10 seeds | Relaxed: some seeds | Binding limit |
| --- | --- | --- | --- | --- |
| Stack | 2–6 | **2–22** | 23 (8/10), 24 (4/10) | Arm height: a 23-block top centre is at 1.52 m, against a reach of at most 1.59 m |
| Unstack | 2–18 | **2–22** | 23 (8/10), 24 (7/10), 25 (4/10) | Arm height |
| Reverse | 2–18 | **2–22** | 23 (7/10), 24 (5/10) | Arm height (the reversed tower) |
| ReStack (tower + 2, 4 or 8 loose) | total 4–18, 2 loose | **total 4–22** | 23 (6–7/10), 24 (2/10) | Arm height (the rebuilt tower) |
| Grid (rows × cols) | ≤ 3 × 3, 4 × 2, 2 × 4, 5 × 1 | **up to 4 × 4 (16)**; also 3 × 4, 4 × 3, 5 × 3, 5 × 2, 2 × 5, 6 × 1 | — | Reachable area: 3 × 5, 4 × 5, 5 × 4, 6 × 2, 2 × 6, 6 × 3, 7 × 2, 1 × 6 and 7 × 1 have a goal outside the region |
| Pyramid (k) | 2–4 | **2–5 (15 blocks)** | — | Source area: the lower-Y side holds at most 19 scattered sources, and k = 6 needs 21 |

- **No simulator failures:** no contact-buffer overflow occurred, and no
  block touched the robot after a goal teleport.
- **Tower margins:** at 22 blocks the top centre is 1.43–1.47 m, at least
  5 mm below the reach measured above its column.
- **Known gap:** in the run with the default buffers, ReStack 2 + 8 and
  4 + 8 lost seed 0. A loose block at (0.39, 0.70) lies inside the region
  (radius 0.799 m), between measured 5 cm cells, and the gripper did not
  reach it. The per-seed check marks such scenes infeasible.

## 4. Contact buffers

- **Default:** MuJoCo's defaults, `nconmax` 100 and `njmax` 500.
  - About 15 contacts are always taken by the robot base, laser, e-stop and
    torso against the table plane.
  - A 20-block tower or a 4 × 5 grid overflows at reset, and the settling
    15-block pyramid overflows too: `Pre-allocated contact buffer is full`,
    after which contacts are dropped.
- **Measured need:** peak usage (`maxuse_con`, `maxuse_efc`) was measured
  with a deliberately large buffer (2000, 8000):

  | Scene | Contacts | Constraint rows |
  | --- | ---: | ---: |
  | 19-block tower, toppling | 142 | 574 |
  | 15-block pyramid | 145 | 587 |
  | 28-block pyramid | 267 | 1,074 |
  | 27 blocks packed 3 × 3 × 3 | 129 | 523 |
  | 40-block tower, toppling | 306 | 1,230 |

  The worst case is about 8 contacts and 31 rows per block.
- **Chosen:** `nconmax` 1000 and `njmax` 4000.
  - This leaves about three times the worst case measured.
  - At 31 rows per block it holds roughly 125 blocks, far more than the
    22–25 the arm can stack.
- **Cost:** MuJoCo sizes its stack automatically, roughly with njmax
  squared. Measured on an Unstack scene with 12 blocks:

  | nconmax, njmax | Stack | Build time | ms per env step |
  | --- | ---: | ---: | ---: |
  | 100, 500 (default) | 13 MB | 0.36 s | 3.79 |
  | 500, 2000 | 166 MB | 0.49 s | 3.79 |
  | 1000, 4000 (chosen) | 638 MB | 0.87 s | 3.77 |
  | 2000, 8000 | 2,495 MB | 2.44 s | 3.78 |

  Step time does not depend on the buffer size. 8000 rows would cost
  2.5 GB per environment with no benefit.
- **Two model builds:** the environment builds its MuJoCo model twice, the
  second time through the MRO in `GymRoboticsGroundTruthSupportEnv`. Both
  constructors receive the override.

## 5. Probe adaptations

These are diagnostics in the run folder, not repository code:
- **Arm park point:** the default probe parks the arm at (0.35, 0, 1.0 m)
  before teleporting a goal, which is inside the relaxed tower sites. The
  relaxed probe parks it 0.60 m high at whichever of (0.40, ±0.60),
  (0.75, 0) or (0.60, ±0.45) is farthest from every goal block. It also
  records robot–block contacts after the teleport; there were none.
  - The first run with the old park point is kept as
    `physical_relaxed/scenes_park035.jsonl`.
- **Unstack goal cells:** the probe's Unstack goal cells are restricted to
  the region. The default probe's cells reach 0.24 m behind the tower, which
  is under the robot when the tower is at X 0.40.
- **High reach:** reach above 1.55 m is measured again with a 2.2 m ceiling
  (`reach_high.jsonl`).

## 6. Interpretation limits

- **Two changes at once:** relaxed scenes change the layout distribution
  even at the trained count.
  - Stack blocks may lie 0.70 m to the side.
  - Towers move up to 0.14 m towards the base.
  - Goal layouts move 0.15 m towards the base.

  A checker must therefore run the relaxed reset at the trained count as a
  baseline. Failures there are layout failures, and the count curve should be
  read relative to that baseline.
- **Necessary, not sufficient:** feasibility here covers the start, one goal
  state and reach. Carrying a block over a tall tower, or past blocks on the
  wider table, may still fail, and such failures count against the policy.
  - The 2026-10-28 best-site test found that the open gripper approaching a
    19-block tower knocks it over. The probe does not test the approach.
- **Grid and Pyramid could go further.** Their sources stay on the lower-Y
  side and their goals start at Y = 0.10. Using all free reachable area for
  sources, or centring the Grid goals at Y = 0, would fit more blocks. That
  would change those tasks' layout semantics further, so it is not part of
  this option.
