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
| Grid, Pyramid | Goals centred at X 0.50 (shifted forward to X ≥ 0.35) and on the robot's midline (Y = 0); every goal must lie in the region, else `ValueError`. Sources anywhere in the region at least 0.105 m (Chebyshev) from every goal centre and from each other |

- **Contact buffers:** `nconmax` 1000 and `njmax` 4000
  (`RELAXED_CONTACT_BUFFER`), written into the generated model XML. Section 4
  explains the choice.
- **Unchanged:**
  - Scattered separation (0.10 m; 0.105 m for ReStack and goal sources);
  - the 0.10 m initial-gripper clearance;
  - resting heights;
  - the side on which the tower tasks keep their towers and loose blocks.
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
- **Location:** `runs/cap-generalization/20261028-max-blocks/`, produced by
  `physical_probe_relaxed.py` and `physical_summary_relaxed.py`.
  - `physical_relaxed_v2/` holds the tower tasks.
  - `physical_relaxed_v3/` holds Grid and Pyramid, after their goals moved
    to the midline and their sources to any free area. Every Grid shape from
    2 × 2 to 30 cells and Pyramid k up to 8 were probed.
  - `physical_relaxed/` is the earlier run with the default buffers.
- **Seeds:** 0–9 per layout. Towers and Stack were tested up to 26 blocks,
  and Pyramid up to k = 7.

A seed is feasible when all of these hold:
- the reset scene passes `validate_start` and stays within 5 mm for 100
  steps;
- a teleported goal stays within 5 mm and satisfies the postcondition;
- every raised top centre is below the reach measured above its column;
- every table-level block centre is reached exactly;
- every initial and goal column is served by Moves within the primitive's
  50-step budget (`synthesis/cap/generalization/reach.py`).

The last condition was added on 2026-10-29 (Revision 9, Section 3.3.1). The
reach of the probe servoed up to 1,000 steps per segment: from the start
posture the arm rises to about 1.0 m quickly and then creeps at about
0.2 mm per step, so the earlier tower limit of 22 blocks (reach up to
1.594 m) was not available to a Move. The re-scoring is in
`runs/cap-generalization/20261029-budget-feasibility/`.

| Task | Default: feasible on all 10 seeds (long-servo reach) | Relaxed: all 10 seeds | Relaxed: some seeds | Binding limit |
| --- | --- | --- | --- | --- |
| Stack | 2–6 | **2–6** | 7–10 (9/10), 11 (8/10), 12 (2/10) | One Move's height (about 1.0 m); seed 5 starts a block in front of the body |
| Unstack | 2–18 | **2–10** | 11 (7/10), 12 (3/10) | One Move's height |
| Reverse | 2–18 | **2–10** | 11 (7/10), 12 (3/10) | One Move's height (also the reversed tower) |
| ReStack (tower + 2, 4 or 8 loose) | total 4–18, 2 loose | **total 4–10** | 11 (28/30), 12 (1/29) | One Move's height (the rebuilt tower) |
| Grid (rows × cols) | ≤ 3 × 3, 4 × 2, 2 × 4, 5 × 1 | **23 shapes up to 6 × 4 (24)** | — | A goal cell at X 0.35 m on the midline is not served (1 × 4, 1 × 5, 3 × 4, 3 × 5, 5 × 4). 6 or more columns are wider than the region. 9 × 2, 7 × 3, 8 × 3, 5 × 5, 9 × 3, 7 × 4 and 6 × 5 leave too little free area for the sources |
| Pyramid (k) | 2–4 | **2–5 (15 blocks)** | — | k = 6 and 7 put a goal at X 0.35 m on the midline; the k = 8 bottom row (0.455 m) is wider than the region |

- **No simulator failures:** no contact-buffer overflow occurred, and no
  block touched the robot after a goal teleport.
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
    10–12 blocks one Move can stack.
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
- **Grid and Pyramid layouts move the most.** Their goals now sit on the
  robot's midline, and sources surround them on both sides. These are the
  largest layout changes in relaxed mode, so their trained-count baseline
  matters most.

## 7. Test-case set

The feasible seeds are collected in a versioned test-case set:
`synthesis/cap/generalization/testsets/relaxed_v2.json`, with its builder
`synthesis/cap/generalization/testset.py`. Revision 9, Section 3.6 has the
details and the seeds per block count. `relaxed_v1` (1,524 cases) used the
long-servo reach and is withdrawn; `relaxed_v2` is its subset that passes the
50-step reach, with identical initial states.

| Task | Cases | Block counts |
| --- | ---: | --- |
| Stack | 96 | 2–12 |
| Unstack | 100 | 2–12 |
| Reverse | 100 | 2–12 |
| ReStack | 158 | totals 4–12, with 2, 4 or 8 loose blocks |
| Grid | 227 | 4–24 (23 shapes) |
| Pyramid | 40 | k = 2–5 (3–15) |
| **Total** | **721** | |
