# Relaxed reset bounds for generalization tests

- **Status:** implemented 2026-10-29 (commits `7d9ec08` environment, `3f82cdb`
  CaP). Default off.
- **Requested by:** the user, for the block-count generalization checker
  ([CaP Revision 9](../cap/docs/revision_9_generalization_checker.md),
  Section 3.5). A first plan covered Stack only. The user then asked for one
  switch for every task.
- **Option:** `relax_boundaries_for_generalization_test`. It is available
  - on the environment constructor;
  - on the `make_roboverify_*` factories and `make_goal_env`;
  - as a `RunConfig` field in CaP.
- **Purpose:** with the default resets, Stack stops at 6 blocks, Grid at
  3 × 3 and Pyramid at k = 4. Those limits come from sampling rectangles and
  goal placement, not from the arm. With the switch on, every task uses the
  region the arm was measured to reach, so the limit becomes physical.

## 1. Default bounds that limit block counts

| Task | Default bound | Where | Limit it causes |
| --- | --- | --- | --- |
| Stack | X 0.54–0.70, Y −0.20–0.20, radius ≤ 0.70 m | `stack_reset.py` | 6 blocks on all seeds, 7 on 4/10 |
| Unstack, Reverse | Tower at X 0.54–0.70, Y 0.15–0.25 | `tower_reset.py` | Reach at the far corner is 1.22 m. 19 blocks only on some seeds |
| ReStack | Tower and loose blocks in one Y strip (0.15–0.25), loose X 0.40–0.70 | `restack_reset.py` | At most 2 loose blocks |
| Grid, Pyramid | Goals centred on the initial gripper (X 0.65); radius ≤ 0.8531 m; sources in a small rectangle | `fpp_construction_env.py` | Grid 3 × 3 (the far cell of 4 × 3 is out of reach); Pyramid k = 5 sources do not fit |

All coordinates are base-relative, in metres.

## 2. Relaxed geometry

One shared region (`relaxed_workspace.py`): **X ≥ 0.35 m and XY radius ≤
0.80 m.**
- **Reach map:** every 5 cm cell of this region was reached by the empty
  gripper at Z ≥ 0.64 m. Points between cells near the high-|Y| edge are not
  guaranteed (Section 3); the per-seed feasibility check covers them. Reach was measured path-aware: straight, or rise,
  traverse and descend, with 2 mm tolerance. The maps are the 2026-10-28 probe
  plus the relaxed run, which added the |Y| > 0.60 m cells.
- **Excluded near the base:** X ≤ 0.30 m near Y = 0 lies under the robot and
  is unreachable.

| Task | Relaxed layout |
| --- | --- |
| Stack | `b0` (the tower base, from `base_block_id`) first, in X 0.40–0.60, Y −0.20–0.20; the other blocks anywhere in the region |
| Unstack, Reverse | Tower in X 0.40–0.60, Y 0.15–0.25 (still on the higher-Y side) |
| ReStack | Tower as Unstack; loose blocks anywhere in the region with Y ≥ 0.15. The 2-loose-block capacity check is skipped |
| Grid, Pyramid | Goals centred at X 0.50 (shifted forward to X ≥ 0.35), first row at Y 0.10 as before; every goal must lie in the region, else `ValueError`. Sources anywhere in the region with Y ≤ −0.08 |

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
  random draws as before. A unit test pins a Stack layout from before the
  change. Archived seeds, collection, CaP runs and the Section 6.2
  invariant-learning experiment are unaffected. CaP records the switch in
  `environment_options` only when it is on.

## 3. Measured limits (relaxed physical probe)

- **Probe:** the Revision 9 Section 3.3 probe, run with the switch on.
- **Location:** `runs/cap-generalization/20261028-max-blocks/physical_relaxed/`;
  scripts `physical_probe_relaxed.py` and `physical_summary_relaxed.py`.
- **Seeds:** 0–9 per layout.

A seed is feasible when all of these hold:
- the reset scene passes `validate_start` and stays within 5 mm for 100
  steps;
- a teleported goal stays within 5 mm and satisfies the postcondition;
- every raised top centre is below the reach measured above its column;
- every table-level block centre is reached exactly.

| Task | Default reset: feasible on all 10 seeds | Relaxed: feasible on all 10 seeds | Relaxed binding limit |
| --- | --- | --- | --- |
| Stack | 2–6 | **2–19** | 20 blocks overflow the contact buffer |
| Unstack | 2–18 | **2–19** | Contact buffer |
| Reverse | 2–18 | **2–19** | Contact buffer |
| ReStack (tower + loose) | total 4–18, 2 loose | **total 4–19, with 2, 4 or 8 loose** | Contact buffer. Tower 2 + 8 loose and 4 + 8 each lost seed 0: a loose block at (0.39, 0.70) is inside the region (radius 0.799 m), between measured 5 cm cells, but the gripper did not reach it. The per-seed check marks such scenes infeasible |
| Grid (rows × cols) | ≤ 3 × 3, 4 × 2, 2 × 4, 5 × 1 | **up to 4 × 4 (16)**; also 3 × 4, 4 × 3, 5 × 3, 5 × 2, 2 × 5, 6 × 1 | Workspace: 3 × 5, 6 × 2, 2 × 6, 6 × 3, 7 × 2, 1 × 6 and 7 × 1 have a goal outside the region. 4 × 5 and 5 × 4 (20) overflow the contact buffer |
| Pyramid (k) | 2–4 | **2–4** | k = 5 (15 blocks): the dense goal overflows the contact buffer while settling, and a block falls. k = 6 (21) overflows at reset |

Tower margins at 19 blocks (top centre 1.29–1.32 m) are at least 0.14 m below
the measured reach.

## 4. The next limit: MuJoCo's contact buffer

- **The buffer:** the generated XML keeps MuJoCo's default `nconmax` of 100.
- **Fixed overhead:** about 15 contacts are always taken by the robot base,
  laser, e-stop and torso against the table plane.
- **What that leaves:** about 85 contacts for blocks.
  - A 20-block tower or a 4 × 5 grid exceeds that at reset.
  - So does the 15-block pyramid, because each upper block rests on two
    blocks.
- **Raising it:** that is a separate environment change (Revision 9,
  decision D5). After the change, tall-tower stability and the arm's
  approach to the top have to be checked again.
  - The 2026-10-28 best-site test found that a 19-block tower is knocked over
    by the arm on approach.
  - The probe above does not test the approach.

## 5. Probe adaptations

These are diagnostics in the run folder, not repository code:
- **Arm park point:** the default probe parks the arm at (0.35, 0, 1.0 m)
  before teleporting a goal, which is inside the relaxed tower sites. The
  relaxed probe parks it 0.60 m high at whichever of (0.40, ±0.60),
  (0.75, 0) or (0.60, ±0.45) is farthest from every goal block. It also
  records robot–block contacts after the teleport; there were none.
  - The first run with the old park point is kept as `scenes_park035.jsonl`.
- **Unstack goal cells:** the probe's Unstack goal cells are restricted to
  the region. The default probe's cells reach 0.24 m behind the tower, which
  is under the robot when the tower is at X 0.40.

## 6. Interpretation limits

- **Two changes at once:** relaxed scenes change the layout distribution
  even at the trained count.
  - Stack blocks may lie 0.70 m to the side.
  - Goal layouts move 0.15 m towards the base.

  A checker must therefore run the relaxed reset at the trained count as a
  baseline. Failures there are layout failures, and the count curve should be
  read relative to that baseline.
- **Necessary, not sufficient:** feasibility here covers the start, one goal
  state and reach. Carrying a block over a tall tower, or past blocks on the
  wider table, may still fail, and such failures count against the policy.
