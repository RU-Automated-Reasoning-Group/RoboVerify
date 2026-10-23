# Tower workspace and Reverse placement investigation

2026-10-23. The user adopted the measured workspace for both Unstack and Reverse:
robot-base-relative **X in [0.54, 0.70] m and Y in [0.15, 0.25] m**, independent
of block count. This expands the former 4 cm by 2 cm source rectangle to
16 cm by 10 cm, while keeping the source to one side of the initial arm.
The shared production reset now uses these bounds. Primitive controllers and
their motion contracts are unchanged. Reverse's commanded destination remains
world XY **(1.25, 0.60) m**, using the existing table-relative offsets.

The large bottom-two-block offset in Reverse is reproducible. The first block
lands on the negative-X side of the commanded destination; subsequent blocks
land on its positive-X side. The controller tracks the gripper, while the block
moves within its grasp. All placements command the same table-relative XY;
`b` supplies the support height, so later XY targets do not follow the first
block's actual landing position. Similar offsets among the upper blocks make
their mutual alignment look better.

## Workspace evidence

The [experiment module](tower_workspace.py) patches only the shared reset sampler
within each trial, then executes the existing complete program with the production
controllers. Unstack uses the existing scoped extrema references from
`unstack_lateral.py`; Reverse uses `examples/reverse.py`. No primitive budget,
collision mask, task predicate or success tolerance was changed.

| Candidate-region executions | Unstack | Reverse |
| --- | ---: | ---: |
| Two blocks, 5 by 5 grid | 25/25 valid | 25/25 valid |
| Three blocks, same grid | 25/25 valid | 25/25 valid |
| Four blocks, same grid | 25/25 valid | 25/25 valid |
| Four blocks, 25 random interior points | 25/25 valid | 25/25 valid |
| Total | **100/100** | **100/100** |

Grid X values were 0.54, 0.58, 0.62, 0.66, 0.70 m; Y values were 0.15,
0.175, 0.20, 0.225, 0.25 m. Random coordinates used NumPy `default_rng(73)`;
all sampled points were retained. Validation includes task entry/exit conditions
and convergence of every primitive. The maximum primitive used 18 of 50 steps.
No candidate reset had a reported robot–block contact. Each Unstack invocation
was bounded by a 55-second external timeout; all completed, with reported run
times below 19 seconds for these batches.

The preceding exploratory four-block grid deliberately extended beyond this
candidate region: X = {0.54, 0.62, 0.70, 0.74}, Y = {0.10, 0.15, 0.25, 0.30}.
Unstack passed 13/16 and Reverse 14/16. Failures were retained:

| Task | Base-relative source XY (m) | Failure |
| --- | --- | --- |
| Unstack | (0.74, 0.10) | Transfer exhausted 50 steps |
| Unstack | (0.74, 0.25), (0.74, 0.30) | Source approach exhausted 50 steps |
| Reverse | (0.70, 0.10) | Final reversal condition false despite converged primitives |
| Reverse | (0.74, 0.30) | Source approach exhausted 50 steps |

Thus extending to 0.74 m is not supported by the unchanged programs. The failure
at (0.70, 0.10) is an execution failure, not evidence of an initial collision.

A separate **60-reset geometry probe** used 2–4 blocks, X = {0.54, 0.62, 0.66,
0.70}, and Y = {0, 0.05, 0.10, 0.15, 0.25}. Eight starts already overlapped
robot collision geometry: two three-block starts on the centerline and six
four-block starts at Y = 0 or 0.05. At X = 0.66, Y = 0, the four-block tower
overlaps both fingers and the gripper body. The two-block probes had no such
contacts. These are reset contact queries, not complete executions or a proof
of swept-arm clearance. The active model still excludes head collisions.

The adopted common Y interval also shifts the two- and three-block Unstack
row toward higher Y compared with the former count-dependent centering. It
preserves the 15 cm spacing and right-to-left execution. The evidence supports
a larger candidate reset region; it is not a continuous-workspace safety proof
or a certified success rate.

After adopting the reset, fresh production-factory executions passed **30/30**:
seeds 0–4 for both tasks at each of 2–4 blocks. These runs use no reset override
or payload-feedback intervention. Thirteen focused reset, execution, diagnostic
and snapshot-replay tests passed. The Reverse destination test checks actual
gripper placement endpoints against the 2 mm controller tolerance and validates
the reversal postcondition separately; the earlier 1 cm payload-centering
assertion was an empirical property of the old test start, not a skill contract.
The new four-block seed 0 reaches 10.64 mm block offset on one axis despite a
converged gripper and valid reversal, consistent with the limitation below.

Production rechecks: `runs/unstack-lateral/20261023-112001-b7d469a-2-blocks`,
`20261023-112007-b7d469a-3-blocks`, `20261023-112014-b7d469a-4-blocks`; and
`runs/tower-workspace/20261023-112002-b7d469a-reverse-2-production-expanded`,
`20261023-112009-b7d469a-reverse-3-production-expanded`,
`20261023-112017-b7d469a-reverse-4-production-expanded`.
Historical Unstack motion-verification results retain their original narrower
input region; expanding the reset does not extend those recorded proofs.

## Why the bottom two Reverse blocks look misaligned

The five current four-block demonstrations (seeds 0–4) were replayed from their
exact archived initial physics states. Every baseline replay reproduced the
original actions and observations within 1e-8. Every intervention restored the
same `gt_state` and all saved simulator arrays exactly. Final adjacent-center
XY distances, ordered from bottom to top, were:

| Adjacent blocks | Production controller | Diagnostic block XY feedback |
| --- | ---: | ---: |
| First–second | **11.18–12.30 mm** | **1.35–1.49 mm** |
| Second–third | 2.57–3.02 mm | 0.15–0.55 mm |
| Third–fourth | 0.31–0.57 mm | 0.52–0.62 mm |

For seed 0, the measured sequence explains the largest gap:

| Measurement | First transferred block | Second transferred block |
| --- | ---: | ---: |
| Block minus gripper X after Pick | -1.94 mm | -0.39 mm |
| Block minus gripper X after lift | -3.48 mm | +0.74 mm |
| Block minus gripper X after lowering | -3.94 mm | +5.26 mm |
| Block X minus commanded destination after Release | -5.36 mm | +5.38 mm |
| Block X minus commanded destination at program end | -5.19 mm | +6.78 mm |

The first block is about 14.1 mm above the gripper at the end of lowering:
the gripper reaches approximately Z = 0.4253 m while the block center is at
0.4393 m. It then drops onto the table during Release and shifts another
1.7 mm in negative X. The second block is only about 0.7 mm above its gripper
at lowering. Later placements also move already-placed blocks slightly.
Gripper-frame measurements confirm relative movement within the grasp; a
changing gripper orientation alone does not explain the offsets. The precise
contact-mechanical origin of the first grasp's different drift is not isolated
by this experiment.

This is a payload tracking limitation of the primitive's centered-attachment
assumption. The loop executes all four transfers in the correct order. A
2 mm gripper stopping tolerance does not bound block-center error. Also, the
task's ON* geometry permits less than 25 mm error on each horizontal axis, so
passing the reversal postcondition does not mean visually centered stacking.

## Paired interventions

All four variants use the same five starts and the same instruction budgets.
Only the experimental controller or Pick tolerance changes.

| Variant | Complete, validated executions | Result |
| --- | ---: | --- |
| Production baseline | 5/5 | Reproduces the bottom-two gap |
| Pick tolerance tightened from 2 mm to 0.5 mm | 0/5 | Pick descent exhausts its 50-step budget |
| Measured block XY feedback during table-directed Moves | 5/5 | Bottom-two gap reduced to 1.35–1.49 mm |
| Measured block XYZ feedback during the same Moves | 0/5 | Lowering exhausts its 50-step budget |

The successful diagnostic still uses the same fixed destination. During transport
and lowering it adjusts gripper XY using the observed carried block's XY error;
Z continues to use the ordinary gripper controller. It retains gain 20, 2 mm
stopping tolerance and the 50-step budget. Release is unchanged. The XYZ
extension is **not** an accepted execution, even when its final picture looks
aligned. No failed or timed-out primitive is counted as successful.

These paired interventions support block-position-aware XY control as a useful
fix direction. They do not establish a production controller or its compatibility
with the geometric verifier. The scoped interventions have distinct diagnostic
fingerprints, and their archives must not be treated as production-program demos.

**Decision: defer payload-aware placement.** The user chose to retain this idea
as a note, without adopting it. Measuring the block–gripper offset and adjusting
placement introduces a stronger primitive than the current gripper-target skill.
Future adoption requires an explicit contract for sensing, grasp-offset changes,
tracking and release, and a review of the verification assumptions. Existing
production primitives, their contracts and default execution remain unchanged.

![Measured final placements, seed 0](../../runs/tower-workspace/20261023-110734-d3fbc5e-investigation-summary/artifacts/placement-comparison.png)

[Diagnostic XY-feedback video](../../runs/tower-workspace/20261023-110043-d3fbc5e-reverse-4-payload-video/artifacts/videos/trial-000-payload_xy.mp4),
[original video for the same seed](../../demos/reverse/4-blocks-table-loop-video/videos/seed_0000.mp4),
[standalone SVG comparison](../../runs/tower-workspace/20261023-110734-d3fbc5e-investigation-summary/artifacts/placement-comparison.svg).

## Reproduction and artifacts

Use the simulator environment from [AGENTS.md](../../../AGENTS.md), then run
from `roboverify/`. Each workspace command below is one bounded batch. Repeat
the grid command with `--task unstack` and block counts 2 and 3 for the full grid.

```bash
timeout --signal=TERM --kill-after=2s 55s uv run python -m synthesis.experiment.tower_workspace \
  --task reverse --num-blocks 4 --x .54 .58 .62 .66 .70 \
  --y .15 .175 .20 .225 .25 --run-name candidate-grid

timeout --signal=TERM --kill-after=2s 55s uv run python -m synthesis.experiment.tower_workspace \
  --task unstack --num-blocks 4 --x .54 .70 --y .15 .25 \
  --samples 25 --seeds 73 --run-name candidate-random

uv run python -m synthesis.experiment.tower_workspace --task reverse --num-blocks 4 \
  --x .54 .62 .66 .70 --y 0 .05 .10 .15 .25 --initial-only --run-name clearance

timeout --signal=TERM --kill-after=2s 55s uv run python -m synthesis.experiment.tower_workspace \
  --task reverse --num-blocks 4 --seeds 0 1 2 3 4 \
  --archive demos/reverse/4-blocks-5-trajectories-table-loop/demonstrations.npz \
  --variants baseline tight_pick payload_xy payload_xyz --save-traces --run-name placement-final

uv run python -m synthesis.experiment.report --glob 'runs/tower-workspace/*' --table
```

The paired command uses an existing current archive; collect fresh Reverse demos
first if it is absent. Tests generate their own traces instead of reading saved
demos. Seven focused tests passed across `test_tower_workspace`, `test_reverse`
and `test_unstack_reset`, including exact replay, scoped intervention cleanup,
distinct executable identity and the centerline contact probe.

The [aggregate summary](../../runs/tower-workspace/20261023-110734-d3fbc5e-investigation-summary/artifacts/summary.json)
lists the eight candidate-region runs and the paired run. Every execution run
contains bounded `artifacts/summary.json` and per-trial measurements. Paired
full-state traces are in
[`20261023-110343-d3fbc5e-reverse-4-placement-final`](../../runs/tower-workspace/20261023-110343-d3fbc5e-reverse-4-placement-final/artifacts/summary.json).
The exploratory failures are retained in the
[Reverse grid](../../runs/tower-workspace/20261023-105630-d3fbc5e-reverse-4-grid4/artifacts/summary.json)
and [Unstack grid](../../runs/tower-workspace/20261023-105646-d3fbc5e-unstack-4-grid4/artifacts/summary.json).
