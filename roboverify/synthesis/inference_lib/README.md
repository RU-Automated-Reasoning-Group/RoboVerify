# Collecting demonstrations and learning loop invariants

Use full-state NPZ archives for collection, synthesis, and inference. The pipeline
requires archives matching the current task specification; observation-only and
loop-head JSON demonstration inputs are unsupported. Generated archives are not
bundled in the repository. The four local demonstration archives were deleted
at the user's request. A separate fresh four-block archive was collected for the
[2026-10-01 PyEDA verification recheck](../cfg/VERIFICATION.md#provided-stack-verification).
The commands below describe collection when the requested archive is absent.
Preserved historical reports and videos describe their original controller configurations.

## Cross-environment audit

The 2026-10-23 audit collected seeds **0–14** for each configuration below,
using the shared `collect_demos` entry point. Directories are relative to
`roboverify/demos/pipeline-audit-20261023/`; each accepted collection contains
`demonstrations.npz`, `collection.json` and `audit.json`.

| Task | Configuration | Valid / requested | Directory |
| --- | --- | ---: | --- |
| Stack | 4 blocks | 15/15 | `stack/` |
| Unstack | 4 blocks | 15/15 | `unstack/` |
| Reverse | 4 blocks | 15/15 | `reverse/` |
| Restack | 5 blocks, initial tower 3 | 15/15 | `restack-3/` |
| Restack | 5 blocks, initial tower 4 | 15/15 | `restack-4/` |
| Grid | 3×3, 9 blocks | 15/15 | `grid/` |
| Pyramid | 4+3+2+1, 10 blocks | 15/15 | `pyramid-recheck/` |

All **105** accepted traces have converged primitives and valid pre/postconditions.
The separate reusable archive audit checks seed coverage, current program/task
identities, initial-state restoration, full action replay, simulator task success,
and success after 50 additional final settling steps. Final action replay matched
all observations exactly. Maximum primitive use was 24/50 steps.
Run an audit on any of these provided-program archives with:

```bash
uv run python -m synthesis.entry.audit_demos \
  --demos demos/pipeline-audit-20261023/stack/demonstrations.npz \
  --output demos/pipeline-audit-20261023/stack/audit.json
```

It expects seeds 0–14 by default; use `--seeds` for a different exact set. Unstack
audit invocations also require the external 55-second timeout shown above.
`summary.json` in the audit root indexes every collection and audit report.
These checks are empirical execution validation; synthesis and formal proof
support are listed separately in the [project support matrix](../../../README.md#project-status).

The original Pyramid batch in `pyramid/` rejected seed 12 after a high approach
exhausted 50 steps, missing by 5.714 mm. Its diagnostic archive remains in
`pyramid/diagnostics/seed_0012.npz`. The source region was moved 4 cm closer and
all the same seeds were recollected in `pyramid-recheck/`. The
[failed-run video](../../demos/pipeline-audit-20261023/pyramid/videos/seed_0012_failed_replay.mp4)
replays the original saved actions and begins failing at 24.55 s. It was rendered
from the archive, not from a new rollout. No failed seed was dropped or substituted.

## Collect Stack demonstrations

Run from `roboverify/`:

```bash
unset LD_PRELOAD
export LD_LIBRARY_PATH="$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia"
uv run python -m synthesis.entry.collect_demos \
  --program synthesis.examples.stack:build_program \
  --num-blocks 3 --num-trajectories 5 --save-video
```

All seven tasks have a default supplied program; `--program` is optional and
selects a replacement factory. The editable factory in `synthesis/examples/stack.py`
uses explicit Pick, Move,
and Release primitives. Its transfer and placement waypoints use the base block
`b0` for X/Y alignment and the current tower top `b` for Z. A custom
`--program module:factory` or
`--program path/to/program.py:factory` must return a `Program` from
`factory(context, *, num_blocks)`. Named and numeric physical operands, Assign,
Get, Skip, and flat While loops are supported for tower tasks. PickPlace is
rejected. Grid/Pyramid/Build3D collection additionally accepts nested goal loops as
described below. Numeric operands get fixed aliases; their identities are not learned.

Collection runs exactly N distinct seeds, defaulting to five seeds starting at
zero. Use `--seed-start 10` for another consecutive range or `--seeds 3 8 12`
for explicit seeds. An explicit trajectory count must match the explicit list.
Failed seeds are retained as failures, never replaced by easier seeds.

Before running the DSL, collection holds the reset gripper position for exactly
**50 control steps** with the gripper open. The full state after those steps
(S50, the 51st state counting reset) becomes demonstration **state zero**. The
first 50 states and preparation actions are excluded from the archive, loop
traces, and video. The trajectory timeout includes this preparation. Every
trace records `initialization.source` and `initialization.settling_steps`.

Synthesis, candidate execution for invariant inference/verification, and
standalone MCMC restore this archived full state when restarting a demo. They
do not reconstruct it by calling reset with the seed, and do not settle it
again. Seeds identify trajectories; snapshots define their actual starting
states. Later segments use their saved snapshot or replay only the recorded
program actions from state zero. Archives replay their own stored states.
Recollect archives made before the settling policy to obtain settled starts, and
archives made before the equal-height precondition to match the current task
identity. A current file format alone does not establish task compatibility.

Default output:

```text
demos/stack/3-blocks-5-trajectories/
  demonstrations.npz
  collection.json
  videos/seed_0000.mp4
  videos/seed_0001.mp4
  ...
```

Repeated default collections use numbered suffixes. `--output-dir demos/my-stack-demonstrations` chooses a new destination; an existing explicit
directory is rejected. Use the actual archive path printed by collection in
subsequent commands; examples below assume the first default collection. There
is no collection run-name flag.

Every accepted trajectory must finish normally, start with unstacked,
pairwise-scattered blocks at one height level, and end with all blocks in the tower
rooted at b0. The entry premise is `forall x,y. Higher(x,y)`; since Higher means
at least as high within the configured tolerance, both ordered pairs enforce
one initial height level (at most 1 mm spread by default). This is an entry
condition, not a requirement on later loop states or the final tower.
Transient success is insufficient. The accepted archive is published only if
all requested trajectories pass. Diagnostic archives and the per-seed report
retain failures. `--max-loop-iterations` defaults to 100 and
`--trajectory-timeout-seconds` to 60; exhaustion is incomplete execution.

`--save-video` records the same execution, headlessly, at fixed **20 FPS**. Each
seed gets its own MP4, including partial failed executions where possible.
`--render` independently displays a live window. The ffmpeg executable is
required for videos. Encoding failures are reported separately and produce a
nonzero command result without discarding valid trajectory data. Frames are
streamed rather than retained in memory. Rendering adds no simulator steps;
recording starts after the 50-step preparation.

## Collect Unstack demonstrations

The existing lateral groundtruth now runs through shared execution, without an
experiment-specific wrapper:

```bash
timeout --signal=TERM --kill-after=2s 55s \
  uv run python -m synthesis.entry.collect_demos --task unstack \
  --num-blocks 4 --num-trajectories 15 --trajectory-timeout-seconds 12
```

The adopted groundtruth and default is `synthesis.examples.unstack:build_program`.
Its guard explicitly
requires `b_prime != b0`, retaining the source base and moving each clear upper
block 15 cm toward decreasing Y. `--num-blocks 1` is supported by the collector:
the already-unstacked scene exits with zero iterations and no primitive actions.
This strengthens the original guard rather than defining a separate selectable
program; the difference and proof scope are recorded in
[discrepancy 10](../../../PAPER-DISCREPANCIES.md#10-the-historical-unstack-oracle-does-not-establish-its-final-task-condition).
`Y_min` and
`Z_m` select physical block coordinates at instruction entry; `gripper` captures
the current gripper position. Each Move freezes its target throughout execution.
These are coordinate references for Move/Release, not physical Pick operands.
The reference semantics enter fingerprints. Collection and supplied-program
verification opt in for execution/inference; ordinary synthesis loading rejects
them. Motion checking encodes them with independent extrema witnesses, with
remaining geometric proof obligations documented in the verification guide.
The shared pipeline now learns an
invariant and proves the current Unstack abstraction; see the
[tower verification guide](../cfg/VERIFICATION.md#provided-unstack-and-reverse-verification).
Direct reset, controller tolerances and step budgets are unchanged. The old
experiment wrapper has been removed. Use smaller seed batches when rendering
so each Unstack invocation remains within the 60-second wall-clock limit.

The strengthened-guard recheck collected **60 valid demonstrations**, 15 seeds
(0–14) each for 1, 2, 3 and 4 blocks, under
`demos/unstack-base-guard-20261023/{N}-blocks/`. Each directory contains
`demonstrations.npz`, `collection.json` and `audit.json`. Every trajectory passed
exact action replay and retained physical task success after 50 extra steps;
iteration counts are respectively 0, 1, 2 and 3. All 15 four-block executions
have identical actions and observations to the previous guard's archived starts;
the comparison is in `guard-comparison.json` at the collection root. The new guard
changes executable fingerprints, so use these fresh archives with the current
factory rather than the earlier collection-audit Unstack archive.

Reproduce for each N in 1, 2, 3, 4 (choose a fresh output directory):

```bash
uv run python -m synthesis.entry.collect_demos --task unstack --num-blocks 4 \
  --num-trajectories 15 --seed-start 0 \
  --output-dir demos/unstack-base-guard-20261023/4-blocks
uv run python -m synthesis.entry.audit_demos \
  --demos demos/unstack-base-guard-20261023/4-blocks/demonstrations.npz \
  --output demos/unstack-base-guard-20261023/4-blocks/audit.json
```

## Collect Reverse demonstrations

Reverse starts one tower to the robot's right (higher Y), with base-relative
X in [0.54, 0.70] m and Y in [0.15, 0.25] m, the same source workspace as Unstack.
The source order is the existing
canonical order with b0 at the bottom. The physical program in
[`synthesis.examples.reverse`](../examples/reverse.py) starts with `b <- tbl`.
One While loop handles all N blocks, moving the original top first and the
original base last. After every release, `b <- b_prime` updates the support
for the next block. Bottom-to-top order [0, 1, 2, 3] becomes [3, 2, 1, 0].

The guard selects a clear physical block ON* b0, excluding the current `b`.
Initially `b` is the table. After the last transfer, b0 is also `b`, so the guard
is false. There is no Get or physical transfer before the loop, and no separate
first-iteration branch.

The loop uses ordinary Pick, Move and Release primitives. Its destination Move
is `Move(tbl, tbl, b, (dx, dy, 0.05))`: table XY positions the column and `b`
supplies its support height. `dx` and `dy` are ordinary trainable parameters;
the groundtruth supplies (+0.45, -0.15) m. With the current physical table origin
(0.8, 0.75, 0.4), the commanded column is (1.25, 0.60), on the robot's other side.
This change exposes the placement parameters for learning; no optimization run
is claimed here.

Move/Release resolve the physical table at its actual XY origin and use
reference Z = table surface minus half a block (0.375 m here). Thus the same
+0.05 m placement offset yields 0.425 m for a table placement, or one full block
above an existing block's center. This is a controller coordinate convention;
relational tbl remains an isolated geometry-free marker in predicates. Pick
still requires a physical block. Missing table geometry fails explicitly.
The table-reference convention is recorded in table-using program fingerprints.

Transfers and retreats reference tbl at `N*0.05 + 0.10` m above that support
reference, 10 cm above the highest nominal block center. The physical program
supports 2–4 blocks. Primitive limits remain 50 steps and tolerances remain 2 mm.

Run from `roboverify/` with the MuJoCo environment configured:

```bash
timeout --signal=TERM --kill-after=2s 55s \
  uv run python -m synthesis.entry.collect_demos \
  --task reverse --program synthesis.examples.reverse:build_program \
  --num-blocks 4 --seeds 0 1 2 3 4 --max-loop-iterations 4
```

Add `--save-video` to record the same execution at 20 FPS. Reverse starts directly
from its reset, without Stack's 50-step settling phase. Saved full states replay
exactly. The task precondition is one tower above b0 with frozen ON* equal to its
entry geometry. Its final condition is
`ON_star_final(x,y) iff ON_star_entry(y,x)` for every physical-block pair.
An unchanged tower, scattered blocks, or an incomplete transfer is rejected.
Every primitive must converge and the loop must exit normally. The relational
postcondition specifies block order; the regression tests also check leftward
placement and top-first transfer order.

**2026-10-23 single-loop validation:** all 15 fresh executions passed, five seeds
(0–4) each with 2, 3 and 4 blocks. Each program made exactly N loop iterations;
every physical control event occurred inside that loop. The 54 focused controller,
Reverse/Unstack, collection, replay and motion tests also passed, including a
nonzero physical base ID and archive/replay with `b = tbl` and `b = block`.
Collections are under `demos/reverse/{2,3,4}-blocks-5-trajectories-table-loop/`,
with `demonstrations.npz` and `collection.json` in each directory.
The aggregate is `demos/reverse/table-loop-validation.json`; all transfers were
top-first and final XY error relative to the intended column stayed below
9.72 mm. Maximum primitive budget use was 17/50 steps. A representative video is
`demos/reverse/4-blocks-table-loop-video/videos/seed_0000.mp4`.
The older `*-lateral` collections and video retain the previous program, which
handled the first block outside its loop; their fingerprints differ.

These results validate the tested simulator executions. The integrated
synthesis CLI still exposes Stack and Unstack. The historical standalone Reverse
verifier encoded a different task and has been removed; see
[discrepancy 39](../../../PAPER-DISCREPANCIES.md#39-the-historical-reverse-verifier-encodes-unstack-rather-than-reversal).

## Collect Restack demonstrations

Restack uses the two-phase program adapted from Fig. 16: free the designated b0,
then build a complete tower on it. The source tower and loose blocks start on
the higher-Y side of the robot. Temporary placements form a row toward lower Y;
b0 is the last parked block and becomes the base of the new tower there.

```bash
uv run python -m synthesis.entry.collect_demos --task restack --num-blocks 4 --save-video
uv run python -m synthesis.entry.collect_demos --task restack --num-blocks 5 --initial-tower-blocks 3 --save-video
uv run python -m synthesis.entry.collect_demos --task restack --num-blocks 5 --initial-tower-blocks 4 --save-video
```

`--num-blocks` is the total (default four). `--initial-tower-blocks` specifies
how many start in the tower (default max(2, N-1)); the others start loose.
With five blocks, heights three and four produce respectively two and one loose
blocks. Tower membership/order, its XY, loose-block XY and b0's interior level
are randomized. A two-block source places b0 on top. The source tower shares
Unstack/Reverse's base-relative X [0.54, 0.70] m, Y [0.15, 0.25] m workspace;
loose blocks may extend to X = 0.40 m. The bounded sampler preserves at least
105 mm separation between columns and refuses layouts that do not fit.
At most three separate source columns fit this region; configuring a count
is not a guarantee of physical success for arbitrary heights.

The default factory is [`synthesis.examples.restack:build_program`](../examples/restack.py).
Explicit `--program` accepts a replacement primitive program. Both phases run
through ordinary While/Program execution; the collector saves both loops and
full-state snapshots after 50 settling steps. Restore/replay uses those states
without settling again. Accepted runs require primitive convergence, normal
termination, separated/comparable source blocks, and all physical blocks ON* b0
at the end. Budgets and failure/archive handling match the other collection tasks.

The supplied program corrects the paper's first-loop guard, which would continue
selecting b0 after placing it on the table. `p <- tbl` tracks the last parked
block, the first guard excludes p, and `p <- b_prime` ends each iteration.
Once p is b0 that loop exits. The second loop starts `b <- b0` and handles every
remaining clear block. Table-relative XY offsets and clearance adapt the paper's
waypoints to the lateral setup, using unchanged primitive controllers. These
changes and the Table 8 precondition correction are documented in
[discrepancy 41](../../../PAPER-DISCREPANCIES.md#41-restack-needs-a-terminating-first-loop-and-an-executable-lateral-layout).
This is supplied-program execution/collection; Restack synthesis and formal
verification have not been added.

Five-seed collections passed for four blocks and both five-block layouts above.
All 15 filmed trajectories formed a complete physical tower, with converged
primitives. Data and videos are in `demos/restack/4-blocks-5-trajectories/`,
`demos/restack/5-blocks-3-tower-5-trajectories/` and
`demos/restack/5-blocks-4-tower-5-trajectories/`; the aggregate diagnostic is
`demos/restack/validation.json`. The six Restack tests and 36 shared regression
tests passed, including exact snapshot/action replay of both phases.

## Collect Grid and Pyramid demonstrations

Pyramid source center X is initial-gripper X minus 0.14 m; Grid uses minus
0.10 m. This preserves source separation while making the ten-block Pyramid
source corner reachable at the top goal clearance.

The collector now uses the provided goal-traversal groundtruth by default for
these tasks. Configure dimensions, rather than specifying a conflicting total
block count:

```bash
uv run python -m synthesis.entry.collect_demos --task grid --grid-rows 3 --grid-cols 2 --save-video
uv run python -m synthesis.entry.collect_demos --task pyramid --bottom-row-blocks 3 --save-video
uv run python -m synthesis.entry.collect_demos --task pyramid --bottom-row-blocks 4 --save-video
```

These create five trajectories each unless `--num-trajectories` or `--seeds` is
specified. Pyramid widths 3 and 4 create 3+2+1 and 4+3+2+1 layers respectively.
For the requested nine-block Grid, use `--task grid --grid-rows 3 --grid-cols 3`.
Explicit `--program synthesis.examples.grid:build_program` and
`--program synthesis.examples.pyramid:build_program` select the same defaults;
custom factories use the existing `module:factory` / `path.py:factory` interface.

Both programs execute ordinary nested While loops, with r links within goal
rows and d links between row heads. Only goals have traversal pointers. The
collector holds the initial gripper for 50 settling steps, then records the
actual primitive execution. Accepted demonstrations start with scattered blocks
and empty goals, finish with every goal occupied, and have converged primitives.
Release checks use the payload's actual goal occupancy and preserve earlier
occupied goals; the Release reference itself supplies only retreat height.
An incomplete or invalid seed prevents publication of the combined archive,
while diagnostic archives retain completed and failed attempts.

Default output directories include dimensions:

```text
demos/grid/3x2-6-blocks-5-trajectories/
demos/pyramid/3-bottom-row-6-blocks-5-trajectories/
demos/pyramid/4-bottom-row-10-blocks-5-trajectories/
```

Each contains `demonstrations.npz`, `collection.json` and optional `videos/`.
Goal archives retain full simulator snapshots/actions, immutable goal coordinates
and topology, and separate block/goal bindings. Events preserve outer/inner loop
paths, invocation IDs, entry indices and logical cursor changes between actions.
Exact snapshot restoration and action replay restore goal bindings, including
event bindings supplied via `DemoSegment.goal_bindings`. Mismatched row layouts
are rejected. These optional version-2 fields preserve existing tower archives.

**2026-10-23 collection validation:** all 15 demonstrations passed on seeds 0–4
for the three layouts above, with all videos saved. Maximum final goal errors
were 9.57 mm (Grid), 8.51 mm (six-block Pyramid), and 10.16 mm (ten-block Pyramid),
within the shared 20 mm occupancy tolerance. The requested 3×3 Grid also passed
on seed 0, with nine transfers and at most 9.05 mm final goal error; its archive
is `demos/grid/3x3-9-blocks-1-trajectories/demonstrations.npz`.
The aggregate is `demos/goal-collection-validation.json`. All 77 focused
collection, replay, runtime/controller, task and existing inference tests passed.

The collector explicitly opts into goal programs in the shared loader. The
synthesis/verification entry points retain their existing restrictions, and
`DemoStore.from_archive` / `loop_store` reject goal traces rather than dropping
their goal state. Collection support does not establish goal-aware synthesis,
invariant learning or formal verification.

## Collect Build3D demonstrations

Build3D creates an `m × n × k` rectangular arrangement: `m` rows per layer,
`n` columns per row, and `k` vertical layers. All dimensions default to 2 and
must be positive integers; the block count is their product. The environment
uses Grid's 10 cm horizontal spacing for finger clearance, with each higher
block directly above its support at 5 cm vertical spacing. Goal centers are
on the higher-Y side of the robot and scattered source blocks on the lower-Y
side. Source center X is initial-gripper X minus 0.14 m; the target grid is
centered 0.04 m closer in X than the initial gripper.

```bash
uv run python -m synthesis.entry.collect_demos --task build3d \
  --build-rows 2 --build-cols 3 --build-layers 2 \
  --num-trajectories 5 --save-video
```

The default groundtruth is [`synthesis.examples.build3d`](../examples/build3d.py).
It reuses the unchanged six-primitive Grid/Pyramid transfer, with approach and
transport Z equal to the current goal Z plus 0.15 m. Its three loops follow
goal-only links: `r` within a row, `d` between row heads within a layer, and `u`
between layer heads. The last row's `d` is null, so the entire current layer
finishes before the next layer starts. Physical blocks have no traversal links.

```text
u <- h0
while u != null:
    d <- u
    while d != null:
        r <- d
        while r != null and exists b_prime: not AtGoal(b_prime):
            Move(b_prime, b_prime, r, (0, 0, 0.15))
            Pick(b_prime)
            Move(b_prime, b_prime, r, (0, 0, 0.15))
            Move(r, r, r, (0, 0, 0.15))
            Move(r, r, r, (0, 0, 0))
            Release(r, 0.15)
            r <- r.r
        d <- d.d
    u <- u.u
```

Default output names include all three dimensions, for example
`demos/build3d/2x3x2-12-blocks-5-trajectories/`. The shared collector, archive
auditor and optional `execute_goal_program --task build3d` all support this task.
Collection uses the same 50-step initial settling, controller convergence checks,
identity-independent occupancy, and preservation of earlier goals as the 2D
tasks. Version-2 archives add `layer_row_counts` and `u` links only for Build3D;
existing Grid/Pyramid serialization is unchanged. Snapshots and events preserve
all three cursors, including logical cursor changes between physical actions.
Loading a snapshot with a different layer partition is rejected even when its
flattened row lengths match.

**Physical validation:** 25/25 demonstrations passed, with exact final-state
action replay and successful occupancy after 50 additional settling steps:

| Dimensions | Blocks | Seeds | Archive directory under `demos/build3d/` |
| --- | ---: | --- | --- |
| 2×2×2 | 8 | 0–14 | `2x2x2-15-seeds/` |
| 2×3×2 | 12 | 0–4 | `2x3x2-5-seeds/` |
| 2×2×3 | 12 | 0–4 | `2x2x3-5-seeds/` |

Each directory contains `demonstrations.npz`, `collection.json` and `audit.json`.
All final replay errors were zero; the maximum primitive used 18 of 50 steps.
An additional twelve-block video is in
`demos/build3d/2x3x2-video/videos/seed_0000.mp4`.
The 22 focused Build3D and existing 2D goal tests passed. They cover single-row,
single-column and single-layer topology, all three loop exits, archive replay,
dimension conflicts, and synthesis-loader rejection.

Dimensions are configurable rather than hardcoded to these tested shapes.
The finite robot workspace still limits physical layouts: source sampling fails
explicitly when separated blocks cannot fit, out-of-workspace goals are rejected,
and unreachable motions fail collection through the normal primitive budgets.
These runs do not establish success for arbitrary sizes. As for Grid/Pyramid,
Build3D currently supports execution and collection, not synthesis or formal
verification.

## Higher height tolerance

`Higher(x,y)` evaluates `z(x) >= z(y) - tolerance`. The default tolerance is
**0.001 m (1 mm)**, compared with a **0.05 m** block side. This treats small
contact-induced height differences as the same level while retaining the order
of separated block levels. `Higher(x,x)` remains true, and the table remains an
isolated logical marker.

Both `collect_demos` and `synthesize_cfg` accept `--higher-tolerance METRES`.
Use `--higher-tolerance 0` for exact comparison. Values must be finite,
nonnegative and below half a block length (0.025 m). Collection metadata and
pipeline configuration record the setting. Archived coordinates are never
rewritten; existing observations can be re-evaluated at another tolerance.
New candidate executions record their active setting as well.

For direct Python calls, use a scoped setting:

```python
from synthesis.inference_lib.demo_store import InvInference
from synthesis.util.on import using_higher_tolerance

with using_higher_tolerance(0.001):
    # Runtime guards, predicate search, invariant learning and verification
    # invoked here share this setting. It is restored when the scope exits.
    invariant = InvInference(store, loop_id, vocabulary, context)
```

The numeric comparison, low-level Z3 translation, and geometric counterexample
realization share the formula. A `LowLevelContext` captures the active setting
when constructed and also accepts an explicit `higher_tolerance` argument.
Abstract Higher axioms and placement WP rules remain unchanged. Tolerant
comparison is not transitive for arbitrary continuous heights; interpreting it
as an order requires a suitable separated-level domain. The threshold is not
a controller tolerance and does not establish physical/model equivalence.

Compare already recorded Stack loop heads and normal exits with ideal placements.
Use your own collection path; this example assumes a local 500-trajectory archive:

```bash
uv run python -m synthesis.experiment.compare_stack_heights \
  --demos demos/stack/4-blocks-500-trajectories/demonstrations.npz \
  --higher-tolerance 0.001
```

The script supports the supplied single-tower Stack program, reconstructing
50 mm levels from recorded `b`/`b_prime` placements instead of rounding observed
heights. It checks every ordered physical-block pair, reports exact and tolerant
Higher results, and separately checks ON*, frozen ON*, Scattered and equality.
It also checks Higher reflexivity, totality and transitivity on each saved state.
Reports go under `runs/height-comparison/`; `--output-dir` and `--run-name` change
the destination. A difference in any checked predicate produces a nonzero exit;
a Higher match must not hide an independent Scattered mismatch. This analysis can
read saved states from an earlier task specification without making that archive
compatible with the current synthesis/verification pipeline.

## Stack reset workspace

New Stack resets sample block centers relative to the robot base: X is
0.54–0.70 m forward, Y is within ±0.20 m, and horizontal distance from the base
is at most **0.70 m**. Every layout retains at least
0.10 m separation in X or Y between blocks and 0.10 m horizontal clearance from
the initial gripper. Blocks start at the resting table height; b0 is unchanged.
Sampling restarts a crowded layout with bounded retries and reports an error if
it cannot fit the requested count; it never expands the region as a fallback.

The bound applies to every newly sampled layout, independently of seed. It is
an XY workspace restriction, not a proof of reachability at every height or for
every tower size. Archives collected with earlier reset bounds keep their saved
layouts; recollect them to use this region.

## Primitive controller settings

`Pick`/`Move`/`Release` and their ByName variants share the controllers in
`api/control.py`. A named instruction only resolves its operands before running
the same controller. Every primitive keeps a default **50-step total budget**;
Pick shares it across approach, opening, descent, and closing.

| Setting | Pick / PickByName | Move / MoveByName | Release / ReleaseByName |
| --- | --- | --- | --- |
| Position tolerance | 0.002 m | 0.002 m | 0.002 m |
| Proportional gain | 20 | 20 | 20 |
| Step limit | 50 | 50 | 50 |

All three primitives use 3D gripper-position error. Release freezes X/Y before
opening, then corrects sideways drift while retreating to the target Z.
These are controller tolerances, not bounds on final block placement.
Gripper opening uses the summed finger positions with threshold 0.052 m and
margin 0.001 m. Open/closed tests are complementary.

Customize one instruction or share an immutable configuration:

```python
from synthesis.api.control import ControlConfig
from synthesis.api.instructions import MoveByName

control = ControlConfig(position_tolerance=0.002, gain=20.0)
move = MoveByName(
    "b0", "b0", "b",
    target_offset=[0, 0, 0.05],
    limit=50,
    control=control,
)
```

The action helper `get_move_action` computes the proportional XYZ command and
divides all three coordinates by `max(1, max(abs(XYZ)))`. This preserves direction
when bounding the command to [-1, 1]. The active CEE-US Fetch backend uses the
same operation for direct actions; it no longer clips Cartesian axes separately.
The independent gripper command is bounded separately. The helper has no
tolerance argument: the controller applies the configured tolerance to its
stopping test. Release's fixed XY target survives every retreat step; it is not
reanchored to the drifting gripper position. The active CEE-US Fetch model disables
both collision masks on the head geometries, retaining their appearance and mass.

Controller settings and the action-scaling version are preserved during
ID-to-name conversion and included in program descriptions/fingerprints. Archives
from the earlier component-clipping controller must be recollected for matching
supplied-program verification. The later head-contact/Release update adds no new
version mechanism. The current [15-seed collections](#cross-environment-audit)
use these settings. These are fixed settings, separate from the waypoint offsets
optimized by CEM. Scaling improves direction fidelity; it does
not establish physical refinement of the motion proof (review entry 36).

Each instruction retains `last_control_result` (convergence, steps, final phase,
and position error when applicable), also saved in its `instruction_end` event.
A step limit stops that instruction without claiming convergence. Collection
rejects a trace containing an unconverged primitive, even if its final task
predicate happens to hold. Convergence alone does not certify a grasp or goal.

The Stack example uses a 0.10 m transfer height above the current top, then
lowers to 0.05 m. For four-block, three-placement demonstrations, collect with
a three-iteration cap:

```bash
uv run python -m synthesis.entry.collect_demos \
  --program synthesis.examples.stack:build_program \
  --num-blocks 4 --num-trajectories 5 --max-loop-iterations 3 --save-video \
  --output-dir demos/stack/4-blocks-5-trajectories-precise
```

Move controls the gripper site, so controller convergence does not guarantee
exact centering of a held block. The task's ON relation allows 25 mm per axis;
its tolerance is distinct from the controller's stopping tolerance. The 50-step
preparation removes the initial robot transient before demonstration recording.
See [review entry 24](../../../PAPER-DISCREPANCIES.md#24-the-first-stack-placement-inherits-a-transient-robot-state-and-a-grasp-offset)
for the saved-state policy. Recollect after changing controller settings;
program fingerprints identify the executable used by each archive.

### Controller findings and current validation

Uniform XYZ scaling preserves the requested command direction; endpoint
convergence does not guarantee a straight physical path or exact carried-block
centering. The adopted settings and limitations are recorded in the
[controller findings](../experiment/CONTROLLER-PATHS.md). Historical controller
comparison, tuning and video-rendering runners have been retired.

Reproduce current executions with `collect_demos` and validate their snapshots,
actions and settled task success with `audit_demos`, following the
[cross-environment audit](#cross-environment-audit). Use `--save-video` during
collection for current demonstration videos.

## Run either pipeline mode

These commands show the two interfaces. For the checked supplied Stack result,
use the [complete configuration and motion premises](../cfg/VERIFICATION.md#provided-stack-verification).

```bash
uv run python -m synthesis.entry.synthesize_cfg \
  --mode full --task stack --num-blocks 3 --quotient \
  --demos demos/stack/3-blocks-5-trajectories/demonstrations.npz
uv run python -m synthesis.entry.synthesize_cfg \
  --mode verify --task stack --num-blocks 3 \
  --program synthesis.examples.stack:build_program \
  --demos demos/stack/3-blocks-5-trajectories/demonstrations.npz
uv run python -m synthesis.experiment.report --run runs/cfg/latest
```

Full mode synthesizes from the archive. Verify mode starts from the supplied
program, whose executable fingerprint must match the collected source. Both
execute the current candidate from the saved initial simulator states, infer
invariants, and perform the same symbolic and motion verification with feedback.
All symbolic inference calls `InvInference` → `inference.loop_inference`, the
intended partition-based algorithm. There is no learner-selection flag or fallback.
Verification-only mode may enter resynthesis later; this is recorded explicitly.
See [the integrated workflow](../cfg/VERIFICATION.md).

For inference alone from collected runtime loop events:

```bash
uv run python -m synthesis.entry.inference \
  --demos demos/stack/3-blocks-5-trajectories/demonstrations.npz \
  --task stack --loop-id 1
```

This standalone inference CLI uses the default 1 mm Higher tolerance and has no
`--higher-tolerance` option. For another threshold, call the Python inference API
inside `using_higher_tolerance(...)` as shown above.

## Truth-table minimization

Choose `--invariant-minimizer sympy` (default) or `--invariant-minimizer pyeda`
in `synthesize_cfg` (full or verify), `learn_invariant`, the standalone
`inference` CLI, or the tower verification CLIs. Instrumented runs record
`invariant_minimizer` in their resolved configuration. PyEDA 0.29.0 is included
in the project dependencies; `uv sync` installs it. Building its C extension
requires a C compiler and Python development headers if no wheel is available.

For example, append `--invariant-minimizer pyeda` to the
[complete Stack verification command](../cfg/VERIFICATION.md#provided-stack-verification),
or run inference alone on your current archive:

```bash
uv run python -m synthesis.entry.inference \
  --demos demos/stack/3-blocks-5-trajectories/demonstrations.npz \
  --task stack --loop-id 1 --invariant-minimizer pyeda
```

The intended partition learner, selected predicates, and truth-table completion
policies stay the same. Phi accepts every row outside U; phi-prime accepts only
S. Unobserved rows are **not don't-cares**. PyEDA's
[Espresso minimizer](https://pyeda.readthedocs.io/en/latest/2llm.html) uses a
heuristic search for a smaller equivalent Boolean expression; a globally minimum
formula is not guaranteed. It can reduce minimization time, but truth-table
enumeration, predicate selection and Z3 verification still contribute to runtime.
End-to-end speedups require measuring the workload.

For tower contexts with the top-existence axiom, quantified Z3 satisfiability
queries can dominate inference. Tautology and redundancy pruning first try a
small finite model of the same assertions. **Only SAT is used**: it witnesses
non-tautology/non-redundancy, so the clause is retained. Finite UNSAT or UNKNOWN
cannot remove a clause; that still requires an unrestricted proof. Redundancy
queries have a 10-second default budget and retain clauses on UNKNOWN. The
symbolic verifier can also use a finite SAT witness to resolve an inconclusive
premise-consistency query after proving the VC in the unbounded domain. This
never replaces the unbounded validity proof with finite checking.

On the same 60 Unstack states, the 2026-10-23 timing investigation measured
Boolean minimization at 0.021 s (SymPy) and 0.072 s (PyEDA), compared with about
241 s of tautology checks for each backend before the finite-witness shortcut.
The subsequent shared PyEDA run took 68 s overall: approximately 11 s setup and
candidate replay, 41 s learning/coverage and 16 s symbolic verification. Motion
verification was excluded. See the [tower verification results](../cfg/VERIFICATION.md#provided-unstack-and-reverse-verification).

SymPy retains its existing POS policy: above five selected predicates, it builds
unminimized CNF to avoid expensive minimization. PyEDA minimizes at those sizes
too. For CNF it minimizes the DNF of rejected rows and complements the result
using De Morgan's laws, preserving the clause form required by inference.
Expressions are converted to SymPy Boolean nodes for the existing Z3 translator,
without invoking SymPy minimization on the PyEDA path. Equivalent expressions
can have different clause layouts and redundancy-check costs; coverage and formal
verification remain required.

For a single Python call:

```python
invariant, clauses = InvInference(store, loop_id, vocabulary, context, minimizer="pyeda")
```

To select a backend for nested inference calls throughout CFG verification,
standalone CEGIS, or direct `inference.loop_inference` calls:

```python
from synthesis.inference_lib.minimization import using_invariant_minimizer

with using_invariant_minimizer("pyeda"):
    invariant, clauses = InvInference(store, loop_id, vocabulary, context)
```

The scoped setting is restored on exit, including exceptions. An explicit
`InvInference` option overrides the enclosing scope for that call. The standalone
experiment's Python API accepts `ExperimentConfig(invariant_minimizer="pyeda")`.

## Recording and invariant data

Archives retain full simulator snapshots, controls, mocap and solver arrays,
actions, observation/action indices, aliases, instruction boundaries, and loop
events. `cfg.recordings.save_traces/load_traces` handle this one current format.
`--reset-mode replay` remains the pipeline default; `reset` restores a segment
snapshot directly. At the beginning of a newly collected Stack demo, both modes
restore the settled state. For later segments, `reset` restores that segment
snapshot and `replay` restores state zero then replays the recorded prefix.
Neither mode repeats the discarded 50-step preparation.
Observations alone cannot restore a segment.

Runtime events identify each loop path, invocation, iteration, continuing head,
and normal guard-false exit, including zero-iteration loops. Frozen geometry
belongs to that loop invocation. Budget failures are never normal exits.
Candidates use their own execution traces; the expert recordings remain separate
for imitation and resynthesis. Persistent in-scope aliases become invariant
constants; selected guard witnesses are retained as metadata, not assumed to
remain defined at loop exit. Symbolic preservation covers every matching witness.

`DemoStore` remains the in-memory inference adapter. `from_archive` extracts
heads and exits, and `save_diagnostic` writes solver/debugging samples only.
The `on_loop_head` callback records successful guard bindings before
bodies; the richer `on_event` interface also reports normal exits and instruction
boundaries. Physical candidate execution does not require an invariant.

Tests use generated scenes and recordings. Golden literal fixtures remain
historical learner regressions, not accepted demonstrations or correctness targets.
