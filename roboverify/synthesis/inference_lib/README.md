# Collecting demonstrations and learning loop invariants

Use one full-state NPZ archive for collection, synthesis, and inference. Old
observation-only NumPy/pickle datasets and loop-head JSON inputs are removed;
recollect demonstrations instead of converting them.

## Collect Stack demonstrations

Run from `roboverify/`:

```bash
unset LD_PRELOAD
export LD_LIBRARY_PATH="$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia"
uv run python -m synthesis.entry.collect_demos \
  --program synthesis.examples.stack:build_program \
  --num-blocks 3 --num-trajectories 5 --save-video
```

The editable factory in `synthesis/examples/stack.py` uses explicit Pick, Move,
and Release primitives. Its transfer and placement waypoints use the base block
`b0` for X/Y alignment and the current tower top `b` for Z. A custom
`--program module:factory` or
`--program path/to/program.py:factory` must return a `Program` from
`factory(context, *, num_blocks)`. Named and numeric physical operands, Assign,
Get, Skip, and flat While loops are supported. PickPlace and nested loops are
rejected. Numeric operands get fixed aliases; their identities are not learned.

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
program actions from state zero. Recollect older demonstrations to adopt the
settled start; existing archives always replay their own stored states.

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
directory is rejected. There is no collection run-name flag.

Every accepted trajectory must finish normally, start with unstacked,
pairwise-scattered blocks, and end with all blocks in the tower rooted at b0.
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

## Stack reset workspace

New Stack resets sample block centers relative to the robot base: X is
0.54–0.70 m forward, Y is within ±0.20 m, and horizontal distance from the base
is at most **0.70 m**. This replaces the former square around the initial
gripper, which allowed distant approach targets. Every layout retains at least
0.10 m separation in X or Y between blocks and 0.10 m horizontal clearance from
the initial gripper. Blocks start at the resting table height; b0 is unchanged.
Sampling restarts a crowded layout with bounded retries and reports an error if
it cannot fit the requested count; it never expands the region as a fallback.

The bound applies to every newly sampled layout, independently of seed. It is
an XY workspace restriction, not a proof of reachability at every height or for
every tower size. Existing archives restore their saved initial states; recollect
to use the new region. Candidate runs restore their archived starts.

## Primitive controller settings

`Pick`/`Move`/`Release` and their ByName variants share the controllers in
`api/control.py`. A named instruction only resolves its operands before running
the same controller. Every primitive keeps a default **50-step total budget**;
Pick shares it across approach, opening, descent, and closing.

| Setting | Pick / PickByName | Move / MoveByName | Release / ReleaseByName |
| --- | --- | --- | --- |
| Position tolerance | 0.010 m | 0.002 m | 0.002 m |
| Proportional gain | 20 | 20 | 20 |
| Step limit | 50 | 50 | 50 |

Pick and Move use 3D gripper-position error; Release uses vertical error after
opening. These are controller tolerances, not bounds on final block placement.
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

The action helper `get_move_action` computes the proportional command and has
no tolerance argument. The controller applies the configured tolerance to its
stopping test. Controller settings are preserved during ID-to-name conversion
and included in program descriptions/fingerprints; they are fixed settings,
separate from the waypoint offsets optimized by CEM.

Each instruction retains `last_control_result` (convergence, steps, final phase,
and position error when applicable), also saved in its `instruction_end` event.
A step limit stops that instruction without claiming convergence. Collection
rejects a trace containing an unconverged primitive, even if its final task
predicate happens to hold. Convergence alone does not certify a grasp or goal.

The Stack example uses a 0.10 m transfer height above the current top, then
lowers to 0.05 m. Tighter tolerances with the original 0.20 m transfer waypoint
exposed controller stalls in the tested scenes. For the intended four-block
three-placement demonstrations, collect with a three-iteration cap:

```bash
uv run python -m synthesis.entry.collect_demos \
  --program synthesis.examples.stack:build_program \
  --num-blocks 4 --num-trajectories 5 --max-loop-iterations 3 --save-video \
  --output-dir demos/stack/4-blocks-5-trajectories-precise
```

With the former reset region, seeds 0–4 finished in three iterations with
20 FPS videos, but a broader check passed only 96/100: seeds 38, 46, 73, and 85
failed the third Pick's approach; the failures were retained during diagnosis.
With the bounded reset region above and identical program/controller settings,
the earlier collection before settling passed **500/500** seeds (0–499): exactly
three iterations, all 15 primitives converged, and at most 22 steps per primitive.
Ten separately rendered runs (seeds 0, 38, 46, 73, 85, 150, 250, 350, 450, 499)
had 20 FPS videos. All ten passed validation and reproduced their matching batch
actions exactly; observations agreed within 1e-8.
Those earlier accepted traces exhibit a systematic first-placement offset of
about 13 mm: the initial robot state had not fully settled, and Move controls
the gripper site without compensating for the held block's offset. The task's
25 mm per-axis ON tolerance accepts it. The adopted 50-step preparation reduces
yellow's final X offset to 0.4–1.9 mm on seeds 0, 38, 73, and 499; the normal
collector passes all four in three iterations and fresh-environment replay
reproduces every action. This is not a new 500-seed validation.
The generated collections, videos, and diagnostics were removed during cleanup;
use the collection command above to create new demonstrations. See
[review entry 24](../../../PAPER-DISCREPANCIES.md#24-the-first-stack-placement-inherits-a-transient-robot-state-and-a-grasp-offset)
for the diagnosis; controller convergence does not certify block centering.
This validates that collection, not all possible scenes or formal verification;
see [review entry 23](../../../PAPER-DISCREPANCIES.md#23-numeric-and-named-release-use-different-physical-stopping-tolerances).
Recollect after changing controller settings; older fingerprints describe the
previous executable. The earlier continuation findings in review entry 22 refer
to the former controllers and waypoints.

## Run either pipeline mode

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
Verification-only mode may enter resynthesis later; this is recorded explicitly.
See [the integrated workflow](../cfg/VERIFICATION.md).

For inference alone from collected runtime loop events:

```bash
uv run python -m synthesis.entry.inference \
  --demos demos/stack/3-blocks-5-trajectories/demonstrations.npz \
  --task stack --loop-id 1
```

## Recording and invariant data

Archives retain full simulator snapshots, controls, mocap and solver arrays,
actions, observation/action indices, aliases, instruction boundaries, and loop
events. `cfg.recordings.save_traces/load_traces` handle this one current format.
`--reset-mode replay` remains the pipeline default; `reset` restores a segment
snapshot directly. For a newly collected Stack demo, both modes start from the
settled state; replay never includes the discarded 50-step preparation.
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
The old `on_loop_head` callback still records successful guard bindings before
bodies; the richer `on_event` interface also reports normal exits and instruction
boundaries. Physical candidate execution does not require an invariant.

Tests use generated scenes and recordings. Golden literal fixtures remain
historical learner regressions, not accepted demonstrations or correctness targets.
