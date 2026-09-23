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
streamed rather than retained in memory. No extra simulator steps are added.

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
snapshot directly. Observations alone cannot restore a segment.

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
