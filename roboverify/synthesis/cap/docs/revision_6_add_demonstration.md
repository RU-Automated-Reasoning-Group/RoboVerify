# Revision 6: configurable demonstration input for CaP

- **Status:** implemented (2026-10-26; Section 6), with the user's approval of this design, and merged into `feature/cap`. The served window is 262,144 tokens (Section 3.5). A real-model smoke trial per task passed for 4 of 6 tasks (Section 6.6). Prefix caching is routed, measured and verified at batch start (Sections 3.6 and 6.8). The loader has not yet been reviewed with Yuning.
- **Goal:** give CaP access to the same selected demonstration data used by RoboVerify's demonstration-driven synthesis, with an explicit on/off setting for ablation.
- **Dependencies:** [task support plan](revision_5_add_tasks.md), an agreed demonstration schema, and validated data. Stack is the first integration target.

## 1. Current behavior and information boundaries

| Information | Current CaP synthesis | Planned demonstration-enabled synthesis |
| --- | --- | --- |
| Task instruction, API descriptions, mode rules | Provided | Retained |
| Demonstration state/action trajectories | Not provided | Selected demonstrations encoded through a shared input representation |
| Demonstration initial states | Not provided | Included as part of demonstration data |
| A new evaluation scene's initial coordinates | Not provided | Separate design choice; not implicitly added by enabling demonstrations |
| Runtime environment queries | Available to the generated policy | Retained; results are not automatically sent back to the LLM |

- RoboVerify uses demonstration states, segment starts, and candidate rollouts during synthesis. Demonstration access does not imply prior knowledge of a held-out test scene.
- Matching demonstrations controls information access, not all differences in representations, compute budgets, or search feedback.
- The October 19 review found a Stack-only collector and no archive in this checkout. Both have since changed (Section 1.1).

### 1.1 Demonstration collector and sample archive (2026-10-25)

- **Collector:** [`synthesis/entry/collect_demos.py`](../../entry/collect_demos.py), run as `python -m synthesis.entry.collect_demos`. It executes a supplied DSL program (by default the task's groundtruth in [`synthesis/examples/`](../../examples/)) on distinct seeds and keeps only valid traces ([`validate_trace`](../../cfg/collection.py): normal completion, converged primitives, shared pre at the first state, shared post at the last). It covers all six tasks: `--task stack|unstack|reverse|restack|grid|pyramid`.
- **Archive format:** one full-state NPZ per collection, written by [`synthesis/cfg/recordings.py`](../../cfg/recordings.py) and read with `load_traces`. Each trace holds the observations, actions, simulator snapshots, runtime events, initial bindings (`b0`) and initialization metadata. The [trace workflow](../../inference_lib/README.md) describes it.
- **Sample collection:** Stack, 3 blocks, seeds 0–2, groundtruth program, with videos:

  ```bash
  python -m synthesis.entry.collect_demos --task stack --num-blocks 3 --seeds 0 1 2 --save-video --output-dir demos/cap-revision6-samples/stack-3-blocks
  ```

  | File | Content |
  | --- | --- |
  | [`demonstrations.npz`](../../../demos/cap-revision6-samples/stack-3-blocks/demonstrations.npz) | 3 valid traces: 78, 83 and 78 states (77, 82, 77 actions), each after 50 settling steps |
  | [`collection.json`](../../../demos/cap-revision6-samples/stack-3-blocks/collection.json) | Collection report: per-seed validation, program serialization, configuration |
  | [seed 0](../../../demos/cap-revision6-samples/stack-3-blocks/videos/seed_0000.mp4), [seed 1](../../../demos/cap-revision6-samples/stack-3-blocks/videos/seed_0001.mp4), [seed 2](../../../demos/cap-revision6-samples/stack-3-blocks/videos/seed_0002.mp4) | 20 FPS videos |

- `demos/` holds generated data and is not committed, so these links work only in a checkout that has the collection. The upstream 2026-10-23 audit archives (`demos/pipeline-audit-20261023/`, all six tasks) are not present in this checkout.

### 1.2 What a demonstration archive contains

Examples below come from the sample `demonstrations.npz` (Stack, 3 blocks, seed 0, trace `d0`: 78 states and 77 actions). The archive is one `np.savez_compressed` file. Its arrays are keyed per trace `d<i>`, and one JSON string `metadata` describes all traces. `load_traces` returns one `DemoTrace` per `d<i>`.

**Arrays per trace** (`n` = blocks, `T` = recorded states):

| Key | Shape (sample) | Content |
| --- | --- | --- |
| `d0_states` | (T, 10 + 12n + 3n + 3) = (78, 58) | Observation at each recorded state, `float32` |
| `d0_actions` | (T − 1, 4) = (77, 4) | Low-level action sent to `env.step`: `[dx, dy, dz, gripper]`, each in [−1, 1]; a negative gripper value closes |
| `d0_indices` | (T,) | For each state, the number of actions executed before it (state t follows action t − 1) |
| `d0_gt` | (T, 82) | Full MuJoCo state (`get_GT_state`) used for exact restoration |
| `d0_ctrl`, `d0_mocap_pos`, `d0_mocap_quat`, `d0_qacc_warmstart`, `d0_qfrc_applied`, `d0_xfrc_applied` | (T, …) | Simulator arrays that complete the snapshot of each state |
| `metadata` | scalar string | JSON: `format: "roboverify-demonstrations"`, `version: 2`, and one entry per trace (below) |

**Observation vector `states[t]`** (the same layout CaP's adapter reads):

| Columns | Meaning |
| --- | --- |
| 0–2 | Gripper position (world frame, m) |
| 3–4 | Finger joint positions |
| 5–7, 8–9 | Gripper linear velocity, finger velocities (per control step) |
| 10 + 12i … 10 + 12i + 2 | Block i position (centre) |
| 10 + 12i + 3 … + 11 | Block i rotation (Euler), linear and angular velocity |
| 10 + 12n … end | Desired goal slots (3n + 3); zeros for the tower tasks |

**Example rows** (rounded; z of a block resting on the table is 0.4249):

```text
t= 0  gripper=[1.3446, 0.7491, 0.5315] fingers=[0.0000, 0.0000]
      b0=[1.3019, 0.9008, 0.4249]  b1=[1.2433, 0.5790, 0.4249]  b2=[1.2508, 0.8001, 0.4249]
      bindings={b0: 0}
t=41  gripper=[1.3019, 0.9008, 0.6273] fingers=[0.0437, 0.0434]
      b0=[1.3019, 0.9008, 0.4248]  b1=[1.3037, 0.9009, 0.4745]  b2=[1.2508, 0.8001, 0.4249]
      bindings={b0: 0, b: 0, b_prime: 1}        # b1 has been placed on b0
t=77  gripper=[1.3018, 0.9008, 0.6749] fingers=[0.0437, 0.0434]
      b0=[1.3019, 0.9008, 0.4247]  b1=[1.3038, 0.9008, 0.4742]  b2=[1.3046, 0.9009, 0.5238]
      bindings={b0: 0, b: 1, b_prime: 2}        # final tower b0, b1, b2

actions[0]  = [-0.5958, -1.0000,  0.0000,  0.0000]   # move toward b1, gripper open
actions[14] = [ 0.0000,  0.0000,  0.0000, -0.2000]   # closing on b1 (Pick ends at state 15)
actions[15] = [-0.0048,  0.0082,  1.0000, -0.2000]   # lift while holding
actions[76] = [ 0.0110,  0.0034,  0.1210,  0.0000]   # last retreat step after Release
```

**Per-trace metadata** (JSON, abridged):

```json
{
  "seed": 0, "task": "stack", "num_blocks": 3,
  "bindings": [{"b0": 0}, {"b0": 0, "b": 0, "b_prime": 1}, "... one per state"],
  "arrays": ["ctrl", "mocap_pos", "mocap_quat", "qacc_warmstart", "qfrc_applied", "xfrc_applied"],
  "metadata": {
    "source": "synthesis.examples.stack:build_program",
    "status": "valid",
    "validation": {"total": 1, "pre_holds": 1, "post_at_end": 1, "post_reached": 1, "issues": []},
    "task_spec": {"task": "stack",
                  "precondition": "and(forall @0,@1.(implies(not(eq(@0, @1)), and(not(ON_star(@0, @1)), Scattered(@0, @1)))), forall @0,@1.(Higher(@0, @1)))",
                  "postcondition": "forall @0.(ON_star(@0, b0))"},
    "initial_bindings": {"b0": 0},
    "initialization": {"source": "reset", "settling_steps": 50},
    "higher_tolerance": 0.001,
    "program": [
      {"instruction": "Assign", "left": "b", "right": "b0"},
      {"instruction": "While",
       "guard": "and(and(not(eq(b_prime, b))), forall @0.(or(eq(@0, b_prime), not(ON_star(@0, b_prime)))))",
       "witnesses": ["b_prime"],
       "body": [
         {"instruction": "PickByName", "operands": [{"type": "BoxName", "val": "b_prime"}], "limit": 50},
         {"instruction": "MoveByName", "operands": ["b_prime", "b_prime", "b"], "offsets": [0.0, 0.0, 0.1]},
         {"instruction": "MoveByName", "operands": ["b0", "b0", "b"], "offsets": [0.0, 0.0, 0.1]},
         {"instruction": "MoveByName", "operands": ["b0", "b0", "b"], "offsets": [0.0, 0.0, 0.05]},
         {"instruction": "ReleaseByName", "operands": ["b_prime"], "release_offset": 0.15},
         {"instruction": "Assign", "left": "b", "right": "b_prime"}]}],
    "fingerprint": "04fdc3a1…", "video": {"path": ".../videos/seed_0000.mp4", "fps": 20, "frames": 78}
  },
  "events": ["... 32 events, below"]
}
```

`program` is abridged: each primitive also records its controller settings (`control`, `action_scaling`), and Move/Release operands are objects `{"type": "BoxName", "val": …}`.

**Events** (32 in `d0`: 14 `instruction_start`/`instruction_end` pairs, `loop_enter`, 2 `loop_head`, `loop_exit`). `index` is the state at which the event occurs, `path` the instruction position in the program, and `bindings` the variable values:

```json
{"kind": "loop_enter", "path": "1", "index": 0, "bindings": {"b0": 0, "b": 0}, "entry_index": 0, "invocation": 0}
{"kind": "loop_head", "path": "1", "index": 0, "bindings": {"b0": 0, "b": 0, "b_prime": 1}, "iteration": 0, "witnesses": ["b_prime"]}
{"kind": "instruction_end", "path": "1.0", "index": 15, "bindings": {"b0": 0, "b": 0, "b_prime": 1},
 "control": {"converged": true, "steps": 15, "phase": "close", "position_error": null}}
{"kind": "instruction_end", "path": "1.4", "index": 41, "bindings": {"b0": 0, "b": 0, "b_prime": 1},
 "control": {"converged": true, "steps": 7, "phase": "retreat", "position_error": 0.0019}}
{"kind": "loop_exit", "path": "1", "index": 77, "bindings": {"b0": 0, "b": 2, "b_prime": 2}, "iteration": 2}
```

**What this means for CaP input (Section 2):**

- **Usable without simulator access:** the block positions per state, the bindings, the primitive sequence with convergence, and the loop structure.
- **The archive also contains the generating program.** It is RoboVerify's groundtruth program, not a demonstration in the sense of observed behaviour. Deciding what CaP may see is the information boundary in Section 1: for example, states and primitive segments only, without `program`.

## 2. Planned input interface

- Add explicit demonstration-input on/off configuration through a `demonstrations` block in `RunConfig`. Settings decided so far:

  | Setting | Default | Meaning |
  | --- | --- | --- |
  | `demonstrations.enabled` | `false` | Off keeps the existing prompts unchanged |
  | `demonstrations.waypoint_tolerance` | `0.01` (m) | RDP tolerance for motion waypoints (Section 3.3). Must be finite and positive. Recorded in every trial's configuration and in the serialized block header. |
  | `demonstrations.max_tokens` | `131072` | Limit for the rendered demonstration block (Section 3.5); exceeding it is an explicit failure |

  The remaining options were added with the implementation: `archive`, `trace_ids`, `archive_sha256`, `token_counter` and `tokenizer_url` (Section 6.1).
- Configure archive paths, selected trace IDs, and a deterministic serialization version.
- Validate task identity, block/goal bindings, trajectory completeness, and compatibility with the active primitive/predicate semantics.
- Use the same selected demonstration identities and representation across Baseline, Method 1, and Method 2.
- Keep the existing no-demonstration behavior available; version any new prompt package rather than rewriting frozen packages.

## 3. Representation and prompt design

### 3.1 Information boundary (decided 2026-10-25)

RoboVerify learns from demonstrations, where a demonstration is a trajectory of states and actions. CaP receives the same kind of information and nothing derived from the expert program. What RoboVerify's synthesis actually reads from an archive:

| Archive field | RoboVerify synthesis | CaP input |
| --- | --- | --- |
| `states` (block and gripper positions per step) | Imitation distance, predicate evaluation, PostScore, segment splits (`entry/synthesize_cfg.py`, `cfg/refine.py`, `cfg/straightline.py`) | **Yes** |
| `actions` | Only to reset the simulator to a segment start (`cfg/reset.py:reset_segment`, replay mode) | **Yes**, as gripper open/close events (Section 3.3) |
| `initial_bindings` (`b0`), task name | Yes | **Yes** |
| Simulator snapshots (`gt`, `ctrl`, `mocap_*`, …) | Resetting segments | No (simulator internals) |
| `program`, `events` (instruction boundaries, loop enter/head/exit, variable bindings), `task_spec` text, `fingerprint` | **Not used**: RoboVerify infers segments and loops itself (Algorithms 3–5) | **No** |

`cfg/candidate_traces.py` reads events of RoboVerify's *own candidate* executions, not of the expert demonstrations. Giving CaP the expert's primitive boundaries or loop events would hand it the segmentation and loop structure that RoboVerify must discover, so they are excluded. They may appear only in a separately labelled ablation that reports the extra information.

### 3.2 Token budget

Measured with the Qwen3.8 27B tokenizer (vLLM `/tokenize`) on one seed-0 demonstration per task at the paper's block counts ([task_specification.md](task_specification.md)). The paper uses 15 demonstrations per environment (line 1833); its Fig. 11 x-axis counts simulator samples consumed by search, not demonstrations.

| Task (blocks) | Steps | Per-step text: gripper, fingers, block positions, action | × 15 | Full observation vector per step (73–103 dims) | × 15 | Gripper events only (positions) | × 15 | **Events + RDP waypoints (Section 3.3, positions only)** | **× 15** |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Stack (4) | 131 | 16.5k | 248k | 61k | 908k | 880 (8 frames) | 13k | 1.3k (8 + 14) | 20k |
| Unstack (4) | 158 | 19.9k | 299k | 73k | 1.1M | 1.4k (13) | 21k | 1.9k (13 + 13) | 28k |
| Reverse (4) | 229 | 29.0k | 434k | 106k | 1.6M | 1.9k (17) | 28k | 2.6k (17 + 20) | 38k |
| ReStack (5) | 390 | 56.4k | 846k | 216k | 3.2M | 3.3k (25) | 49k | 4.2k (25 + 27) | 63k |
| Grid (6) | 331 | 53.8k | 807k | 213k | 3.2M | 3.8k (25) | 57k | 4.8k (25 + 28) | 72k |
| Pyramid (6) | 319 | 51.9k | 778k | 205k | 3.1M | 3.8k (25) | 57k | 4.8k (25 + 28) | 72k |

In the last column, “8 + 14” means 8 gripper-event frames plus 14 RDP waypoints; predicate deltas are not yet included.

Fifteen per-step demonstrations do not fit for any task. The Section 3.3 compaction (20k–72k tokens for 15 demonstrations) fits all six, but only after the context upgrade in Section 3.5: today the binding limit is the server's 98,304-token window, not the 200k CaP allowance.

### 3.3 Method: deterministic compaction

The compaction keeps two kinds of frames. Both are computed from `states` and `actions` only.

1. **Event frames.** These are the frames where the gripper command switches between open and close, plus the first and last frame. Each event frame records:
   - every block's position;
   - the DSL predicates that changed since the previous event frame, as in Demo2Code's per-step predicate deltas [1]. The predicates come from the shared scene evaluator, i.e. the abstraction RoboVerify itself evaluates on demonstration states.
2. **Motion waypoints.** These are the corners of the gripper's 3D path, found by Ramer–Douglas–Peucker (RDP) simplification. The tolerance is the configurable `demonstrations.waypoint_tolerance`, default 1 cm (Section 2).
   - Each waypoint records only the gripper position and whether it holds a block. Between two events only the held block moves, so the other positions are not repeated.
   - Waypoints keep the lift, transit and descent structure that gripper events alone lose. For example, the Stack demonstration in Section 3.2 lifts 10 cm after the grasp, moves across at that height, descends onto the tower, then retreats 15 cm before moving to the next block. Move does not plan around obstacles, so this structure is what avoids knocking the tower over.
   - The approach follows KAT's downsampling of action trajectories to a few tens of waypoints [2].

```python
def compact_demonstration(trace, task, settings):
    """Uses trace.states and trace.actions only; never trace.program or trace.events."""
    events = gripper_events(trace.actions) | {first(trace), last(trace)}
    waypoints = motion_waypoints(gripper_path(trace.states),
                                 tolerance=settings.waypoint_tolerance) - events

    frames, previous = [], None
    for t in sorted(events | waypoints):
        if t in waypoints:
            frames.append(Waypoint(t, gripper(trace, t), holding(trace, t)))
        else:
            relations = dsl_relations(task, trace.states[t])     # shared scene evaluator
            frames.append(Event(t, gripper(trace, t), holding(trace, t),
                                block_positions(trace, t),
                                changes(previous, relations)))  # all relations at the first event
            previous = relations
    return serialize(header(task, trace, settings), frames)


def gripper_events(actions):
    """Steps where the gripper command switches between open and close."""

def motion_waypoints(path, tolerance):
    """Corners of the gripper's 3D path: Ramer-Douglas-Peucker simplification."""

def dsl_relations(task, state):
    """True DSL predicates (task API set) on the state, via the shared scene evaluator."""


def run_trial(config, demonstrations, scene):
    assert_disjoint(demonstrations, scene)            # evaluation scene never in demonstrations
    context = render([compact_demonstration(d, config.task, config.demonstrations)
                      for d in demonstrations])
    record(demonstrations, config.demonstrations, context)   # hashes, settings, prompt, tokens
    if tokens(context) > config.demonstrations.max_tokens:
        raise DemonstrationBudgetExceeded              # explicit failure, no silent truncation
    return method(config).synthesize(instruction, demonstrations=context)   # same for all methods
```

Rendered (Stack, 4 blocks, seed 0; positions only, predicate deltas omitted):

```text
event    t=0   open   gripper=(1.345,0.749,0.531)  o0=(1.302,0.901,0.425) o1=(1.243,0.579,0.425) ...
waypoint t=7   open   gripper=(1.244,0.580,0.531)   # above o1 at a safe height
event    t=15  closed gripper=(1.244,0.579,0.424)  o0=(...) o1=(1.243,0.579,0.425) ...
waypoint t=20  closed gripper=(1.243,0.580,0.524)   # lift 10 cm
waypoint t=31  closed gripper=(1.302,0.899,0.524)   # transit above o0
event    t=35  open   gripper=(1.302,0.901,0.475)  o0=(...) o1=(1.304,0.901,0.473) ...
waypoint t=41  open   gripper=(1.302,0.901,0.627)   # retreat 15 cm
waypoint t=46  open   gripper=(1.251,0.801,0.627)   # transit to the next block
...
```

Design notes:

- **Only state and action data.** The event rule reads the gripper command from `actions`. Waypoints, positions and predicate changes come from `states`. The expert's Move targets are not read from the program; they appear only as corners of the observed gripper path.
- **Deltas, not full relation sets.** Demo2Code reports that adding irrelevant predicates hurts [1]. The first event lists the full initial relation set once; later events list only changes.
- **Numbers.** Positions use 3 decimals (millimetres), the precision the controllers resolve.
- **Waypoint tolerance (`demonstrations.waypoint_tolerance`, default 0.01 m).**
  - In the Stack sample, 1 cm keeps every lift, transit and descent corner (rendered above).
  - A larger tolerance merges corners and saves tokens; a smaller one adds controller jitter.
  - The token figures in Section 3.2 are for 1 cm. Changing the value changes the demonstration input, so it is a configuration change: record it, use one value for all methods and tasks within a comparison, and re-measure tokens.
  - The validation below checks the default on the other tasks.
- **Same input for all methods.** Baseline, Method 1 and Method 2 receive identical demonstration context in every policy and repair request.
- **To validate before experiments:**
  - token counts including the predicate deltas, measured as in Section 3.2;
  - a check that every compacted demonstration still identifies each pick, place and safe-height transit (which block moved onto which support, and at what clearance).

### 3.4 Prompt integration and accounting

- The loader was implemented with the user's approval (Section 6.1); a review with Yuning is still open. The serializer above fixes only what the LLM sees.
- Keep full simulator snapshots for restoration, but do not expose simulator internals to the LLM merely because they are present in the archive.
- Version the demonstration block as a new prompt package; do not rewrite frozen packages.
- Record demonstration hashes/IDs, ordering, serialization version, exact rendered prompts and token usage. Exceeding the input budget is an explicit failure, never silent truncation.

### 3.5 Context window upgrade (`config_frozen_v3`)

**Current limits (checked 2026-10-25):**

| Limit | Value | Source |
| --- | ---: | --- |
| Checkpoint context (`max_position_embeddings`, no RoPE scaling) | 262,144 | `Qwen/Qwen3.8-27B-FP8` @ `017b9c7a`, `config.json` |
| **Served window** (`--max-model-len`, input + output) | **98,304** | Both 27B instances (ports 8126, 8127), `/v1/models` |
| KV cache per instance | 587,589 tokens (5.98 × 98,304) | vLLM startup log, `gpu-memory-utilization 0.85` |
| CaP allowance `recovery.context_tokens` (input + output) | 200,000 | `config_frozen_v1/v2` |
| Output cap `max_output_tokens` | 32,768 | `config_frozen_v1/v2` |
| CaP effort budgets (provider-reported total tokens) | 100k initial, 200k per round, 900k per trial | `config_frozen_v1/v2` |

The served window binds first. A request may carry at most 98,304 − 32,768 = 65,536 input tokens, so the 200k CaP allowance is never reached. With 15 compacted demonstrations, Grid and Pyramid need about 57k tokens before the API description, instruction and recovery feedback, which does not fit.

**Upgrade for Revision 6** (a new configuration version, `config_frozen_v3`; v1 and v2 are not edited):

| Setting | v2 | v3 | Reason |
| --- | ---: | ---: | --- |
| Served window `--max-model-len` | 98,304 | **262,144** | The checkpoint's native context. No RoPE/YaRN extension is needed. The KV cache (587,589 tokens) holds two full-length requests per instance. |
| `recovery.context_tokens` | 200,000 | **262,144** | Match the served window |
| `max_output_tokens` | 32,768 | 32,768 (unchanged) | Reasoning budget is not part of this change |
| Waypoint tolerance `demonstrations.waypoint_tolerance` | — | **0.01 m** | Section 3.3; the token figures in Section 3.2 assume it |
| Demonstration block limit `demonstrations.max_tokens` | — | **131,072** | Fits 15 compacted demonstrations for every task (≤ 72k measured for events plus waypoints, Section 3.2), with margin for the predicate deltas. Leaves ≥ 98k for the API description, instruction, recovery feedback and history. |
| Effort budgets (initial / round / trial) | 100k / 200k / 900k | Unchanged, **excluding the demonstration block** | See accounting below |
| Serving: prefix caching | vLLM default | Required, verified at startup | The demonstration block is an identical prefix across calls and trials |

**Token accounting.** Effort budgets count provider-reported total tokens per call. Charging a 20k–72k demonstration block to every call would exhaust the 100k initial budget within a few calls, and would make demonstration-on and demonstration-off runs differ in effective generation budget.

- **v3 charges the demonstration block separately** (`demonstration_tokens`, per call and per trial). The existing budgets continue to bound everything else. Both totals are reported.
- **The demonstration block is rendered first in every request**, before the method's prompt. Identical prefixes then hit vLLM's prefix cache, which bounds the prefill cost.

**Ablation hygiene.** Demonstration-on and demonstration-off runs both use v3 serving and limits, so the demonstrations are the only difference. Prompts that fit in 98,304 tokens are served by the same checkpoint, template and sampling at either window size. The window change is still recorded in each plan's `serving` block.

**When to apply.** Restarting the 27B instances with the larger window interrupts any running batch. Apply it only after the Revision 5 runs (tower tasks, then Grid/Pyramid) have finished, using the drain procedure from [Revision 4.3.1](revision_4_3_1_inference_backend_optimization.md). Then confirm `/v1/models` reports `max_model_len` 262,144 and check the KV-cache size in the startup log.

**Applied 2026-10-26.** Both instances were restarted with `--max-model-len 262144` (`~/vllm-serve/serve.sh`, now `MAX_MODEL_LEN`, default 262,144). `/v1/models` reports 262,144 on ports 8126 and 8127. The KV cache shrank to 537,356 and 570,413 tokens, i.e. 2.05× and 2.18× a full-length request. `config_frozen_v3.SERVING["window_upgrade_status"]` still reads "pending": frozen files are not edited, and this section is the record.

### 3.6 Prefix caching

A prefix cache stores the attention keys and values (the KV cache) that the server computed for a prompt. A later request whose token sequence begins with the same tokens reuses them, and only the remaining tokens need prefill. The cache lives in each vLLM instance's GPU memory and is evicted least-recently-used. It does not store or replay model outputs.

**Why it matters here.** The demonstration block is 26k–87k tokens and is sent with every request of a trial. The repair and recovery requests of one trial also repeat the method prompt and earlier content. Without reuse, every request pays the block's full prefill again.

**What it must not change.** Caching and routing change neither the request (prompt, sampling settings), nor the token accounting (provider `total_tokens` still counts cached tokens), nor the information the model sees. Requests do not share outputs, only the prefill of an identical prefix. Run-to-run numerical differences from batching exist with or without the cache.

**Design.**

| Part | Rule | Why |
| --- | --- | --- |
| Prompt layout | `[demonstration block][method prompt][recovery feedback]`. The block is rendered deterministically and ends in a blank line; its SHA-256 is recorded with every request (Section 6.3). | Only an identical leading token sequence can be reused, so the block, which never changes within a task, goes first. |
| Serving flags | `--enable-prefix-caching` and `--enable-prompt-tokens-details` | The second makes the server report `usage.prompt_tokens_details.cached_tokens`, so hits are measured per request instead of inferred from aggregate metrics. |
| Routing | Each trial sends an affinity key (`X-CaP-Affinity`). The balancer ranks the live instances by rendezvous hashing of the key and uses the top one unless its in-flight count exceeds the least-loaded instance's by more than a slack (default 2), in which case it uses the least-loaded instance. The response names the instance (`X-CaP-Backend`). Requests without a key keep least-in-flight routing. | Least-in-flight routing alone sends consecutive calls of one trial to different instances, which then miss. Per-trial affinity keeps a trial's calls on one instance. Trials of the same task spread over both instances, and each instance caches the block after its first miss. The slack bounds the load imbalance that hashing a few keys can cause. |
| Startup check | Before dispatching a batch with demonstrations on and a vLLM model, the batch runner probes every instance behind the endpoint: the same 4k-token prompt, starting with a random nonce, is sent twice with a one-token output. The second response must report `cached_tokens` > 0. Otherwise the batch refuses to start. | "Required, verified at startup" in Section 3.5. A server restarted without the flags would otherwise silently lose caching (or its measurement) while the run still succeeds. |
| Records | Per request: `usage` (with `cached_tokens`) and `served_by`. Per trial: `cached_prompt_tokens` and calls per instance. Per batch: the probe results in `execution_metadata.json`. | Cache effectiveness is reported next to `demonstration_tokens`. |

**Measured in the smoke run (2026-10-26, before routing affinity).** After the first Stack recovery round the two instances had served 76,048 and 77,616 cached prompt tokens (vLLM `prefix_cache_hits_total`), so the block is reused on this hybrid-attention checkpoint (vLLM uses the `align` Mamba cache mode with prefix caching).

## 4. Experimental design

| Comparison | Controlled inputs | Measurement |
| --- | --- | --- |
| Demonstrations on versus off | Same method, model, condition mode, resource limits, and evaluation scenes | Held-out success, compliance, failures, calls, tokens, and latency |
| CaP versus RoboVerify | Same selected demonstrations and task/evaluation definitions | Policy quality and generalization, with compute/search differences reported |
| DSL-only versus coordinate conditions | Same demonstration selection and scene split | Effect of the expanded condition language and actual use of numeric predicates |

- Synthesize from a fixed demonstration set, freeze each program, and evaluate it on held-out scenes to measure generalization.
- Treat synthesis conditioned on each test scene as a separate experimental setting.
- Keep demonstration input and execution-failure recovery independent; [Revision 4](revision_4_failure_recover.md) remains a separate discussion.

## 5. Acceptance criteria

- Demonstration-off requests preserve the existing prompt behavior.
- Demonstration-on requests contain the configured data consistently across methods, with reproducible hashes and usage accounting.
- Reject incompatible/malformed datasets and prevent evaluation-scene data from entering demonstration inputs unintentionally.
- Validate at least one Stack integration end-to-end before extending to other tasks.
- Preserve original benchmark results and report any new ablation as a separate batch.
- This document did not originally authorize implementation; the user approved implementing it on 2026-10-26 (Section 6). New model experiments still need a separate decision.

## 6. Implementation (2026-10-26)

Implemented on branch `feature/cap-demonstrations`. The LLM summarization in the Backlog is not implemented.

### 6.1 Configuration and loading

`RunConfig.demonstrations` ([config.py](../config.py), `DemonstrationConfig`):

| Setting | Default | Meaning |
| --- | --- | --- |
| `enabled` | `false` | Off leaves every prompt byte-identical (Section 6.5) |
| `archive` | `null` | `demonstrations.npz` written by `cfg/recordings.py`; required when enabled |
| `trace_ids` | `null` | Selected trajectories and their rendering order (`"d0"`, `"d1"`, …); `null` selects all in archive order |
| `archive_sha256` | `null` | Optional pin of the archive bytes |
| `waypoint_tolerance` | `0.01` | RDP tolerance in metres; finite and > 0 |
| `max_tokens` | `131072` | Limit for the rendered block |
| `token_counter` | `server_tokenize` | `server_tokenize`: the vLLM `/tokenize` endpoint; `utf8_bytes`: a conservative byte bound for offline diagnostics |
| `tokenizer_url` | `null` | Default: the model base URL's root plus `/tokenize` |

- Enabling demonstrations requires a prompt package with demonstration templates (`typed_v5`); other packages are rejected.
- `run_cap` exposes the same settings (`--demonstrations ARCHIVE`, `--demonstration-traces`, `--waypoint-tolerance`, `--demonstration-max-tokens`, `--demonstration-token-counter`, `--tokenizer-url`, `--demonstration-archive-sha256`). Batch configurations set `run_config.demonstrations`.
- The loader ([demonstrations.py](../demonstrations.py), `load_demonstrations`) opens the NPZ directly and reads only each trace's `states`, `actions` and `indices`, its task, block count, seed and initial bindings, the goal layout and the initial goal bindings, plus `metadata.status`, `metadata.higher_tolerance` and `metadata.environment` for validation. It does not call `load_traces`, which would also materialize snapshots and events.
- Validation rejects a missing or unpinned archive, an unknown format, unknown trace ids, another task or block count, other layout options (ReStack tower height, Grid rows/columns, Pyramid base), another Higher tolerance, a trace not marked `valid` by collection, malformed or non-finite arrays, actions outside [−1, 1], incomplete recordings (`indices` must be `0..T−1`, i.e. a state after every action), unexpected input bindings, and repeated seeds.
- `seed` is read only for the evaluation-scene check: a trial is rejected (`status: demonstration_rejected`) if its fresh-scene seed, or the seed of its snapshot trace, equals a selected demonstration's seed, if its snapshot is a selected trace of the same archive, or if every block centre of its start lies within 1 mm of a demonstration's first state.

### 6.2 Compaction (`demo-keyframes-v1`)

- **Gripper command:** the latest nonzero gripper action (positive opens, negative closes; zero keeps the command). Before any command the gripper counts as open. State t follows action t − 1, so the command at state t reflects `actions[:t]`.
- **Event frames:** the first and last state and every state whose command differs from the previous state's. Each lists every block position and the DSL atoms of the task API (`ON`, `ON_star`, `Higher`, `Scattered`, plus `ON_star_zero` for Reverse) over distinct blocks, evaluated with `predicates.scene.evaluate` under the configured Higher tolerance. Goal tasks add `AtGoal`/`HasBlock` (`goal_layout.occupancy`, 20 mm). The first event lists every true atom (`holds:`); later events list `+`/`−` changes since the previous event (`changes:`).
- Atoms involving `tbl` are omitted, because in the shared semantics none holds for a block. `Scattered` is listed once per unordered pair. `ON_star_zero` uses the demonstration's first state as entry geometry.
- **Motion waypoints:** RDP on the gripper path, applied separately to the path between consecutive event frames. Distances are to the chord segment, not the infinite line, so a lift followed by a descent along the same vertical keeps its top. Each waypoint lists the gripper position and the held block only.
- **Held block:** with the command closed, the block whose centre is nearest the gripper, if it is within 3 cm.
- **Names and numbers:** blocks are `o<i>` and goals `g<j>` (physical indices). The `inputs:` line gives `b0` (or `h0`), and goal tasks list the layout once per demonstration with `right`/`down` links. Positions use 3 decimals.
- **Header:** the header template in `typed_v5` states the serialization version, waypoint tolerance, Higher tolerance and the format. The block ends with `END OF DEMONSTRATIONS` and a blank line.

### 6.3 Prompt integration and records

- `prompts/typed_v5.json` is `typed_v4` plus a `demonstrations` section (header, footer, serialization version); typed_v3/v4 are unchanged.
- `LoggedModel.complete` prefixes the block to every request: policy, helper, repair and recovery requests of all three methods. The method prompt follows unchanged, so the block is an identical prefix across calls, methods and trials.
- `run()` prepares the block once per trial, and every recovery round reuses it.
- Each attempt writes:
  - `artifacts/trial_001/demonstrations/context.txt`: the exact block;
  - `demonstrations/manifest.json`: archive path and SHA-256, trace ids in order, per-demonstration seed, source hash (states, actions, bindings, layout) and text hash, frame steps, settings, serialization version, prompt-package hash, block hash, bytes and token count with its measurement method.
- `config.json` contains the manifest as `demonstration_input`. Each request's `prompt_metadata.json` and `generation_end` event record `demonstration_context_sha256`, `demonstration_tokens` and `method_prompt_sha256`.
- A block over `max_tokens` fails the trial before any request (`status: budget_exhausted`, `budget_exhausted_by: demonstration_tokens`). Recovery then stops with `demonstration_input_rejected`; rejected scenes stop recovery the same way.

### 6.4 Token accounting

- The block is tokenized once per trial. With `server_tokenize`, the runner sends `{"model", "prompt": block, "add_special_tokens": false}` to `/tokenize`.
- Every request is charged `demonstration_tokens` = that count, separately. The effort budgets (`max_tokens`, round and trial totals) receive the provider's `total_tokens` minus the block count (`budget.effort_tokens`, clamped at 0).
- The recovery reservation bounds only the method prompt plus output. The context check (`recovery.context_tokens`) also counts the block.
- Results report `known_tokens` (effort, the existing meaning when demonstrations are off), `demonstration_tokens` (per trial, summed over requests), `demonstration_block_tokens` and `provider_total_tokens`. Batch summaries subtract each request's recorded block tokens from `known_tokens` and show `(+N demo)` in the token column.
- Joining the block and the method prompt could change the count at the boundary. The block ends in a blank line, and the Stack Method 2 request measured exactly 25,838 + 2,264 = 28,102 tokens (block + method prompt = whole).
- Tests use a fixture tokenizer with exactly additive counts and a fixture model that reports usage. They check that effort tokens are identical with demonstrations on and off, and that an effort budget the off trial exactly uses up still suffices with demonstrations on.

### 6.5 Configuration `config_frozen_v3`

[configs/config_frozen_v3.py](../configs/config_frozen_v3.py) is v2 plus:

- `prompt_version: typed_v5`;
- `recovery.context_tokens: 262144`;
- `demonstrations` enabled with the defaults above and the pinned archive per task (`ARCHIVES`; `configuration(task, demonstrations=False)` gives the ablation's off arm).

`SERVING` records `max_model_len: 262144` and `enable_prefix_caching: true`, marked pending. v1 and v2 are unchanged. With demonstrations off, v3 renders exactly v2's prompts.

### 6.6 Data and validation

**Archives.** `synthesis.entry.collect_demos` with the provided groundtruth programs, seeds 1000–1014, at the paper's counts (Stack/Unstack/Reverse 4 blocks, ReStack 5 with a 3-block tower, Grid 3×2, Pyramid 3+2+1). All 15 seeds were valid for every task. The archives are in `demos/cap-revision6/<task>/demonstrations.npz`, with collection reports and logs. They are not committed; v3 pins their SHA-256. Since 2026-10-30, [`collect_demos.sh`](../collect_demos.sh) re-collects the archives pinned by the final configurations. A re-collection reproduces every array and metadata field except the per-trace wall time `elapsed_seconds`, so the loader also accepts the pinned archive's content digest (`demonstrations.CONTENT_PINS`) and records `pin_match`. The rendered demonstration block was checked to be byte-identical for all six tasks on the authors' machine. A collection that matches neither digest (another simulator build) is installed anyway; `revision6_experiment --accept-unpinned-demonstrations` (passed by `run_final.sh`) runs it and records the pinned and used hashes as an override.

**Tests.**

- [tests/test_demonstrations.py](../tests/test_demonstrations.py) (23 tests, synthetic archives and fixture environments) covers:
  - configuration validation;
  - prompt parity: all requests of generate-only trials for 6 tasks × 3 methods × 2 condition modes, hashed before the change ([golden file](../tests/testdata/demonstrations_off_prompt_hashes.json)) and compared with typed_v4 and typed_v5 with demonstrations off;
  - command events, RDP and tolerance;
  - pick/place structure and deltas;
  - determinism;
  - absence of program, events, task_spec, fingerprint, source and snapshot values from the block;
  - loader rejections;
  - the server tokenize request;
  - identical prefixes in every request of all three methods;
  - token accounting;
  - the explicit over-budget failure;
  - evaluation-scene rejection (seed and geometry);
  - recovery stop;
  - reservation and context accounting.
- [configs/test_config_frozen_v3.py](../configs/test_config_frozen_v3.py) covers v3 against v2.

**Measured tokens.** Tokens of the rendered block for 15 demonstrations per task: the whole block including header and predicate deltas. Counts use the checkpoint's own tokenizer (`Qwen/Qwen3.8-27B-FP8` @ `017b9c7a`, `tokenizer.json`, no special tokens). It gave token ids identical to the server's `/tokenize` on a reference string; the server was stopped during this measurement.

| Task (blocks) | States per demo | Events + waypoints per demo | Block tokens (15 demos) | Per demo | Positions only (no `holds`/`changes`) | Fits the current 98,304 window? |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Stack (4) | 112–136 | 8 + 11 | 25,838 | 1,696–1,705 | 20,573 | Yes |
| Unstack (4) | 158–161 | 14 + 8 | 35,636 | 2,334–2,369 | 29,965 | Yes |
| Reverse (4) | 214–232 | 18 + 11 | 46,814 | 3,093–3,106 | 38,611 | Yes, but recovery feedback may exceed it |
| ReStack (5) | 379–400 | 26 + 17 | 75,974 | 5,039–5,047 | 64,109 | No |
| Grid (6) | 325–345 | 26 + 17 | 83,581 | 5,526–5,609 | 74,691 | No |
| Pyramid (6) | 312–330 | 26 + 17 | 86,752 | 5,740–5,812 | 74,693 | No |

- All six blocks fit `max_tokens` = 131,072. The predicate deltas add 5k–12k tokens.
- The Stack Method 2 policy request is 28,102 input tokens (block 25,838 + method prompt 2,264). At the current window a request may carry 98,304 − 32,768 = 65,536 input tokens. ReStack, Grid and Pyramid therefore need the Section 3.5 upgrade, as predicted.
- Grid and Pyramid demonstrations close the gripper while empty in transit, so they have 26 events per demonstration (12 of them empty-handed close/open switches).

**Waypoint check (default 0.01 m, all 90 demonstrations).** For every transfer the check finds, in the full state sequence:

- the true approach corner above the block;
- the lift corner (where lateral motion starts after the grasp);
- the transit end above the support;
- the retreat corner after the release;
- the maximum transit height.

It then compares them with the kept frames.

- The largest distance from a true corner to the nearest kept frame was 1.9 mm, except one Pyramid retreat corner at 7.2 mm. The compacted maximum transit height matched within 0.2 mm.
- Every place event identifies the moved block and its support exactly as the full state does: `+ON(block,support)` for block supports, and the resting table height with no `ON` atom for table placements.
- Pyramid's upper-layer blocks straddle two lower blocks and are identified by `+AtGoal`/`+HasBlock` and their positions.
- Transfers per demonstration:
  - Stack: 3 (onto o0, o1, o2, at about 5 cm transit clearance above the higher of grasp and release);
  - Unstack: 3 (to the table at about 15 cm);
  - Reverse: 4 (table, then the new tower at 10 to 15 cm);
  - ReStack: 6;
  - Grid and Pyramid: 6 each (at about 15 cm).
- Artifacts are in `runs/cap-revision6/20261026-compaction-check/` (rendered blocks, `compaction_check.json`, `tokens_local_tokenizer.json`).

**End-to-end, scripted responses** (`runs/cap-revision6/20261026-scripted-stack/`): Stack, seed 0, real MuJoCo, config v3 (recovery on, videos off). The scripted Stack policy was used for all three methods, after one invalid response for Methods 1 and 2.

- All six trials (demonstrations on and off) completed with `task_spec_success`.
- With demonstrations on, every request began with the same block (one context hash across methods), and `demonstration_tokens` was 25,838 per request.
- The method prompts after the block were byte-identical to the demonstrations-off requests.

**Real-model smoke trial (2026-10-26).** One trial per task, scene seed 0, `config_frozen_v3` with demonstrations on, served window 262,144, before routing affinity (Section 6.8). Launcher: `runs/cap-revision6/launch_frozen_v3.py ROOT TASK TRIALS SEED_START MAX_WORKERS [off]`; results in `runs/cap-revision6/20261026-smoke-v3-demos/`.

| Task | `task_spec_success` | Rounds | Calls | Effort tokens | Block tokens per call | Provider total |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Stack | Yes | 1 | 2 | 40,997 | 25,838 | 92,673 |
| Unstack | Yes | 0 | 1 | 15,552 | 35,636 | 51,188 |
| Reverse | No (4 rounds used: Move/Pick non-convergence, then goal failures) | 4 | 6 | 155,171 | 46,814 | 436,055 |
| ReStack | No (every round: Move or Pick non-convergence) | 4 | 7 | 163,022 | 75,974 | 694,840 |
| Grid | Yes | 0 | 1 | 12,520 | 83,581 | 96,101 |
| Pyramid | Yes | 0 | 1 | 16,403 | 86,752 | 103,155 |

- The server's `/tokenize` gave exactly the block counts measured offline above, for all six tasks.
- Grid and Pyramid requests (about 96k and 103k provider tokens in one call) exceed the old 98,304 window, so they confirm the upgrade.
- One trial per task is a smoke test of the pipeline, not a success-rate estimate. The Reverse and ReStack failures are the Move/Pick non-convergence that dominates the Revision 5 failures as well.

### 6.7 Deviations and open points

- **Event frame index:** an event is the first state after the first action with the new command (t = k + 1). The Section 3.3 illustration shows the closed event at the end of the closing steps. In the 3-block sample of Section 1.2, closing actions 13 and 14 put the closed event at t = 14, not 15.
- **RDP is applied between consecutive event frames** rather than to the whole path at once. Event frames are always kept, so this only makes the corners around them more stable. Distances use the chord segment.
- **Empty-handed gripper switches are events:** the Grid/Pyramid programs close the gripper during empty transits, which adds event frames with full block positions.
- **Not committed:** the archives (`demos/` is generated data). Regenerate the parity golden file only for an intentional prompt change ([tests/prompt_parity.py](../tests/prompt_parity.py)).
- **Open:** a loader review with Yuning. The smoke trial, live `/tokenize` counts, the window upgrade and the prefix-caching check are done (Sections 3.5, 3.6, 6.6 and 6.8).

### 6.8 Prefix caching (`cap-prefix-cache-v1`)

Implements Section 3.6 in [serving/prefix_cache.py](../serving/prefix_cache.py) and [serving/balancer.py](../serving/balancer.py).

- **Affinity key.** `HTTPModel(..., affinity_key=...)` sends `X-CaP-Affinity` for `provider: vllm` only; other providers never receive it. Batch workers use `affinity_key(batch_dir, job_id)`, which is stable, so a resumed trial returns to its instance. `runner.run` creates one key per trial otherwise, shared by every recovery round. The key is a 32-hex-digit hash and carries no path or name.
- **Balancer.** The balancer formerly kept in `~/vllm-serve/balancer.py` is now `python -m synthesis.cap.serving.balancer PORT BACKENDS_JSON [--slack N]`. It runs with the serving venv (`aiohttp`) and `roboverify/` on `PYTHONPATH`. `choose_backend` ranks instances by rendezvous hashing, so removing one instance moves only the keys that ranked it first. Requests without a key keep least-in-flight routing. `/balancer/status` lists the instances, the in-flight counts and the slack.
- **Serving.** `~/vllm-serve/serve.sh` adds `--enable-prompt-tokens-details`; prefix caching was already on.
- **Startup check.** `execute_batch` calls `verify_prefix_caching` when a pending job has demonstrations on, a vLLM model and no scripted responses or fixture. The probe sends a 4,373-token prompt twice to every instance listed by `/balancer/status`, without the request gate. The batch refuses to start unless each second response reports `cached_tokens` > 0. Each session's probe is stored in `execution_metadata.json` (`sessions[].prefix_cache`).
- **Records.** `generation_end` events add `served_by`; `usage` already contains `prompt_tokens_details.cached_tokens`. With demonstrations on, results add `prefix_cache`: `cached_prompt_tokens`, `cache_unreported_calls` and `served_by` (requests per instance), per round and per trial. Budgets do not read them.
- **Tests.** [tests/test_prefix_cache.py](../tests/test_prefix_cache.py) (12 tests) covers:
  - routing: least-in-flight without a key, stickiness within the slack, fallback beyond it, spread over instances, and stability when another instance is removed;
  - the header for vLLM only, and the reported instance and cached tokens;
  - ledger recording through round budgets;
  - the probe through a balancer and against a plain endpoint, and refusal when reuse or its report is missing;
  - which jobs require the probe.
- **Deployment check (2026-10-26).** After restarting both instances with the new flag, the probe verified both: the second request had 3,920 of 4,373 prompt tokens cached, and its time fell from 1.52 s to 0.19 s.
- **Trial check (2026-10-26, `runs/cap-revision6/20261026-prefix-cache-check/`, scene seed 1).**
  - Stack succeeded in one call; the batch probe was recorded in `execution_metadata.json`.
  - Reverse used all 4 recovery rounds and ended `primitive_failed`. All 6 of its requests were served by one instance (8127).

  | Round | Purpose | Prompt tokens | Cached | Seconds |
  | ---: | --- | ---: | ---: | ---: |
  | 0 | policy | 49,287 | 0 | 572 |
  | 1 | policy | 57,297 | 48,608 | 388 |
  | 2 | policy | 58,507 | 48,608 | 421 |
  | 3 | policy | 58,250 | 48,608 | 404 |
  | 3 | repair | 49,684 | 0 | 435 |
  | 4 | policy | 59,242 | 49,392 | 341 |

  - The trial reused 195,216 of 332,267 prompt tokens.
  - **The round-3 repair request missed even though it shares the 46,814-token block.** Our likely explanation (not yet confirmed in vLLM's source): this checkpoint mixes attention with Mamba-style layers, and in vLLM's `align` cache mode a recurrent state is kept only at the block-aligned end of each cached prompt. A request can then resume only from such a point. Each policy request extends round 0's cached prompt (hit at 48,608 tokens). The repair prompt diverges after the block, before that point, so no saved state applies. Round 4 extends the repair prompt and hits at 49,392.
  - Consequence: for this checkpoint, reuse is guaranteed only for requests that extend an earlier request's whole prompt, not for every request that shares the block. Generation time dominates these requests (minutes) in any case.

### 6.9 Settled criterion and `config_frozen_v4` (2026-10-26)

[`config_frozen_v4`](../configs/config_frozen_v4.py) is v3 plus `post_settling_steps: 50`, which selects `task-spec-v2` ([settled postcondition plan](../../cfg/docs/settled_postcondition.md)), and two re-collected archives. Every fresh reset now settles for 50 steps, including Unstack and Reverse, which v3's archives recorded from unsettled starts.

**Re-collected archives.** Same settings as v3 (4 blocks, seeds 1000–1014, provided groundtruth programs), with `--post-settling-steps 50`; all 15 seeds valid for both tasks. Each trace stores its settled check in `metadata.validation.settled`. `audit_demos --post-settling-steps 50` replays every trace exactly and reproduces the stored settled result (15/15 each, `settled_audit.json`).

| Task | Archive | SHA-256 | States (v3 → v4) | Demonstration block, UTF-8 bytes (v3 → v4) |
| --- | --- | --- | ---: | ---: |
| Unstack | `demos/cap-revision6-v4/unstack/demonstrations.npz` | `81a19a15…37a50b` | 2,383 → 2,359 | 56,767 → 56,346 |
| Reverse | `demos/cap-revision6-v4/reverse/demonstrations.npz` | `78a389ed…fad5` | 3,333 → 3,297 | 74,944 → 75,358 |

Server token counts of the new blocks were not measured (no model-server requests were made); the byte sizes change by less than 1%.

**Revalidation of the other archives.** `synthesis.entry.revalidate_settled --post-settling-steps 50` restored each trace's last snapshot, settled it and scored it with the shared checker. Stack, ReStack, Grid and Pyramid: 15/15 each; reports in `settled_validation.json` next to each archive, whose NPZ files and v3 SHA-256 pins are unchanged. The superseded v3 Unstack and Reverse archives also pass (15/15 each, not written to their directories). `demos/pipeline-audit-20261023/` is not present in this checkout and was not revalidated.

## Backlog (not implemented)

- **LLM recursive summarization of demonstrations** (Demo2Code-style [1]): summarize each compacted demonstration with the LLM, summarize the set into one task specification, and give the specification to the synthesis request instead of the raw compaction.
  - Intended as an ablation: does LLM pattern extraction help beyond the deterministic compaction?
  - Its calls and tokens would be charged to the trial budget.
  - Risks: numbers may be misstated or invented, and pattern extraction overlaps with synthesis itself, which blurs attribution.
  - Deferred by user decision (2026-10-25); not part of `config_frozen_v3`.

## References

1. Huaxiaoyue Wang, Gonzalo Gonzalez-Pumariega, Yash Sharma, Sanjiban Choudhury. *Demo2Code: From Summarizing Demonstrations to Synthesizing Code via Extended Chain-of-Thought.* NeurIPS 2023. [arXiv:2305.16744](https://arxiv.org/abs/2305.16744), [project page](https://portal.cs.cornell.edu/demo2code)
2. Norman Di Palo, Edward Johns. *Keypoint Action Tokens Enable In-Context Imitation Learning in Robotics.* RSS 2024. [arXiv:2403.19578](https://arxiv.org/abs/2403.19578)
3. Noah Patton, Kia Rahmani, Meghana Missula, Joydeep Biswas, Işıl Dillig. *Programming-by-Demonstration for Long-Horizon Robot Tasks.* POPL 2024. [arXiv:2305.03129](https://arxiv.org/abs/2305.03129)
4. *Abstraction-Refinement-guided Program Synthesis for Robot Learning from Demonstrations* (RoboScribe). OOPSLA 2025. [SPLASH 2025 page](https://2025.splashcon.org/details/OOPSLA/59/Abstraction-Refinement-guided-Program-Synthesis-for-Robot-Learning-from-Demonstration)
5. *ExoPredicator: Learning Abstract Models of Dynamic Worlds for Robot Planning.* [arXiv:2509.26255](https://arxiv.org/abs/2509.26255)
6. Jacky Liang, Wenlong Huang, Fei Xia, Peng Xu, Karol Hausman, Brian Ichter, Pete Florence, Andy Zeng. *Code as Policies: Language Model Programs for Embodied Control.* ICRA 2023. [arXiv:2209.07753](https://arxiv.org/abs/2209.07753)

References 3–5 synthesize programs or abstractions from demonstrations without an LLM context bottleneck; they are related work for RoboVerify rather than inputs to the CaP design.
