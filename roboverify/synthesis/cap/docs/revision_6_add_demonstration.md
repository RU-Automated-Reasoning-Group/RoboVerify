# Revision 6: configurable demonstration input for CaP

- **Status:** planned; not implemented. RoboVerify's demonstration collector is available (Section 1.1); the schema for CaP input still needs to be agreed with Yuning.
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

- Add explicit demonstration-input on/off configuration; exact option names remain to be designed.
- Configure archive paths, selected trace IDs, and a deterministic serialization version.
- Validate task identity, block/goal bindings, trajectory completeness, and compatibility with the active primitive/predicate semantics.
- Use the same selected demonstration identities and representation across Baseline, Method 1, and Method 2.
- Keep the existing no-demonstration behavior available; version any new prompt package rather than rewriting frozen packages.

## 3. Representation and prompt design

- Agree on the schema with Yuning before implementing the loader/serializer.
- Evaluate an explicit state/action representation that retains relevant object relations, positions, and action transitions; document any sampling or compression.
- Keep full simulator snapshots for restoration, but do not expose simulator internals to the LLM merely because they are present in the archive.
- Do not provide the expert program source unless it is deliberately included as a shared input to the compared approaches.
- Specify whether policy, helper, and repair requests receive the same demonstration context or documented subsets. Avoid accidental differences between methods.
- Record demonstration hashes/IDs, ordering, serialization settings, exact rendered prompts, and token usage. Exceeding the input budget must be handled explicitly, not by silent truncation.

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
- This document does not authorize implementing these changes or launching new model experiments yet.
