# Revision 7.2: demonstration-style (RDP) keyframes in recovery feedback

- **Status:** implemented 2026-10-26 (Section 4); experiment in Section 5, results in [revision_7_2_report.md](revision_7_2_report.md).
- **Goal:** show the model the failed rollout the same way it sees a demonstration: gripper-command events with block positions and DSL predicate changes, plus Ramer-Douglas-Peucker (RDP) corners of the gripper path. The method is a configuration option; the legacy keyframes stay the default.

## 1. Current recovery keyframes (`legacy`, cap-recovery-v1/v2)

A recovery request carries the evidence object built by `recovery.select_relevant_evidence`. Its `keyframes` item is built from the recorded rollout (`execution/rollout_001/trajectory.npz`) as follows:

1. Collect the *important* timeline events (`primitive_start`, `primitive_end`, `controller_phase`, `query_failure`, `trial_failure`, `python_exception`, `evaluation_failure`).
2. The *latest* events are those from the last `primitive_start` onward (or the single last important event).
3. Candidate state indices: 0, the last recorded state, and every `state_index` of a latest event.
4. Only three are kept: **state 0, the second-smallest candidate index (the state at which the last primitive started, when there is one), and the last state**.
5. Each frame lists `state_index`, `gripper_xyz` and `blocks_xyz` (every block, full float precision).

A legacy frame has no gripper command, no held block and no predicate values. Everything between the entry state and the start of the last primitive is absent. For a failure after several transfers (for example, a Reverse tower rebuilt in the wrong order), the model sees only the start, the last action and the end. The other evidence items describe only the end of the execution too: `failure_context` (events of the last primitive), `supporting_context` (the last 40 timeline events), `primitive_diagnostics` (targets and errors of selected actions) and the terminal goal clauses.

Example (Reverse, demonstrations on, round 0 → round 1; 85 states): frames at t = 0, 34 (start of the failing Release) and 84.

## 2. Demonstration keyframes (`demo-keyframes-v1`, Revision 6)

`demonstrations.compact(demonstration, waypoint_tolerance)`:

- **Event frames:** the first and last state and every state at which the gripper command (the latest nonzero gripper action: open or closed) changes. Each lists the gripper, the held block (within 0.030 m of a closed gripper), every block position (3 decimals), and the DSL atoms of the task API: all true atoms at the first event, then `+`/`-` changes since the previous event.
- **Waypoint frames:** RDP corners of the gripper's 3D path between consecutive events (tolerance 0.010 m), listing only the gripper and the held block.

## 3. Change: `keyframe_method`

New fields of `RecoveryFeedbackConfig` (`recovery.feedback`):

| Field | Default | Meaning |
| --- | --- | --- |
| `keyframe_method` | `"legacy"` | `"legacy"`: Section 1, unchanged. `"rdp"`: the rollout compacted with `demonstrations.compact` and rendered with the demonstration line format |
| `keyframe_waypoint_tolerance` | 0.01 | RDP tolerance in metres (the demonstrations use 0.01) |
| `keyframe_bytes` | 8192 | byte budget of the RDP keyframe item |

- `rdp` requires the `fixed_items` evidence layout (all frozen configurations since v1 use it). In that layout the evidence cap (`evidence_bytes`) is a guard: no other item is sized by leftover bytes, so the keyframes cannot crowd out other evidence.
- The RDP item contains `method`, `version` (`recovery-keyframes-v1`), the tolerance, a short format description (self-contained, because the demonstration arm may be off), a summary line, the inputs/goals lines and the rendered frames. Blocks are named `o<i>` (physical block i, the same index as `block_<i>` and `object_handle_to_physical_id`), goals `g<j>`.
- **Budget:** the item gets `min(keyframe_bytes, room under the evidence cap)`. If the frames do not fit, it keeps the first frame and the most recent frames (nearest the failure), declares the omitted span (`omitted_frames`), and lists every true atom at the first event after the gap. `item_delivery.keyframes` is then `{"partial": kept, "available": total}`.
- Defaults leave `default_profile` and every earlier prompt unchanged.

Measured on the 132 recovery requests of the Revision 7.1 demonstrations-on arm, rebuilt offline with `rdp`:

| Task | Requests | Legacy evidence, max bytes | RDP evidence (8,192-byte keyframe budget), max bytes | Truncated |
| --- | ---: | ---: | ---: | ---: |
| Stack | 7 | 22,210 | 25,201 | 0 |
| Unstack | 7 | 20,666 | 22,520 | 0 |
| Reverse | 57 | 22,267 | 27,132 | 4 |
| ReStack | 57 | 22,641 | 29,228 | 4 |
| Grid | 4 | 21,687 | 25,682 | 1 |

Complete RDP renderings are 1–9 KB (median) and up to 26 KB for the longest Reverse rollout (1,776 states, 83 events).

**Budget for the experiments.** Later CaP experiments use Claude Haiku 5.5 and Opus 5.5, whose context windows hold far more than these sizes (decision 2026-10-26: raise budgets so that items fit). The Revision 7.2 RDP arm therefore uses `keyframe_bytes` 65,536 and `evidence_bytes` 98,304, so every rollout is delivered complete. Because the evidence cap is only a guard under `fixed_items`, the other evidence items are identical to the legacy arm's.

## 4. Implementation

- [`config.py`](../config.py): the three fields and their validation (`KEYFRAME_METHODS`).
- [`recovery_keyframes.py`](../recovery_keyframes.py): `rdp_keyframes(trace, tolerance, budget)`, reusing `demonstrations.compact`, `render_frame`, `relations` and the new `render_inputs` (split out of `render_demonstration`; the demonstration block is byte-identical).
- [`recovery.py`](../recovery.py): `select_relevant_evidence` replaces the legacy item when `keyframe_method` is `rdp`, records `item_delivery.keyframes` and `keyframe_method`. `load_seed` now checks the seed against `RecoveryConfig.success_field`: with `task_spec_success`, a round 0 that ended physically successful but failed the settled specification is a valid seed (recovery was triggered for it).
- [`experiments/revision72_keyframe_experiment.py`](../experiments/revision72_keyframe_experiment.py): `select`, `run`, `report` (Section 5).
- [`tests/test_recovery_keyframes.py`](../tests/test_recovery_keyframes.py): defaults and validation; complete rendering in the demonstration format; truncation keeps the first and latest frames and lists full atoms after the gap; an end-to-end recovery request with `rdp`; the experiment variant changes only the three keyframe/guard fields of `config_frozen_v6`.

## 5. Experiment

- **Legacy arm (`legacy` on, `rdp` off):** the Revision 7.1 demonstrations-on arm, unchanged ([`runs/cap-revision7-1/20261026-demo-ablation-v6/demos-on`](../../../runs/cap-revision7-1/20261026-demo-ablation-v6/demos-on/report.md)): `config_frozen_v6`, Claude Haiku 5.5 on OpenRouter (reasoning effort medium), demonstrations on, scene seeds 0–19 per task.
- **RDP arm (`rdp` on, `legacy` off):** every trial of the legacy arm whose round 0 failed under `task_spec_success` (49 of 120) is restarted from its archived round-0 attempt (`seed_attempt`), with the same configuration plus the `rdp` overrides. Round 0, its charged usage, the initial state and the demonstrations are shared; only recovery rounds 1–4 are new.
- Trials that succeeded at round 0 are identical in both arms and are not rerun.
- Seed manifest: `runs/cap-revision7-2/20261026-rdp-keyframes/seeds.json` (read-only, SHA-256 of every seed result). RDP batches: `runs/cap-revision7-2/20261026-rdp-keyframes/rdp/<model>/<task>/`; paired table: `.../rdp/paired_report.md`.
- Outcome: final task-spec success per seed in both arms, paired; recovery rounds, effort tokens, keyframe delivery.
- Caveat: each arm is one sample of rounds 1–4 per seed; a seed that differs between arms also reflects sampling variation.
