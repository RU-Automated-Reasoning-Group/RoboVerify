# CaP experiment execution

Follow the repository-wide [AGENTS.md](../../../AGENTS.md) for environment setup,
project decisions, implementation conventions, and validation. This file adds
experiment guidance specific to `synthesis/cap/`.

- Before starting any task under `synthesis/cap/`, read the [README](README.md)
  overview and the sections relevant to the task. Read historical experiment
  reports only when needed.
- Use `task_spec_success` under `task-spec-v2` (`post_settling_steps: 50`,
  [configs/config_final_v1.py](configs/config_final_v1.py)) as the success
  criterion for new experiments, analyses and reports. final_v1 adds a CaP-only
  check (`final_release_check`, `task-spec-v2+final-release-v1`, on by default
  and pinned off in `config_frozen_v1`–`v7`): a rollout
  that ends with a block touching the gripper fails with status
  `unreleased_block` ([Revision 7.4](docs/revision_7_4_check_final_release.md)).
  The shared checker
  `cfg/task_success.py` decides success on the state 50 unrecorded settling
  steps after the rollout, so that CaP is scored like RoboVerify
  (`collect_demos --post-settling-steps 50`). `post_settling_steps: 0`
  (`task-spec-v1`, the last recorded state) remains the default only to
  reproduce earlier results; see the
  [settled-success notes](../cfg/docs/settled_postcondition.md). Set
  `recovery.success_field: task_spec_success` explicitly in new recovery
  configurations. Report the end-of-rollout postcondition and the physical
  evaluator (`success`, `settled_environment_success`) only as diagnostics, and
  score RoboVerify executions in a comparison with the same criterion. See [Revision 5, §3.5](docs/revision_5_add_tasks.md#35-result-fields)
  and [Revision 4.3 report, §3.1 and §4](docs/revision_4_3_report.md#31-success-criterion-task_spec_success).
- Use the frozen CaP baseline configuration for RoboVerify comparisons, without
  per-task re-tuning: the version's model (v1–v6: Qwen3.8 27B FP8 with
  `reasoning_effort=medium` and the reasoning guard; v7 and final_v1: Claude Haiku 5.5 or
  Opus 5.5, reasoning effort medium), `cap_method_2` with `dsl_only` and `typed_v3` (`typed_v4` in v2), primitive prompt
  `p0101` (convergence criteria and motion-planning limits on), recovery on with
  `max_rounds: 4` and feedback profile `full`
  (`profile_settings("full")`), and `task_spec_success` as the success field.
  The full table (budgets, history, enrichments) is in the
  [Revision 4.3 report, §3.5](docs/revision_4_3_report.md#35-frozen-configuration).
  Build new hosted-model experiments from
  [configs/config_final_v2.py](configs/config_final_v2.py) (`--frozen final_v2`):
  final_v1's prompts byte for byte, sent with provider prompt caching, pinned
  OpenRouter routing, demonstration tokens measured once per batch and a
  first-response warm-up ([Revision 7.5](docs/revision_7_5_prompt_cache.md)).
  [configs/config_final_v1.py](configs/config_final_v1.py) (`--frozen final_v1`)
  is the final configuration for all six tasks: v7 plus the final release check and
  one instruction sentence stating it (`task-instructions-v3`,
  [Revision 7.4](docs/revision_7_4_check_final_release.md)). v7 runs Claude Haiku 5.5 (default) or
  Opus 5.5 on OpenRouter with 64k output, larger token budgets and RDP recovery
  keyframes ([Revision 7.3](docs/revision_7_3_frozen_v7.md)) on top of v6: v4 (v3's goal interface, `typed_v4`/`typed_v5`
  prompts, `positions-v2` and demonstration input, plus `post_settling_steps: 50`
  and the Unstack/Reverse archives re-collected from settled starts) with the
  primitive prompt package `primitive-prompt-v2`, which drops v1's sentence
  telling the model to express lift, transfer and descent stages
  ([Revision 7](docs/revision_7_update_prompt_and_config.md), v5), and the
  general task instructions `task-instructions-v2`
  ([Revision 7.1](docs/revision_7_1_general_task_spec.md), v6).
  Use it for both the demonstration-on and -off arms (`configuration(task,
  demonstrations=False, ...)`). Earlier results used v1–v4 (v3: Revision 6
  demonstrations; v1–v4 prompts contain the motion-strategy sentence; v1–v5
  Grid/Pyramid instructions state the goal counts); every
  fresh reset now settles for 50 steps, so Unstack and Reverse scenes differ
  from runs made before 2026-10-26.
- Keep every prompt an API description: what each primitive does and does not
  do, its parameters, limits and failure behavior. Never state a task
  decomposition or motion strategy in words (for example lift, transfer,
  descend, approach heights, waypoint order) in the initial, repair or recovery
  prompts. Strategy may reach the model only through demonstrations (when
  enabled) and measured feedback ([Revision 7, §3](docs/revision_7_update_prompt_and_config.md#3-rules-for-later-changes)).
  State every task as RoboVerify's specification: one policy that takes every
  initial state satisfying the precondition, for any number of blocks, to the
  postcondition. Instructions say that the number of blocks (and the goal
  layout size) is not given, and never state a particular count; the evaluated
  counts are run settings (`TASKS`) only
  ([Revision 7.1](docs/revision_7_1_general_task_spec.md)).
  Recovery keyframes are `recovery.feedback.keyframe_method`: `legacy` (dataclass
  default and v1–v6: entry, start of the last primitive, end) or `rdp` (v7: the
  failed rollout compacted and laid out like a demonstration, as a plain-text
  section after the evidence object; needs `fixed_items`).
  Size `keyframe_bytes` and `evidence_bytes` so that the frames fit, and report
  partial delivery ([Revision 7.2](docs/revision_7_2_recovery_key_frame.md)).
  Initial settling is `RunConfig.initial_settling_steps`, **on by default**
  (50 steps after every fresh reset, all six tasks, all frozen versions). Keep
  the default; set 0 only to reproduce an earlier unsettled Unstack/Reverse run,
  and record that override with the results. Restored snapshots are never
  settled again. See the [cfg agent guide](../cfg/AGENTS.md#settling). v3–v6 need the 262,144-token served window (v7 and final_v1 target hosted models)
  ([Revision 6, §3.5](docs/revision_6_add_demonstration.md#35-context-window-upgrade-config_frozen_v3)). `configuration(task,
  seed=...)` returns the run config, model config and instruction; use it rather
  than restating values; it also records the server-side
  settings (`SERVING`, including the reasoning guard) and the per-task block
  counts. Never edit a frozen version. Change the configuration only when a task
  fails because the CaP interface does not expose or describe something the task
  needs (for example, the prompt is silent about `tbl` or goal handles); low
  success alone is a result, not a reason. Record such a change as a new
  `config_frozen_vN.py` and apply it to all six tasks (§3.6).
- Build demonstration-input experiments (Revision 6) from
  [configs/config_final_v1.py](configs/config_final_v1.py) (earlier runs:
  v3, v6, v7), for both the demonstrations-on and demonstrations-off arms, so that the
  demonstrations are the only difference. Its `SERVING` block records the
  262,144-token window and prefix caching (applied 2026-10-26, Revision 6,
  §3.5). Report `demonstration_tokens` next to the
  effort tokens (`known_tokens`). Never include an evaluation scene among the
  demonstrations.
- Use the paper's block counts for each task (Stack/Unstack/Reverse 4, ReStack
  5, Grid/Pyramid 6; ReStack's 3-block tower is only drawn in Fig. 10d), and
  check each task's pre/postconditions, in
  [task_specification.md](docs/task_specification.md). Repository
  defaults differ, so set counts explicitly and use the same scenes for CaP and
  RoboVerify.
- The repository's 60-second Unstack limit governs `synthesize_cfg`'s process
  alarm, not CaP trials. Every CaP task uses the configured trial and recovery
  time limits.
- For new experiments, use `video_mode: "deferred"` by default. Save full rollout
  trajectories, then render videos through the shared background queue. Use
  `inline` only for an explicit comparison/debugging need and `off` when videos
  are explicitly unnecessary. When adapting an old configuration, set
  `video_mode` explicitly: legacy `save_video: true` retains inline behavior.
  Keep video completion/failures separate from policy outcomes and recovery.
- Run independent experiment trials in parallel by default. Maximize useful
  utilization of authorized CPU, GPU, memory, and model-request capacity to reduce
  experiment completion time.
- Use the shared CaP batch runner and optional GPU scheduler. Select only GPUs in
  the explicit allowlist, coordinate reservations across concurrent batches, and
  respect global and per-GPU worker limits and provider request limits.
- Choose concurrency from measured worker memory and observed throughput. Increase
  it when resources permit; reduce it when contention, memory pressure, encoding,
  I/O, or request throttling lowers throughput. High GPU utilization alone is not
  evidence of faster experiments.
- Preserve independent trial processes, fixed initial-state replay, per-trial
  budgets, artifact ownership, and the declared experiment denominator. Keep all
  recovery attempts in the same trial and trial GPU assignment. Deferred rendering
  jobs reserve their own fixed assignments from the shared GPU ledger; trial and
  render workers must respect the combined batch worker limit and per-GPU limits.
- Record the scheduling configuration, actual assignments, queue time, active
  time, resource measurements, and failures. Resume pending trials without
  duplicating live workers, completed trials, or paid model requests.
- Use serial execution when a dependency, resource limit, debugging procedure,
  or controlled comparison requires it, and record the reason.
