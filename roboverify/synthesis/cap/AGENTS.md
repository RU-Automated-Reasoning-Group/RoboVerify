# CaP experiment execution

Follow the repository-wide [AGENTS.md](../../../AGENTS.md) for environment setup,
project decisions, implementation conventions, and validation. This file adds
experiment guidance specific to `synthesis/cap/`.

- Before starting any task under `synthesis/cap/`, read the [README](README.md)
  overview and the sections relevant to the task. Read historical experiment
  reports only when needed.
- Use `task_spec_success` (`task-spec-v1`) as the default success criterion for
  new experiments, analyses and reports, so that CaP is scored like RoboVerify
  (`cfg/collection.py:validate_trace`). Set `recovery.success_field:
  task_spec_success` explicitly in new recovery configurations. Report the
  physical evaluator (`success`, `settled_environment_success`) only as a
  diagnostic, and score RoboVerify executions in a comparison with the same
  criterion. See [Revision 5, §3.5](docs/revision_5_add_tasks.md#35-result-fields)
  and [Revision 4.3 report, §3.1 and §4](docs/revision_4_3_report.md#31-success-criterion-task_spec_success).
- Use the frozen CaP baseline configuration for RoboVerify comparisons, without
  per-task re-tuning: Qwen3.8 27B FP8 (`reasoning_effort=medium`, reasoning
  guard on), `cap_method_2` with `dsl_only` and `typed_v3` (`typed_v4` in v2), primitive prompt
  `p0101` (convergence criteria and motion-planning limits on), recovery on with
  `max_rounds: 4` and feedback profile `full`
  (`profile_settings("full")`), and `task_spec_success` as the success field.
  The full table (budgets, history, enrichments) is in the
  [Revision 4.3 report, §3.5](docs/revision_4_3_report.md#35-frozen-configuration).
  Build new experiments from [configs/config_frozen_v2.py](configs/config_frozen_v2.py),
  the current version for all six tasks (v1 plus the Grid/Pyramid goal interface,
  `typed_v4` and `positions-v2`; tower-task prompts unchanged). `configuration(task,
  seed=...)` returns the run config, model config and instruction; use it rather
  than restating values; it also records the server-side
  settings (`SERVING`, including the reasoning guard) and the per-task block
  counts. Never edit a frozen version. Change the configuration only when a task
  fails because the CaP interface does not expose or describe something the task
  needs (for example, the prompt is silent about `tbl` or goal handles); low
  success alone is a result, not a reason. Record such a change as a new
  `config_frozen_vN.py` and apply it to all six tasks (§3.6).
- Build demonstration-input experiments (Revision 6) from
  [configs/config_frozen_v3.py](configs/config_frozen_v3.py), for both the
  demonstrations-on and demonstrations-off arms, so that the demonstrations are
  the only difference. Its `SERVING` block records the intended 262,144-token
  window and prefix caching. That upgrade is pending (Revision 6, §3.5); until it
  is applied, only Stack, Unstack and Reverse requests with 15 demonstrations fit
  the served 98,304-token window. Report `demonstration_tokens` next to the
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
