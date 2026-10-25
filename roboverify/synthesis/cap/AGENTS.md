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
