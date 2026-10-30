# CaP agent guide

Follow the repository-wide [AGENTS.md](../../../AGENTS.md) for environment setup,
project decisions, implementation conventions and validation. This file adds
what is specific to `synthesis/cap/`.

## Orientation

- **What this is.** Code as Policies (CaP) baselines for RoboVerify. An LLM
  writes a Python policy against RoboVerify's `Pick`/`Move`/`Release` DSL and
  predicates (`ON_star`, `Higher`, `Scattered`, goal predicates); the policy
  runs in RoboVerify's MuJoCo Fetch environment and is scored with RoboVerify's
  task specification. Six tasks: Stack, Unstack, Reverse, ReStack, Grid,
  Pyramid ([task_specification.md](docs/task_specification.md)).
- **Read first.** The [README Quick start](README.md#quick-start-run-the-final-experiment)
  (setup, one-command run, output layout), then the
  [module docs](docs/module_docs/README.md), which describe what the code does
  now. Revision documents under `docs/` are design history; read one only when
  the task touches it. The rest of the README records earlier experiments.
- **Final method.** `cap_method_2`: whole-program generation, the shared static
  DSL checker ([checker.py](checker.py), [validation.py](validation.py)) with
  whole-candidate repair, checked execution, then up to four recovery rounds
  that feed measured failure evidence back to the model
  ([recovery.py](recovery.py)). `cap_google_baseline` (pinned upstream CaP) and
  `cap_method_1` (per-helper generation and local repair) are earlier
  comparison methods.
- **Code path of one trial.** [runner.py](runner.py) → `cap_method_2/` →
  [checker.py](checker.py) → [adapter.py](adapter.py)/[runtime.py](runtime.py)
  (DSL bindings and rollout) → [tasks.py](tasks.py) (evaluation) →
  [recovery.py](recovery.py). Batches: [batch.py](batch.py) runs every trial in
  its own [batch_worker.py](batch_worker.py) process. Prompts:
  [prompt_package.py](prompt_package.py) with versioned packages in `prompts/`.
  Demonstrations: [demonstrations.py](demonstrations.py).

## Current state (2026-10-30)

- **Final configuration.** [config_final_v3](configs/config_final_v3.py),
  one version for both arms (`configuration(task, demonstrations=False)` for
  off). It is [config_final_v2](configs/config_final_v2.py) (on) and
  [config_final_v2_wo_demo](configs/config_final_v2_wo_demo.py) (off) with the
  static DSL analysis budget `max_analysis_steps` raised to 1,000,000 for both.
  It uses `config_final_v1`'s prompts byte for byte with provider prompt
  caching: Claude Opus 5.5 (or Haiku 5.5) on
  OpenRouter, reasoning effort medium, `cap_method_2`, `dsl_only`, `typed_v5`,
  primitive prompt `primitive-prompt-v2`, recovery on (4 rounds, feedback
  profile `full`, RDP keyframes), block counts Stack/Unstack/Reverse 4,
  ReStack 5, Grid 2 × 3, Pyramid 3-2-1.
- **Final results.** Run under final_v2 / final_v2_wo_demo: 300/300 trials
  succeed in both arms (50 per task, Opus 5.5); demonstrations raise round-0
  success on most tasks. No final_v3 run exists yet. See the
  [demonstrations-on](docs/final_v1_report.md) and
  [demonstrations-off](docs/final_v2_wo_demo_report.md) reports.
- **Generalization (Revision 9, implemented).** The final programs replayed on
  721 physically feasible cases of other block counts (`relaxed_v2`): no CaP or
  supplied RoboVerify program generalizes to every count
  ([report](docs/revision_9_report.md)).
- **Verification (Revision 8).** Running RoboVerify's verifier on CaP programs
  is a plan, not implemented ([plan](docs/revision_8_verification.md)).

## Running experiments

- Run the final experiment with
  `bash synthesis/cap/run_final.sh --demo on|off --task TASK --model MODEL`
  from `roboverify/`, after `bash synthesis/cap/collect_demos.sh` for
  `--demo on`. It calls
  [experiments/revision6_experiment.py](experiments/revision6_experiment.py)
  with `--frozen final_v3 --demonstrations on|off` and no GPU reservations unless `--gpu UUID` is given. Use the module
  directly for other frozen versions or this host's GPU allowlist (its default
  without `--gpu`/`--no-gpu-scheduler`).
- Read results through `ROOT/report.md`, `<model>/<task>/summary.json` and the
  per-trial `report.txt`. Never `cat` `metrics.jsonl` or `stdout.log`; use
  `python -m synthesis.experiment.report --run DIR` or a bounded grep.
- Before any paid run, check `OPENROUTER_API_KEY` is set and confirm the run's
  size with the user. Keep at most 2–3 workers per task per OpenRouter key;
  30 concurrent workers triggered HTTP 429s.
- Tests: `uv run python -m unittest discover -s synthesis/cap -t . -p 'test_*.py'`.
  Offline tests use scripted responses and make no model requests.

## Rules for experiments, analyses and reports

- **Success criterion.** Use `task_spec_success` under
  `task-spec-v2+final-release-v1`: the shared pre/postcondition checked by
  `cfg/task_success.py` after 50 unrecorded settling steps
  (`post_settling_steps: 50`), and a rollout that ends with a block touching
  the gripper fails with status `unreleased_block`
  ([Revision 7.4](docs/revision_7_4_check_final_release.md)). Set
  `recovery.success_field: task_spec_success` in new recovery configurations.
  Report the end-of-rollout postcondition and the physical evaluator
  (`success`, `settled_environment_success`) only as diagnostics. Score
  RoboVerify executions in a comparison with the same criterion
  ([settled-success notes](../cfg/docs/settled_postcondition.md)).
- **Frozen configurations.** Build new experiments from `config_final_v3`
  via `configuration(task, demonstrations, seed=...)`, which returns
  the run config, model config and instruction; do not restate its values.
  Never edit a frozen version (`config_frozen_v1`–`v7`, `config_final_*`).
  Change the configuration only when a task fails because the CaP interface
  does not expose or describe something the task needs; low success alone is a
  result, not a reason. Record a change as a new version applied to all six
  tasks. Use the same configuration for every task, without per-task tuning.
  Record any run-time override (for example `--max-analysis-steps`) with the
  results; the driver writes it to `comparison_plan.json`.
- **Prompts describe the API, never a strategy.** State what each primitive
  does and does not do, its parameters, limits and failure behavior. Never
  state a task decomposition or motion strategy (lift, transfer, descend,
  approach heights, waypoint order) in initial, repair or recovery prompts.
  Strategy reaches the model only through demonstrations (when enabled) and
  measured feedback ([Revision 7, §3](docs/revision_7_update_prompt_and_config.md#3-rules-for-later-changes)).
  State every task as RoboVerify's specification: one policy for every initial
  state satisfying the precondition and any number of blocks; never give a
  block or goal count in the instruction
  ([Revision 7.1](docs/revision_7_1_general_task_spec.md)). Add a new prompt
  package version rather than editing an existing one.
- **Demonstrations.** The on and off arms must differ only in the
  demonstration block. The six pinned archives are not in git;
  `bash synthesis/cap/collect_demos.sh` re-collects them, deterministically,
  and must run before any demonstrations-on experiment. The loader accepts an
  archive by its file SHA-256 pin or by the pinned archive's content digest
  (`demonstrations.CONTENT_PINS`, which excludes only per-trace wall time) and
  records `pin_match`. An archive that matches neither (a collection on
  another machine) runs only with `--accept-unpinned-demonstrations`, which
  `run_final.sh` passes and which records both hashes in the batch's
  `comparison_plan.json` overrides. Report such runs as unpinned. Never include an evaluation scene among the demonstrations (the
  runner rejects such a scene before any request). Report
  `demonstration_tokens` next to the effort tokens (`known_tokens`).
- **Scenes.** Use the paper's block counts per task, as the frozen `TASKS`
  set, and the same scenes (seeds) for CaP and RoboVerify. Fresh resets settle
  for 50 steps (`RunConfig.initial_settling_steps`, on by default); restored
  snapshots are never settled again. Unstack and Reverse scenes made before
  2026-10-26 were unsettled and are not comparable. The repository's
  60-second Unstack limit applies to `synthesize_cfg`, not to CaP trials.
- **Denominators.** Keep failed, interrupted and budget-exhausted trials in
  the denominator; never replace a trial because of its outcome. Rerun an
  interrupted trial with its original seed and record the rerun.
- **Videos.** Use `video_mode: "deferred"` (the default): record full
  trajectories, render in a background queue, and keep video failures separate
  from policy outcomes. When adapting an old configuration, set `video_mode`
  explicitly (legacy `save_video: true` means inline).

## Parallel execution and GPUs

- Run independent trials in parallel through the shared batch runner. Choose
  concurrency from measured worker memory, throughput and provider rate
  limits; high GPU utilization alone is not evidence of a faster experiment.
- GPU reservations are optional. Select only GPUs in an explicit allowlist,
  share one reservation directory across concurrent batches, and respect the
  global and per-GPU worker limits. Trial and deferred-render workers share
  these limits; recovery rounds keep their trial's GPU.
- Preserve independent trial processes, fixed initial-state replay, per-trial
  budgets and artifact ownership. Resume pending trials without duplicating
  live workers, completed trials or paid requests. Record scheduling settings,
  assignments, queue and active time and failures. Use serial execution only
  when a dependency, resource limit, debugging procedure or controlled
  comparison requires it, and record the reason.
