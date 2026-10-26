# Entry points and configuration

## Entry points

- **One trial:** [`runner.run`](../../runner.py#L623-L633).
  - It prepares the demonstration block once per trial if demonstrations are on.
  - With recovery on, it calls [`recovery.run_recovery`](../../recovery.py#L879) (the recovery loop). Otherwise it calls [`runner.run_once`](../../runner.py#L157) for a single attempt.
- **One attempt:** [`runner.run_once`](../../runner.py#L157-L613) builds the logger, LLM client, environment, adapter, recorder and method, then runs synthesis, rollout and evaluation for one round.
- **Batches:** [`batch.create_batch`](../../batch.py#L42) writes a manifest of independent trials, and [`batch.execute_batch`](../../batch.py#L769) dispatches them. Each trial runs `runner.run` in its own process through [`batch_worker.main`](../../batch_worker.py#L13).
- **Command line:** [`entry/run_cap.py`](../../../entry/run_cap.py) (one trial) and [`entry/run_cap_batch.py`](../../../entry/run_cap_batch.py) (batch).

## Configuration

- [`RunConfig`](../../config.py#L314): task, block count, method, condition mode, prompt package, initial-attempt budgets, and the nested settings below.
- [`RecoveryConfig`](../../config.py#L135): rounds, per-round and per-trial budgets, success field, feedback items ([`RecoveryFeedbackConfig`](../../config.py#L76)) and history mode ([`RecoveryHistoryConfig`](../../config.py#L113)).
- [`DemonstrationConfig`](../../config.py#L234): archive, trace selection, waypoint tolerance, block token limit, token counter.
- [`PrimitivePromptConfig`](../../config.py#L195) and [`PrimitiveFeedbackConfig`](../../config.py#L207): primitive prompt blocks and feedback enrichments.
- [`ModelConfig`](../../config.py#L12): endpoint, model, output cap, sampling, retries.

## Frozen configurations

Experiments use a frozen version; frozen files are never edited. Each exposes `configuration(task, ...)`, which returns `(RunConfig, ModelConfig, instruction)`.

| Version | Use | Code |
| --- | --- | --- |
| v1 | Stack, Unstack, Reverse, ReStack | [`config_frozen_v1.py`](../../configs/config_frozen_v1.py#L50-L120), [`configuration`](../../configs/config_frozen_v1.py#L216) |
| v2 | v1 plus the Grid/Pyramid goal interface (`typed_v4`) | [`config_frozen_v2.py`](../../configs/config_frozen_v2.py) |
| v3 | v2 plus demonstration input (`typed_v5`), for demonstrations on/off | [`config_frozen_v3.py`](../../configs/config_frozen_v3.py#L126) |

Frozen v1 settings that matter for the pseudocode: Method 2, `dsl_only`, at most 5 checker attempts per synthesis, recovery on with 4 rounds, success field `task_spec_success`, feedback profile `full`, and history mode `executed_programs`.
