# Revision 4: Haiku 5.5 recovery comparison — 20 trials per setting

- **120 trials:** the original five corrected trials plus fifteen new trials for each of the six settings.
- Stack, three blocks, one shared archived initial scene; `dsl_only`; `typed_v3`; Haiku 5.5 medium reasoning; 16,384 output tokens per request; three parallel workers.
- Both cohorts use the corrected recovery implementation from `df0a42a`. The extension is recorded at `231b368`, whose changes from that implementation are documentation only.
- All failures are retained. The earlier CAP-RECOVERY-1 diagnostic batch is excluded.

## Accumulated results

| Method | Recovery off | On: first attempt | On: final | Initially failed trials recovered | Tokens off | Tokens on | Calls off / on |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | 9/20 | 8/20 | 19/20 | 11 | 164,348 | 456,633 | 30 / 68 |
| Method 1 | 10/20 | 10/20 | 20/20 | 10 | 120,686 | 258,352 | 20 / 33 |
| Method 2 | 14/20 | 17/20 | 20/20 | 3 | 186,554 | 235,812 | 22 / 24 |

- Recovery benefit is measured from the first attempt to the final outcome within the recovery-on group. Off/on groups use independent model generations; their initial-success difference is not a recovery gain.
- Recovery receives extra failure observations and additional generation/execution budget. This experiment does not compare against independent restarts at matched cost.
- Twenty generations per setting on one fixed scene do not establish generalization across new layouts or tasks.

## Original and additional cohorts

| Method | Setting | Original five | Additional fifteen | Combined twenty | Final DSL pass |
| --- | --- | --- | --- | --- | --- |
| Baseline | Recovery off | 1/5 | 8/15 | 9/20 | 18/20 |
| Baseline | Recovery on | 5/5 | 14/15 | 19/20 | 19/20 |
| Method 1 | Recovery off | 3/5 | 7/15 | 10/20 | 20/20 |
| Method 1 | Recovery on | 5/5 | 15/15 | 20/20 | 20/20 |
| Method 2 | Recovery off | 2/5 | 12/15 | 14/20 | 20/20 |
| Method 2 | Recovery on | 5/5 | 15/15 | 20/20 | 20/20 |

- [Original corrected 30-trial report](../../../runs/cap-revision4-haiku55/20261019-225124/report.md); [historical five-trial analysis](revision_4_recovery_initial5_report.md).
- [Additional 90-trial report](../../../runs/cap-revision4-haiku55/20261020-001210-additional15/report.md). New local trial IDs 1–15 correspond to accumulated trial IDs 6–20.
- [All 120 trials, including each trial’s token usage](../../../runs/cap-revision4-haiku55/20261020-001210-additional15/accumulated/report.md); [all attempt trajectories and videos](../../../runs/cap-revision4-haiku55/20261020-001210-additional15/accumulated/artifacts.md).

## Recovery outcomes

| Method | Initially unsuccessful, recovery on | Recovered | Final failures | Additional rounds used |
| --- | --- | --- | --- | --- |
| Baseline | 12 | 11 | 1 | 16 |
| Method 1 | 10 | 10 | 0 | 13 |
| Method 2 | 3 | 3 | 0 | 4 |

| Recovery stop reason | Trials |
| --- | --- |
| preparation_failure_no_rollout | 1 |
| success | 59 |

### Unsuccessful recovery-enabled trials

| Method | Accumulated trial | Status | Reason | Artifacts |
| --- | --- | --- | --- | --- |
| Baseline | 12 | failed | SyntaxError: invalid syntax (executed.py, line 1) | [Manifest](../../../runs/cap-revision4-haiku55/20261020-001210-additional15/trials/cap_google_baseline-dsl_only-007-recovery-on/cap/20261020-002049-231b368-cap_google_baseline-recovery-76acf51a/artifacts/trial_001/manifest.json) |

- The helper response included prose and Markdown, so Python compilation failed before rollout. Runtime recovery requires an executed policy trajectory; this preparation failure is recorded and remains in the denominator.

## Resource accounting

| Included batch | Reported tokens | Model calls | Known provider cost |
| --- | --- | --- | --- |
| Original corrected five per setting | 348,076 | 47 | $0.102141 |
| Additional fifteen per setting | 1,074,309 | 150 | $0.309952 |
| **Combined 120 trials** | **1,422,385** | **197** | **$0.412093** |

- 153 prepared attempts, 151 policy rollouts, and 33 additional recovery rounds. Missing provider usage: 0 calls.
- Usage includes policy, helper, static-repair, transport-retry and recovery requests. Reasoning is included in completion tokens and is not added twice.
- Initial attempt: 50,000 tokens, 12 calls, 600 seconds. Each recovery: 100,000 tokens, 16 calls, 600 seconds. Whole trial: 450,000 tokens, 76 calls, 3,000 seconds. At most four recovery rounds; 10,000 execution/query ticks and simulation steps per rollout.
- Feedback limit: 8,192 under the conservative UTF-8 byte bound. Per-request output remains 16,384 tokens. These settings match the original corrected batch.

## Audit and scope

- All 153 attempt starts match the archived full initial snapshot; 151 attempts entered policy rollout. Initial policy prompts match within methods across settings and cohorts. Prepared and delivered recovery feedback match, with correct history counts and serialized bounds. Artifact issues: 0.
- All 153 saved videos match their trajectory frame/state indices, including the initial-state-only recordings for two preparation failures. Calls and token totals reconcile against request/response artifacts.
- All 92 successful final policies passed the DSL checker. Baseline has one failed source audit (recovery-off trial 16, which also failed physical execution) and two incomplete preparation outputs (off trial 6 and on trial 12). No successful-but-DSL-invalid final policy was observed.
- Physical/logical terminal-goal disagreements: 2 across attempts. The physical goal, logical postcondition and source DSL compliance remain separate measurements.
- Recovery-off preserves the original normal-completion/physical-success metric. Recovery-on additionally requires the shared logical postcondition. Source compliance alone does not establish physical task completion or formal verification.
- The implementation previously passed 118 CaP tests, controller regressions and the physical recovery preflight. This extension changes neither the implementation nor the experiment configuration.
- [Machine-readable accumulated results and per-trial audit](../../../runs/cap-revision4-haiku55/20261020-001210-additional15/accumulated/summary.json); [extension configuration and cohort mapping](../../../runs/cap-revision4-haiku55/20261020-001210-additional15/comparison_plan.json); [reproducible artifact-only accumulation script](../../../runs/cap-revision4-haiku55/20261020-001210-additional15/audit_accumulate.py).

## Excluded diagnostic batch

- The 30-trial batch `20261019-224320` remains excluded because of CAP-RECOVERY-1 (the latest failure was duplicated in earlier history). Its results are not included in the 120 trials or usage table above.
- [Diagnostic evidence](../../../runs/cap-revision4-haiku55/20261019-224320/analysis.md) and the [history fix](revision_4_failure_recover.md#cap-recovery-1-frozen-history-in-delivered-feedback) are preserved.
