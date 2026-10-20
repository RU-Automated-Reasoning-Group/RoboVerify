# Revision 4: shared failure recovery — Haiku and MiniMax

- **420 included trials:** 120 Haiku 5.5 trials (20 per setting) and 300 MiniMax M3.1 Flash free trials (50 per setting). Each model has six settings: three methods × recovery off/on.
- Shared setup: Stack, three blocks, the same archived initial scene, `dsl_only`, `typed_v3`, medium reasoning, and a 16,384-token output cap per model request.
- All attempts start from the saved initial state. Recovery permits up to four replacement programs after the initial program, for at most five rollouts.
- Results remain separated by model. No trial was rerun or replaced to produce this merged report; the excluded Haiku diagnostic batch and the MiniMax compatibility probe are outside the 420 trials.

## 1. Results across models

| Model | Method | Recovery off | On: initial program | On: final | Initially failed trials recovered |
| --- | --- | --- | --- | --- | --- |
| Haiku 5.5 | Baseline | 9/20 | 8/20 | 19/20 | 11 |
| Haiku 5.5 | Method 1 | 10/20 | 10/20 | 20/20 | 10 |
| Haiku 5.5 | Method 2 | 14/20 | 17/20 | 20/20 | 3 |
| MiniMax M3.1 Flash free | Baseline | 5/50 | 4/50 | 42/50 | 38 |
| MiniMax M3.1 Flash free | Method 1 | 4/50 | 5/50 | 40/50 | 35 |
| MiniMax M3.1 Flash free | Method 2 | 6/50 | 7/50 | 40/50 | 33 |

- **On: initial program / first attempt** means round 0, before any recovery. **On: final** means the final outcome after zero to four recovery rounds.
- **Tokens off** and **Tokens on** in the detailed tables are the total input plus output tokens across all trials in that model/setting, not averages. On includes the initial attempt and all recovery calls; both include helper generation and static repair where applicable. Reasoning tokens are already included in output usage.
- Off/on groups use independent initial generations. Measure recovery gain within the recovery-on group, from its initial to final success count.
- Model sampling behavior, request timeout, and pacing differ across cohorts. A shared medium-effort label does not equate reasoning-token usage. This is a fixed-scene comparison with extra feedback and budget for recovery, not a matched-cost restart study or a test of layout generalization.

| Model cohort | Trials | Total tokens | Model calls | Known provider cost |
| --- | ---: | ---: | ---: | ---: |
| Haiku 5.5 | 120 | 1,422,385 | 197 | $0.412093 |
| MiniMax M3.1 Flash free | 300 | 5,315,581 | 692 | $0.000000 |

## 2. Haiku 5.5 — 20 trials per setting

- **120 trials:** the original five corrected trials plus fifteen new trials for each of the six settings.
- Stack, three blocks, one shared archived initial scene; `dsl_only`; `typed_v3`; Haiku 5.5 medium reasoning; 16,384 output tokens per request; three parallel workers.
- Both cohorts use the corrected recovery implementation from `df0a42a`. The extension is recorded at `231b368`, whose changes from that implementation are documentation only.
- All failures are retained. The earlier CAP-RECOVERY-1 diagnostic batch is excluded.

### Accumulated results

| Method | Recovery off | On: first attempt | On: final | Initially failed trials recovered | Tokens off | Tokens on | Calls off / on |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | 9/20 | 8/20 | 19/20 | 11 | 164,348 | 456,633 | 30 / 68 |
| Method 1 | 10/20 | 10/20 | 20/20 | 10 | 120,686 | 258,352 | 20 / 33 |
| Method 2 | 14/20 | 17/20 | 20/20 | 3 | 186,554 | 235,812 | 22 / 24 |

- Recovery benefit is measured from the first attempt to the final outcome within the recovery-on group. Off/on groups use independent model generations; their initial-success difference is not a recovery gain.
- Recovery receives extra failure observations and additional generation/execution budget. This experiment does not compare against independent restarts at matched cost.
- Twenty generations per setting on one fixed scene do not establish generalization across new layouts or tasks.

### Original and additional cohorts

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

### Recovery outcomes

| Method | Initially unsuccessful, recovery on | Recovered | Final failures | Additional rounds used |
| --- | --- | --- | --- | --- |
| Baseline | 12 | 11 | 1 | 16 |
| Method 1 | 10 | 10 | 0 | 13 |
| Method 2 | 3 | 3 | 0 | 4 |

| Recovery stop reason | Trials |
| --- | --- |
| preparation_failure_no_rollout | 1 |
| success | 59 |

#### Unsuccessful recovery-enabled trials

| Method | Accumulated trial | Status | Reason | Artifacts |
| --- | --- | --- | --- | --- |
| Baseline | 12 | failed | SyntaxError: invalid syntax (executed.py, line 1) | [Manifest](../../../runs/cap-revision4-haiku55/20261020-001210-additional15/trials/cap_google_baseline-dsl_only-007-recovery-on/cap/20261020-002049-231b368-cap_google_baseline-recovery-76acf51a/artifacts/trial_001/manifest.json) |

- The helper response included prose and Markdown, so Python compilation failed before rollout. Runtime recovery requires an executed policy trajectory; this preparation failure is recorded and remains in the denominator.

### Resource accounting

| Included batch | Reported tokens | Model calls | Known provider cost |
| --- | --- | --- | --- |
| Original corrected five per setting | 348,076 | 47 | $0.102141 |
| Additional fifteen per setting | 1,074,309 | 150 | $0.309952 |
| **Combined 120 trials** | **1,422,385** | **197** | **$0.412093** |

- 153 prepared attempts, 151 policy rollouts, and 33 additional recovery rounds. Missing provider usage: 0 calls.
- Usage includes policy, helper, static-repair, transport-retry and recovery requests. Reasoning is included in completion tokens and is not added twice.
- Initial attempt: 50,000 tokens, 12 calls, 600 seconds. Each recovery: 100,000 tokens, 16 calls, 600 seconds. Whole trial: 450,000 tokens, 76 calls, 3,000 seconds. At most four recovery rounds; 10,000 execution/query ticks and simulation steps per rollout.
- Feedback limit: 8,192 under the conservative UTF-8 byte bound. Per-request output remains 16,384 tokens. These settings match the original corrected batch.

### Audit and scope

- All 153 attempt starts match the archived full initial snapshot; 151 attempts entered policy rollout. Initial policy prompts match within methods across settings and cohorts. Prepared and delivered recovery feedback match, with correct history counts and serialized bounds. Artifact issues: 0.
- All 153 saved videos match their trajectory frame/state indices, including the initial-state-only recordings for two preparation failures. Calls and token totals reconcile against request/response artifacts.
- All 92 successful final policies passed the DSL checker. Baseline has one failed source audit (recovery-off trial 16, which also failed physical execution) and two incomplete preparation outputs (off trial 6 and on trial 12). No successful-but-DSL-invalid final policy was observed.
- Physical/logical terminal-goal disagreements: 2 across attempts. The physical goal, logical postcondition and source DSL compliance remain separate measurements.
- Recovery-off preserves the original normal-completion/physical-success metric. Recovery-on additionally requires the shared logical postcondition. Source compliance alone does not establish physical task completion or formal verification.
- The implementation previously passed 118 CaP tests, controller regressions and the physical recovery preflight. This extension changes neither the implementation nor the experiment configuration.
- [Machine-readable accumulated results and per-trial audit](../../../runs/cap-revision4-haiku55/20261020-001210-additional15/accumulated/summary.json); [extension configuration and cohort mapping](../../../runs/cap-revision4-haiku55/20261020-001210-additional15/comparison_plan.json); [reproducible artifact-only accumulation script](../../../runs/cap-revision4-haiku55/20261020-001210-additional15/audit_accumulate.py).

### Excluded diagnostic batch

- The 30-trial batch `20261019-224320` remains excluded because of CAP-RECOVERY-1 (the latest failure was duplicated in earlier history). Its results are not included in the 120 trials or usage table above.
- [Diagnostic evidence](../../../runs/cap-revision4-haiku55/20261019-224320/analysis.md) and the [history fix](revision_4_failure_recover.md#cap-recovery-1-frozen-history-in-delivered-feedback) are preserved.

## 3. MiniMax M3.1 Flash free — 50 trials per setting

- **Status: completed; 300/300 planned trials finished.** Target: 50 trials in each of six settings.
- Stack, three blocks, the same full initial snapshot as the Haiku comparison; `dsl_only`; `typed_v3`.
- Model: `minimax/minimax-m3.1-flash:free`; reasoning explicitly `medium`; response cap 16,384 tokens; temperature and top_p omitted.
- Three parallel workers. Request timeout: 300 seconds. Shared request admission interval: 6 seconds. Batch safety threshold: 800 wire requests.
- Runtime recovery and DSL checking are unchanged from the corrected implementation. Commit `29473fe` adds optional pacing and resumable quota admission; all 124 CaP tests pass.

### Results

| Method | Off success / finished | On: initial program / finished | On: final / finished | Recovered | Tokens off | Tokens on | Calls off / on |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | 5/50 | 4/50 | 42/50 | 38 | 358,675 | 1,560,818 | 65 / 195 |
| Method 1 | 4/50 | 5/50 | 40/50 | 35 | 307,989 | 1,336,328 | 55 / 162 |
| Method 2 | 6/50 | 7/50 | 40/50 | 33 | 366,830 | 1,384,941 | 60 / 155 |

- The initial program is round 0, before any recovery. Final success permits up to four recovery rounds. No failed trial is replaced; pending trials are not failures.
- Off/on initial programs are independent generations. Recovery gain is the initial-to-final improvement within the recovery-on group.
- [Per-trial results and token usage](../../../runs/cap-revision4-minimax31flash/20261020-152924-free-300/analysis.md); [attempt trajectories/videos](../../../runs/cap-revision4-minimax31flash/20261020-152924-free-300/artifacts.md); [machine-readable audit](../../../runs/cap-revision4-minimax31flash/20261020-152924-free-300/experiment_audit.json).

### Included pilot and request budget

- The first five trials per setting (30 total, included in the target) used 72 requests: 2.40 requests per trial. The pilot projection for 300 trials was 720 requests.
- A new trial is admitted only if the batch threshold and current free quota can cover its maximum call allowance and the remaining allowances of active trials. A refusal drains active workers and preserves unstarted jobs as pending.
- Quota checks are account reads, not model generations. A separate compatibility probe used one inference request, 265 tokens and zero reported cost; it is excluded from trial counts.

### Usage and outcomes

- Completed-trial usage: **5,315,581 tokens**, **692 model calls**, known provider cost **$0.000000**. Missing usage: 0 calls.
- Wire requests admitted: 692. Prepared attempts: 592; policy rollouts: 582; recovery rounds: 292.
- Usage includes policy, helper, static-repair, transport and recovery calls. Reasoning tokens are included in completion tokens and are not counted twice.

| Recovery stop reason | Trials |
| --- | --- |
| max_recovery_rounds | 23 |
| preparation_failure_no_rollout | 5 |
| success | 122 |

| Method | Recovery | Final DSL pass / finished |
| --- | --- | --- |
| Baseline | off | 44/50 |
| Baseline | on | 46/50 |
| Method 1 | off | 50/50 |
| Method 1 | on | 50/50 |
| Method 2 | off | 50/50 |
| Method 2 | on | 49/50 |

### Final failure categories

| Method | Recovery | Primitive failed | Execution/resource budget exhausted | Goal failed | Other runtime failure | Generation failed |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | off | 20 | 14 | 3 | 8 | 0 |
| Baseline | on | 3 | 1 | 0 | 3 | 1 |
| Method 1 | off | 35 | 9 | 1 | 1 | 0 |
| Method 1 | on | 6 | 2 | 2 | 0 | 0 |
| Method 2 | off | 23 | 15 | 6 | 0 | 0 |
| Method 2 | on | 7 | 1 | 0 | 1 | 1 |

These are final runner statuses, not independently established root causes. Recovery-on counts describe the last attempt after recovery stopped. Earlier failed attempts remain in the linked per-trial records.

All 137 successful final policies passed the saved DSL audit. This batch contains no physically successful final policy with a non-passing DSL audit; this does not establish that prompting always enforces the DSL.

### Audit and interpretation

- Artifact issues: 0. Exact initial-state matches: 592/592. Verified video/frame mappings: 592.
- Initial prompts match the corresponding Haiku hashes. Every delivered recovery payload is compared with the prepared artifact; history length and serialized feedback bounds are checked.
- Planning and case-study Markdown files were added or edited while the batch was running. Later attempts may record a dirty worktree; the audit checks the documented Markdown changes and verifies that runtime sources remain unchanged. [Documentation provenance](../../../runs/cap-revision4-minimax31flash/20261020-152924-free-300/documentation_provenance.json).
- [Baseline trial 6: all five programs and delivered feedback](case_study/revision_4_failure_recovery_cases/minimax_baseline_dsl_only_trial_006.md).
- Physical/logical terminal-goal disagreements: 12 across attempts. Physical success, the logical postcondition and source DSL compliance are recorded separately.
- Recovery-off retains the original physical-goal/normal-completion metric. Recovery-on additionally requires the shared logical postcondition. Exceptions after reaching a goal are not silently relabeled as normal successful rollouts.
- This is a fixed-scene generation study. Recovery uses extra requests and failure observations; the experiment is not a matched-cost comparison with independent restarts or a generalization test across layouts.
- Model, native sampling behavior, request timeout and request pacing differ from Haiku. The same reasoning-effort label does not imply equal thinking-token usage.
- [Frozen configuration](../../../runs/cap-revision4-minimax31flash/20261020-152924-free-300/comparison_plan.json); [pilot estimate](../../../runs/cap-revision4-minimax31flash/20261020-152924-free-300/pilot_estimate.json); [request ledger](../../../runs/cap-revision4-minimax31flash/20261020-152924-free-300/request_gate.json); [validation logs](../../../runs/cap-revision4-minimax31flash/20261020-152924-free-300/validation/full_cap_suite.txt).
