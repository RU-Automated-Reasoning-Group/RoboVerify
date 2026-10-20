# MiniMax M3.1 Flash free: shared recovery experiment

Results are also included in the [combined Revision 4 report](revision_4_recovery_report.md).

- **Status: completed; 300/300 planned trials finished.** Target: 50 trials in each of six settings.
- Stack, three blocks, the same full initial snapshot as the Haiku comparison; `dsl_only`; `typed_v3`.
- Model: `minimax/minimax-m3.1-flash:free`; reasoning explicitly `medium`; response cap 16,384 tokens; temperature and top_p omitted.
- Three parallel workers. Request timeout: 300 seconds. Shared request admission interval: 6 seconds. Batch safety threshold: 800 wire requests.
- Runtime recovery and DSL checking are unchanged from the corrected implementation. Commit `29473fe` adds optional pacing and resumable quota admission; all 124 CaP tests pass.

## Results

| Method | Off success / finished | On: initial program / finished | On: final / finished | Recovered | Tokens off | Tokens on | Calls off / on |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | 5/50 | 4/50 | 42/50 | 38 | 358,675 | 1,560,818 | 65 / 195 |
| Method 1 | 4/50 | 5/50 | 40/50 | 35 | 307,989 | 1,336,328 | 55 / 162 |
| Method 2 | 6/50 | 7/50 | 40/50 | 33 | 366,830 | 1,384,941 | 60 / 155 |

- The initial program is round 0, before any recovery. Final success permits up to four recovery rounds. No failed trial is replaced; pending trials are not failures.
- Off/on initial programs are independent generations. Recovery gain is the initial-to-final improvement within the recovery-on group.
- [Per-trial results and token usage](../../../runs/cap-revision4-minimax31flash/20261020-152924-free-300/analysis.md); [attempt trajectories/videos](../../../runs/cap-revision4-minimax31flash/20261020-152924-free-300/artifacts.md); [machine-readable audit](../../../runs/cap-revision4-minimax31flash/20261020-152924-free-300/experiment_audit.json).

## Included pilot and request budget

- The first five trials per setting (30 total, included in the target) used 72 requests: 2.40 requests per trial. The pilot projection for 300 trials was 720 requests.
- A new trial is admitted only if the batch threshold and current free quota can cover its maximum call allowance and the remaining allowances of active trials. A refusal drains active workers and preserves unstarted jobs as pending.
- Quota checks are account reads, not model generations. A separate compatibility probe used one inference request, 265 tokens and zero reported cost; it is excluded from trial counts.

## Usage and outcomes

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

## Final failure categories

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

## Audit and interpretation

- Artifact issues: 0. Exact initial-state matches: 592/592. Verified video/frame mappings: 592.
- Initial prompts match the corresponding Haiku hashes. Every delivered recovery payload is compared with the prepared artifact; history length and serialized feedback bounds are checked.
- Planning and case-study Markdown files were added or edited while the batch was running. Later attempts may record a dirty worktree; the audit checks the documented Markdown changes and verifies that runtime sources remain unchanged. [Documentation provenance](../../../runs/cap-revision4-minimax31flash/20261020-152924-free-300/documentation_provenance.json).
- [Baseline trial 6: all five programs and delivered feedback](case_study/revision_4_failure_recovery_cases/minimax_baseline_dsl_only_trial_006.md).
- Physical/logical terminal-goal disagreements: 12 across attempts. Physical success, the logical postcondition and source DSL compliance are recorded separately.
- Recovery-off retains the original physical-goal/normal-completion metric. Recovery-on additionally requires the shared logical postcondition. Exceptions after reaching a goal are not silently relabeled as normal successful rollouts.
- This is a fixed-scene generation study. Recovery uses extra requests and failure observations; the experiment is not a matched-cost comparison with independent restarts or a generalization test across layouts.
- Model, native sampling behavior, request timeout and request pacing differ from Haiku. The same reasoning-effort label does not imply equal thinking-token usage.
- [Frozen configuration](../../../runs/cap-revision4-minimax31flash/20261020-152924-free-300/comparison_plan.json); [pilot estimate](../../../runs/cap-revision4-minimax31flash/20261020-152924-free-300/pilot_estimate.json); [request ledger](../../../runs/cap-revision4-minimax31flash/20261020-152924-free-300/request_gate.json); [validation logs](../../../runs/cap-revision4-minimax31flash/20261020-152924-free-300/validation/full_cap_suite.txt).

## Haiku reference

- [Haiku cohort in the combined report](revision_4_recovery_report.md#2-haiku-55--20-trials-per-setting). These are separate model cohorts; no MiniMax trial is accumulated into the Haiku totals.
