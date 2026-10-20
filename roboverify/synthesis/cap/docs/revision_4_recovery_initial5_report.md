# Revision 4: initial five-trial Haiku 5.5 recovery comparison

- Stack, three blocks; `dsl_only`; `typed_v3`; Haiku 5.5 medium reasoning; 16,384 output tokens per request.
- Five independent generations per method and recovery setting, on one shared archived scene. Three parallel workers.
- Recovery permits four extra synthesis/rollout rounds, always starting from the same full initial snapshot. No failed trial was replaced.
- The initial prompts are byte-identical across settings within each method. Recovery-on adds conservative request reservations, runtime evidence and additional effort. This is not a comparison against matched-cost independent restarts.

| Method | Recovery off | Recovery on: first attempt | Recovery on: final | Failures recovered | Tokens off | Tokens on | Calls off / on |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cap_google_baseline | 1/5 | 2/5 | 5/5 | 3 | 30,252 | 149,142 | 5 / 20 |
| cap_method_1 | 3/5 | 3/5 | 5/5 | 2 | 31,204 | 52,801 | 5 / 7 |
| cap_method_2 | 2/5 | 5/5 | 5/5 | 0 | 41,577 | 43,100 | 5 / 5 |

Total: 348,076 reported tokens, 47 model calls, 37 prepared attempts, 37 policy rollouts. Provider-reported cost: $0.102141. Reasoning is included in completion usage, not added twice.

## Recovery details

| Method | Trial | Initial status | Recovery rounds | Final status | Stop reason |
| --- | --- | --- | --- | --- | --- |
| cap_google_baseline | 1 | completed | 0 | completed | success |
| cap_google_baseline | 2 | primitive_failed | 1 | completed | success |
| cap_google_baseline | 3 | primitive_failed | 1 | completed | success |
| cap_google_baseline | 4 | completed | 0 | completed | success |
| cap_google_baseline | 5 | budget_exhausted | 3 | completed | success |
| cap_method_1 | 1 | completed | 0 | completed | success |
| cap_method_1 | 2 | primitive_failed | 1 | completed | success |
| cap_method_1 | 3 | completed | 0 | completed | success |
| cap_method_1 | 4 | primitive_failed | 1 | completed | success |
| cap_method_1 | 5 | completed | 0 | completed | success |
| cap_method_2 | 1 | completed | 0 | completed | success |
| cap_method_2 | 2 | completed | 0 | completed | success |
| cap_method_2 | 3 | completed | 0 | completed | success |
| cap_method_2 | 4 | completed | 0 | completed | success |
| cap_method_2 | 5 | completed | 0 | completed | success |

## Failure inventory

Counts below are failed attempts, not independent trials.

| Method | Recovery | Status | Reason | Count |
| --- | --- | --- | --- | --- |
| cap_google_baseline | off | budget_exhausted | BudgetExceeded: Execution/query step budget exhausted | 1 |
| cap_google_baseline | off | goal_failed |  | 1 |
| cap_google_baseline | off | primitive_failed | PrimitiveFailure: Move did not converge within its shared primitive budget | 2 |
| cap_google_baseline | on | budget_exhausted | BudgetExceeded: Execution/query step budget exhausted | 3 |
| cap_google_baseline | on | primitive_failed | PrimitiveFailure: Move did not converge within its shared primitive budget | 2 |
| cap_method_1 | off | primitive_failed | PrimitiveFailure: Move did not converge within its shared primitive budget | 2 |
| cap_method_1 | on | primitive_failed | PrimitiveFailure: Move did not converge within its shared primitive budget | 2 |
| cap_method_2 | off | primitive_failed | PrimitiveFailure: Move did not converge within its shared primitive budget | 3 |

## Artifact checks

- All 37 recorded starts match the reference full snapshot; all initial prompt hashes match within methods.
- Every recovery prompt includes the current failure and the required earlier summaries; feedback remains within its recorded bound.
- Token and call counts reconcile with request/response artifacts across every attempt.
- Physical/logical goal disagreements: 2. Both criteria and DSL source compliance remain separately recorded.
- Five samples per group on one scene are exploratory; they do not establish generalization across layouts/tasks.

[Per-trial report](../../../runs/cap-revision4-haiku55/20261019-225124/report.md) · [Attempt trajectories/videos](../../../runs/cap-revision4-haiku55/20261019-225124/artifacts.md) · [Machine-readable audit](../../../runs/cap-revision4-haiku55/20261019-225124/experiment_audit.json) · [Frozen comparison plan](../../../runs/cap-revision4-haiku55/20261019-225124/comparison_plan.json)

## Interpretation and scope

- Five initially unsuccessful recovery-on trials became successful: Baseline trials 2 and 3 after one revision each, Baseline trial 5 after three revisions, and Method 1 trials 2 and 4 after one revision each.
- Method 2's recovery-on group succeeded on all initial attempts. Its difference from recovery-off (2/5 versus 5/5) is therefore **not an observed recovery gain**; the groups use independent model generations.
- Baseline trial 5 exhausted 10,000 execution/query ticks in its first three rollouts. The fourth rollout succeeded. Its recorded policy text and actual helper bindings differ: earlier generated helpers did not implement the visible inline helper's intended block-selection/placement sequence. The successful revision put the primitive sequence directly in the invoked policy function. This is an observed workflow/source-binding issue, not evidence that a physical primitive alone caused those budget failures.
- Two unsuccessful terminal states satisfy the relational DSL postcondition but fail the physical goal: Baseline recovery-off trial 2 (`goal_failed`) and Method 2 recovery-off trial 5 (Move nonconvergence). Neither is counted as success. There are no physically successful states with a false DSL postcondition in this batch, so the stricter recovery-on success gate does not change this comparison's outcomes.
- All 30 final outputs pass the shared DSL checker. Source compliance does not establish task completion or formal verification.

## Implementation and validation

- Shared recovery defaults off and supports all three methods. Enabled trials permit four recovery rounds, restore the full initial snapshot and create a fresh execution context before each rollout, and accumulate bounded textual failure history.
- Initial attempt: 50,000 tokens, 12 calls, 600 seconds. Each recovery: 100,000 tokens, 16 calls, 600 seconds. Whole trial: 450,000 tokens, 76 calls, 3,000 seconds. Each rollout retains 10,000 execution/query ticks and 10,000 simulation steps. Per-response output cap: 16,384; feedback cap: 8,192 using a conservative UTF-8 byte bound. Original primitive control limits/tolerances are unchanged.
- Baseline retains its original execution workflow and post-hoc DSL audit. Optional recovery on DSL noncompliance is implemented but disabled in this experiment.
- **118 CaP tests pass**, plus the controller regression suite and scripted six-group physical preflight. Additional failure-trigger cases cover normal goal failure, primitive failure, tick exhaustion and unavailable evaluation. All 37 videos have matching frame/state indices; no provider usage is missing.
- Implementation commits: `555bedf` (shared recovery) and `df0a42a` (freeze feedback history). The corrected batch uses `df0a42a`.

## Excluded diagnostic batch

- The earlier 30-trial batch `20261019-224320` is retained separately and excluded from the tables above. Its artifact audit discovered **CAP-RECOVERY-1**: a shared mutable history list caused the latest failure to be duplicated in the earlier-failure summaries sent to the model.
- The fix copies history when constructing feedback. Tests now compare the prepared and delivered feedback and verify the exact number of earlier failures in each recovery round. The entire comparison was rerun; no individual trial was selectively replaced.
- Diagnostic batch: 307,349 tokens, 41 calls, provider-reported cost $0.092757. Corrected batch: 348,076 tokens, 47 calls, $0.102141. Combined actual usage: 655,425 tokens, 88 calls, $0.194898. Only the corrected 30 trials contribute to the main results.

[Excluded diagnostic details](../../../runs/cap-revision4-haiku55/20261019-224320/analysis.md) · [Validation logs](../../../runs/cap-revision4-haiku55/20261019-225124/validation/cap_suite.txt)
