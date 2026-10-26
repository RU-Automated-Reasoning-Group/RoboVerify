# Recovery loop and budgets

[`run_recovery`](../../recovery.py#L879-L1171) is Loop 3: it repeats synthesis and rollout until an attempt succeeds or a limit is reached.

## Loop

For round 0 to `max_rounds` (frozen: 4, so at most 5 attempts) ([`recovery.py`](../../recovery.py#L936-L1094)):

1. Stop if the trial budget is used up.
2. Run one attempt with [`run_once`](../../runner.py#L157), passing the previous round's feedback. Round 0 has no feedback.
3. Collect the attempt and flag a candidate identical to an earlier one. It still runs; repeats are recorded, not skipped.
4. If `success_field` is true, stop with `success`.
5. Otherwise build feedback ([Feedback builder](feedback_builder.md)) and continue.

All rounds restore the same initial snapshot. If round 0 started from a fresh scene, its saved initial state is reused for later rounds.

## Stop reasons

`success`, `max_recovery_rounds`, `trial_budget_exhausted`, `missing_initial_snapshot`, `missing_essential_evidence`, `demonstration_input_rejected`, or a feedback-builder reason such as `preparation_failure_no_rollout` or `required_feedback_exceeds_token_limit`.

## Budgets

Two layers apply at once ([`Budget`](../../budget.py#L21), [`RoundBudget`](../../recovery.py#L30)):

| Scope | Frozen v1 limits |
| --- | --- |
| Round 0 | 100k tokens, 12 calls, 3,600 s |
| Each recovery round | 200k tokens, 16 calls, 3,600 s |
| Whole trial | 900k tokens, 76 calls, 18,000 s, 50,000 execution and simulation steps |
| Each rollout | 10,000 execution ticks, 10,000 simulation steps |

A request must fit both the round and the trial budget; the round budget reserves tokens before each call and reconciles them with the reported usage.

## Result

[`recovery_result.json`](../../recovery.py#L1096-L1159) reports first-attempt and final success, the round of the first `task_spec_success`, rounds used, rollouts, stop reason, every attempt's result, and total calls, tokens and time.
