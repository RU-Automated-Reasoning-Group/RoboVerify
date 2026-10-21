# Revision 4.2: primitive prompt and feedback experiment

**Status: complete; 220/220 trials finished.** Pending trials are not observed failures.

MiniMax M3.1 Flash free; medium reasoning; 16,384 output tokens; Stack with three blocks; `dsl_only`; `typed_v3` plus the selected primitive prompt supplement. All trials enable recovery, with up to four repairs/five rollouts from the same full initial state.
The 60 full-treatment trials cover all three methods. The 160 ablation trials cover Method 1/2 with one Section 3 candidate group removed. The full groups are reused as controls; no duplicate control cohort is added.

[Immutable manifest](../../../runs/cap-revision42/20261021-174848-minimax-220/manifest.json) · [Configuration and source hashes](../../../runs/cap-revision42/20261021-174848-minimax-220/comparison_plan.json) · [Machine-readable analysis](../../../runs/cap-revision42/20261021-174848-minimax-220/revision42_analysis.json) · [Per-trial CSV](../../../runs/cap-revision42/20261021-174848-minimax-220/revision42_trials.csv)

## 1. Main comparison

| Method | Finished / planned | Initial success | Final success | Initial nonconvergence | Final nonconvergence | # of tokens used |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 20/20 | 8/20 | 18/20 | 6/20 | 1/20 | 393,895 |
| Method 1 | 20/20 | 6/20 | 18/20 | 7/20 | 2/20 | 397,139 |
| Method 2 | 20/20 | 10/20 | 20/20 | 5/20 | 0/20 | 353,868 |

Ratios use finished trials; planned counts remain visible until completion. Nonconvergence includes Pick, Move and Release, not all runtime failures.

![Full treatment outcomes](plots/revision42_full_outcomes.png)

## 2. Candidate-level ablations

`no_prompt` disables all four general-prompt additions. `no_target`, `no_phases`, and `no_context` each disable only their named feedback enrichment. Legacy API text and legacy feedback facts remain present.

| Method | Profile | Finished / planned | Final success | Initial nonconvergence | Final nonconvergence | Any nonconvergence | Nonconverged / executed rollouts | Tokens |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Method 1 | full | 20/20 | 18/20 | 7/20 | 2/20 | 8/20 | 15/41 | 397,139 |
| Method 1 | no_prompt | 20/20 | 19/20 | 14/20 | 0/20 | 16/20 | 24/56 | 481,067 |
| Method 1 | no_target | 20/20 | 19/20 | 7/20 | 0/20 | 8/20 | 10/37 | 349,255 |
| Method 1 | no_phases | 20/20 | 20/20 | 7/20 | 0/20 | 8/20 | 11/36 | 357,175 |
| Method 1 | no_context | 20/20 | 20/20 | 4/20 | 0/20 | 5/20 | 5/32 | 267,157 |
| Method 2 | full | 20/20 | 20/20 | 5/20 | 0/20 | 8/20 | 10/35 | 353,868 |
| Method 2 | no_prompt | 20/20 | 16/20 | 11/20 | 2/20 | 16/20 | 27/64 | 599,750 |
| Method 2 | no_target | 20/20 | 20/20 | 6/20 | 0/20 | 9/20 | 15/41 | 428,629 |
| Method 2 | no_phases | 20/20 | 18/20 | 5/20 | 1/20 | 5/20 | 10/35 | 366,233 |
| Method 2 | no_context | 20/20 | 19/20 | 6/20 | 0/20 | 7/20 | 10/33 | 383,863 |

![Ablation task success](plots/revision42_ablation_success.png)

![Ablation nonconvergence](plots/revision42_ablation_nonconvergence.png)

## 3. Recovery transitions and failure exposure

| Method | Profile | Nonconv → nonconv / eligible transitions | Same primitive | Next task success | Next other execution failure | Next preparation failure | Terminal nonconv without next attempt | Pick / Move / Release failed rollouts |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Baseline | full | 3/9 | 1 | 6 | 0 | 0 | 1 | 3 / 6 / 1 |
| Method 1 | full | 7/13 | 3 | 5 | 1 | 0 | 2 | 2 / 9 / 4 |
| Method 1 | no_context | 0/5 | 0 | 3 | 2 | 0 | 0 | 1 / 3 / 1 |
| Method 1 | no_phases | 3/11 | 2 | 6 | 2 | 0 | 0 | 4 / 4 / 3 |
| Method 1 | no_prompt | 7/24 | 5 | 12 | 5 | 0 | 0 | 4 / 19 / 1 |
| Method 1 | no_target | 2/10 | 1 | 6 | 2 | 0 | 0 | 6 / 2 / 2 |
| Method 2 | full | 2/10 | 1 | 8 | 0 | 0 | 0 | 3 / 6 / 1 |
| Method 2 | no_context | 3/10 | 1 | 6 | 1 | 0 | 0 | 4 / 5 / 1 |
| Method 2 | no_phases | 5/9 | 4 | 3 | 1 | 0 | 1 | 4 / 4 / 2 |
| Method 2 | no_prompt | 8/25 | 3 | 9 | 8 | 0 | 2 | 5 / 20 / 2 |
| Method 2 | no_target | 6/15 | 3 | 9 | 0 | 0 | 0 | 6 / 8 / 1 |

A transition is eligible only when an actual next attempt exists. Repeated failure labels do not establish the same defect, and disappearance of nonconvergence is not necessarily success. Attempts/transitions within a trial are dependent; these plots do not assign independent-sample confidence intervals to them.

## 4. Validation, resources, and comparison scope

- Artifact audit issues: **0**; verified delivered recovery payloads: 226; saved videos: 446. Independent video decoding checks: enabled.
- Reported tokens: 4,378,031; calls: 493; calls without returned usage: 0. Reasoning is included in output usage once.
- Source compliance, physical task success, and logical postconditions remain distinct. Baseline retains the pinned original executor and no pre-execution DSL gate. No preparation/API repair mechanism was added.
- GPU assignments are fixed through each trial; request pacing and quota reservations are independent controls. Credential values are absent from manifests/logs; only environment-variable identities are recorded.
- All groups use independent policy generations on one fixed scene. Ablating general prompts can change the initial candidate; this is an end-to-end ablation, not a paired repair comparison from identical failed programs.
- Historical MiniMax recovery-on results were Baseline 42/50, Method 1 40/50, Method 2 40/50 final success; initial nonconvergence 24/50, 28/50, 19/50; final nonconvergence 3/50, 6/50, 7/50. These are a separate historical cohort with different execution dates/request concurrency, not a contemporaneous control.
- The 11-setting scripted physical preflight and minimal credential probes are excluded from the 220 live trials. Full validation logs and preflight audits are retained with the experiment artifacts.
- Requested workers: 18; observed peak active workers: 14. [GPU/resource telemetry](../../../runs/cap-revision42/20261021-174848-minimax-220/execution_metadata.json)
- Wire requests by credential environment: `{"OPENROUTER_API_KEY": 361, "OPENROUTER_API_KEY_2": 44, "OPENROUTER_API_KEY_3": 44, "OPENROUTER_API_KEY_4": 44}`.

## 5. Interpretation of the completed results

- **Full treatment:** 56/60 tasks succeed: Baseline 18/20, Method 1 18/20, and Method 2 20/20. The eight ablation groups contribute another 151/160 successes. The overall 207/220 is a workload total across different treatments, not a single-method success estimate.
- **The clearest descriptive prompt effect is in the initial attempt.** Removing the general-prompt additions raises initial nonconvergence from 7/20 to 14/20 in Method 1 and from 5/20 to 11/20 in Method 2. Initial success falls from 6/20 to 2/20 and from 10/20 to 3/20, respectively. Both comparisons keep the same feedback configuration, which has not yet been delivered at initial generation.
- **The full configuration does not uniformly dominate the feedback ablations.** Method 1 finishes at 18/20 with all features, versus 19/20 without target derivation and 20/20 without phase diagnostics or action context. Method 2 finishes at 20/20 full and without targets, 18/20 without phases, and 19/20 without context. With 20 trials per setting and independently generated initial candidates, these small final-success differences do not establish a reliable ranking of the feedback components.
- **Identical initial prompts were verified** across full/no_target/no_phases/no_context within each method. Therefore their differences in initial success/nonconvergence are sampling outcomes, not effects of feedback that was never sent before the first rollout. The no_prompt initial prompts also match the historical typed_v3 prompts byte-for-byte for Method 1/2.

### 5.1 Exploratory pooled initial-prompt comparison

The following pools the four prompt-on profiles within each method **only for the initial attempt**, before their feedback settings can apply. It is an additional descriptive analysis, not a replacement for the prespecified per-profile results or a paired repair experiment.

| Method | General prompt | Initial success | Initial nonconvergence |
| --- | --- | ---: | ---: |
| Method 1 | On, four profiles pooled | 37/80 (46.2%) | 25/80 (31.2%) |
| Method 1 | Off | 2/20 (10.0%) | 14/20 (70.0%) |
| Method 2 | On, four profiles pooled | 42/80 (52.5%) | 22/80 (27.5%) |
| Method 2 | Off | 3/20 (15.0%) | 11/20 (55.0%) |

### 5.2 Failures and budget effects

| Method | Profile | Final nonconvergence | Final execution/query budget stop | Final goal failure | Preparation stop | Feedback-size stop |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | full | 1 | 0 | 0 | 1 | 1 |
| Method 1 | full | 2 | 0 | 0 | 0 | 0 |
| Method 1 | no_context | 0 | 0 | 0 | 0 | 0 |
| Method 1 | no_phases | 0 | 0 | 0 | 0 | 0 |
| Method 1 | no_prompt | 0 | 1 | 0 | 0 | 0 |
| Method 1 | no_target | 0 | 1 | 0 | 0 | 0 |
| Method 2 | full | 0 | 0 | 0 | 0 | 0 |
| Method 2 | no_context | 0 | 0 | 0 | 1 | 0 |
| Method 2 | no_phases | 1 | 0 | 1 | 0 | 0 |
| Method 2 | no_prompt | 2 | 1 | 1 | 0 | 0 |
| Method 2 | no_target | 0 | 0 | 0 | 0 | 0 |

The feedback-size column overlaps final nonconvergence: that trial ended with a failed Pick, then stopped while preparing another feedback request. All other failure columns distinguish their terminal outcome/stop paths.
- Baseline full trial 2 stops before rollout on a helper/source SyntaxError. Its original preparation behavior is preserved.
- Method 2 no_context trial 1 stops during initial preparation because a request reservation exceeds the fixed round token allowance. Its 19/20 final result is not evidence that removing action context prevented repair of an executed failure; this trial never reached that stage.
- Baseline full trial 20 executes three unsuccessful candidates (execution/query budget failure, NameError, Pick nonconvergence), then stops with `required_feedback_exceeds_token_limit`. Offline reconstruction of the same saved attempt gives **8,300 bytes** for the combined required diagnostics plus metadata, even with zero optional action-context records and zero optional legacy supporting events, exceeding the **8,192-byte cap by 108 bytes**. The legacy selector on the same recorded events fits at 8,061 bytes. This is a feedback-packing limitation of the new combination, not evidence that the model used all four repairs unsuccessfully. No additional candidate was generated or rerun.
[Feedback-size diagnostic](../../../runs/cap-revision42/20261021-174848-minimax-220/feedback_budget_diagnostic.json) · [Affected trial](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_google_baseline-dsl_only-020-recovery-on-full/cap/20261021-184244-29d9ebf-cap_google_baseline-recovery-b33d7805)
- Delivered action context also depends on the shared byte cap: of 126 feedback payloads with action-context enrichment enabled, 125 have context candidates, 118 include at least one selected action, 98 omit at least one candidate under the cap, and 7 omit every candidate. One payload has no eligible candidate. These omissions are explicit. Removing another enrichment can free space for more context or legacy events, so these are ablations of the bounded end-to-end system, not strictly identical payloads with only one block deleted.
[Actual feedback-feature coverage](../../../runs/cap-revision42/20261021-174848-minimax-220/feedback_feature_coverage.json)
- A follow-up should improve packing of overlapping facts/metadata without increasing budgets silently, and use matched failed programs/states for direct repair comparisons. These observations motivate further work; no post-run implementation change or selective rerun was made.

### 5.3 Historical comparison and scope

| Method | Earlier MiniMax final success | Current full final success | Earlier initial nonconvergence | Current full initial nonconvergence | Earlier final nonconvergence | Current full final nonconvergence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 42/50 (84%) | 18/20 (90%) | 24/50 (48%) | 6/20 (30%) | 3/50 (6%) | 1/20 (5%) |
| Method 1 | 40/50 (80%) | 18/20 (90%) | 28/50 (56%) | 7/20 (35%) | 6/50 (12%) | 2/20 (10%) |
| Method 2 | 40/50 (80%) | 20/20 (100%) | 19/50 (38%) | 5/20 (25%) | 7/50 (14%) | 0/20 (0%) |

The [earlier recovery report](revision_4_recovery_report.md) supplies the historical cohort. These are fresh generations on the same fixed Stack scene, with different dates, request concurrency, and credential routing. They are not paired trials or a contemporaneous all-features-off control. Twenty trials on one scene do not establish robustness across layouts/tasks or that each new component helps independently.

## 6. Completed validation and reproducibility

- Implementation commit: `29d9ebf`. All **62 frozen Python/JSON source files** matched their recorded hashes at batch completion. Runtime changes were not made during the batch.
- After completion, the four prompt blocks and heading were relocated to [prompts/primitive_v1.json](../prompts/primitive_v1.json). All 16 supplement renderings/provenance records and 96 complete API renderings remain identical; 13 focused prompt tests pass. This resource-only refactor is subsequent to the recorded experiment source snapshot.
- Tests: **189 passed, one opt-in hardware test skipped** (190 discovered, including the controller suite); focused feature/report/request-control tests: **24 passed**. Final scripted preflight: **11/11 recover successfully**, with 22 rollouts and zero artifact-audit issues.
- Live data: **220/220 trials**, exactly **20 in each of 11 settings**; **446 prepared attempts**, **444 actual rollouts**, **446 exact initial-state matches**, **446 independently checked videos**, and **226 matching delivered recovery payloads**. Two initial preparation failures remain in the trial denominator.
- All **207 successful final policies pass the recorded DSL audit**. There are no observed caught/masked primitive failures that escape the nonconvergence classification. Source validity and physical task success remain separate metrics.
- Usage: **493 model calls / 493 wire requests**, **4,378,031 reported tokens**, **zero missing-usage calls**, and **provider-reported cost $0.00**. The four minimal compatibility probes used a separate 913 tokens at reported cost zero and are excluded.
- Credential request counts: original key environment **361**; the three additional credential environments **44 each**. Values remain outside repository artifacts. Daily caps and pacing were respected; no trial was replaced.
- Resource observations: peak **14 active workers** under an 18-worker ceiling; first dispatch to last worker exit **58.10 minutes**; sampled maximum worker RSS **1023.63 MiB**, sampled attributed GPU memory **142.52 MiB**. Quota reservations and request latency constrain throughput even when GPU memory is available.
[Final experiment audit](../../../runs/cap-revision42/20261021-174848-minimax-220/final_experiment_audit.json) · [Per-trial/video audit](../../../runs/cap-revision42/20261021-174848-minimax-220/revision42_audit.json) · [Final preflight audit](../../../runs/cap-revision42-validation/20261021-final-preflight/revision42_audit.json) · [Validation logs](../../../runs/cap-revision42-validation/20261021-final-preflight/validation)

The numerical tables and figures are generated by `synthesis.cap.revision42_report`. This interpretation is retained separately as [final_interpretation.md](../../../runs/cap-revision42/20261021-174848-minimax-220/final_interpretation.md) so it can be restored after regenerating numerical tables.

## 7. Per-trial results

| Method | Profile | Trial | Initial status | Final status | Success | # of tokens used | Calls | Artifacts |
| --- | --- | ---: | --- | --- | --- | ---: | ---: | --- |
| Baseline | full | 1 | primitive_failed | completed | True | 14973 | 2 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_google_baseline-dsl_only-001-recovery-on-full/cap/20261021-174853-29d9ebf-cap_google_baseline-recovery-e2f7152e) |
| Baseline | full | 2 | failed | failed | False | 7707 | 1 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_google_baseline-dsl_only-002-recovery-on-full/cap/20261021-175107-29d9ebf-cap_google_baseline-recovery-574d3ab6) |
| Baseline | full | 3 | completed | completed | True | 10626 | 2 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_google_baseline-dsl_only-003-recovery-on-full/cap/20261021-175159-29d9ebf-cap_google_baseline-recovery-55570028) |
| Baseline | full | 4 | completed | completed | True | 7282 | 1 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_google_baseline-dsl_only-004-recovery-on-full/cap/20261021-175424-29d9ebf-cap_google_baseline-recovery-6229987a) |
| Baseline | full | 5 | completed | completed | True | 6469 | 1 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_google_baseline-dsl_only-005-recovery-on-full/cap/20261021-175646-29d9ebf-cap_google_baseline-recovery-a197caad) |
| Baseline | full | 6 | completed | completed | True | 9146 | 2 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_google_baseline-dsl_only-006-recovery-on-full/cap/20261021-175905-29d9ebf-cap_google_baseline-recovery-3c224538) |
| Baseline | full | 7 | goal_failed | completed | True | 38905 | 5 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_google_baseline-dsl_only-007-recovery-on-full/cap/20261021-175940-29d9ebf-cap_google_baseline-recovery-89356a75) |
| Baseline | full | 8 | primitive_failed | completed | True | 24080 | 4 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_google_baseline-dsl_only-008-recovery-on-full/cap/20261021-180231-29d9ebf-cap_google_baseline-recovery-a71bb449) |
| Baseline | full | 9 | primitive_failed | completed | True | 24145 | 3 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_google_baseline-dsl_only-009-recovery-on-full/cap/20261021-180606-29d9ebf-cap_google_baseline-recovery-2a15c66b) |
| Baseline | full | 10 | primitive_failed | completed | True | 19520 | 2 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_google_baseline-dsl_only-010-recovery-on-full/cap/20261021-180723-29d9ebf-cap_google_baseline-recovery-5f997bc1) |
| Baseline | full | 11 | primitive_failed | completed | True | 28683 | 3 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_google_baseline-dsl_only-011-recovery-on-full/cap/20261021-180952-29d9ebf-cap_google_baseline-recovery-1bc6732a) |
| Baseline | full | 12 | goal_failed | completed | True | 17255 | 2 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_google_baseline-dsl_only-012-recovery-on-full/cap/20261021-181332-29d9ebf-cap_google_baseline-recovery-e027c859) |
| Baseline | full | 13 | completed | completed | True | 10374 | 2 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_google_baseline-dsl_only-013-recovery-on-full/cap/20261021-181608-29d9ebf-cap_google_baseline-recovery-18273342) |
| Baseline | full | 14 | completed | completed | True | 6552 | 1 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_google_baseline-dsl_only-014-recovery-on-full/cap/20261021-182114-29d9ebf-cap_google_baseline-recovery-e5b91f93) |
| Baseline | full | 15 | completed | completed | True | 16573 | 2 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_google_baseline-dsl_only-015-recovery-on-full/cap/20261021-182522-29d9ebf-cap_google_baseline-recovery-55579b3e) |
| Baseline | full | 16 | primitive_failed | completed | True | 44326 | 4 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_google_baseline-dsl_only-016-recovery-on-full/cap/20261021-182651-29d9ebf-cap_google_baseline-recovery-64676157) |
| Baseline | full | 17 | completed | completed | True | 17832 | 2 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_google_baseline-dsl_only-017-recovery-on-full/cap/20261021-183134-29d9ebf-cap_google_baseline-recovery-cf5e9069) |
| Baseline | full | 18 | budget_exhausted | completed | True | 40501 | 5 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_google_baseline-dsl_only-018-recovery-on-full/cap/20261021-183237-29d9ebf-cap_google_baseline-recovery-5f41f3d6) |
| Baseline | full | 19 | failed | completed | True | 16780 | 2 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_google_baseline-dsl_only-019-recovery-on-full/cap/20261021-183824-29d9ebf-cap_google_baseline-recovery-4a15d562) |
| Baseline | full | 20 | budget_exhausted | primitive_failed | False | 32166 | 4 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_google_baseline-dsl_only-020-recovery-on-full/cap/20261021-184244-29d9ebf-cap_google_baseline-recovery-b33d7805) |
| Method 1 | full | 1 | budget_exhausted | primitive_failed | False | 67712 | 6 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_1-dsl_only-001-recovery-on-full/cap/20261021-174851-29d9ebf-cap_method_1-recovery-e5eade75) |
| Method 1 | full | 2 | primitive_failed | completed | True | 15730 | 2 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_1-dsl_only-002-recovery-on-full/cap/20261021-175006-29d9ebf-cap_method_1-recovery-ffa13d52) |
| Method 1 | full | 3 | budget_exhausted | completed | True | 20360 | 2 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_1-dsl_only-003-recovery-on-full/cap/20261021-175145-29d9ebf-cap_method_1-recovery-2e00dbde) |
| Method 1 | full | 4 | goal_failed | completed | True | 17074 | 2 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_1-dsl_only-004-recovery-on-full/cap/20261021-175338-29d9ebf-cap_method_1-recovery-f7eae9b4) |
| Method 1 | full | 5 | completed | completed | True | 8019 | 1 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_1-dsl_only-005-recovery-on-full/cap/20261021-175604-29d9ebf-cap_method_1-recovery-081664b7) |
| Method 1 | full | 6 | completed | completed | True | 7664 | 1 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_1-dsl_only-006-recovery-on-full/cap/20261021-175804-29d9ebf-cap_method_1-recovery-3dda6f13) |
| Method 1 | full | 7 | completed | completed | True | 9327 | 1 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_1-dsl_only-007-recovery-on-full/cap/20261021-180219-29d9ebf-cap_method_1-recovery-060395d5) |
| Method 1 | full | 8 | primitive_failed | completed | True | 16818 | 2 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_1-dsl_only-008-recovery-on-full/cap/20261021-180351-29d9ebf-cap_method_1-recovery-e6c6c3d1) |
| Method 1 | full | 9 | completed | completed | True | 8200 | 1 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_1-dsl_only-009-recovery-on-full/cap/20261021-180556-29d9ebf-cap_method_1-recovery-1ebc2be0) |
| Method 1 | full | 10 | budget_exhausted | completed | True | 16250 | 2 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_1-dsl_only-010-recovery-on-full/cap/20261021-180926-29d9ebf-cap_method_1-recovery-b2c6c35c) |
| Method 1 | full | 11 | completed | completed | True | 6783 | 1 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_1-dsl_only-011-recovery-on-full/cap/20261021-180957-29d9ebf-cap_method_1-recovery-f56dc733) |
| Method 1 | full | 12 | completed | completed | True | 7098 | 1 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_1-dsl_only-012-recovery-on-full/cap/20261021-181428-29d9ebf-cap_method_1-recovery-5242ff2f) |
| Method 1 | full | 13 | primitive_failed | completed | True | 31436 | 3 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_1-dsl_only-013-recovery-on-full/cap/20261021-181753-29d9ebf-cap_method_1-recovery-e739b2de) |
| Method 1 | full | 14 | primitive_failed | primitive_failed | False | 60541 | 5 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_1-dsl_only-014-recovery-on-full/cap/20261021-182009-29d9ebf-cap_method_1-recovery-f43a8f50) |
| Method 1 | full | 15 | budget_exhausted | completed | True | 16628 | 2 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_1-dsl_only-015-recovery-on-full/cap/20261021-182322-29d9ebf-cap_method_1-recovery-ed06f57b) |
| Method 1 | full | 16 | budget_exhausted | completed | True | 18395 | 2 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_1-dsl_only-016-recovery-on-full/cap/20261021-182703-29d9ebf-cap_method_1-recovery-86b4c39a) |
| Method 1 | full | 17 | budget_exhausted | completed | True | 15992 | 2 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_1-dsl_only-017-recovery-on-full/cap/20261021-183056-29d9ebf-cap_method_1-recovery-2715eee9) |
| Method 1 | full | 18 | primitive_failed | completed | True | 16627 | 2 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_1-dsl_only-018-recovery-on-full/cap/20261021-183756-29d9ebf-cap_method_1-recovery-a989945a) |
| Method 1 | full | 19 | primitive_failed | completed | True | 17133 | 2 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_1-dsl_only-019-recovery-on-full/cap/20261021-183838-29d9ebf-cap_method_1-recovery-1f6afd4a) |
| Method 1 | full | 20 | primitive_failed | completed | True | 19352 | 2 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_1-dsl_only-020-recovery-on-full/cap/20261021-184411-29d9ebf-cap_method_1-recovery-4be0ca77) |
| Method 1 | no_prompt | 1 | budget_exhausted | budget_exhausted | False | 41401 | 5 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_1-dsl_only-001-recovery-on-no_prompt/cap/20261021-174851-29d9ebf-cap_method_1-recovery-78449cd3) |
| Method 1 | no_prompt | 2 | primitive_failed | completed | True | 16833 | 2 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_1-dsl_only-002-recovery-on-no_prompt/cap/20261021-174853-29d9ebf-cap_method_1-recovery-dc30848e) |
| Method 1 | no_prompt | 3 | primitive_failed | completed | True | 22780 | 3 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_1-dsl_only-003-recovery-on-no_prompt/cap/20261021-175223-29d9ebf-cap_method_1-recovery-d79f7ee9) |
| Method 1 | no_prompt | 4 | completed | completed | True | 10548 | 2 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_1-dsl_only-004-recovery-on-no_prompt/cap/20261021-175459-29d9ebf-cap_method_1-recovery-cca90ae0) |
| Method 1 | no_prompt | 5 | primitive_failed | completed | True | 20554 | 2 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_1-dsl_only-005-recovery-on-no_prompt/cap/20261021-175634-29d9ebf-cap_method_1-recovery-1fbd24b0) |
| Method 1 | no_prompt | 6 | budget_exhausted | completed | True | 33135 | 4 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_1-dsl_only-006-recovery-on-no_prompt/cap/20261021-175921-29d9ebf-cap_method_1-recovery-53184678) |
| Method 1 | no_prompt | 7 | primitive_failed | completed | True | 21118 | 3 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_1-dsl_only-007-recovery-on-no_prompt/cap/20261021-180222-29d9ebf-cap_method_1-recovery-166b4a25) |
| Method 1 | no_prompt | 8 | primitive_failed | completed | True | 15420 | 2 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_1-dsl_only-008-recovery-on-no_prompt/cap/20261021-180439-29d9ebf-cap_method_1-recovery-ffd60660) |
| Method 1 | no_prompt | 9 | primitive_failed | completed | True | 14515 | 2 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_1-dsl_only-009-recovery-on-no_prompt/cap/20261021-180655-29d9ebf-cap_method_1-recovery-cc1516f0) |
| Method 1 | no_prompt | 10 | primitive_failed | completed | True | 24133 | 3 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_1-dsl_only-010-recovery-on-no_prompt/cap/20261021-180857-29d9ebf-cap_method_1-recovery-c84af19d) |
| Method 1 | no_prompt | 11 | primitive_failed | completed | True | 40812 | 4 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_1-dsl_only-011-recovery-on-no_prompt/cap/20261021-181158-29d9ebf-cap_method_1-recovery-3c06eb22) |
| Method 1 | no_prompt | 12 | goal_failed | completed | True | 27170 | 3 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_1-dsl_only-012-recovery-on-no_prompt/cap/20261021-181432-29d9ebf-cap_method_1-recovery-f2ae9815) |
| Method 1 | no_prompt | 13 | primitive_failed | completed | True | 13204 | 2 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_1-dsl_only-013-recovery-on-no_prompt/cap/20261021-181734-29d9ebf-cap_method_1-recovery-77ff34ce) |
| Method 1 | no_prompt | 14 | primitive_failed | completed | True | 31687 | 4 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_1-dsl_only-014-recovery-on-no_prompt/cap/20261021-181810-29d9ebf-cap_method_1-recovery-7bbc58b9) |
| Method 1 | no_prompt | 15 | completed | completed | True | 5350 | 1 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_1-dsl_only-015-recovery-on-no_prompt/cap/20261021-182152-29d9ebf-cap_method_1-recovery-2665ab24) |
| Method 1 | no_prompt | 16 | primitive_failed | completed | True | 15354 | 2 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_1-dsl_only-016-recovery-on-no_prompt/cap/20261021-182751-29d9ebf-cap_method_1-recovery-5b46dc1d) |
| Method 1 | no_prompt | 17 | primitive_failed | completed | True | 50918 | 6 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_1-dsl_only-017-recovery-on-no_prompt/cap/20261021-182925-29d9ebf-cap_method_1-recovery-aefbc7c4) |
| Method 1 | no_prompt | 18 | primitive_failed | completed | True | 24024 | 3 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_1-dsl_only-018-recovery-on-no_prompt/cap/20261021-183413-29d9ebf-cap_method_1-recovery-051ac791) |
| Method 1 | no_prompt | 19 | primitive_failed | completed | True | 33206 | 4 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_1-dsl_only-019-recovery-on-no_prompt/cap/20261021-184124-29d9ebf-cap_method_1-recovery-198aab78) |
| Method 1 | no_prompt | 20 | goal_failed | completed | True | 18905 | 2 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_1-dsl_only-020-recovery-on-no_prompt/cap/20261021-184406-29d9ebf-cap_method_1-recovery-0b5aebe8) |
| Method 1 | no_target | 1 | primitive_failed | completed | True | 16333 | 2 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_1-dsl_only-001-recovery-on-no_target/cap/20261021-174853-29d9ebf-cap_method_1-recovery-522e0afe) |
| Method 1 | no_target | 2 | primitive_failed | completed | True | 18033 | 2 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_1-dsl_only-002-recovery-on-no_target/cap/20261021-175004-29d9ebf-cap_method_1-recovery-baeb4e50) |
| Method 1 | no_target | 3 | budget_exhausted | completed | True | 25837 | 3 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_1-dsl_only-003-recovery-on-no_target/cap/20261021-175131-29d9ebf-cap_method_1-recovery-d5b2f55a) |
| Method 1 | no_target | 4 | completed | completed | True | 8081 | 1 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_1-dsl_only-004-recovery-on-no_target/cap/20261021-175529-29d9ebf-cap_method_1-recovery-b603e12c) |
| Method 1 | no_target | 5 | primitive_failed | completed | True | 22997 | 2 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_1-dsl_only-005-recovery-on-no_target/cap/20261021-175734-29d9ebf-cap_method_1-recovery-6a44d7d6) |
| Method 1 | no_target | 6 | completed | completed | True | 7676 | 1 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_1-dsl_only-006-recovery-on-no_target/cap/20261021-175927-29d9ebf-cap_method_1-recovery-02fe4288) |
| Method 1 | no_target | 7 | primitive_failed | completed | True | 19226 | 2 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_1-dsl_only-007-recovery-on-no_target/cap/20261021-180027-29d9ebf-cap_method_1-recovery-be8c919f) |
| Method 1 | no_target | 8 | primitive_failed | completed | True | 22794 | 2 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_1-dsl_only-008-recovery-on-no_target/cap/20261021-180312-29d9ebf-cap_method_1-recovery-9bff15e1) |
| Method 1 | no_target | 9 | budget_exhausted | completed | True | 16387 | 2 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_1-dsl_only-009-recovery-on-no_target/cap/20261021-180650-29d9ebf-cap_method_1-recovery-140aba58) |
| Method 1 | no_target | 10 | completed | completed | True | 7820 | 1 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_1-dsl_only-010-recovery-on-no_target/cap/20261021-180850-29d9ebf-cap_method_1-recovery-19d1621f) |
| Method 1 | no_target | 11 | completed | completed | True | 7850 | 1 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_1-dsl_only-011-recovery-on-no_target/cap/20261021-181158-29d9ebf-cap_method_1-recovery-00953a11) |
| Method 1 | no_target | 12 | completed | completed | True | 7970 | 1 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_1-dsl_only-012-recovery-on-no_target/cap/20261021-181444-29d9ebf-cap_method_1-recovery-66203c0f) |
| Method 1 | no_target | 13 | primitive_failed | completed | True | 19007 | 2 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_1-dsl_only-013-recovery-on-no_target/cap/20261021-181722-29d9ebf-cap_method_1-recovery-ef461fe7) |
| Method 1 | no_target | 14 | budget_exhausted | completed | True | 32244 | 4 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_1-dsl_only-014-recovery-on-no_target/cap/20261021-181935-29d9ebf-cap_method_1-recovery-99173550) |
| Method 1 | no_target | 15 | budget_exhausted | budget_exhausted | False | 51108 | 6 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_1-dsl_only-015-recovery-on-no_target/cap/20261021-182158-29d9ebf-cap_method_1-recovery-014da07f) |
| Method 1 | no_target | 16 | completed | completed | True | 7343 | 1 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_1-dsl_only-016-recovery-on-no_target/cap/20261021-182608-29d9ebf-cap_method_1-recovery-99697494) |
| Method 1 | no_target | 17 | completed | completed | True | 10974 | 1 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_1-dsl_only-017-recovery-on-no_target/cap/20261021-183047-29d9ebf-cap_method_1-recovery-ecfda474) |
| Method 1 | no_target | 18 | completed | completed | True | 6232 | 1 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_1-dsl_only-018-recovery-on-no_target/cap/20261021-183529-29d9ebf-cap_method_1-recovery-1150fb5f) |
| Method 1 | no_target | 19 | completed | completed | True | 6301 | 1 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_1-dsl_only-019-recovery-on-no_target/cap/20261021-184030-29d9ebf-cap_method_1-recovery-7b185811) |
| Method 1 | no_target | 20 | primitive_failed | completed | True | 35042 | 3 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_1-dsl_only-020-recovery-on-no_target/cap/20261021-184152-29d9ebf-cap_method_1-recovery-0f32c915) |
| Method 1 | no_phases | 1 | completed | completed | True | 7296 | 1 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_1-dsl_only-001-recovery-on-no_phases/cap/20261021-174850-29d9ebf-cap_method_1-recovery-347cd606) |
| Method 1 | no_phases | 2 | completed | completed | True | 7756 | 1 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_1-dsl_only-002-recovery-on-no_phases/cap/20261021-175008-29d9ebf-cap_method_1-recovery-df536c5d) |
| Method 1 | no_phases | 3 | completed | completed | True | 7183 | 1 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_1-dsl_only-003-recovery-on-no_phases/cap/20261021-175214-29d9ebf-cap_method_1-recovery-b0a99d4f) |
| Method 1 | no_phases | 4 | primitive_failed | completed | True | 25518 | 3 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_1-dsl_only-004-recovery-on-no_phases/cap/20261021-175333-29d9ebf-cap_method_1-recovery-99757255) |
| Method 1 | no_phases | 5 | completed | completed | True | 8518 | 1 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_1-dsl_only-005-recovery-on-no_phases/cap/20261021-175617-29d9ebf-cap_method_1-recovery-7234d236) |
| Method 1 | no_phases | 6 | primitive_failed | completed | True | 15177 | 2 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_1-dsl_only-006-recovery-on-no_phases/cap/20261021-175747-29d9ebf-cap_method_1-recovery-9a20187c) |
| Method 1 | no_phases | 7 | goal_failed | completed | True | 18597 | 2 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_1-dsl_only-007-recovery-on-no_phases/cap/20261021-180007-29d9ebf-cap_method_1-recovery-7c20f7a3) |
| Method 1 | no_phases | 8 | completed | completed | True | 13425 | 2 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_1-dsl_only-008-recovery-on-no_phases/cap/20261021-180236-29d9ebf-cap_method_1-recovery-876e7ce7) |
| Method 1 | no_phases | 9 | primitive_failed | completed | True | 19957 | 3 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_1-dsl_only-009-recovery-on-no_phases/cap/20261021-180545-29d9ebf-cap_method_1-recovery-fa42e028) |
| Method 1 | no_phases | 10 | budget_exhausted | completed | True | 27217 | 3 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_1-dsl_only-010-recovery-on-no_phases/cap/20261021-180807-29d9ebf-cap_method_1-recovery-bd5ade1d) |
| Method 1 | no_phases | 11 | primitive_failed | completed | True | 18985 | 2 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_1-dsl_only-011-recovery-on-no_phases/cap/20261021-181005-29d9ebf-cap_method_1-recovery-030101bb) |
| Method 1 | no_phases | 12 | budget_exhausted | completed | True | 16610 | 2 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_1-dsl_only-012-recovery-on-no_phases/cap/20261021-181342-29d9ebf-cap_method_1-recovery-aecc5780) |
| Method 1 | no_phases | 13 | completed | completed | True | 8673 | 1 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_1-dsl_only-013-recovery-on-no_phases/cap/20261021-181622-29d9ebf-cap_method_1-recovery-e576f91c) |
| Method 1 | no_phases | 14 | completed | completed | True | 16922 | 2 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_1-dsl_only-014-recovery-on-no_phases/cap/20261021-181839-29d9ebf-cap_method_1-recovery-96691d23) |
| Method 1 | no_phases | 15 | completed | completed | True | 9401 | 1 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_1-dsl_only-015-recovery-on-no_phases/cap/20261021-182529-29d9ebf-cap_method_1-recovery-6998214b) |
| Method 1 | no_phases | 16 | completed | completed | True | 6901 | 1 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_1-dsl_only-016-recovery-on-no_phases/cap/20261021-182709-29d9ebf-cap_method_1-recovery-82e369c3) |
| Method 1 | no_phases | 17 | primitive_failed | completed | True | 59679 | 6 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_1-dsl_only-017-recovery-on-no_phases/cap/20261021-183156-29d9ebf-cap_method_1-recovery-90694268) |
| Method 1 | no_phases | 18 | primitive_failed | completed | True | 39301 | 4 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_1-dsl_only-018-recovery-on-no_phases/cap/20261021-183731-29d9ebf-cap_method_1-recovery-aaa0a165) |
| Method 1 | no_phases | 19 | primitive_failed | completed | True | 23085 | 2 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_1-dsl_only-019-recovery-on-no_phases/cap/20261021-183937-29d9ebf-cap_method_1-recovery-869284cd) |
| Method 1 | no_phases | 20 | completed | completed | True | 6974 | 1 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_1-dsl_only-020-recovery-on-no_phases/cap/20261021-184206-29d9ebf-cap_method_1-recovery-c518e55a) |
| Method 1 | no_context | 1 | completed | completed | True | 7004 | 1 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_1-dsl_only-001-recovery-on-no_context/cap/20261021-174853-29d9ebf-cap_method_1-recovery-112aa46e) |
| Method 1 | no_context | 2 | completed | completed | True | 7706 | 1 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_1-dsl_only-002-recovery-on-no_context/cap/20261021-175114-29d9ebf-cap_method_1-recovery-fc28f313) |
| Method 1 | no_context | 3 | goal_failed | completed | True | 15968 | 2 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_1-dsl_only-003-recovery-on-no_context/cap/20261021-175158-29d9ebf-cap_method_1-recovery-b0be19c4) |
| Method 1 | no_context | 4 | completed | completed | True | 6830 | 1 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_1-dsl_only-004-recovery-on-no_context/cap/20261021-175416-29d9ebf-cap_method_1-recovery-61128f6c) |
| Method 1 | no_context | 5 | primitive_failed | completed | True | 20787 | 2 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_1-dsl_only-005-recovery-on-no_context/cap/20261021-175642-29d9ebf-cap_method_1-recovery-c389001b) |
| Method 1 | no_context | 6 | goal_failed | completed | True | 15192 | 2 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_1-dsl_only-006-recovery-on-no_context/cap/20261021-175739-29d9ebf-cap_method_1-recovery-b4e8482c) |
| Method 1 | no_context | 7 | completed | completed | True | 5957 | 1 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_1-dsl_only-007-recovery-on-no_context/cap/20261021-180020-29d9ebf-cap_method_1-recovery-a920a985) |
| Method 1 | no_context | 8 | completed | completed | True | 15063 | 2 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_1-dsl_only-008-recovery-on-no_context/cap/20261021-180339-29d9ebf-cap_method_1-recovery-bae54f32) |
| Method 1 | no_context | 9 | completed | completed | True | 7253 | 1 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_1-dsl_only-009-recovery-on-no_context/cap/20261021-180452-29d9ebf-cap_method_1-recovery-dc3c96f3) |
| Method 1 | no_context | 10 | completed | completed | True | 7722 | 1 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_1-dsl_only-010-recovery-on-no_context/cap/20261021-180828-29d9ebf-cap_method_1-recovery-4ad51bfa) |
| Method 1 | no_context | 11 | completed | completed | True | 8512 | 1 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_1-dsl_only-011-recovery-on-no_context/cap/20261021-181053-29d9ebf-cap_method_1-recovery-04386634) |
| Method 1 | no_context | 12 | budget_exhausted | completed | True | 16815 | 2 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_1-dsl_only-012-recovery-on-no_context/cap/20261021-181527-29d9ebf-cap_method_1-recovery-5ec5b0b3) |
| Method 1 | no_context | 13 | completed | completed | True | 5861 | 1 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_1-dsl_only-013-recovery-on-no_context/cap/20261021-181628-29d9ebf-cap_method_1-recovery-ca478b65) |
| Method 1 | no_context | 14 | completed | completed | True | 5911 | 1 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_1-dsl_only-014-recovery-on-no_context/cap/20261021-181902-29d9ebf-cap_method_1-recovery-c68766cd) |
| Method 1 | no_context | 15 | primitive_failed | completed | True | 23500 | 3 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_1-dsl_only-015-recovery-on-no_context/cap/20261021-182333-29d9ebf-cap_method_1-recovery-470a46a1) |
| Method 1 | no_context | 16 | completed | completed | True | 6765 | 1 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_1-dsl_only-016-recovery-on-no_context/cap/20261021-182606-29d9ebf-cap_method_1-recovery-c90223fd) |
| Method 1 | no_context | 17 | budget_exhausted | completed | True | 37056 | 4 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_1-dsl_only-017-recovery-on-no_context/cap/20261021-182940-29d9ebf-cap_method_1-recovery-c0c39587) |
| Method 1 | no_context | 18 | primitive_failed | completed | True | 28732 | 3 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_1-dsl_only-018-recovery-on-no_context/cap/20261021-183246-29d9ebf-cap_method_1-recovery-ce0e8abd) |
| Method 1 | no_context | 19 | completed | completed | True | 6810 | 1 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_1-dsl_only-019-recovery-on-no_context/cap/20261021-184033-29d9ebf-cap_method_1-recovery-ba5f5817) |
| Method 1 | no_context | 20 | primitive_failed | completed | True | 17713 | 2 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_1-dsl_only-020-recovery-on-no_context/cap/20261021-184337-29d9ebf-cap_method_1-recovery-178bc075) |
| Method 2 | full | 1 | completed | completed | True | 7022 | 1 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_2-dsl_only-001-recovery-on-full/cap/20261021-174852-29d9ebf-cap_method_2-recovery-89bd400f) |
| Method 2 | full | 2 | primitive_failed | completed | True | 18075 | 2 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_2-dsl_only-002-recovery-on-full/cap/20261021-174853-29d9ebf-cap_method_2-recovery-8915e59b) |
| Method 2 | full | 3 | completed | completed | True | 16415 | 2 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_2-dsl_only-003-recovery-on-full/cap/20261021-175205-29d9ebf-cap_method_2-recovery-1f0f32fa) |
| Method 2 | full | 4 | completed | completed | True | 6728 | 1 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_2-dsl_only-004-recovery-on-full/cap/20261021-175513-29d9ebf-cap_method_2-recovery-2e080d1c) |
| Method 2 | full | 5 | primitive_failed | completed | True | 31542 | 3 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_2-dsl_only-005-recovery-on-full/cap/20261021-175540-29d9ebf-cap_method_2-recovery-9c71bf26) |
| Method 2 | full | 6 | goal_failed | completed | True | 33248 | 3 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_2-dsl_only-006-recovery-on-full/cap/20261021-175737-29d9ebf-cap_method_2-recovery-8ccf776d) |
| Method 2 | full | 7 | goal_failed | completed | True | 16752 | 2 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_2-dsl_only-007-recovery-on-full/cap/20261021-180027-29d9ebf-cap_method_2-recovery-74010a60) |
| Method 2 | full | 8 | completed | completed | True | 7338 | 1 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_2-dsl_only-008-recovery-on-full/cap/20261021-180345-29d9ebf-cap_method_2-recovery-2b515ca9) |
| Method 2 | full | 9 | budget_exhausted | completed | True | 25708 | 3 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_2-dsl_only-009-recovery-on-full/cap/20261021-180505-29d9ebf-cap_method_2-recovery-a494323c) |
| Method 2 | full | 10 | budget_exhausted | completed | True | 16897 | 2 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_2-dsl_only-010-recovery-on-full/cap/20261021-180741-29d9ebf-cap_method_2-recovery-86c2d7d2) |
| Method 2 | full | 11 | completed | completed | True | 12239 | 2 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_2-dsl_only-011-recovery-on-full/cap/20261021-181242-29d9ebf-cap_method_2-recovery-78049b56) |
| Method 2 | full | 12 | completed | completed | True | 8697 | 1 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_2-dsl_only-012-recovery-on-full/cap/20261021-181319-29d9ebf-cap_method_2-recovery-ad283150) |
| Method 2 | full | 13 | completed | completed | True | 7092 | 1 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_2-dsl_only-013-recovery-on-full/cap/20261021-181807-29d9ebf-cap_method_2-recovery-cc5cdb0e) |
| Method 2 | full | 14 | primitive_failed | completed | True | 27347 | 2 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_2-dsl_only-014-recovery-on-full/cap/20261021-181915-29d9ebf-cap_method_2-recovery-8b53e141) |
| Method 2 | full | 15 | completed | completed | True | 7913 | 1 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_2-dsl_only-015-recovery-on-full/cap/20261021-182535-29d9ebf-cap_method_2-recovery-b8beb6f3) |
| Method 2 | full | 16 | primitive_failed | completed | True | 20839 | 2 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_2-dsl_only-016-recovery-on-full/cap/20261021-182744-29d9ebf-cap_method_2-recovery-d1e68a82) |
| Method 2 | full | 17 | goal_failed | completed | True | 53065 | 5 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_2-dsl_only-017-recovery-on-full/cap/20261021-183015-29d9ebf-cap_method_2-recovery-a70c17cc) |
| Method 2 | full | 18 | completed | completed | True | 8670 | 1 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_2-dsl_only-018-recovery-on-full/cap/20261021-183616-29d9ebf-cap_method_2-recovery-8221cd3a) |
| Method 2 | full | 19 | primitive_failed | completed | True | 19951 | 2 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_2-dsl_only-019-recovery-on-full/cap/20261021-183948-29d9ebf-cap_method_2-recovery-eab89ba0) |
| Method 2 | full | 20 | completed | completed | True | 8330 | 1 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_2-dsl_only-020-recovery-on-full/cap/20261021-184259-29d9ebf-cap_method_2-recovery-341619a2) |
| Method 2 | no_prompt | 1 | primitive_failed | completed | True | 24535 | 3 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_2-dsl_only-001-recovery-on-no_prompt/cap/20261021-174852-29d9ebf-cap_method_2-recovery-335b01ff) |
| Method 2 | no_prompt | 2 | goal_failed | primitive_failed | False | 51281 | 6 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_2-dsl_only-002-recovery-on-no_prompt/cap/20261021-174852-29d9ebf-cap_method_2-recovery-8c52beb5) |
| Method 2 | no_prompt | 3 | budget_exhausted | completed | True | 43372 | 5 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_2-dsl_only-003-recovery-on-no_prompt/cap/20261021-175209-29d9ebf-cap_method_2-recovery-0d8d8a8f) |
| Method 2 | no_prompt | 4 | primitive_failed | completed | True | 19229 | 2 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_2-dsl_only-004-recovery-on-no_prompt/cap/20261021-175519-29d9ebf-cap_method_2-recovery-6402e72b) |
| Method 2 | no_prompt | 5 | primitive_failed | primitive_failed | False | 49527 | 5 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_2-dsl_only-005-recovery-on-no_prompt/cap/20261021-175707-29d9ebf-cap_method_2-recovery-bd0cd9a6) |
| Method 2 | no_prompt | 6 | primitive_failed | budget_exhausted | False | 44138 | 5 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_2-dsl_only-006-recovery-on-no_prompt/cap/20261021-175933-29d9ebf-cap_method_2-recovery-558bf303) |
| Method 2 | no_prompt | 7 | primitive_failed | completed | True | 16264 | 2 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_2-dsl_only-007-recovery-on-no_prompt/cap/20261021-180053-29d9ebf-cap_method_2-recovery-c6301f14) |
| Method 2 | no_prompt | 8 | primitive_failed | completed | True | 16937 | 2 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_2-dsl_only-008-recovery-on-no_prompt/cap/20261021-180424-29d9ebf-cap_method_2-recovery-4b55ec0a) |
| Method 2 | no_prompt | 9 | primitive_failed | completed | True | 48576 | 5 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_2-dsl_only-009-recovery-on-no_prompt/cap/20261021-180721-29d9ebf-cap_method_2-recovery-956b6166) |
| Method 2 | no_prompt | 10 | failed | completed | True | 23113 | 3 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_2-dsl_only-010-recovery-on-no_prompt/cap/20261021-180927-29d9ebf-cap_method_2-recovery-6d17936a) |
| Method 2 | no_prompt | 11 | budget_exhausted | completed | True | 38225 | 5 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_2-dsl_only-011-recovery-on-no_prompt/cap/20261021-181219-29d9ebf-cap_method_2-recovery-31ee3e14) |
| Method 2 | no_prompt | 12 | budget_exhausted | completed | True | 31186 | 4 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_2-dsl_only-012-recovery-on-no_prompt/cap/20261021-181343-29d9ebf-cap_method_2-recovery-e82b18c2) |
| Method 2 | no_prompt | 13 | completed | completed | True | 12793 | 1 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_2-dsl_only-013-recovery-on-no_prompt/cap/20261021-181726-29d9ebf-cap_method_2-recovery-4b53de2a) |
| Method 2 | no_prompt | 14 | primitive_failed | completed | True | 44993 | 4 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_2-dsl_only-014-recovery-on-no_prompt/cap/20261021-182005-29d9ebf-cap_method_2-recovery-0e4b3148) |
| Method 2 | no_prompt | 15 | primitive_failed | completed | True | 16848 | 2 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_2-dsl_only-015-recovery-on-no_prompt/cap/20261021-182406-29d9ebf-cap_method_2-recovery-8267b68e) |
| Method 2 | no_prompt | 16 | primitive_failed | completed | True | 28670 | 3 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_2-dsl_only-016-recovery-on-no_prompt/cap/20261021-182702-29d9ebf-cap_method_2-recovery-de991047) |
| Method 2 | no_prompt | 17 | completed | completed | True | 15875 | 2 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_2-dsl_only-017-recovery-on-no_prompt/cap/20261021-182905-29d9ebf-cap_method_2-recovery-99f3c60c) |
| Method 2 | no_prompt | 18 | budget_exhausted | goal_failed | False | 52412 | 5 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_2-dsl_only-018-recovery-on-no_prompt/cap/20261021-183311-29d9ebf-cap_method_2-recovery-8a841724) |
| Method 2 | no_prompt | 19 | primitive_failed | completed | True | 16182 | 2 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_2-dsl_only-019-recovery-on-no_prompt/cap/20261021-184035-29d9ebf-cap_method_2-recovery-e8fff1ac) |
| Method 2 | no_prompt | 20 | completed | completed | True | 5594 | 1 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_2-dsl_only-020-recovery-on-no_prompt/cap/20261021-184234-29d9ebf-cap_method_2-recovery-53bb63ea) |
| Method 2 | no_target | 1 | completed | completed | True | 7165 | 1 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_2-dsl_only-001-recovery-on-no_target/cap/20261021-174852-29d9ebf-cap_method_2-recovery-6eb8a3aa) |
| Method 2 | no_target | 2 | completed | completed | True | 6502 | 1 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_2-dsl_only-002-recovery-on-no_target/cap/20261021-175112-29d9ebf-cap_method_2-recovery-70540730) |
| Method 2 | no_target | 3 | primitive_failed | completed | True | 19631 | 2 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_2-dsl_only-003-recovery-on-no_target/cap/20261021-175225-29d9ebf-cap_method_2-recovery-ad6efd80) |
| Method 2 | no_target | 4 | primitive_failed | completed | True | 62689 | 6 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_2-dsl_only-004-recovery-on-no_target/cap/20261021-175447-29d9ebf-cap_method_2-recovery-c78bd94e) |
| Method 2 | no_target | 5 | primitive_failed | completed | True | 17917 | 2 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_2-dsl_only-005-recovery-on-no_target/cap/20261021-175550-29d9ebf-cap_method_2-recovery-5fb4b40f) |
| Method 2 | no_target | 6 | completed | completed | True | 8536 | 1 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_2-dsl_only-006-recovery-on-no_target/cap/20261021-175814-29d9ebf-cap_method_2-recovery-17359a24) |
| Method 2 | no_target | 7 | completed | completed | True | 14742 | 2 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_2-dsl_only-007-recovery-on-no_target/cap/20261021-180101-29d9ebf-cap_method_2-recovery-c1c1e51d) |
| Method 2 | no_target | 8 | goal_failed | completed | True | 39339 | 4 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_2-dsl_only-008-recovery-on-no_target/cap/20261021-180300-29d9ebf-cap_method_2-recovery-6d9aa095) |
| Method 2 | no_target | 9 | completed | completed | True | 7451 | 1 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_2-dsl_only-009-recovery-on-no_target/cap/20261021-180612-29d9ebf-cap_method_2-recovery-8d7978c9) |
| Method 2 | no_target | 10 | completed | completed | True | 8172 | 1 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_2-dsl_only-010-recovery-on-no_target/cap/20261021-180738-29d9ebf-cap_method_2-recovery-fde5b065) |
| Method 2 | no_target | 11 | primitive_failed | completed | True | 46540 | 5 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_2-dsl_only-011-recovery-on-no_target/cap/20261021-181025-29d9ebf-cap_method_2-recovery-ab28834b) |
| Method 2 | no_target | 12 | primitive_failed | completed | True | 18845 | 2 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_2-dsl_only-012-recovery-on-no_target/cap/20261021-181524-29d9ebf-cap_method_2-recovery-a5e03cb0) |
| Method 2 | no_target | 13 | goal_failed | completed | True | 21972 | 2 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_2-dsl_only-013-recovery-on-no_target/cap/20261021-181754-29d9ebf-cap_method_2-recovery-4becc715) |
| Method 2 | no_target | 14 | completed | completed | True | 10329 | 1 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_2-dsl_only-014-recovery-on-no_target/cap/20261021-181949-29d9ebf-cap_method_2-recovery-39ca7080) |
| Method 2 | no_target | 15 | budget_exhausted | completed | True | 45450 | 4 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_2-dsl_only-015-recovery-on-no_target/cap/20261021-182243-29d9ebf-cap_method_2-recovery-1f2e793e) |
| Method 2 | no_target | 16 | primitive_failed | completed | True | 18005 | 2 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_2-dsl_only-016-recovery-on-no_target/cap/20261021-182636-29d9ebf-cap_method_2-recovery-d84a6460) |
| Method 2 | no_target | 17 | completed | completed | True | 6302 | 1 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_2-dsl_only-017-recovery-on-no_target/cap/20261021-183033-29d9ebf-cap_method_2-recovery-6dc3c2ad) |
| Method 2 | no_target | 18 | budget_exhausted | completed | True | 42601 | 4 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_2-dsl_only-018-recovery-on-no_target/cap/20261021-183349-29d9ebf-cap_method_2-recovery-1d70cd2e) |
| Method 2 | no_target | 19 | completed | completed | True | 6938 | 1 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_2-dsl_only-019-recovery-on-no_target/cap/20261021-183945-29d9ebf-cap_method_2-recovery-d2e0c6ad) |
| Method 2 | no_target | 20 | goal_failed | completed | True | 19503 | 2 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_2-dsl_only-020-recovery-on-no_target/cap/20261021-184156-29d9ebf-cap_method_2-recovery-4529c9d3) |
| Method 2 | no_phases | 1 | completed | completed | True | 9066 | 1 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_2-dsl_only-001-recovery-on-no_phases/cap/20261021-174851-29d9ebf-cap_method_2-recovery-cf734aaf) |
| Method 2 | no_phases | 2 | primitive_failed | completed | True | 20095 | 2 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_2-dsl_only-002-recovery-on-no_phases/cap/20261021-175016-29d9ebf-cap_method_2-recovery-7defff6c) |
| Method 2 | no_phases | 3 | completed | completed | True | 7192 | 1 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_2-dsl_only-003-recovery-on-no_phases/cap/20261021-175241-29d9ebf-cap_method_2-recovery-f63ffa46) |
| Method 2 | no_phases | 4 | failed | completed | True | 18777 | 2 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_2-dsl_only-004-recovery-on-no_phases/cap/20261021-175446-29d9ebf-cap_method_2-recovery-76c4aa4a) |
| Method 2 | no_phases | 5 | primitive_failed | completed | True | 23516 | 2 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_2-dsl_only-005-recovery-on-no_phases/cap/20261021-175625-29d9ebf-cap_method_2-recovery-58c26956) |
| Method 2 | no_phases | 6 | completed | completed | True | 6615 | 1 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_2-dsl_only-006-recovery-on-no_phases/cap/20261021-175933-29d9ebf-cap_method_2-recovery-26f83264) |
| Method 2 | no_phases | 7 | primitive_failed | completed | True | 35671 | 4 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_2-dsl_only-007-recovery-on-no_phases/cap/20261021-180018-29d9ebf-cap_method_2-recovery-29eb189d) |
| Method 2 | no_phases | 8 | budget_exhausted | goal_failed | False | 47427 | 5 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_2-dsl_only-008-recovery-on-no_phases/cap/20261021-180302-29d9ebf-cap_method_2-recovery-91ee260f) |
| Method 2 | no_phases | 9 | completed | completed | True | 7969 | 1 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_2-dsl_only-009-recovery-on-no_phases/cap/20261021-180658-29d9ebf-cap_method_2-recovery-8aed2e9e) |
| Method 2 | no_phases | 10 | completed | completed | True | 16840 | 2 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_2-dsl_only-010-recovery-on-no_phases/cap/20261021-180805-29d9ebf-cap_method_2-recovery-1781c13e) |
| Method 2 | no_phases | 11 | completed | completed | True | 14208 | 2 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_2-dsl_only-011-recovery-on-no_phases/cap/20261021-181007-29d9ebf-cap_method_2-recovery-e0e43d10) |
| Method 2 | no_phases | 12 | completed | completed | True | 7948 | 1 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_2-dsl_only-012-recovery-on-no_phases/cap/20261021-181329-29d9ebf-cap_method_2-recovery-a6402e7a) |
| Method 2 | no_phases | 13 | primitive_failed | primitive_failed | False | 61456 | 5 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_2-dsl_only-013-recovery-on-no_phases/cap/20261021-181727-29d9ebf-cap_method_2-recovery-e623ed33) |
| Method 2 | no_phases | 14 | completed | completed | True | 7022 | 1 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_2-dsl_only-014-recovery-on-no_phases/cap/20261021-182131-29d9ebf-cap_method_2-recovery-b4181924) |
| Method 2 | no_phases | 15 | completed | completed | True | 6610 | 1 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_2-dsl_only-015-recovery-on-no_phases/cap/20261021-182234-29d9ebf-cap_method_2-recovery-f01b6f54) |
| Method 2 | no_phases | 16 | primitive_failed | completed | True | 35141 | 3 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_2-dsl_only-016-recovery-on-no_phases/cap/20261021-182810-29d9ebf-cap_method_2-recovery-3138b363) |
| Method 2 | no_phases | 17 | completed | completed | True | 6968 | 1 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_2-dsl_only-017-recovery-on-no_phases/cap/20261021-183005-29d9ebf-cap_method_2-recovery-b818368f) |
| Method 2 | no_phases | 18 | completed | completed | True | 17683 | 2 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_2-dsl_only-018-recovery-on-no_phases/cap/20261021-183616-29d9ebf-cap_method_2-recovery-21cd3599) |
| Method 2 | no_phases | 19 | completed | completed | True | 7589 | 1 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_2-dsl_only-019-recovery-on-no_phases/cap/20261021-184052-29d9ebf-cap_method_2-recovery-057d11f0) |
| Method 2 | no_phases | 20 | completed | completed | True | 8440 | 1 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_2-dsl_only-020-recovery-on-no_phases/cap/20261021-184133-29d9ebf-cap_method_2-recovery-d80dbf59) |
| Method 2 | no_context | 1 | budget_exhausted | budget_exhausted | False | 21436 | 2 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_2-dsl_only-001-recovery-on-no_context/cap/20261021-174851-29d9ebf-cap_method_2-recovery-e29be6d5) |
| Method 2 | no_context | 2 | primitive_failed | completed | True | 19515 | 2 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_2-dsl_only-002-recovery-on-no_context/cap/20261021-174953-29d9ebf-cap_method_2-recovery-4e20ffcb) |
| Method 2 | no_context | 3 | budget_exhausted | completed | True | 21138 | 2 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_2-dsl_only-003-recovery-on-no_context/cap/20261021-175315-29d9ebf-cap_method_2-recovery-1707d721) |
| Method 2 | no_context | 4 | primitive_failed | completed | True | 35994 | 3 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_2-dsl_only-004-recovery-on-no_context/cap/20261021-175318-29d9ebf-cap_method_2-recovery-520eb672) |
| Method 2 | no_context | 5 | completed | completed | True | 6610 | 1 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_2-dsl_only-005-recovery-on-no_context/cap/20261021-175716-29d9ebf-cap_method_2-recovery-db41d3f8) |
| Method 2 | no_context | 6 | completed | completed | True | 7880 | 1 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_2-dsl_only-006-recovery-on-no_context/cap/20261021-175906-29d9ebf-cap_method_2-recovery-9ab13799) |
| Method 2 | no_context | 7 | primitive_failed | completed | True | 25227 | 3 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_2-dsl_only-007-recovery-on-no_context/cap/20261021-180038-29d9ebf-cap_method_2-recovery-cf14ee38) |
| Method 2 | no_context | 8 | completed | completed | True | 19240 | 2 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_2-dsl_only-008-recovery-on-no_context/cap/20261021-180259-29d9ebf-cap_method_2-recovery-66735dbf) |
| Method 2 | no_context | 9 | completed | completed | True | 14073 | 2 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_2-dsl_only-009-recovery-on-no_context/cap/20261021-180520-29d9ebf-cap_method_2-recovery-85bfd412) |
| Method 2 | no_context | 10 | primitive_failed | completed | True | 43942 | 4 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_2-dsl_only-010-recovery-on-no_context/cap/20261021-180748-29d9ebf-cap_method_2-recovery-1bc59c75) |
| Method 2 | no_context | 11 | completed | completed | True | 8741 | 1 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_2-dsl_only-011-recovery-on-no_context/cap/20261021-181213-29d9ebf-cap_method_2-recovery-c8dc87bf) |
| Method 2 | no_context | 12 | goal_failed | completed | True | 47776 | 5 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_2-dsl_only-012-recovery-on-no_context/cap/20261021-181247-29d9ebf-cap_method_2-recovery-1b868b0f) |
| Method 2 | no_context | 13 | completed | completed | True | 5965 | 1 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_2-dsl_only-013-recovery-on-no_context/cap/20261021-181757-29d9ebf-cap_method_2-recovery-542c3ad6) |
| Method 2 | no_context | 14 | completed | completed | True | 7528 | 1 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_2-dsl_only-014-recovery-on-no_context/cap/20261021-181854-29d9ebf-cap_method_2-recovery-7dc95bb7) |
| Method 2 | no_context | 15 | budget_exhausted | completed | True | 22493 | 2 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_2-dsl_only-015-recovery-on-no_context/cap/20261021-182217-29d9ebf-cap_method_2-recovery-f11f0f9d) |
| Method 2 | no_context | 16 | primitive_failed | completed | True | 23081 | 2 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_2-dsl_only-016-recovery-on-no_context/cap/20261021-182833-29d9ebf-cap_method_2-recovery-f6cc77c4) |
| Method 2 | no_context | 17 | completed | completed | True | 7366 | 1 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_2-dsl_only-017-recovery-on-no_context/cap/20261021-182918-29d9ebf-cap_method_2-recovery-33fda9de) |
| Method 2 | no_context | 18 | primitive_failed | completed | True | 32023 | 3 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_2-dsl_only-018-recovery-on-no_context/cap/20261021-183646-29d9ebf-cap_method_2-recovery-69b57cae) |
| Method 2 | no_context | 19 | completed | completed | True | 6886 | 1 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_2-dsl_only-019-recovery-on-no_context/cap/20261021-184046-29d9ebf-cap_method_2-recovery-7b7da39a) |
| Method 2 | no_context | 20 | completed | completed | True | 6949 | 1 | [Run](../../../runs/cap-revision42/20261021-174848-minimax-220/trials/cap_method_2-dsl_only-020-recovery-on-no_context/cap/20261021-184309-29d9ebf-cap_method_2-recovery-fb40bdf1) |

`*` marks incomplete returned token usage. No failed trial is replaced.
