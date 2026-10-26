# Revision 7.2 report: RDP recovery keyframes

- **Date:** 2026-10-26. Design: [revision_7_2_recovery_key_frame.md](revision_7_2_recovery_key_frame.md).
- **Question:** does showing the failed rollout in the demonstration keyframe format (gripper-command events with predicate changes, plus RDP waypoints) help recovery more than the legacy three keyframes?
- **Answer (one sample per seed):** on the 49 trials whose round 0 failed, RDP keyframes recovered 40 and legacy keyframes 33. Discordant seeds: 11 recovered only with RDP, 4 only with legacy (exact McNemar, two-sided p = 0.12). The gain is concentrated in Reverse (16 vs 10; 7 vs 1 discordant, p = 0.07). Not significant at this sample size.

## 1. Setup

| | Legacy arm (`legacy` on, `rdp` off) | RDP arm (`rdp` on, `legacy` off) |
| --- | --- | --- |
| Source | Revision 7.1 demonstrations-on arm, unchanged ([report](../../../runs/cap-revision7-1/20261026-demo-ablation-v6/demos-on/report.md)) | [`runs/cap-revision7-2/20261026-rdp-keyframes/rdp`](../../../runs/cap-revision7-2/20261026-rdp-keyframes/rdp/report.md), git `3a7dba1` (clean) |
| Round 0 | its own | the same archived attempt (`seed_attempt`), with its charged usage |
| Rounds 1–4 | legacy keyframes (entry, start of last primitive, end) | RDP keyframes (`recovery-keyframes-v1`, tolerance 0.01 m) |
| Keyframe budget / evidence cap | – / 32,768 bytes | 65,536 / 98,304 bytes |
| Everything else | `config_frozen_v6`, Claude Haiku 5.5 (OpenRouter, reasoning medium), demonstrations on, `cap_method_2`, `dsl_only`, recovery `full` with up to 4 rounds, `task-spec-v2` (50 settling steps) | same |

- Seeds: every trial of the legacy arm whose round 0 failed under `task_spec_success`: 49 of 120 ([seed manifest](../../../runs/cap-revision7-2/20261026-rdp-keyframes/seeds.json)). The 71 trials that succeeded at round 0 are identical in both arms and were not rerun. Round-0 failures: 35 primitive nonconvergence, 6 goal not reached, 3 budget exhausted, 3 other execution errors, 2 Stack rollouts that ran to the end but failed the settled task specification.
- The evidence cap is a guard under the `fixed_items` layout, so the other evidence items are the same in both arms; only the keyframe item differs.
- Pyramid has no seed (20/20 at round 0).

## 2. Results

### 2.1 Per task (seeded trials only)

| Task | Seeds | Legacy final | RDP final | Both | Only legacy | Only RDP | Neither | Mean recovery rounds, legacy / RDP | Mean effort tokens, legacy / RDP |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| Stack | 4 | 4/4 | 4/4 | 4 | 0 | 0 | 0 | 1.75 / 1.00 | 45,314 / 29,163 |
| Unstack | 5 | 5/5 | 5/5 | 5 | 0 | 0 | 0 | 1.40 / 1.60 | 40,928 / 45,904 |
| Reverse | 20 | 10/20 | 16/20 | 9 | 1 | 7 | 3 | 2.85 / 2.20 | 89,484 / 75,618 |
| ReStack | 18 | 12/18 | 13/18 | 9 | 3 | 4 | 2 | 3.17 / 3.17 | 125,356 / 138,689 |
| Grid | 2 | 2/2 | 2/2 | 2 | 0 | 0 | 0 | 2.00 / 2.50 | 43,872 / 63,510 |
| **All** | **49** | **33/49** | **40/49** | 29 | 4 | 11 | 5 | | |

Effort tokens are the trial's charged tokens including round 0 (shared), excluding the demonstration block.

### 2.2 Whole task set (120 trials, round 0 shared)

| Task | Legacy arm final | RDP arm final (round-0 successes + RDP recoveries) |
| --- | --- | --- |
| Stack | 20/20 | 20/20 |
| Unstack | 20/20 | 20/20 |
| Reverse | 10/20 | 16/20 |
| ReStack | 14/20 | 15/20 |
| Grid | 20/20 | 20/20 |
| Pyramid | 20/20 | 20/20 |
| **All** | **104/120** | **111/120** |

### 2.3 Evidence size and delivery

| Arm | Recovery requests | Evidence bytes, median | Evidence bytes, max | Requests over 32,768 bytes |
| --- | ---: | ---: | ---: | ---: |
| Legacy | 132 | 18,300 | 22,641 | 0 |
| RDP | 118 | 21,507 | 43,943 | 5 |

All 118 RDP keyframe items were delivered complete. Five requests exceeded the legacy 32,768-byte cap; with that cap and an 8,192-byte keyframe budget they would have been truncated (Section 3 of the design). Cost of the RDP arm on OpenRouter: about $2.0.

### 2.4 Remaining failures (RDP arm)

| Task | Trials | Last attempt |
| --- | ---: | --- |
| Reverse | 4 | five rounds executed to the end, goal not reached (`reverse-02`, `-08`, `-09`, `-19`) |
| ReStack | 3 | Move nonconvergence |
| ReStack | 2 | Release nonconvergence |

## 3. Observations

- Reverse benefits most. In the legacy arm, 8 of its 10 Reverse failures executed all five rounds to the end without reaching the goal, so the error lies in the sequence of transfers, which the legacy frames (start, last primitive, end) do not show; the RDP frames list every pick and release with the ON/ON_star changes. The RDP arm also needs fewer rounds there (2.20 vs 2.85) and fewer tokens.
- ReStack is mixed (4 vs 3 discordant). Its remaining failures in both arms are mostly primitive nonconvergence, where the failure location and primitive diagnostics, shared by both arms, carry the relevant information.
- Stack, Unstack and Grid recover in both arms; the differences there are in rounds and tokens only.
- Each seed has one sample of rounds 1–4 per arm. The legacy rounds were sampled in the Revision 7.1 run and the RDP rounds here; a discordant seed also reflects sampling variation. A repeat of both arms from the same seeds would separate the method effect from sampling noise.

## 4. Per-seed results

Links open each arm's trial report (policy sources, rounds, videos are reachable from it).

| Seed | Round-0 failure | Legacy | RDP | RDP stop / last failure | Legacy trial | RDP trial |
| --- | --- | --- | --- | --- | --- | --- |
| stack-06 | PrimitiveFailure: Pick did not converge within its | yes (round 2) | yes (round 1) | success | [report](../../../runs/cap-revision7-1/20261026-demo-ablation-v6/demos-on/claude-haiku-5.5/stack/trials/cap_method_2-dsl_only-007/report.txt) | [report](../../../runs/cap-revision7-2/20261026-rdp-keyframes/rdp/claude-haiku-5.5/stack/trials/cap_method_2-dsl_only-001-seed06/report.txt) |
| stack-08 | PrimitiveFailure: Pick did not converge within its | yes (round 1) | yes (round 1) | success | [report](../../../runs/cap-revision7-1/20261026-demo-ablation-v6/demos-on/claude-haiku-5.5/stack/trials/cap_method_2-dsl_only-009/report.txt) | [report](../../../runs/cap-revision7-2/20261026-rdp-keyframes/rdp/claude-haiku-5.5/stack/trials/cap_method_2-dsl_only-001-seed08/report.txt) |
| stack-12 | ran to the end; settled task spec failed | yes (round 1) | yes (round 1) | success | [report](../../../runs/cap-revision7-1/20261026-demo-ablation-v6/demos-on/claude-haiku-5.5/stack/trials/cap_method_2-dsl_only-013/report.txt) | [report](../../../runs/cap-revision7-2/20261026-rdp-keyframes/rdp/claude-haiku-5.5/stack/trials/cap_method_2-dsl_only-001-seed12/report.txt) |
| stack-16 | ran to the end; settled task spec failed | yes (round 3) | yes (round 1) | success | [report](../../../runs/cap-revision7-1/20261026-demo-ablation-v6/demos-on/claude-haiku-5.5/stack/trials/cap_method_2-dsl_only-017/report.txt) | [report](../../../runs/cap-revision7-2/20261026-rdp-keyframes/rdp/claude-haiku-5.5/stack/trials/cap_method_2-dsl_only-001-seed16/report.txt) |
| unstack-04 | PrimitiveFailure: Move did not converge within its | yes (round 1) | yes (round 2) | success | [report](../../../runs/cap-revision7-1/20261026-demo-ablation-v6/demos-on/claude-haiku-5.5/unstack/trials/cap_method_2-dsl_only-005/report.txt) | [report](../../../runs/cap-revision7-2/20261026-rdp-keyframes/rdp/claude-haiku-5.5/unstack/trials/cap_method_2-dsl_only-001-seed04/report.txt) |
| unstack-05 | PrimitiveFailure: Move did not converge within its | yes (round 1) | yes (round 1) | success | [report](../../../runs/cap-revision7-1/20261026-demo-ablation-v6/demos-on/claude-haiku-5.5/unstack/trials/cap_method_2-dsl_only-006/report.txt) | [report](../../../runs/cap-revision7-2/20261026-rdp-keyframes/rdp/claude-haiku-5.5/unstack/trials/cap_method_2-dsl_only-001-seed05/report.txt) |
| unstack-08 | PrimitiveFailure: Move did not converge within its | yes (round 2) | yes (round 2) | success | [report](../../../runs/cap-revision7-1/20261026-demo-ablation-v6/demos-on/claude-haiku-5.5/unstack/trials/cap_method_2-dsl_only-009/report.txt) | [report](../../../runs/cap-revision7-2/20261026-rdp-keyframes/rdp/claude-haiku-5.5/unstack/trials/cap_method_2-dsl_only-001-seed08/report.txt) |
| unstack-11 | PrimitiveFailure: Move did not converge within its | yes (round 1) | yes (round 1) | success | [report](../../../runs/cap-revision7-1/20261026-demo-ablation-v6/demos-on/claude-haiku-5.5/unstack/trials/cap_method_2-dsl_only-012/report.txt) | [report](../../../runs/cap-revision7-2/20261026-rdp-keyframes/rdp/claude-haiku-5.5/unstack/trials/cap_method_2-dsl_only-001-seed11/report.txt) |
| unstack-15 | PrimitiveFailure: Move did not converge within its | yes (round 2) | yes (round 2) | success | [report](../../../runs/cap-revision7-1/20261026-demo-ablation-v6/demos-on/claude-haiku-5.5/unstack/trials/cap_method_2-dsl_only-016/report.txt) | [report](../../../runs/cap-revision7-2/20261026-rdp-keyframes/rdp/claude-haiku-5.5/unstack/trials/cap_method_2-dsl_only-001-seed15/report.txt) |
| reverse-00 | PrimitiveFailure: Move did not converge within its | yes (round 3) | yes (round 1) | success | [report](../../../runs/cap-revision7-1/20261026-demo-ablation-v6/demos-on/claude-haiku-5.5/reverse/trials/cap_method_2-dsl_only-001/report.txt) | [report](../../../runs/cap-revision7-2/20261026-rdp-keyframes/rdp/claude-haiku-5.5/reverse/trials/cap_method_2-dsl_only-001-seed00/report.txt) |
| reverse-01 | PrimitiveFailure: Move did not converge within its | yes (round 1) | yes (round 1) | success | [report](../../../runs/cap-revision7-1/20261026-demo-ablation-v6/demos-on/claude-haiku-5.5/reverse/trials/cap_method_2-dsl_only-002/report.txt) | [report](../../../runs/cap-revision7-2/20261026-rdp-keyframes/rdp/claude-haiku-5.5/reverse/trials/cap_method_2-dsl_only-001-seed01/report.txt) |
| reverse-02 | BudgetExceeded: Execution/query step budget exhaus | no | no | max_recovery_rounds: completed | [report](../../../runs/cap-revision7-1/20261026-demo-ablation-v6/demos-on/claude-haiku-5.5/reverse/trials/cap_method_2-dsl_only-003/report.txt) | [report](../../../runs/cap-revision7-2/20261026-rdp-keyframes/rdp/claude-haiku-5.5/reverse/trials/cap_method_2-dsl_only-001-seed02/report.txt) |
| reverse-03 | BudgetExceeded: Execution/query step budget exhaus | no | yes (round 1) | success | [report](../../../runs/cap-revision7-1/20261026-demo-ablation-v6/demos-on/claude-haiku-5.5/reverse/trials/cap_method_2-dsl_only-004/report.txt) | [report](../../../runs/cap-revision7-2/20261026-rdp-keyframes/rdp/claude-haiku-5.5/reverse/trials/cap_method_2-dsl_only-001-seed03/report.txt) |
| reverse-04 | PrimitiveFailure: Move did not converge within its | yes (round 1) | yes (round 2) | success | [report](../../../runs/cap-revision7-1/20261026-demo-ablation-v6/demos-on/claude-haiku-5.5/reverse/trials/cap_method_2-dsl_only-005/report.txt) | [report](../../../runs/cap-revision7-2/20261026-rdp-keyframes/rdp/claude-haiku-5.5/reverse/trials/cap_method_2-dsl_only-001-seed04/report.txt) |
| reverse-05 | PrimitiveFailure: Move did not converge within its | yes (round 2) | yes (round 1) | success | [report](../../../runs/cap-revision7-1/20261026-demo-ablation-v6/demos-on/claude-haiku-5.5/reverse/trials/cap_method_2-dsl_only-006/report.txt) | [report](../../../runs/cap-revision7-2/20261026-rdp-keyframes/rdp/claude-haiku-5.5/reverse/trials/cap_method_2-dsl_only-001-seed05/report.txt) |
| reverse-06 | PrimitiveFailure: Move did not converge within its | yes (round 1) | yes (round 1) | success | [report](../../../runs/cap-revision7-1/20261026-demo-ablation-v6/demos-on/claude-haiku-5.5/reverse/trials/cap_method_2-dsl_only-007/report.txt) | [report](../../../runs/cap-revision7-2/20261026-rdp-keyframes/rdp/claude-haiku-5.5/reverse/trials/cap_method_2-dsl_only-001-seed06/report.txt) |
| reverse-07 | goal_failed | yes (round 3) | yes (round 1) | success | [report](../../../runs/cap-revision7-1/20261026-demo-ablation-v6/demos-on/claude-haiku-5.5/reverse/trials/cap_method_2-dsl_only-008/report.txt) | [report](../../../runs/cap-revision7-2/20261026-rdp-keyframes/rdp/claude-haiku-5.5/reverse/trials/cap_method_2-dsl_only-001-seed07/report.txt) |
| reverse-08 | PrimitiveFailure: Move did not converge within its | yes (round 1) | no | max_recovery_rounds: completed | [report](../../../runs/cap-revision7-1/20261026-demo-ablation-v6/demos-on/claude-haiku-5.5/reverse/trials/cap_method_2-dsl_only-009/report.txt) | [report](../../../runs/cap-revision7-2/20261026-rdp-keyframes/rdp/claude-haiku-5.5/reverse/trials/cap_method_2-dsl_only-001-seed08/report.txt) |
| reverse-09 | BudgetExceeded: Execution/query step budget exhaus | no | no | max_recovery_rounds: completed | [report](../../../runs/cap-revision7-1/20261026-demo-ablation-v6/demos-on/claude-haiku-5.5/reverse/trials/cap_method_2-dsl_only-010/report.txt) | [report](../../../runs/cap-revision7-2/20261026-rdp-keyframes/rdp/claude-haiku-5.5/reverse/trials/cap_method_2-dsl_only-001-seed09/report.txt) |
| reverse-10 | PrimitiveFailure: Move did not converge within its | yes (round 3) | yes (round 3) | success | [report](../../../runs/cap-revision7-1/20261026-demo-ablation-v6/demos-on/claude-haiku-5.5/reverse/trials/cap_method_2-dsl_only-011/report.txt) | [report](../../../runs/cap-revision7-2/20261026-rdp-keyframes/rdp/claude-haiku-5.5/reverse/trials/cap_method_2-dsl_only-001-seed10/report.txt) |
| reverse-11 | PrimitiveFailure: Move did not converge within its | no | yes (round 1) | success | [report](../../../runs/cap-revision7-1/20261026-demo-ablation-v6/demos-on/claude-haiku-5.5/reverse/trials/cap_method_2-dsl_only-012/report.txt) | [report](../../../runs/cap-revision7-2/20261026-rdp-keyframes/rdp/claude-haiku-5.5/reverse/trials/cap_method_2-dsl_only-001-seed11/report.txt) |
| reverse-12 | goal_failed | no | yes (round 3) | success | [report](../../../runs/cap-revision7-1/20261026-demo-ablation-v6/demos-on/claude-haiku-5.5/reverse/trials/cap_method_2-dsl_only-013/report.txt) | [report](../../../runs/cap-revision7-2/20261026-rdp-keyframes/rdp/claude-haiku-5.5/reverse/trials/cap_method_2-dsl_only-001-seed12/report.txt) |
| reverse-13 | PrimitiveFailure: Release did not converge within  | yes (round 1) | yes (round 1) | success | [report](../../../runs/cap-revision7-1/20261026-demo-ablation-v6/demos-on/claude-haiku-5.5/reverse/trials/cap_method_2-dsl_only-014/report.txt) | [report](../../../runs/cap-revision7-2/20261026-rdp-keyframes/rdp/claude-haiku-5.5/reverse/trials/cap_method_2-dsl_only-001-seed13/report.txt) |
| reverse-14 | goal_failed | no | yes (round 4) | success | [report](../../../runs/cap-revision7-1/20261026-demo-ablation-v6/demos-on/claude-haiku-5.5/reverse/trials/cap_method_2-dsl_only-015/report.txt) | [report](../../../runs/cap-revision7-2/20261026-rdp-keyframes/rdp/claude-haiku-5.5/reverse/trials/cap_method_2-dsl_only-001-seed14/report.txt) |
| reverse-15 | PrimitiveFailure: Move did not converge within its | no | yes (round 1) | success | [report](../../../runs/cap-revision7-1/20261026-demo-ablation-v6/demos-on/claude-haiku-5.5/reverse/trials/cap_method_2-dsl_only-016/report.txt) | [report](../../../runs/cap-revision7-2/20261026-rdp-keyframes/rdp/claude-haiku-5.5/reverse/trials/cap_method_2-dsl_only-001-seed15/report.txt) |
| reverse-16 | PrimitiveFailure: Move did not converge within its | no | yes (round 2) | success | [report](../../../runs/cap-revision7-1/20261026-demo-ablation-v6/demos-on/claude-haiku-5.5/reverse/trials/cap_method_2-dsl_only-017/report.txt) | [report](../../../runs/cap-revision7-2/20261026-rdp-keyframes/rdp/claude-haiku-5.5/reverse/trials/cap_method_2-dsl_only-001-seed16/report.txt) |
| reverse-17 | goal_failed | yes (round 1) | yes (round 1) | success | [report](../../../runs/cap-revision7-1/20261026-demo-ablation-v6/demos-on/claude-haiku-5.5/reverse/trials/cap_method_2-dsl_only-018/report.txt) | [report](../../../runs/cap-revision7-2/20261026-rdp-keyframes/rdp/claude-haiku-5.5/reverse/trials/cap_method_2-dsl_only-001-seed17/report.txt) |
| reverse-18 | PrimitiveFailure: Move did not converge within its | no | yes (round 4) | success | [report](../../../runs/cap-revision7-1/20261026-demo-ablation-v6/demos-on/claude-haiku-5.5/reverse/trials/cap_method_2-dsl_only-019/report.txt) | [report](../../../runs/cap-revision7-2/20261026-rdp-keyframes/rdp/claude-haiku-5.5/reverse/trials/cap_method_2-dsl_only-001-seed18/report.txt) |
| reverse-19 | goal_failed | no | no | max_recovery_rounds: goal_failed | [report](../../../runs/cap-revision7-1/20261026-demo-ablation-v6/demos-on/claude-haiku-5.5/reverse/trials/cap_method_2-dsl_only-020/report.txt) | [report](../../../runs/cap-revision7-2/20261026-rdp-keyframes/rdp/claude-haiku-5.5/reverse/trials/cap_method_2-dsl_only-001-seed19/report.txt) |
| restack-00 | PrimitiveFailure: Move did not converge within its | yes (round 4) | yes (round 4) | success | [report](../../../runs/cap-revision7-1/20261026-demo-ablation-v6/demos-on/claude-haiku-5.5/restack/trials/cap_method_2-dsl_only-001/report.txt) | [report](../../../runs/cap-revision7-2/20261026-rdp-keyframes/rdp/claude-haiku-5.5/restack/trials/cap_method_2-dsl_only-001-seed00/report.txt) |
| restack-01 | CapError: No object satisfies get condition | yes (round 1) | yes (round 3) | success | [report](../../../runs/cap-revision7-1/20261026-demo-ablation-v6/demos-on/claude-haiku-5.5/restack/trials/cap_method_2-dsl_only-002/report.txt) | [report](../../../runs/cap-revision7-2/20261026-rdp-keyframes/rdp/claude-haiku-5.5/restack/trials/cap_method_2-dsl_only-001-seed01/report.txt) |
| restack-02 | PrimitiveFailure: Move did not converge within its | no | yes (round 4) | success | [report](../../../runs/cap-revision7-1/20261026-demo-ablation-v6/demos-on/claude-haiku-5.5/restack/trials/cap_method_2-dsl_only-003/report.txt) | [report](../../../runs/cap-revision7-2/20261026-rdp-keyframes/rdp/claude-haiku-5.5/restack/trials/cap_method_2-dsl_only-001-seed02/report.txt) |
| restack-03 | PrimitiveFailure: Release did not converge within  | yes (round 1) | yes (round 4) | success | [report](../../../runs/cap-revision7-1/20261026-demo-ablation-v6/demos-on/claude-haiku-5.5/restack/trials/cap_method_2-dsl_only-004/report.txt) | [report](../../../runs/cap-revision7-2/20261026-rdp-keyframes/rdp/claude-haiku-5.5/restack/trials/cap_method_2-dsl_only-001-seed03/report.txt) |
| restack-04 | PrimitiveFailure: Move did not converge within its | yes (round 2) | yes (round 3) | success | [report](../../../runs/cap-revision7-1/20261026-demo-ablation-v6/demos-on/claude-haiku-5.5/restack/trials/cap_method_2-dsl_only-005/report.txt) | [report](../../../runs/cap-revision7-2/20261026-rdp-keyframes/rdp/claude-haiku-5.5/restack/trials/cap_method_2-dsl_only-001-seed04/report.txt) |
| restack-05 | PrimitiveFailure: Move did not converge within its | yes (round 2) | yes (round 2) | success | [report](../../../runs/cap-revision7-1/20261026-demo-ablation-v6/demos-on/claude-haiku-5.5/restack/trials/cap_method_2-dsl_only-006/report.txt) | [report](../../../runs/cap-revision7-2/20261026-rdp-keyframes/rdp/claude-haiku-5.5/restack/trials/cap_method_2-dsl_only-001-seed05/report.txt) |
| restack-06 | PrimitiveFailure: Move did not converge within its | no | yes (round 2) | success | [report](../../../runs/cap-revision7-1/20261026-demo-ablation-v6/demos-on/claude-haiku-5.5/restack/trials/cap_method_2-dsl_only-007/report.txt) | [report](../../../runs/cap-revision7-2/20261026-rdp-keyframes/rdp/claude-haiku-5.5/restack/trials/cap_method_2-dsl_only-001-seed06/report.txt) |
| restack-07 | PrimitiveFailure: Move did not converge within its | no | yes (round 1) | success | [report](../../../runs/cap-revision7-1/20261026-demo-ablation-v6/demos-on/claude-haiku-5.5/restack/trials/cap_method_2-dsl_only-008/report.txt) | [report](../../../runs/cap-revision7-2/20261026-rdp-keyframes/rdp/claude-haiku-5.5/restack/trials/cap_method_2-dsl_only-001-seed07/report.txt) |
| restack-08 | CapError: No object satisfies get condition | yes (round 3) | yes (round 2) | success | [report](../../../runs/cap-revision7-1/20261026-demo-ablation-v6/demos-on/claude-haiku-5.5/restack/trials/cap_method_2-dsl_only-009/report.txt) | [report](../../../runs/cap-revision7-2/20261026-rdp-keyframes/rdp/claude-haiku-5.5/restack/trials/cap_method_2-dsl_only-001-seed08/report.txt) |
| restack-09 | PrimitiveFailure: Move did not converge within its | yes (round 4) | yes (round 4) | success | [report](../../../runs/cap-revision7-1/20261026-demo-ablation-v6/demos-on/claude-haiku-5.5/restack/trials/cap_method_2-dsl_only-010/report.txt) | [report](../../../runs/cap-revision7-2/20261026-rdp-keyframes/rdp/claude-haiku-5.5/restack/trials/cap_method_2-dsl_only-001-seed09/report.txt) |
| restack-10 | PrimitiveFailure: Move did not converge within its | no | no | max_recovery_rounds: PrimitiveFailure: Move did not converge withi | [report](../../../runs/cap-revision7-1/20261026-demo-ablation-v6/demos-on/claude-haiku-5.5/restack/trials/cap_method_2-dsl_only-011/report.txt) | [report](../../../runs/cap-revision7-2/20261026-rdp-keyframes/rdp/claude-haiku-5.5/restack/trials/cap_method_2-dsl_only-001-seed10/report.txt) |
| restack-11 | PrimitiveFailure: Move did not converge within its | yes (round 4) | no | max_recovery_rounds: PrimitiveFailure: Release did not converge wi | [report](../../../runs/cap-revision7-1/20261026-demo-ablation-v6/demos-on/claude-haiku-5.5/restack/trials/cap_method_2-dsl_only-012/report.txt) | [report](../../../runs/cap-revision7-2/20261026-rdp-keyframes/rdp/claude-haiku-5.5/restack/trials/cap_method_2-dsl_only-001-seed11/report.txt) |
| restack-12 | PrimitiveFailure: Move did not converge within its | yes (round 4) | no | max_recovery_rounds: PrimitiveFailure: Release did not converge wi | [report](../../../runs/cap-revision7-1/20261026-demo-ablation-v6/demos-on/claude-haiku-5.5/restack/trials/cap_method_2-dsl_only-013/report.txt) | [report](../../../runs/cap-revision7-2/20261026-rdp-keyframes/rdp/claude-haiku-5.5/restack/trials/cap_method_2-dsl_only-001-seed12/report.txt) |
| restack-13 | PrimitiveFailure: Move did not converge within its | no | yes (round 4) | success | [report](../../../runs/cap-revision7-1/20261026-demo-ablation-v6/demos-on/claude-haiku-5.5/restack/trials/cap_method_2-dsl_only-014/report.txt) | [report](../../../runs/cap-revision7-2/20261026-rdp-keyframes/rdp/claude-haiku-5.5/restack/trials/cap_method_2-dsl_only-001-seed13/report.txt) |
| restack-14 | CapError: No object satisfies get condition | yes (round 1) | yes (round 2) | success | [report](../../../runs/cap-revision7-1/20261026-demo-ablation-v6/demos-on/claude-haiku-5.5/restack/trials/cap_method_2-dsl_only-015/report.txt) | [report](../../../runs/cap-revision7-2/20261026-rdp-keyframes/rdp/claude-haiku-5.5/restack/trials/cap_method_2-dsl_only-001-seed14/report.txt) |
| restack-15 | PrimitiveFailure: Move did not converge within its | no | no | max_recovery_rounds: PrimitiveFailure: Move did not converge withi | [report](../../../runs/cap-revision7-1/20261026-demo-ablation-v6/demos-on/claude-haiku-5.5/restack/trials/cap_method_2-dsl_only-016/report.txt) | [report](../../../runs/cap-revision7-2/20261026-rdp-keyframes/rdp/claude-haiku-5.5/restack/trials/cap_method_2-dsl_only-001-seed15/report.txt) |
| restack-17 | PrimitiveFailure: Move did not converge within its | yes (round 4) | yes (round 2) | success | [report](../../../runs/cap-revision7-1/20261026-demo-ablation-v6/demos-on/claude-haiku-5.5/restack/trials/cap_method_2-dsl_only-018/report.txt) | [report](../../../runs/cap-revision7-2/20261026-rdp-keyframes/rdp/claude-haiku-5.5/restack/trials/cap_method_2-dsl_only-001-seed17/report.txt) |
| restack-18 | PrimitiveFailure: Move did not converge within its | yes (round 3) | no | max_recovery_rounds: PrimitiveFailure: Move did not converge withi | [report](../../../runs/cap-revision7-1/20261026-demo-ablation-v6/demos-on/claude-haiku-5.5/restack/trials/cap_method_2-dsl_only-019/report.txt) | [report](../../../runs/cap-revision7-2/20261026-rdp-keyframes/rdp/claude-haiku-5.5/restack/trials/cap_method_2-dsl_only-001-seed18/report.txt) |
| grid-07 | PrimitiveFailure: Move did not converge within its | yes (round 2) | yes (round 1) | success | [report](../../../runs/cap-revision7-1/20261026-demo-ablation-v6/demos-on/claude-haiku-5.5/grid/trials/cap_method_2-dsl_only-008/report.txt) | [report](../../../runs/cap-revision7-2/20261026-rdp-keyframes/rdp/claude-haiku-5.5/grid/trials/cap_method_2-dsl_only-001-seed07/report.txt) |
| grid-19 | goal_failed | yes (round 2) | yes (round 4) | success | [report](../../../runs/cap-revision7-1/20261026-demo-ablation-v6/demos-on/claude-haiku-5.5/grid/trials/cap_method_2-dsl_only-020/report.txt) | [report](../../../runs/cap-revision7-2/20261026-rdp-keyframes/rdp/claude-haiku-5.5/grid/trials/cap_method_2-dsl_only-001-seed19/report.txt) |
