# Progress report: October 19–22, 2026

Covers all work after the October 19 meeting (14:00 Eastern). Setting throughout: Stack, three blocks, one archived initial scene, `dsl_only`, `typed_v3` prompts, medium reasoning, 16,384 output tokens per request.

- **Revision 4: shared failure recovery.** All three methods can now revise a failed policy from text feedback about the failed rollout. They get up to four repairs, and every rollout starts from the same saved initial state.
- **Recovery makes Methods 1 and 2 nearly always succeed.**
  - Haiku 5.5: final success 20/20 for both methods, up from 10/20 and 14/20 without recovery.
  - MiniMax M3.1 Flash, a weaker free model: 40/50 for both methods, up from 4/50 and 6/50.
- **The weaker model exposed primitive nonconvergence as the main remaining failure.**
  - Pick, Move and Release failures make up 92 of the 159 initial failures and 16 of the 29 unrecovered trials.
  - The reviewed cases point to wrong policy targets (offset/reference confusion, missing transport waypoints) rather than controller defects.
- **Revision 4.2: four prompt additions and three feedback enrichments, ablated across 3 models (780 trials).**
  - The prompt additions are the effective change: removing them raises initial nonconvergence in every method/model pair.
  - The feedback enrichments add no detectable benefit over the existing feedback. `only_prompt` matches or beats `full`.
- **Next:**
  1. Add every earlier executed program to the recovery request.
  2. Add demonstration input.
  3. Test three more tasks.

## 1. Revision 4: shared failure recovery

### 1.1 What is collected and injected into the next prompt

When a rollout fails, a shared, deterministic feedback module (no extra LLM call) collects evidence. It serializes that evidence as text and adds it to the next synthesis request. The table lists what the delivered feedback actually contains.

| Feedback field | Content |
| --- | --- |
| Latest outcome | Round, candidate hash, failure status and reason. Physical goal, logical postcondition and DSL compliance are reported separately. Repeated candidates are flagged. |
| Current executable source | Policy and the helper sources that actually ran (Baseline helpers can differ from the visible policy text). Includes the execution-time binding map. |
| Failure location | Source unit and line, statement, failing primitive with its arguments, and the simulation step |
| Controller result | Convergence, steps used out of 50, final position error, and the fixed target computed when the primitive was called |
| Keyframes | Gripper and block XYZ at up to three recorded states: the initial state, the start of the last primitive before the failure, and the failure/terminal state. Coinciding states collapse to two or one; 749 of 810 delivered feedbacks carry three. |
| Supporting events | Recent primitive calls, query results, predicates and assignments, chosen deterministically under the size cap. Omitted events are counted. |
| Goal evidence | Truth value of each goal clause (e.g. `ON_star(block_i, b0)`) at the terminal state |
| Failed `get` | Up to 16 candidate evaluations for a query that found no witness |
| Earlier failures | A short summary of each earlier failed attempt: candidate, outcome, reason. **Earlier programs themselves are not included.** |
| Bookkeeping | Object-handle → physical-ID map, units/frame, Higher tolerance, explicit markers for missing fields |

- **Size limit:** feedback is capped at 8,192 bytes, a conservative stand-in for tokens. If the required evidence does not fit, recovery stops with an explicit reason instead of silently truncating.
- **Design details:**
  - [shared feedback module](revision_4_failure_recover.md#4-shared-feedback-module)
  - [evidence selection rules](revision_4_failure_recover.md#41-deterministic-evidence-selection)
  - [where recovery enters each method's pipeline](revision_4_failure_recover.md#5-injection-into-the-synthesis-stage)
  - [rollout protocol and logging](revision_4_failure_recover.md#6-rollout-protocol-history-and-logging)
  - [budgets](revision_4_failure_recover.md#8-recommended-layered-budgets)
  - [recovery prompt](revision_4_failure_recover.md#9-proposed-recovery-prompt)
- **Budgets:**

  | Scope | Tokens | Calls | Time |
  | --- | ---: | ---: | ---: |
  | Initial attempt | 50k | 12 | 600 s |
  | Each recovery round | 100k | 16 | 600 s |
  | Whole trial | 450k | 76 | 3,000 s |

- **Bug fixed before the reported runs (CAP-RECOVERY-1):** the first batch delivered the latest failure twice, once as detailed evidence and once in the earlier-failure history. The history is now frozen before delivery. That batch is excluded from all results.

### 1.2 Results: recovery off vs. on

Full report: [revision_4_recovery_report.md](revision_4_recovery_report.md).

| Model | Method | Recovery off | On: initial program | On: final |
| --- | --- | ---: | ---: | ---: |
| Haiku 5.5 | Baseline | 9/20 | 8/20 | 19/20 |
| Haiku 5.5 | Method 1 | 10/20 | 10/20 | **20/20** |
| Haiku 5.5 | Method 2 | 14/20 | 17/20 | **20/20** |
| MiniMax M3.1 Flash free | Baseline | 5/50 | 4/50 | 42/50 |
| MiniMax M3.1 Flash free | Method 1 | 4/50 | 5/50 | 40/50 |
| MiniMax M3.1 Flash free | Method 2 | 6/50 | 7/50 | 40/50 |

- **With Haiku, Methods 1 and 2 reach 100% after recovery.** The first repair is the most productive round: it accounts for 19 of Haiku's 24 recovered trials and 52 of MiniMax's 106.
- **Cost:** recovery uses more calls and tokens than a single attempt. For example, Haiku Baseline uses 456,633 tokens with recovery on vs. 164,348 with it off.
- **Not tested here:** whether recovery beats simply generating more independent programs for the same cost.

### 1.3 Stress test with a weaker model: what MiniMax exposed

We ran the free MiniMax M3.1 Flash model to see which weaknesses a less capable model would surface. Then we reviewed every recovery step offline: all 325 transitions from feedback to replacement program, across 159 initially failed trials ([Section 4](revision_4_recovery_report.md#4-analysis-of-initially-failed-recovery-on-trials)).

- **Primitive nonconvergence dominates.**

  | Failure | Initial failures | Unrecovered trials |
  | --- | ---: | ---: |
  | Move | 75 | 9 |
  | Pick | 12 | 5 |
  | Release | 5 | 2 |
  | **Total** | **92 of 159** | **16 of 29** |

- **What goes wrong:**
  - Confusing absolute coordinates with relative offsets. For example, a target 1.4 m away is commanded where 0.1 m was intended.
  - Moving straight to the placement without first lifting and transferring.
  - Choosing a Release retreat height that is too low.
- **The feedback already contained the target and the residual.** The model still had to work out by itself how the target followed from the reference block and the offset. 128 of the 130 successful repairs changed a Move call.
- **Other gaps found** (kept in the [backlog](revision_4_unimplemented.md)):
  - Loop-progress summaries.
  - Namespace diagnosis: 8 of 10 NameErrors come from Baseline's `exec` scoping, not from a missing assignment.
  - Explanations for queries that find no witness.
  - **Earlier programs are absent from history.** 169 feedbacks summarize earlier failures without their source code.

## 2. Revision 4.2: prompt and feedback changes for primitive nonconvergence

### 2.1 Changes

Design: [revision_4_2_fix_primitive_noncovergence.md](revision_4_2_fix_primitive_noncovergence.md). Each change can be switched on or off independently, and all three methods share one rendering path.

| Kind | Switch | Adds |
| --- | --- | --- |
| Prompt | `explain_target_offsets` | `offset = desired_gripper − reference`, how to account for the carried block's displacement, and a numeric right/wrong example |
| Prompt | `explain_convergence_criteria` | When each phase of Pick, Move and Release counts as complete; the 3D 2 mm tolerance; one 50-step budget shared by all phases |
| Prompt | `explain_partial_execution` | A failed primitive may already have moved objects or opened the gripper; nothing is rolled back |
| Prompt | `explain_motion_planning_limits` | Move has no obstacle avoidance or automatic waypoints; lift, transfer and descent must be written explicitly |
| Feedback | `target_derivation` | For each axis: reference block and its coordinate, the offset, and the resulting fixed target |
| Feedback | `controller_phases` | Which controller phases completed, which phase failed, the measured value against its threshold, and steps per phase |
| Feedback | `action_context` | The preceding converged action and the local Pick → transport sequence, with geometry before and after |

The prompt additions are present from the first attempt. The feedback enrichments only take effect after an executed failure.

### 2.2 Experiment

Full report: [revision_4_2_report.md](revision_4_2_report.md). Recovery is on in every trial, and each group has 20 trials.

- **Six profiles:**
  - `full`
  - `no_prompt`
  - `no_target`, `no_phases`, `no_context`: each turns off one feedback enrichment
  - `only_prompt`: prompt additions on, all three feedback enrichments off
- **Models:**
  - MiniMax M3.1 Flash and Haiku 5.5 run all 18 method × profile groups each: 360 trials per model.
  - Qwen3.8 27B runs only `only_prompt`, as a stress test: 60 trials.
  - Model identity is the only difference between cohorts.
- **Totals:** 780 trials, audited with 0 issues. Provider cost: Haiku $1.44, Qwen $1.87, MiniMax free.

**Final success, out of 20:**

| Model | Method | full | no_prompt | no_target | no_phases | no_context | only_prompt |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MiniMax | Baseline | 18 | 14 | 19 | 17 | 18 | 20 |
| MiniMax | Method 1 | 18 | 19 | 19 | 20 | 20 | 20 |
| MiniMax | Method 2 | 20 | 16 | 20 | 18 | 19 | 19 |
| Haiku | Baseline | 20 | 17 | 19 | 20 | 17 | 18 |
| Haiku | Method 1 | 20 | 20 | 20 | 20 | 20 | 20 |
| Haiku | Method 2 | 20 | 20 | 20 | 20 | 20 | 20 |
| Qwen3.8 27B | Baseline / M1 / M2 | — | — | — | — | — | 16 / 20 / 19 |

**Initial nonconvergence, with vs. without the prompt additions (`full` → `no_prompt`):**

| Model | Baseline | Method 1 | Method 2 |
| --- | ---: | ---: | ---: |
| MiniMax | 6 → 11 | 7 → 14 | 5 → 11 |
| Haiku | 1 → 10 | 1 → 10 | 0 → 5 |

![Ablation nonconvergence](plots/revision42_ablation_nonconvergence.png)

### 2.3 Findings

- **The prompt additions are what help.**
  - Without them, initial nonconvergence roughly doubles for MiniMax and rises about tenfold for Haiku.
  - Initial success drops sharply. Example: MiniMax Baseline goes from 8/20 to 1/20.
  - With MiniMax, `no_prompt` also uses the most tokens in every method; Baseline uses about twice as many as `full`.
- **The feedback enrichments add no detectable benefit.**
  - `only_prompt` (prompt additions, original feedback) finishes at least as well as `full`: MiniMax 20/20/19 vs. 18/18/20.
  - Among initially failed MiniMax trials, it recovers 33 of 34, vs. 32 of 36 for `full`.
  - Both feedback-size stops occurred with all three enrichments on, because the extra evidence pushed the feedback past the 8 KB cap.
  - With 20 trials per group, small differences cannot rank the components or show that they hurt.
- **The remaining nonconvergences are errors in the policy's logic, not missing information.** We reviewed each MiniMax final nonconvergence. Examples:
  - The release retreat is too low, so the next pick knocks the tower over.
  - The Move target is referenced to a block that is moving with the gripper, which puts it above reachable height.
  - Pick samples its target while the blocks are still settling after an impact.
  - A loop never terminates.
  - Z is counted twice in an offset.
- **Haiku's remaining losses are all Baseline.** Six of nine are helper SyntaxErrors that stop the trial before rollout.
- **Qwen3.8 27B falls between MiniMax and Haiku.**
  - Initial success is 12/16/17, final success 16/20/19.
  - All 7 initial nonconvergences ended without a final nonconvergence.
  - 3 of its 5 losses are recovery responses that hit the 16,384-token output cap. The other 2 are Baseline SyntaxErrors.
- **First successful-but-DSL-noncompliant Baseline policy:** MiniMax `no_phases` trial 11.
  - The policy builds a three-element offset list and then indexes it (`displacement[0]`).
  - `typed_v3` forbids this explicitly: "Only Position3 literal indexing 0/1/2 is allowed". So this is a genuine violation, not checker over-strictness.
  - Baseline has no pre-execution gate, so it remains a physical success that fails the audit.

## 3. Infrastructure and fixes

| Change | Effect |
| --- | --- |
| [Revision 3](revision_3.md): checker fixes CAP-CHECK-1/2 (`shared-contract-v3`) | Closes a gap where values carried across loop iterations hid action provenance. Removes a false rejection of generic helpers. 106 tests passing. |
| Request pacing and quota admission | A shared request gate. A trial starts only if the remaining quota covers its full 76-call allowance. Batches can resume safely, which makes free-tier runs possible. |
| [Revision 4.1](revision_4_1_gpu_scheduling.md): GPU scheduling | Reserves GPUs from an allowlist across processes. Scripted 6-trial check: 49.69 s with one worker vs. 30.14 s with two. |
| [Revision 4.2.1](revision_4_2_1_deferred_video_rendering.md): deferred video | Videos render on a shared GPU queue after the trial, so recovery doesn't wait for rendering. Every video is checked against its trajectory. |
| Recovery history analysis ([recovery_analysis.py](../recovery_analysis.py)) | Offline review of every recovery transition from saved artifacts only, with no model calls |
| Multi-model launcher and report | `--model`, `--variant PROFILE=METHOD`, and paid/free admission. The report merges batches by model, distinguishes models by hatching in plots, and lists the terminal path of every unsuccessful trial. |

## 4. Limits

- All results come from one task (Stack, three blocks) and one fixed initial scene. They say nothing yet about generalization to other layouts or tasks.
- Recovery gets extra calls and failure evidence. We have no matched-cost comparison against independent restarts.
- Groups have 20 trials each, so differences of 1–2 trials are noise.
- Independent initial generations confound comparisons between feedback profiles. A clean feedback comparison needs a fixed set of failed programs and initial states, branched under each profile.

## 5. Next steps

1. **Minor fix: include every earlier executed program in recovery.**
   - Today the history accumulates failure summaries but not the programs that produced them.
   - Add each round's executed source and helper bindings to the recovery request:
     - deduplicate identical bodies;
     - attach a short edit/outcome summary per round;
     - stay within the existing feedback budget.
   - Design: [historical programs and edit/outcome ledger](revision_4_unimplemented.md#14-historical-programs-and-the-editoutcome-ledger).
   - Evaluate by branching from the same failed programs, with history off vs. on.
2. **Add demonstration input** ([Revision 6](revision_6_add_demonstration.md)).
   - Give all three methods the same demonstrations RoboVerify uses, with an on/off switch for ablation.
   - Waiting on the demonstration format and data from Yuning.
3. **Test three more tasks** ([Revision 5](revision_5_add_tasks.md)): Unstack, Reverse and ReStack, which already have CaP adapters.
   - Shared task specification and demonstrations.
   - Matched scenes across methods.
   - Unstack keeps its 60-second limit, which covers all recovery rounds.

## Appendix A. Commits since the last meeting

| Area | Commits |
| --- | --- |
| Revision 3 checker fixes | `070716d`, `35155a0` |
| Recovery design | `df610a1`, `5541188`, `bfa333c`, `91c851f`, `308e495`, `13cf3f4`, `d519605`, `bb6acdc`, `bd1f7e3`, `8650f5c`, `0a902fd` |
| Recovery implementation and fix | `555bedf` (shared recovery + paired batches), `df0a42a` (CAP-RECOVERY-1) |
| Recovery experiments and analysis | `231b368`, `212cd89` (Haiku), `29473fe` (pacing/admission), `bf70364`, `f6a9f54` (MiniMax), `236ce71` (history analysis), `f772b84` (case review, revision split) |
| GPU scheduling (4.1) | `b772a23` |
| Revision 4.2 prompt/feedback | `29d9ebf`, `3ac2b04`, `8887e14` |
| Deferred video (4.2.1) | `7b34aa7` |
| Multi-model ablations and reports | `3f0c0f9`, `8411dbb`, `6f55e0f`, `fb3374f`, `f3ebe71`, `cb90bc2`, `270d5e5`, `024fa38` |

Validation: all 181 tests under `synthesis/cap/` pass at the current commit (2 opt-in hardware tests skipped). Every reported batch passed its artifact audit: initial-state hashes, prepared vs. delivered feedback, video/trajectory alignment, and usage reconciliation.
