# Revision 4.3: primitive prompt ablation plan

- **Status:** design agreed except the open point in Section 8; no experiment run yet.
- **Purpose:** decide which of the four Revision 4.2 primitive prompt blocks to keep.
- **Order of experiments:**
  1. This prompt study runs first, with feedback fixed.
  2. The [feedback ablation](revision_4_3_ablation_experiment_plan.md) then runs with the selected prompt set fixed.

  Separating the two keeps the number of groups small.

## 1. Factors

The four Boolean switches under `RunConfig.primitive_prompt` ([Revision 4.2, Section 3.1.1](revision_4_2_fix_primitive_noncovergence.md#311-four-independent-general-prompt-switches)):

| Switch | Block content |
| --- | --- |
| `explain_target_offsets` | Offset = desired gripper position − reference position, carried-block compensation, numeric example |
| `explain_convergence_criteria` | Phase-completion criteria and the shared 50-step budget |
| `explain_partial_execution` | A failed primitive may already have changed the scene; nothing is rolled back |
| `explain_motion_planning_limits` | No automatic waypoints or obstacle avoidance; lift/transfer/descent must be explicit |

**Decided.**
- Full 2⁴ factorial: all 16 combinations, including all-off and all-on.
- Feedback options stay at one fixed setting in every cell (Section 7).

## 2. Scale

**Decided.**
- 50 trials per cell × 16 cells × 3 methods × 2 models = **4,800 trials** as planned. The 9B half was stopped on 2026-10-22 because its round-0 success was at the floor (1 of 291); the factorial is completed for the Qwen3.8 27B only (2,400 trials). See the [report](revision_4_3_report.md#13-qwen35-9b-stopped).
- Models and serving are as in the [feedback ablation plan](revision_4_3_ablation_experiment_plan.md#6-models-and-local-serving):
  - Qwen3.8 27B FP8 on GPU 6;
  - Qwen3.5 9B BF16 on GPU 7;
  - checkpoint sampling defaults;
  - 32,768 output tokens.
- Setting: Stack, three blocks, the saved initial scene, `dsl_only`, `typed_v3`.
- Cells are interleaved in a recorded randomized schedule, so server load and time effects do not align with any cell.

### 2.1 Serving details (decided before launch)

- The 27B uses `reasoning_effort=medium` through the chat template. The template default, `xhigh`, exhausted the 32k output cap without code in every smoke round 0.
- The 9B template has no effort option; thinking is enabled.
- No thinking-token budget is applied to either model.
- Time and token budgets are scaled for local decoding (see the [report](revision_4_3_report.md#11-configuration)).
- After the 27B batch finishes, GPU 6 serves a second 9B instance behind the same balancer.

## 3. Outcomes

- **Primary: round-0 task success.**
  - The prompt acts before any feedback exists.
  - Round 0 is far from the ceiling.
  - It is the outcome where Revision 4.2 found the clear prompt effect.
- **Secondary:**
  - round-0 primitive nonconvergence, split by Pick / Move / Release;
  - round-0 failure-class distribution;
  - preparation stops (e.g. Baseline SyntaxErrors);
  - DSL compliance;
  - added input tokens.
- **Final success after recovery** is reported as a secondary outcome (Section 7). It is not used to select prompts.

## 4. Analysis

- **Model:** logistic regression `success ~ four switches + two-way interactions + method + model`, fitted on the round-0 outcome.
- **Main effects.**
  - Each switch's main effect compares 400 vs 400 trials within one model × method stratum, which detects roughly 7–9 percentage points.
  - Pooled, it compares 2,400 vs 2,400, which detects roughly 3–4 points.
- **Strata.** Per-method and per-model estimates are reported descriptively. Interactions with method or model are tested before pooling.

## 5. Selection rule (fixed before the run)

1. Keep a block whose pooled main effect is positive and whose confidence interval excludes harm.
2. Drop a block whose effect is near zero, for a shorter prompt with fewer variables.
3. If a two-way interaction is significant, choose the combination with the highest predicted success. When predictions are within noise, choose the combination with the fewest blocks.
4. Use one prompt set for all methods and models, so the feedback ablation remains comparable across strata. Deviate only if a block is clearly harmful for a specific model or method, and record that exception.

**Limitation:** one initial scene and one task. The selection is optimized for this setting and is not a claim about other layouts or tasks.

## 6. Link to the feedback ablation

- Round-0 attempts from the selected cell were generated under the final prompt. Its failed attempts can seed Stage A of the feedback ablation, with additional round-0 attempts generated only to fill class quotas.
- No separate pilot is needed for prompt rendering: all 16 combinations already render deterministically, as tested in Revision 4.2.
- A short throughput pilot of local serving runs before the full batch. Its data are excluded from the main analysis.

## 7. Recovery and feedback setting

**Decided:** recovery on, with the Revision 4 default feedback in every cell.

- The feedback is `cap-recovery-v1`: legacy items, summaries history, and primitive enrichments off, under the original `legacy_fill` 8,192-byte layout. This matches Revision 4.2 `only_prompt` and needs no Revision 4.3 feedback code.
- The token budgets of the feedback ablation apply: 200k per recovery round and 900k per trial; call and time limits unchanged.
- Round-0 success stays the primary outcome. Final success and recovery of initially failed trials are secondary and are not used for selection.

## 8. Open point

- Confirmation of the selection rule in Section 5.
