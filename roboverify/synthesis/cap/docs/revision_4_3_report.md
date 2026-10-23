# Revision 4.3: experiment report

**Status: prompt factorial completed; selected prompt set `p0001` (motion-planning limits only, Section 1.5.4). Feedback ablation not started.** Pending trials are not observed failures.

- **Prompt factorial:** [plan](revision_4_3_prompt_ablation_plan.md). Full 2⁴ factorial over the four primitive prompt blocks; 50 trials per cell × 16 cells × 3 methods × 2 models = 4,800 trials.
- **Feedback ablation:** [plan](revision_4_3_ablation_experiment_plan.md). Runs after the prompt study, with the selected prompt set fixed.
- **Implementation under test:** [Revision 4.3 design and validation](revision_4_3_configurable_feedback_and_full_policy_history.md).
- **Inference backend:** [Revision 4.3.1: inference backend optimization](revision_4_3_1_inference_backend_optimization.md) records serving, scheduling and failure-handling changes.

## 1. Prompt factorial

### 1.1 Configuration

| Item | Qwen3.8 27B | Qwen3.5 9B |
| --- | --- | --- |
| Checkpoint | `Qwen/Qwen3.8-27B-FP8` @ `017b9c7a` (official FP8) | `Qwen/Qwen3.5-9B` @ `c2022362` (official BF16) |
| Serving | vLLM 0.32.0 on GPU 6, port 8106 | vLLM 0.32.0 on GPU 7, behind a local least-in-flight balancer (port 8117). A second instance on GPU 6 is added after the 27B batch finishes. |
| Chat template | `enable_thinking=true`, `reasoning_effort=medium` | `enable_thinking=true`; the template has no effort option |
| Batch | `runs/cap-revision43/20261022b-prompt-factorial-qwen38-27b` (32 workers) | `runs/cap-revision43/20261022b-prompt-factorial-qwen35-9b` (64 workers) |

Shared settings:

- **Task:** Stack, three blocks, the saved initial scene, `dsl_only`, `typed_v3`.
- **Recovery:** on, with Revision 4 default feedback (`cap-recovery-v1`, primitive enrichments off).
- **Sampling:** checkpoint `generation_config` defaults (temperature 1.0, top_p 0.95, top_k 20 for the 27B checkpoint).
- **Output cap:** 32,768 tokens; no thinking-token budget.
- **Token budgets:** 100k initial, 200k per recovery round, 900k per trial.
- **Time budgets:** request timeout 2,400 s, 3,600 s per round, 18,000 s per trial. These are raised because local decoding runs at roughly 25–75 tokens/s per request.
- **Call limits:** unchanged.
- **Code:** launcher [revision43_prompt_experiment.py](../revision43_prompt_experiment.py), run from a pinned worktree at commit `b1a6ea9`. Both contain the Revision 4.2 runtime code (`45d6f98`), unaffected by the concurrent Revision 4.3 implementation.

### 1.2 Pre-run findings

- **27B reasoning effort.** The Qwen3.8 chat template defaults to `reasoning_effort=xhigh`, which adds the system instruction "think carefully … validate key assumptions, consider plausible alternatives".
  - In the first smoke run, all three 27B round-0 requests reasoned for 118k–121k characters, hit the 32,768-token cap and produced no code.
  - The text was not looping: about 82–87% of lines were unique.
  - With `reasoning_effort=medium`, which adds no instruction, four requests on the same real round-0 prompt finished with 5,791–11,937 reasoning tokens. This matches the OpenRouter Qwen3.8 27B batch (median 7,049, p90 10,978).
  - A vLLM `thinking_token_budget` was tested and works, but is not used: effort control changes how long the model reasons without truncating it.
- **OpenRouter reasoning effort.**
  - Per OpenRouter's documentation, `effort` maps to a hard reasoning budget only for budget-based models (Anthropic: `max_tokens × ratio`, medium 0.5). For Qwen, support "varies by model".
  - Recorded OpenRouter Qwen usage shows no hard 50% cap: 29 of 89 calls exceeded 8,192 reasoning tokens, and 3 stopped at the 16,384 output cap.
- **9B smoke.** All 14 calls finished (3k–28k output tokens). The three trials ended in a goal or runtime failure, execution-budget exhaustion, and primitive nonconvergence.
- **Smoke artifacts:** `runs/cap-revision43-validation/` (excluded from all results).
- **Aborted first launch.**
  - The first full launch (`20261022-prompt-factorial-*`, about 15 minutes, 24 finished 27B trials) was stopped and relaunched in full.
  - Cause: the GPU scheduler caps each device at the smallest `slot_limit` among live reservations, so the two batches shared 16 worker slots per GPU. LLM concurrency was too low (27B: 17 requests, 440 tokens/s).
  - The relaunch shares 48 slots per GPU with 128 MiB per worker (commit `b1a6ea9`). Throughput rose to 886 tokens/s (27B, 32 requests) and 2,352 tokens/s (9B, 67 requests).
  - The aborted directories are kept, each with an `ABORTED.md` note, and are excluded from all analyses.
  - Expected completion: about 13 hours for the 27B batch, about 16 hours for the 9B batch including the GPU 6 handover.

### 1.3 Qwen3.5 9B stopped (2026-10-22)

- **Decision:** stop the 9B batch and move its GPU to the 27B. The 9B is also dropped from the feedback ablation ([plan, Section 6](revision_4_3_ablation_experiment_plan.md#6-model-and-local-serving)).
- **Reason: floor effect.** Over 291 completed trials:
  - round-0 success in 1 trial, so the primary outcome cannot separate the 16 cells;
  - 20 of 290 initially failed trials recovered (about 7%);
  - 96 trials stopped with `preparation_failure_no_rollout`.
- **State of the batch:** 291 trials completed, 56 were running when the launcher was stopped, and 2,053 were never started. The batch is not resumed, and its data are excluded from prompt selection. They are kept as a descriptive record.
- **27B move to two GPUs.**
  - The 27B batch was drained, not interrupted. A placeholder reservation in the shared GPU ledger lowered the slot limit on GPUs 6 and 7 to 1, so no new trials were dispatched while running ones finished.
  - Port 8106 then becomes a least-in-flight balancer over two 27B instances (GPU 6 on port 8126, GPU 7 on port 8127), with identical checkpoint, template and server flags.
  - The batch resumes with 64 concurrent trials, using a resume-time override that each session records in `execution_metadata.json`. Trial definitions are unchanged.
  - Outcome: the drain ended without interrupted trials, and the batch resumed at 22:29 EDT. In the first hour on two GPUs it completed 225 trials, compared with 114 per hour on one GPU. Expected completion is about 07:00 EDT on 2026-10-23.

### 1.4 Server failure and rerun (2026-10-23)

- **Failure.** At 06:08:16 EDT, both 27B vLLM instances (GPUs 6 and 7) stopped in the same second with `CUDA error: unspecified launch failure`.
  - The simultaneous failure on two independent processes points to a host-level GPU fault. Kernel logs are not readable to us, so the cause is not confirmed.
  - The batch then kept dispatching, and every new request failed with `TransportError: Model HTTP status 503: no backend available`. It finished at 06:29.
- **Affected trials: 192 of 2,400.** These are all trials that ended after the failure.
  - 161 lost round 0 itself.
  - 31 had a valid round-0 failure, but their recovery was cut off.
  - No transport errors occurred before the failure.
- **Treatment.** All 192 trials are excluded, and their exact job definitions are rerun as fresh trials in `runs/cap-revision43/20261023-prompt-factorial-qwen38-27b-rerun`, with the same identifiers, cells and methods.
  - All 192 are excluded because trials in flight at the failure are biased toward slow, mostly failing trials. Keeping only the 31 with a valid round 0 would bias the estimate.
  - Every trial in the rerun batch is a complete draw under the same configuration. The analysis combines the 2,208 unaffected trials with the rerun.
  - This is a documented infrastructure replacement, not a replacement of model failures.
- **Serving for the rerun.** Six 27B instances (GPUs 0, 1, 2, 3, 6, 7; ports 8120–8123, 8126, 8127) with identical checkpoint, template and flags, all behind the balancer on port 8106. The rerun uses 96 concurrent trials.
- **Safeguard.** Dispatch now checks the model endpoint (`GET /v1/models`) every 10 seconds. If the endpoint fails, the batch stops dispatching and pauses, so pending trials are not consumed (commit `bcd80f5`).

### 1.5 Results

**Data:** 2,400 trials (Qwen3.8 27B): the 2,208 unaffected trials of the main batch plus the 192 reruns (Section 1.4). Every cell × method has exactly 50 trials, and no rerun trial had a transport error. Primary outcome: round-0 task success. Analysis: logistic regression with method as a covariate.

#### 1.5.1 Prompt blocks under test

Each block is a fixed text in [`prompts/primitive_v1.json`](../prompts/primitive_v1.json). It is appended after the header "Additional primitive contract explanations:" to the initial and recovery prompts of all methods. Each example below is the actual round-0 prompt of a Baseline trial in the cell where only that block is on; the block starts at the linked line. Prompt files live under `runs/`, which git ignores, so these links work only in a checkout that has the run folders.

| Block (switch) | Content | Text | Example prompt |
| --- | --- | --- | --- |
| Target offsets (`explain_target_offsets`) | Offset = desired gripper coordinate − reference coordinate; carried-block compensation; numeric example (X 1.30 → 1.40 needs dx = 0.10, not 1.40) | [line 5](../prompts/primitive_v1.json#L5) | [p1000 Baseline trial 1, round 0](../../../runs/cap-revision43/20261022b-prompt-factorial-qwen38-27b/trials/cap_google_baseline-dsl_only-001-recovery-on-p1000/cap/20261022-212230-b1a6ea9-cap_google_baseline-recovery-95c7aa74/attempts/round_000/cap/20261022-212234-b1a6ea9-cap_google_baseline-stack-2110d9fd/artifacts/trial_001/lmp/call_0001/prompt.txt#L107) |
| Convergence criteria (`explain_convergence_criteria`) | Pick/Move/Release phase-completion tests, tolerances, the shared 50-step budget | [line 6](../prompts/primitive_v1.json#L6) | [p0100 Baseline trial 1, round 0](../../../runs/cap-revision43/20261022b-prompt-factorial-qwen38-27b/trials/cap_google_baseline-dsl_only-001-recovery-on-p0100/cap/20261022-212231-b1a6ea9-cap_google_baseline-recovery-3446aaaf/attempts/round_000/cap/20261022-212235-b1a6ea9-cap_google_baseline-stack-52aaf61a/artifacts/trial_001/lmp/call_0001/prompt.txt#L107) |
| Partial execution (`explain_partial_execution`) | A primitive may change the scene before it fails; nothing is rolled back; later statements do not run | [line 7](../prompts/primitive_v1.json#L7) | [p0010 Baseline trial 1, round 0](../../../runs/cap-revision43/20261022b-prompt-factorial-qwen38-27b/trials/cap_google_baseline-dsl_only-001-recovery-on-p0010/cap/20261022-212842-b1a6ea9-cap_google_baseline-recovery-75223e0f/attempts/round_000/cap/20261022-212844-b1a6ea9-cap_google_baseline-stack-e1ecbe9a/artifacts/trial_001/lmp/call_0001/prompt.txt#L107) |
| Motion-planning limits (`explain_motion_planning_limits`) | Move has no obstacle avoidance or inserted waypoints; lift, transfer and descent must be explicit; no universal safe height | [line 8](../prompts/primitive_v1.json#L8) | [p0001 Baseline trial 1, round 0](../../../runs/cap-revision43/20261022b-prompt-factorial-qwen38-27b/trials/cap_google_baseline-dsl_only-001-recovery-on-p0001/cap/20261022-212231-b1a6ea9-cap_google_baseline-recovery-b7347428/attempts/round_000/cap/20261022-212236-b1a6ea9-cap_google_baseline-stack-29c36dad/artifacts/trial_001/lmp/call_0001/prompt.txt#L107) |
| All four blocks | — | — | [p1111 Baseline trial 1, round 0](../../../runs/cap-revision43/20261022b-prompt-factorial-qwen38-27b/trials/cap_google_baseline-dsl_only-001-recovery-on-p1111/cap/20261022-212229-b1a6ea9-cap_google_baseline-recovery-3cd90c78/attempts/round_000/cap/20261022-212234-b1a6ea9-cap_google_baseline-stack-62a84497/artifacts/trial_001/lmp/call_0001/prompt.txt#L107) |

#### 1.5.2 Main effects

| Block | Round-0 success off → on | Odds ratio (95% CI) | p | Final success off → on |
| --- | --- | --- | --- | --- |
| Target offsets | 51.2% → 52.3% | 1.05 (0.88–1.25) | 0.56 | 84.0% → 83.7% |
| Convergence criteria | 51.8% → 51.8% | 1.00 (0.84–1.20) | 0.97 | 84.8% → 82.8% |
| Partial execution | 52.4% → 51.2% | 0.94 (0.79–1.12) | 0.51 | 83.8% → 83.8% |
| **Motion-planning limits** | **32.1% → 71.5%** | **5.32 (4.47–6.33)** | **< 0.001** | **74.3% → 93.3%** |

- **Block × block interactions:** none (likelihood-ratio test p = 0.50 for round-0 success, 0.82 for final success; the smallest single term is P:M, p = 0.09).
- **Block × method:** significant for round-0 success (p = 0.039), and entirely due to the size of the motion-planning effect (M × method p = 0.007; the other three blocks p ≥ 0.17).
  - Its odds ratio is about 3.7 for Baseline, 5.6 for Method 1 and 7.4 for Method 2. It is strongly positive in every method.
  - Round-0 success without → with the block: Baseline 36.2% → 68.0%, Method 1 32.5% → 72.8%, Method 2 27.5% → 73.8%.
  - Final success shows no block × method interaction (p = 0.70).
- **The other three blocks show no effect,** whether the motion-planning block is on or off. Among the 1,200 trials with it on, their odds ratios are 0.95–1.10, all p ≥ 0.44.

#### 1.5.3 Secondary outcomes (motion-planning block off → on)

- **Round-0 primitive nonconvergence:** 49.2% → 19.1% of trials. By primitive, Move 314 → 105, Pick 155 → 69, Release 121 → 55 trials.
- **Round-0 execution-budget exhaustion:** 12.4% → 2.4%.
- **Preparation stops** (`preparation_failure_no_rollout`, any round): 15.6% → 5.0% of trials.
- **Median tokens per trial:** 31.0k → 11.8k, mainly because fewer recovery rounds are needed.

#### 1.5.4 Selected prompt set

Under the [selection rule](revision_4_3_prompt_ablation_plan.md#5-selection-rule-fixed-before-the-run), the selected set is **`p0001`**: only `explain_motion_planning_limits` on.

- **Rule 1:** keep motion-planning limits. Its pooled effect is positive, and its confidence interval excludes harm.
- **Rule 2:** drop target offsets, convergence criteria and partial execution. Their effects are near zero, and dropping them gives a shorter prompt.
- **Rule 3:** does not apply; no block × block interaction is significant.
- **Rule 4:** one set for all methods. The method interaction changes only the size of the motion-planning benefit, not its direction, so no exception is needed.
- For reference, round-0 success by method (Baseline / M1 / M2): `p0000` 34 / 32 / 32%, `p0001` 70 / 64 / 76%, `p1111` 72 / 78 / 76%. The extra three blocks in `p1111` have no significant effect when motion-planning limits is on (Section 1.5.2).

**Limitation:** one model, one task and one initial scene (see the plan).

### 1.6 Cell table (round-0 and final success, 150 trials per cell)

![Prompt factorial success by cell](plots/revision43_prompt_cells.png)

- **Bars:** round-0 (orange) and final (blue) task success per cell, with 95% Wilson intervals; 150 trials per cell, three methods pooled.
- **Order:** cells are grouped by the motion-planning block (green band: on). Within each group they are sorted by round-0 success. Letters mark the blocks that are on (T target offsets, C convergence criteria, P partial execution, M motion-planning limits).
- **Reading:** the two groups separate cleanly on both measures, while the other three blocks do not order the cells within a group (Section 1.5.2).
- **Source:** generated by `synthesis.cap.revision43_prompt_report` from the main batch, with the 192 rerun trials replacing their originals (Section 1.4).

#### 1.6.1 Per method (50 trials per cell)

Same layout, one figure per method. Rows keep the pooled cell order above, so the three figures compare row by row.

![Prompt factorial success by cell, Baseline](plots/revision43_prompt_cells_baseline.png)

![Prompt factorial success by cell, Method 1](plots/revision43_prompt_cells_method1.png)

![Prompt factorial success by cell, Method 2](plots/revision43_prompt_cells_method2.png)

- **The motion-planning split holds in every method.** Round-0 success off → on: Baseline 36.2% → 68.0%, Method 1 32.5% → 72.8%, Method 2 27.5% → 73.8%. Final success: 76.5% → 92.5%, 75.5% → 93.0%, 71.0% → 94.5%.
- **The gap widens from Baseline to Method 2.** This is the block × method interaction in Section 1.5.2.
- **Single cells are noisy at this size.** With 50 trials per cell, a cell's 95% interval spans roughly ±12–14 points, so differences between cells within one group are not interpretable.


| Cell (T C P M) | Round 0 | Final |
| --- | ---: | ---: |
| p1111 | 75.3% | 94.0% |
| p0011 | 74.0% | 92.7% |
| p0111 | 72.0% | 93.3% |
| p0101 | 72.0% | 93.3% |
| p1101 | 70.7% | 94.0% |
| p0001 | 70.0% | 92.7% |
| p1001 | 69.3% | 94.7% |
| p1011 | 68.7% | 92.0% |
| p1000 | 38.0% | 74.0% |
| p0100 | 34.0% | 70.7% |
| p0000 | 32.7% | 78.0% |
| p1010 | 32.7% | 76.7% |
| p1100 | 32.7% | 73.3% |
| p1110 | 31.3% | 70.7% |
| p0010 | 28.7% | 78.0% |
| p0110 | 26.7% | 73.3% |

## 2. Feedback ablation

**Status: not started.** It runs after the prompt factorial selects a prompt set.

### 2.1 Profile definitions that differ from a single-item removal

- **`no_current_source`** removes the latest executed program and sets history to `summaries`.
  - The configuration rejects `executed_programs` without the current source, because the history registry also holds the latest program's bodies.
  - This profile therefore measures the value of all program source in the request (latest and earlier) relative to `full`, with earlier failures still listed as one-line summaries.
  - It is not a test of the latest source alone. (Decided 2026-10-22.)
- **`minimal`** keeps the always-on core: latest outcome, missing-field markers, scene metadata and DSL diagnostics. It adds the current source and uses history `none`.

### 2.2 Seeded-branch validation (2026-10-22)

End-to-end check of the Stage B path on spare capacity of GPU 0, with requests served by the 9B balancer. Artifacts: `runs/cap-revision43-validation/20261022-seeded-check` (excluded from all results).

- **Design:** 6 round-0 failures from 9B cell `p1100` (two per method: four goal failures, one Move nonconvergence, one execution-budget exhaustion) × profiles `full`, `no_current_source`, `minimal` = 18 chains.
- **Mechanics: passed.** All 18 chains completed and the audit reported no issues. Round 0 is identical across the branches of each seed. Item delivery and history mode match each profile:
  - `full`: every applicable item, history `executed_programs` with the source registry;
  - `no_current_source`: current source disabled, history `summaries`;
  - `minimal`: current source only, history `none` (about 1.8 KB per feedback, against 2.9–24 KB for the other two).
- **Outcomes:** 0 of 18 chains recovered.
  - 12 chains used all four rounds.
  - 6 stopped with `preparation_failure_no_rollout`: three responses hit the 32,768-token output cap, two raised SyntaxErrors, and one raised a KeyError. These stops occurred under all three profiles.
  - With six seeds, this does not compare profiles.
