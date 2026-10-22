# Revision 4.3: experiment report

**Status: prompt factorial running; feedback ablation not started.** Pending trials are not observed failures.

- **Prompt factorial:** [plan](revision_4_3_prompt_ablation_plan.md). Full 2⁴ factorial over the four primitive prompt blocks; 50 trials per cell × 16 cells × 3 methods × 2 models = 4,800 trials.
- **Feedback ablation:** [plan](revision_4_3_ablation_experiment_plan.md). Runs after the prompt study, with the selected prompt set fixed.
- **Implementation under test:** [Revision 4.3 design and validation](revision_4_3_configurable_feedback_and_full_policy_history.md).

## 1. Prompt factorial

### 1.1 Configuration

| Item | Qwen3.8 27B | Qwen3.5 9B |
| --- | --- | --- |
| Checkpoint | `Qwen/Qwen3.8-27B-FP8` @ `017b9c7a` (official FP8) | `Qwen/Qwen3.5-9B` @ `c2022362` (official BF16) |
| Serving | vLLM 0.32.0 on GPU 6, port 8106 | vLLM 0.32.0 on GPU 7, behind a local least-in-flight balancer (port 8117). A second instance on GPU 6 is added after the 27B batch finishes. |
| Chat template | `enable_thinking=true`, `reasoning_effort=medium` | `enable_thinking=true`; the template has no effort option |
| Batch | `runs/cap-revision43/20261022-prompt-factorial-qwen38-27b` | `runs/cap-revision43/20261022-prompt-factorial-qwen35-9b` |

Shared settings:

- **Task:** Stack, three blocks, the saved initial scene, `dsl_only`, `typed_v3`.
- **Recovery:** on, with Revision 4 default feedback (`cap-recovery-v1`, primitive enrichments off).
- **Sampling:** checkpoint `generation_config` defaults (temperature 1.0, top_p 0.95, top_k 20 for the 27B checkpoint).
- **Output cap:** 32,768 tokens; no thinking-token budget.
- **Token budgets:** 100k initial, 200k per recovery round, 900k per trial.
- **Time budgets:** request timeout 2,400 s, 3,600 s per round, 18,000 s per trial. These are raised because local decoding runs at roughly 25–75 tokens/s per request.
- **Call limits:** unchanged.
- **Code:** launcher [revision43_prompt_experiment.py](../revision43_prompt_experiment.py), run from a pinned worktree at commits `aeb6552` (27B) and `5e0f9f5` (9B, balancer routing only). Both contain the Revision 4.2 runtime code (`45d6f98`), unaffected by the concurrent Revision 4.3 implementation.

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

### 1.3 Results

Pending.
