# Revision 4.3.1: inference backend optimization

- **Scope:** every change to local model serving and batch dispatch made for the Revision 4.3 experiments (2026-10-22 to 2026-10-23), including the reasoning-termination guard (Section 10). Experiment design and results are in the [Revision 4.3 report](revision_4_3_report.md).
- **Rule kept throughout:** serving changes may alter speed and capacity, never a trial's definition. Checkpoint, chat template, sampling and budgets are identical across instances and over time. Every scheduling change is recorded in the batch artifacts.
- **Location:** the serving scripts live outside the repository in `~/vllm-serve/` on dbar, next to their logs. The repository holds the launcher-side changes. The API key is read from a file outside the repository and passed as `VLLM_API_KEY`; it is never on a command line or in an artifact.

## 1. Summary

| # | Change | Problem | Effect |
| --- | --- | --- | --- |
| 1 | Local vLLM serving | OpenRouter cost and provider variation | Two pinned official checkpoints on A100s |
| 2 | `reasoning_effort=medium` for the 27B | Template default `xhigh` exhausted 32k output tokens without code | Reasoning lengths match OpenRouter |
| 3 | 32k output cap; scaled time and token budgets | Local decoding is slower; outputs doubled | No budget artifacts from serving speed |
| 4 | Shared per-GPU worker limit (48 slots, 128 MiB) | Scheduler took the smallest slot limit across batches → 16 slots per GPU | 27B 440 → 886 tokens/s; 9B 2,352 tokens/s |
| 5 | Least-in-flight HTTP balancer | One endpoint per batch; instances must be added or removed live | Instances added or removed without touching clients |
| 6 | Drain via placeholder reservation; resume-time worker override | Moving a batch to more GPUs without interrupting trials | 27B 114 → 225 trials/h on two GPUs; 0 interrupted |
| 7 | Endpoint health gate | A server crash turned 192 pending trials into instant failures | Dispatch pauses while no backend answers |
| 8 | Rerun of infrastructure-failed trials | 192 trials invalidated by the crash | Clean replacement batch with copied job definitions |
| 9 | Six GPUs behind one balancer | Spare GPUs 0–3 | 6 instances, about 16 requests each |
| 10 | Stage A request limit | Limit below the per-trial reservation capped concurrency at 14 | 88 concurrent trials |
| 11 | Live dispatch limit and GPU release procedure | Lab may ask to free GPUs mid-experiment | Shrink from 6 to 2 GPUs without interrupting trials |
| 12 | Reasoning-termination guard (vLLM logits processor) | About 10% of recovery calls ended the turn inside `<think>` with an empty answer | Empty answers 9/40 → 0/40 on previously failed prompts |

## 2. Serving stack

- **Engine:** vLLM 0.32.0 (PyTorch 2.13, CUDA 13.0) in its own virtual environment (`~/vllm-serve/venv`), on NVIDIA A100-SXM4-80GB.
  - The first launch failed because FlashInfer needed `ninja` to JIT-compile; it was installed into the serving environment.
  - `VLLM_USE_FLASHINFER_SAMPLER=0` keeps sampling on the default kernels.
- **Checkpoints, pinned by revision and served offline (`HF_HUB_OFFLINE=1`):**

  | Model | Checkpoint | Precision | KV cache per instance |
  | --- | --- | --- | --- |
  | Qwen3.8 27B | `Qwen/Qwen3.8-27B-FP8` @ `017b9c7a` | Official FP8; weight-only Marlin kernels on A100 | 520k–588k tokens (5.3–6.0 × 98,304) |
  | Qwen3.5 9B (dropped 2026-10-22) | `Qwen/Qwen3.5-9B` @ `c2022362` | Official BF16; no official FP8 | 1.44M tokens |

- **Server flags (`~/vllm-serve/serve.sh GPU PORT NAME REPO REVISION`):**
  - `--max-model-len 98304`: covers the largest Stage B input plus 32,768 output tokens.
  - `--gpu-memory-utilization 0.85`: leaves about 12 GB per GPU for MuJoCo EGL workers.
  - `--enable-prefix-caching`: hit rate is about 60% in practice, since all trials share the system prompt.
  - `--reasoning-parser qwen3`.
  - `--limit-mm-per-prompt '{"image":0,"video":0}'`.
  - From 2026-10-23 12:13 EDT: `--logits-processors synthesis.cap.serving.think_close_guard:ThinkCloseGuard`, with `roboverify/` on `PYTHONPATH` (Section 10). `serve.sh` includes it; `serve_unguarded.sh` keeps the earlier configuration.
  - `--host 127.0.0.1`, with an API key.
- **Client:** provider `vllm` in `ModelConfig`.
  - It has no request pacing and no free-quota admission.
  - `extra_body.chat_template_kwargs` carries the thinking settings.
  - `transport_retries=1`.

## 3. Generation settings

- **Reasoning effort (27B).** The Qwen3.8 chat template defaults to `reasoning_effort=xhigh`, which injects an instruction to think carefully and consider alternatives.
  - In the smoke run, all three round-0 requests reasoned for 118k–121k characters and hit the output cap with no code.
  - With `medium` (no injected instruction), four requests on the real round-0 prompt used 5,791–11,937 reasoning tokens, matching the OpenRouter batch (median 7,049).
  - `low` asks for brief reasoning and was not used. The 9B template has no effort option.
- **No thinking-token budget.** vLLM's `thinking_token_budget` works but truncates reasoning. Effort control changes how much the model reasons without cutting it off.
  - OpenRouter maps `effort` to a hard budget only for budget-based models. Its Qwen calls showed no hard cap (29 of 89 above 8,192 reasoning tokens).
- **Sampling:** checkpoint `generation_config` defaults (27B: temperature 1.0, top_p 0.95, top_k 20).
- **Output cap:** 32,768 tokens, double the earlier 16,384.
- **Budgets scaled for local decoding** (about 25–75 tokens/s per request under load):
  - request timeout 2,400 s; 3,600 s per round; 18,000 s per trial;
  - 100k initial tokens, 200k per recovery round, 900k per trial.

## 4. Worker GPU scheduling

- **Problem.** The shared GPU scheduler caps each device at the smallest `slot_limit` among live reservations. Two concurrent batches with different limits therefore ran at the lower one: 16 worker slots per GPU. Model concurrency was too low (27B: 17 requests, 440 tokens/s).
- **Fix (commit `b1a6ea9`).**
  - All concurrent batches use one per-GPU limit, 48 slots of 128 MiB each. Measured EGL workers use about 70 MiB.
  - The feedback launcher exposes `--gpu-uuid`, `--workers-per-gpu` and `--worker-memory-mb` (commit `2712ef4`).
- **Effect:** 27B 886 tokens/s with 32 requests; 9B 2,352 tokens/s with 67 requests. The first launch was aborted and relaunched; its directories carry `ABORTED.md`.

## 5. Load balancing (`~/vllm-serve/balancer.py`)

- **Design.** A small aiohttp proxy routes each request to the backend with the fewest in-flight requests.
  - It re-reads its backend list (`backends_27b.json`) on every request, so instances can be added or removed without touching clients.
  - On a connection error it tries up to two other backends.
  - `GET /balancer/status` reports per-backend in-flight counts.
- **Use.**
  - The 9B ran behind port 8117.
  - The 27B moved behind port 8106, which was its original address, so the immutable batch manifests did not change.
- **Instance registration** (`register_backends.sh`): each new instance is added to the list once `/v1/models` returns 200.

## 6. Changing GPUs under a running batch

### 6.1 Adding GPUs: drain and resume (2026-10-22)

The 27B batch had to move from one direct endpoint to a balancer over two GPUs, without interrupting trials. Killing the launcher would have marked 32 running trials as interrupted, and the batch never retries those.

1. **Pause dispatch** with a placeholder reservation (`drain_hold.py`). It holds one live entry with `slot_limit=1` on each worker GPU, so the scheduler refuses new trials while running ones finish. The batch then pauses by itself with "GPU admission timeout; unstarted trials remain pending".
2. **Start the new instance** on GPU 7 while the drain runs.
3. **Cut over** (`cutover_27b.sh`): stop the launcher, stop the 8106 server, start the balancer on 8106, start a fresh instance on GPU 6, register both, and release the hold.
4. **Resume with more workers.** A resume-time `max_workers` override (commit `9cbada5`) is recorded per session in `execution_metadata.json`. Concurrency is a scheduling choice, not part of a trial's definition.

**Result:** the drain took about 21 minutes and interrupted no trials. Throughput rose from 114 to 225 trials per hour.

### 6.2 Removing GPUs without interrupting trials (2026-10-23)

The lab may ask for GPUs back. The procedure:

1. **Lower concurrency first.** Write `{"max_active": N}` to `<batch>/dispatch_limit.json` (commit `1b848ae`).
   - The dispatch gate re-reads the file before every dispatch.
   - Running trials continue; new ones start only below N. `0` pauses dispatch.
   - Every change is logged in `dispatch_limit_events.jsonl`.
   - Rule of thumb: about 16 active trials per remaining instance. Each 27B instance holds about 590k KV tokens, and Stage B requests use about 30k tokens. Oversubscription makes requests queue in vLLM, and queue time counts toward the 2,400 s request timeout.
2. **Release GPUs:** `release_gpus.sh GPU_INDEX...`
   - It removes the GPUs' instances from the balancer and blocks new CaP workers on those GPUs.
   - It waits until the removed instances have no in-flight requests, then stops them.
   - CaP workers already on those GPUs (about 130 MiB each) finish with their trials.
   - It refuses to remove the last backend.
3. **Add GPUs back:** start `serve.sh` on the GPU and register the instance. Raise or delete the dispatch limit afterwards.

**Drill (2026-10-23, `--keep-serving`).** GPU 3's instance was removed while 88 pilot trials were running.
- Its in-flight requests fell from 8 to 0 in 3.5 minutes, while new requests went to the other five instances.
- The instance was then re-added.
- All 90 pilot trials finished with no transport error.

The worker side was not part of the drill; it uses the same placeholder mechanism as Section 6.1.

**Batches started before the dispatch-limit change** do not read the file. This applies to the pilot Stage A. For them, release GPUs only after their trials finish, or accept a higher per-instance load.

## 7. Failure handling

- **Server failure (2026-10-23 06:08:16 EDT).** Both 27B instances failed in the same second with `CUDA error: unspecified launch failure`. Simultaneous failure of two independent processes points to a host-level GPU event; the kernel log is not readable to us.
- **Damage.** The batch kept dispatching for 21 minutes. 192 trials failed with `503 no backend available` (161 in round 0) or lost their recovery rounds.
- **Health gate (commits `bcd80f5`, `8e38c11`).** Before dispatch, both launchers check `GET /v1/models` through the balancer (cached for 10 s). While it fails, no trial starts. Once running trials finish, the batch pauses as `model_endpoint_unavailable` and can be resumed.
- **Rerun (commit `bcd80f5`).** `--rerun-of BATCH --rerun-jobs FILE` copies the selected job definitions into a new batch with new paths. The source stays read only, and the reason is recorded in `comparison_plan.json`.
  - All 192 affected trials were rerun. Trials in flight at a failure are biased toward slow, mostly failing ones, so keeping the subset with a valid round 0 would bias the estimate.
- **Stage A request limit (commit `66c8ab9`).** Stage A keeps recovery enabled with `max_rounds=0`, and the request gate reserves `recovery.total_calls` per active trial. The batch limit was sized by `max_model_calls`, which capped the pilot at 14 concurrent trials. It is now sized by the reserved allowance, with a regression test. The two-minute aborted launch is marked `ABORTED.md`.

## 8. Throughput record

| Setting | Model concurrency | Throughput |
| --- | --- | --- |
| 27B, 1 GPU, 16 worker slots per GPU (aborted) | 17 requests | 440 tokens/s |
| 27B, 1 GPU, 32 workers | 32 requests | 886 tokens/s; 114 trials/h |
| 9B, 1 GPU, 64 workers | 67 requests | 2,352 tokens/s; about 65 trials/h (4 recovery rounds typical) |
| 27B, 2 GPUs, 64 workers | about 30 requests per instance | 225 trials/h |
| 27B, 6 GPUs, 96 workers (rerun) | 16 requests per instance | 192 trials in 35 minutes; about 470 generation tokens/s per instance |

Trials per hour depend on the mix of round-0 successes and recovery chains, so rows are comparable only within one experiment.

## 9. Operational notes

- **Launch background jobs with `setsid nohup … < /dev/null &`.** Children of a plain background command died with the tool shell.
- **Background Python ignores SIGINT.** A non-interactive shell sets SIGINT to ignored for `&` jobs, so `kill -INT` does not run the batch's cleanup. Stop a batch by draining it (Section 6), not by signals.
- **Anchor `pkill`/`pgrep` patterns** (for example `'^\.venv/bin/python -m synthesis\.cap\.…'`). Unanchored patterns matched the invoking shell's own command line and killed it.
- **Count failures from parsed fields, not `grep` on JSON.** A substring search for "503" matched token counts.

## 10. Empty answers: reasoning that ends the turn (2026-10-23)

### 10.1 Symptom

- The pilot Stage B showed 20 of 219 recovery calls ending with `ModelError: Incomplete/empty model response: stop`: `finish_reason=stop`, an empty answer, and all text in the reasoning field. The trial then stops with `preparation_failure_no_rollout`.
- The same pattern runs through the whole prompt factorial:

  | Source | Round-0 calls | Recovery calls |
  | --- | --- | --- |
  | Local 27B, prompt factorial | 11 / 2,356 (0.5%) | 235 / 2,300 (10.2%) |
  | Local 27B, pilot Stage B | — | 20 / 219 (9.1%) |
  | OpenRouter 27B, Revision 4.2 (16k output cap) | 0 / 61 | 0 / 28 (3 hit the cap) |

- **Not the request.** No stop strings are sent, and `stop_reason` is empty, so the model sampled its end-of-turn token `<|im_end|>`. Every affected request is a single user message under 8k tokens.
- **Reasoning length.** Affected responses are long (median 17.9k completion tokens against 9.9k for other recovery calls) and stop mid-line, often in the middle of code drafted inside the reasoning.

### 10.2 Diagnosis

The failure points were re-scored by sending the original prompt plus the generated reasoning and reading the next-token distribution (`max_tokens=1`, `logprobs=20`).

| Check | Result |
| --- | --- |
| End-of-turn probability at the 6 failure points (FP8, production) | 0.51, 0.81, 0.47, 0.31, 0.08, 0.11; e.g. 0.81 right after `\|2(g` |
| Same, with prefix caching off | Identical. Prefix caching of the hybrid model's linear-attention state (`mamba_cache_mode=align`) is not the cause. |
| Same, with the official BF16 checkpoint (`Qwen/Qwen3.8-27B` @ `1d4bf0f2`) | 0.33, 0.59, 0.18, 0.11, 0.01, 0.07. Present without quantization, but FP8 raises it 1.5–8× |
| Along the reasoning (prompt log-probabilities) | Near 0 in the first half of long reasoning, rising after roughly 8–10k tokens (up to 0.27 mid-text) |
| Same text under the `xhigh` effort template | Lower: positions above 0.05 fall from 18 to 2 |

- **Conclusion.** This is the model's behavior: at `medium` effort, long reasoning ends with `<|im_end|>` instead of `</think>`. FP8 quantization amplifies it.
- **Serving BF16 instead would not remove it** and would cut KV capacity about 3× (176k vs about 590k tokens per A100).
- The 27B is a hybrid model: Gated DeltaNet linear-attention layers, with full attention every fourth layer.

### 10.3 Fix: `ThinkCloseGuard`

- **Code:** [`synthesis/cap/serving/think_close_guard.py`](../serving/think_close_guard.py), a vLLM 0.32 logits processor for the V2 model runner, with unit tests in `test_think_close_guard.py`. The tests run in the serving environment and are skipped in the project environment.
- **Phase 1 (reasoning open).** When an admitted request's prompt ends inside an open `<think>` block, the probability of the stop tokens (`<|im_end|>`, `<|endoftext|>`) is moved onto `</think>`:

  ```
  logit[</think>] <- logsumexp(logit[</think>], logit[stop tokens]);  logit[stop tokens] <- -inf
  ```

  The model's decision to stop reasoning is kept; only the token that expresses it changes.
- **Phase 2 (awaiting the answer).** After `</think>`, stop tokens stay blocked until the first non-whitespace answer token. Without this phase, one probe closed the reasoning and then immediately ended the turn, leaving the answer empty again.
- **Otherwise unchanged.** All other probabilities are left alone, and the processor does nothing once the answer has started or when a prompt has no open reasoning block.
  - It tracks state per request slot on the GPU from each step's input token, so there is no per-step host synchronization.
  - It reads a prompt once at admission, including the output so far for preempted requests.
- **Logged log-probabilities are raw.** vLLM reports log-probabilities from the raw logits, before logits processors. The guard's effect therefore shows in sampled tokens, not in logged log-probabilities.

### 10.4 Validation

- **Greedy continuation at three failure points.** Unguarded: end of turn, end of turn, code. Guarded: `</think>` followed by a `python` block; `</think>` followed by an answer; unchanged.
- **Sampling replay** of 40 recovery requests that had ended empty in the prompt factorial, with the original messages and sampling settings, once per server:

  | Server | Empty answer | Answer contains `Move`/`Pick`/`Release` | Length stop | Median completion tokens |
  | --- | --- | --- | --- | --- |
  | Unguarded | 9 / 40 | 31 / 40 | 0 | 12.4k |
  | Guarded | 0 / 40 | 40 / 40 | 0 | 14.8k |

- **Rollout:** all six 27B instances were restarted with the guard at 12:13 EDT on 2026-10-23. Feedback-ablation plans record it under `serving.logits_processors`, and batches also hash its source.

### 10.5 Effect on earlier results

- **Prompt factorial (unguarded).**
  - Round 0, the primary outcome: 11 of 2,400 trials (0.5%) ended this way. The prompt selection stands.
  - Final success, a secondary outcome: of the trials that failed round 0, 183 of 815 without the motion-planning block (22%) and 57 of 342 with it (17%) ended on an empty answer during recovery. Final success is understated in both groups, more so without the block. The report notes this next to the final-success numbers.
- **Feedback-ablation pilot (unguarded):** mechanics only; excluded from all analyses. The main Stage A and Stage B run with the guard.
