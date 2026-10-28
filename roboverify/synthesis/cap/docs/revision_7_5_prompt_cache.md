# Revision 7.5: provider prompt caching for `config_final_v1`

- **Status:** analysis and plan, 2026-10-27. Nothing is implemented.
- **Scope:** Method 2 with recovery and demonstrations under [`config_final_v1`](../configs/config_final_v1.py), on hosted Anthropic models through OpenRouter (Opus 5.5, Haiku 5.5).
- **Constraint:** the prompt text the model sees must not change. Reordering without additions or deletions is allowed only if needed.
- **Evidence:** the `final_v1` smoke run `runs/cap-final-v1/20261027-smoke/` (Opus 5.5, one trial per task, seed 0), plus generate-only renders of every task for seeds 0, 1 and 37.

## 1. Summary

1. **There is no prompt cache today.** Every request is one plain string in one user message (`llm.py`, `HTTPModel.payload`). All nine smoke calls report `cached_tokens: 0` and `cache_write_tokens: 0`. Anthropic models cache only content marked with `cache_control`.
2. **The whole round-0 request is identical across trials of a task**, not just the demonstration block. The evaluation seed changes only the initial block positions, and the round-0 prompt never describes the scene. Renders for seeds 0, 1 and 37 are byte-identical for all six tasks and equal the prompts the smoke actually sent (Section 4).
3. **Within a trial:**
   - Every recovery request starts with the complete round-0 prompt.
   - Every DSL-repair request starts with the demonstration block.
4. **A hidden cost needs no cache at all:** each trial sends the full demonstration block once more, only to count its tokens (`demonstrations.provider_usage_tokens`).
   - This is 32% of the smoke bill: $1.35 of $4.18, which exactly explains the gap between the $2.83 logged on the calls and the $4.18 charged to the key.
   - The count is deterministic per task and model, so it can be measured once.
5. **Recommended change (Section 6):**
   - Measure the demonstration tokens once per batch.
   - Send each request as content parts at boundaries that already exist, with `cache_control` on the demonstration block and on the round-0 prompt.
   - The concatenated text stays byte-identical, and no reordering is needed.
   - Moving the API section ahead of the method header would add only about 2k cached tokens per repair call, so it is not recommended.
6. **Estimated effect for 50 Opus trials per task, counting round 0 only** (a lower bound, Section 7):

   | Setup | Cost |
   | --- | ---: |
   | Today | $155 |
   | Count measured once | $87 |
   | Count measured once, plus caching | about $23 |

   Recovery calls drop from about $0.43 to $0.15 each and repair calls from $0.38 to $0.12 (ReStack figures).

## 2. Requests sent by one Method 2 trial

The framework is in [progress report Oct 22–26, Section 0](progress_report_Oct22-26.md#0-framework-overview-method-2-with-failure-recovery). Under `final_v1`, `LoggedModel.complete` puts the demonstration block in front of every request. `recovery_prompt` appends the recovery feedback to the first request of rounds 1–4 only. The repair requests of Loop 1 carry no recovery feedback.

Notation:
- `D`: demonstration block, which ends with `END OF DEMONSTRATIONS\n`.
- `H₀`: Method 2 policy header ("Generate a complete Python robot policy …").
- `H_r`: repair header ("Revise the complete Python policy …").
- `A`: API section ("Available robot APIs …", shared effects, condition rules, primitive notes). It is 8,631 characters for ReStack.
- `I`: input handles, task instruction and request line.
- `R`: the recovery header and fixed instructions ("RECOVERY REQUEST (cap-recovery-v2) …").
- `F_k`: round k's feedback (current source, bindings, failure evidence and accumulated history).

| Request | When | Text | Same across trials of a task? |
| --- | --- | --- | --- |
| Token count, two calls | once per trial, before round 0 | `D + "Reply with OK."`, then `"Reply with OK."` (16 output tokens) | yes, fully |
| Round-0 policy | once | `D H₀ A I` | yes, fully |
| Repair | after each failed DSL check, up to 4 per round | `D H_r A`, then diagnostics and the current candidate | `D`, `H_r` and `A` only |
| Round-k policy, k ≥ 1 | once per recovery round | `D H₀ A I R F_k` | `D H₀ A I R`; `F_k` is specific to the trial |

These structures were measured on the ReStack smoke trial:

- **Round 1 vs round 0:** the round-1 request (157,316 characters) begins with the whole round-0 request (127,615 characters).
- **Repair vs round 0:** the repair request shares the first 117,186 characters with round 0, which is exactly `D` plus one newline.
- **What follows `D` in the repair request:** `H_r` replaces `H₀`. The API section `A` then matches round 0 byte for byte. Diagnostics and the candidate replace `I`.

## 3. What the evaluation seed changes

`batch.py` gives trial k the seed `seed_start + k − 1`. The seed has two uses:
- `tasks.prepare_environment` calls `np.random.seed(seed)` before building, resetting and settling the environment.
- `check_evaluation_seed` rejects a seed that collides with a demonstration's seed.

Comparing seeds 0 and 1:

| Task | Blocks | Initial block positions | Gripper start | `b0` / goal layout |
| --- | ---: | --- | --- | --- |
| Stack | 4 | differ, up to 31.6 cm | same | `b0` = block 0 |
| Unstack | 4 | tower base differs (2.1 cm for these seeds) | same | `b0` = block 0 |
| Reverse | 4 | tower base differs (2.1 cm) | same | `b0` = block 0 |
| ReStack | 5 | tower and loose blocks differ, up to 13.6 cm | same | `b0` = block 0 |
| Grid | 6 | source blocks differ, up to 28.9 cm | same | goal layout identical |
| Pyramid | 6 | source blocks differ, up to 28.9 cm | same | goal layout identical |

The seed therefore changes the scene, and through it the rollout and the feedback `F_k`. The model's sampled programs also differ between trials (no temperature or seed is set), so repair diagnostics and candidates differ too.

The following do **not** depend on the seed:
- the demonstration selection (fixed seeds 1000–1014 per task, the same block in every trial);
- the instruction, which has no counts or positions (Revision 7.1);
- the API text;
- the recovery header.

## 4. What is identical across trials

Generate-only renders under `final_v1` (demonstrations on, Method 2, a scripted checker failure to reach a repair request), seeds 0, 1 and 37:

| Task | Round-0 characters | Repair characters | Seeds 0/1/37 identical | Equal to smoke round 0 |
| --- | ---: | ---: | --- | --- |
| Stack | 53,974 | 53,123 | yes | yes |
| Unstack | 66,647 | 65,355 | yes | yes |
| Reverse | 86,076 | 84,500 | yes | yes |
| ReStack | 127,615 | 126,195 | yes | yes |
| Grid | 136,993 | 135,414 | yes | yes |
| Pyramid | 143,973 | 142,297 | yes | yes |

The repair prompt here is identical only because the scripted candidate is the same. With a real model, only `D H_r A` repeats.

The demonstration block differs between tasks, and the API section `A` follows it, so tasks share no prefix. Caching is per task.

## 5. Why nothing is cached now

- `HTTPModel.payload` sends `messages=[{"role": "user", "content": prompt}]` with `prompt` as one string. Anthropic caches only up to a content part marked `cache_control`. OpenRouter forwards these marks, but `extra_body` cannot express them, because `payload` refuses to override `messages`.
- A breakpoint marks the end of a cacheable prefix. A later request hits the cache only if its parts up to that breakpoint are identical. A single-part prompt can therefore never hit, because its only possible breakpoint is at the end, after trial-specific text.
- `serving/prefix_cache.py` is vLLM prefix caching for the local server. It does not apply to OpenRouter.
- Every smoke call was served by "Claude Platform on AWS". The cache lives at the upstream provider, so a request routed to another provider would miss.

**Pricing used below.** Fitted from the smoke calls' logged `cost`: $4 per million input tokens and $20 per million output tokens for Opus 5.5 on OpenRouter. Anthropic's documented cache rules apply: a write costs 1.25× input with a 5-minute TTL that every hit refreshes (a 1-hour TTL costs 2×), and a hit costs 0.1× input. Confirm on the OpenRouter bill. Each breakpoint's prefix must exceed a model-specific minimum (1–4k tokens); every breakpoint proposed here is past 26k tokens.

## 6. Design: maximize cache hits without changing the prompt text

### 6.1 Measure the demonstration tokens once (largest saving, no cache needed)

With `token_counter: provider_usage` (`final_v1` with hosted models), `prepare()` runs in every trial and sends `D` once more. The smoke trials paid for 337,325 such input tokens ($1.35).

The block and model are identical in every trial of a task, so:

- Measure the count once per (task, `demonstration_context_sha256`, model, base URL) when the batch is created.
- Store it in the batch manifest. Workers use it after checking the SHA-256 of the block they rendered.
- The recorded `demonstration_tokens` stays the same number, with `token_measurement.method` noting that it was measured once.
- The effort-budget accounting is unchanged.

Savings: the full `D` per trial, about $67 for 50 Opus trials of each of the six tasks (Section 7).

### 6.2 Split each request into content parts at existing boundaries

`LoggedModel.complete` already knows where `D` ends (it concatenates `demonstrations.text + method_prompt`). `recovery_prompt` knows where the original round-0 text ends (`original + "\n\nRECOVERY REQUEST…"`). Send the same text as parts:

| Request | Part 1 | Part 2 | Part 3 |
| --- | --- | --- | --- |
| Round 0 | `D` ✱ | `H₀ A I` ✱ | — |
| Round k ≥ 1 | `D` ✱ | `H₀ A I` ✱ | `R F_k` |
| Repair | `D` ✱ | `H_r A` + diagnostics + candidate | — |
| Token count (if kept, see 6.1) | `D` ✱ | `"Reply with OK."` | — |

✱ means `cache_control: {"type": "ephemeral"}`. That is at most two breakpoints per request; Anthropic allows four.

How each request uses the cache:

- **Round 0 of the first trial of a task:** writes `D` and `D H₀ A I`.
- **Round 0 of every later trial in the TTL window:** reads the whole input.
- **Every recovery request:** reads the round-0 prefix and pays only for `R F_k`.
- **Every repair request:** reads `D`.

The concatenation of the parts is byte-identical to today's string. Keep `prompt.txt` and `prompt_sha256` as the concatenated text, so prompt parity with `final_v1` is checked exactly as now. `request.json` and `request_sha256` change because the message now has parts.

Risk: the model receives several text parts instead of one string. The characters are identical, but tokenization at a boundary may differ slightly. Both boundaries fall at a newline (`…END OF DEMONSTRATIONS\n` | `\nGenerate…`, and `…APIs.` | `\n\nRECOVERY…`). Record the change as a new configuration version (Section 8) rather than calling it `final_v1`.

### 6.3 Reordering (evaluated; not recommended)

- **Repair:** moving `A` in front of the headers (`D A H₀ I` and `D A H_r …`) would let repairs also read `A`. That is about 2.2k tokens (about $0.008 per repair call with Opus). It changes the order the model reads, and both `final_v1` prompt hashes and the Revision 4.3 golden prompts would change. Not worth it.
- **Across tasks:** putting `A` before `D` would give tasks a shared prefix of only about 2.2k tokens. Not worth it.
- **Recovery history:** `F_k` puts the current source first and the accumulated history last. Round k + 1 cannot extend round k's text by reordering alone, because round k's "current" evidence is re-rendered in compact form as history. Reuse beyond `D H₀ A I R` would need content changes, so it is out of scope.

### 6.4 Scheduling, TTL and routing (no prompt change)

- **TTL:** `revision6_experiment` runs one batch per task with up to five workers. Same-task round-0 and recovery requests arrive every few tens of seconds, which keeps a 5-minute cache warm. The 1-hour TTL (2× write) is unnecessary except for sparse, slow batches.
- **First wave:** five workers start together, so up to five requests may each write the cache (1.25×) before any can read it. Options:
  - Accept it: at most four extra writes per task, under about $1.5 per task for Opus.
  - Delay workers 2–5 until worker 1's round-0 response arrives.
  - If 6.1 keeps a count request, send it as `[D ✱, suffix]` so it writes `D` first.
- **Routing:** the cache is held by the upstream provider. Optionally pin it with OpenRouter `provider: {"order": ["…"], "allow_fallbacks": false}` in `extra_body`. Pinning trades availability for hit rate. Without pinning, record `provider` per call (already in `response.json`) and measure the hit rate.

## 7. Expected cost (Opus 5.5, $4/M input, $20/M output)

Per task, one trial that succeeds in round 0. `D` and `P₀` (round-0 prompt tokens) and the output `O` come from the smoke. "Write" is the first trial in a cache window; "hit" is every later one.

| Task | `D` | `P₀` | `O` | Today | 6.1 | 6.1 + 6.2, write | 6.1 + 6.2, hit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Stack | 26,668 | 30,349 | 1,457 | $0.257 | $0.151 | $0.181 | $0.041 |
| Unstack | 34,199 | 38,038 | 2,521 | $0.339 | $0.203 | $0.241 | $0.066 |
| Reverse | 46,109 | 50,117 | 2,492 | $0.435 | $0.250 | $0.300 | $0.070 |
| ReStack | 72,253 | 76,142 | 5,873 | $0.711 | $0.422 | $0.498 | $0.148 |
| Grid | 76,593 | 81,085 | 1,086 | $0.652 | $0.346 | $0.427 | $0.054 |
| Pyramid | 81,503 | 86,035 | 1,581 | $0.702 | $0.376 | $0.462 | $0.066 |
| Six tasks | | | | $3.10 | $1.75 | | $0.45 |
| × 50 trials | | | | $155 | $87 | | about $23 |

Calls after round 0, ReStack and Reverse smoke figures:

| Call | Today | Cached |
| --- | ---: | ---: |
| ReStack recovery (`P₀` + 14.9k feedback, 3.0k output) | $0.425 | $0.151 |
| ReStack repair (`D` + 4.8k, 3.4k output) | $0.376 | $0.116 |
| Reverse recovery (`P₀` + 12.0k feedback, 1.7k output) | $0.283 | $0.103 |

With caching, output tokens dominate the remaining cost (reasoning tokens count as output). A trial that uses all 4 recovery rounds plus a few repairs costs about $0.6–1.2 for ReStack instead of $2–4.

Smoke check: only within-trial reuse applies (one trial per task). It would have cost about $2.5 instead of $2.83 (plus $1.35 for the count requests). Cross-trial reuse is what matters in the 50-trial run.

## 8. Implementation plan (not started)

1. **`demonstrations.py` / `batch.py` (6.1):** a measured-once count. `revision6_experiment.create` (or `create_batch`) renders the block for the task, counts it once with `provider_usage`, and writes the count into the batch manifest with its SHA-256. `prepare()` uses the stored count when the hashes match and falls back to measuring otherwise.
2. **`llm.py` (6.2):** `HTTPModel.payload(prompt, segments=None)`. With `ModelConfig.prompt_cache` on (proposed field: `{"enabled": bool, "ttl": "5m"|"1h"}`), it sends `content` as a list of text parts and marks the parts that end at the given offsets.
   - `LoggedModel.complete` passes the end of `D`.
   - `recovery_prompt` returns the end offset of the original text.
   - Parse `prompt_tokens_details.cached_tokens` / `cache_write_tokens` into the call record, the trial result and the batch summary.
3. **Configuration:** `config_final_v2` = `final_v1` + `prompt_cache` on + measured-once count. Prompt text and every other setting stay unchanged. `final_v1` stays frozen.
4. **Tests:**
   - The concatenated parts equal the `final_v1` prompt for every task and request kind; the round-0 and repair `prompt_sha256` match `final_v1`.
   - `payload` shape with and without caching; the providers' usage fields are parsed.
   - The count is reused only when the hashes match.
   - Demonstrations off: no parts and no change.
5. **Smoke (about $1 with Opus):** Stack, three trials run one after another.
   - Trial 1 should show `cache_write_tokens` ≈ `P₀`.
   - Trials 2–3 should show `cached_tokens` ≈ `P₀` and cost about $0.04–0.07 each.
   - Then one ReStack trial with recovery, to check that the recovery and repair calls read their prefixes.

## 9. Decisions needed

1. Adopt 6.1 (count measured once)? It changes no request the model sees.
2. Adopt 6.2 (content parts with `cache_control`) as `config_final_v2`, with the prompt text unchanged?
3. TTL 5 minutes (recommended) or 1 hour, and whether to stagger the first wave or pin a provider (6.4).
4. Reordering (6.3): recommended not to do it.
