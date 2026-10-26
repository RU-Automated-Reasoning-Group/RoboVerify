# LLM client

[`LoggedModel`](../../llm.py#L207) wraps the model backend. Every LLM call in a trial goes through [`LoggedModel.complete`](../../llm.py#L224-L382).

## What one call does

1. **Compose the request:** append recovery feedback to a policy request, and prepend the demonstration block if on (see [Prompt builder](prompt_builder.md)).
2. **Reserve budget:** when recovery is on, reserve the expected input and output tokens against the round and trial budgets before sending ([`RoundBudget.reserve_request`](../../recovery.py#L58)).
3. **Send:** [`HTTPModel.generate`](../../llm.py#L63) posts an OpenAI-style chat request to the vLLM server. Retryable transport errors are retried with backoff (`transport_retries`).
4. **Log:** prompt, request, response, usage, finish reason and hashes go to `lmp/call_NNNN/` and the timeline ([Logger](logger.md)).
5. **Charge usage:** provider-reported total tokens, minus the demonstration block, are charged to the effort budget.
6. **Return code:** a truncated (`length`) or empty response raises `ModelError` (status `generation_failed`). Otherwise [`extract_source`](../../llm.py#L385) strips one Markdown fence and returns the source.

## Backends

- [`HTTPModel`](../../llm.py#L33): the real model (frozen: Qwen3.8 27B FP8 on vLLM, medium reasoning, 32,768 output tokens).
- [`ScriptedModel`](../../llm.py#L188): fixed responses for tests and scripted end-to-end checks.

The server-side reasoning-termination guard is in [`serving/`](../../serving/).
