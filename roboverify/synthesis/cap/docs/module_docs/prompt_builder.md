# Prompt builder

Assembles the text of each LLM request. All text comes from versioned JSON prompt packages in [`prompts/`](../../prompts/), so a run can be reproduced from its package version and hashes.

## Initial synthesis request (Method 2)

The Method 2 `policy` template is filled with three fields ([`Method2.generate`](../../cap_method_2/method.py#L58-L64)):

| Field | Content | Code |
| --- | --- | --- |
| `{api}` | DSL description: primitives (Pick/Move/Release), predicates, queries (`get`/`exists`/`forall`), observation getters, types and units; then the enabled primitive prompt blocks; then the condition rules for the mode (`dsl_only`) | [`api_description`](../../contracts.py#L206), [`render_api`](../../prompt_package.py#L44), [`primitive_prompts.render`](../../primitive_prompts.py#L54) |
| `{context}` | The bound input handles, e.g. "b0 is the input block", plus `tbl` for table tasks or goal handles for Grid/Pyramid | [`task_context`](../../tasks.py#L41) |
| `{instruction}` | The task instruction, derived from the shared task specification | `INSTRUCTIONS` in each frozen config ([v1](../../configs/config_frozen_v1.py#L126)) |

Two pieces are added around it by the [LLM client](llm_client.md):

- **Demonstration block (if on):** prepended to *every* request, so all requests share one prefix ([`llm.py`](../../llm.py#L232-L243)).
- **Recovery feedback (rounds 1–4):** appended to the *policy* request only ([`llm.py`](../../llm.py#L225-L231), [`recovery_prompt`](../../recovery.py#L587)). Contents: [Feedback builder](feedback_builder.md).

## Repair request

The Method 2 `repair` template contains the same `{api}`, the checker diagnostics and the current candidate source ([`method.py`](../../cap_method_2/method.py#L100-L105)). It does not include recovery feedback.

## Prompt packages

| Package | Adds |
| --- | --- |
| `typed_v3` | Typed API, Method 2 templates (tower tasks, frozen v1) |
| `typed_v4` | Goal API for Grid/Pyramid; tower-task prompts byte-identical to v3 (frozen v2) |
| `typed_v5` | Demonstration header and frame templates; with demonstrations off, byte-identical to v4 (frozen v3) |

The primitive prompt blocks live in [`prompts/primitive_v1.json`](../../prompts/primitive_v1.json). The frozen baseline enables convergence criteria and motion-planning limits (`p0101`).
