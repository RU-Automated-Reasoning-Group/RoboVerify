# Revision 6: configurable demonstration input for CaP

- **Status:** planned; not implemented. Awaiting the demonstration-generation interface and data from Yuning.
- **Goal:** give CaP access to the same selected demonstration data used by RoboVerify's demonstration-driven synthesis, with an explicit on/off setting for ablation.
- **Dependencies:** [task support plan](revision_5_add_tasks.md), an agreed demonstration schema, and validated data. Stack is the first integration target.

## 1. Current behavior and information boundaries

| Information | Current CaP synthesis | Planned demonstration-enabled synthesis |
| --- | --- | --- |
| Task instruction, API descriptions, mode rules | Provided | Retained |
| Demonstration state/action trajectories | Not provided | Selected demonstrations encoded through a shared input representation |
| Demonstration initial states | Not provided | Included as part of demonstration data |
| A new evaluation scene's initial coordinates | Not provided | Separate design choice; not implicitly added by enabling demonstrations |
| Runtime environment queries | Available to the generated policy | Retained; results are not automatically sent back to the LLM |

- RoboVerify uses demonstration states, segment starts, and candidate rollouts during synthesis. Demonstration access does not imply prior knowledge of a held-out test scene.
- Matching demonstrations controls information access, not all differences in representations, compute budgets, or search feedback.
- The current collector supports Stack only. Earlier Stack collections are documented, but no demonstration archive was present in this checkout during the October 19 review.

## 2. Planned input interface

- Add explicit demonstration-input on/off configuration; exact option names remain to be designed.
- Configure archive paths, selected trace IDs, and a deterministic serialization version.
- Validate task identity, block/goal bindings, trajectory completeness, and compatibility with the active primitive/predicate semantics.
- Use the same selected demonstration identities and representation across Baseline, Method 1, and Method 2.
- Keep the existing no-demonstration behavior available; version any new prompt package rather than rewriting frozen packages.

## 3. Representation and prompt design

- Agree on the schema with Yuning before implementing the loader/serializer.
- Evaluate an explicit state/action representation that retains relevant object relations, positions, and action transitions; document any sampling or compression.
- Keep full simulator snapshots for restoration, but do not expose simulator internals to the LLM merely because they are present in the archive.
- Do not provide the expert program source unless it is deliberately included as a shared input to the compared approaches.
- Specify whether policy, helper, and repair requests receive the same demonstration context or documented subsets. Avoid accidental differences between methods.
- Record demonstration hashes/IDs, ordering, serialization settings, exact rendered prompts, and token usage. Exceeding the input budget must be handled explicitly, not by silent truncation.

## 4. Experimental design

| Comparison | Controlled inputs | Measurement |
| --- | --- | --- |
| Demonstrations on versus off | Same method, model, condition mode, resource limits, and evaluation scenes | Held-out success, compliance, failures, calls, tokens, and latency |
| CaP versus RoboVerify | Same selected demonstrations and task/evaluation definitions | Policy quality and generalization, with compute/search differences reported |
| DSL-only versus coordinate conditions | Same demonstration selection and scene split | Effect of the expanded condition language and actual use of numeric predicates |

- Synthesize from a fixed demonstration set, freeze each program, and evaluate it on held-out scenes to measure generalization.
- Treat synthesis conditioned on each test scene as a separate experimental setting.
- Keep demonstration input and execution-failure recovery independent; [Revision 4](revision_4_failure_recover.md) remains a separate discussion.

## 5. Acceptance criteria

- Demonstration-off requests preserve the existing prompt behavior.
- Demonstration-on requests contain the configured data consistently across methods, with reproducible hashes and usage accounting.
- Reject incompatible/malformed datasets and prevent evaluation-scene data from entering demonstration inputs unintentionally.
- Validate at least one Stack integration end-to-end before extending to other tasks.
- Preserve original benchmark results and report any new ablation as a separate batch.
- This document does not authorize implementing these changes or launching new model experiments yet.
