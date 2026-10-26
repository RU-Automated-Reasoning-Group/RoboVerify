# Revision 7: remove the motion-strategy instruction from the primitive prompt (`config_frozen_v5`)

- **Status:** implemented 2026-10-26 (Section 4). No experiment has been run under `config_frozen_v5` yet.
- **Goal:** the CaP prompts must not state, in words, how a manipulation should be decomposed. In particular, they must not tell the model to lift, transfer and then descend. The model can still learn such patterns indirectly, from the demonstrations (Revision 6) or from its own feedback.
- **Scope:** one sentence of the primitive-contract prompt. Everything else in `config_frozen_v4` is unchanged.

## 1. Finding (prompt audit, 2026-10-26)

Every frozen configuration enables the primitive-prompt cell `p0101`: convergence criteria and motion-planning limits on, target offsets and partial execution off ([Revision 4.3 report, §3.3](revision_4_3_report.md)). The motion-planning block in [`prompts/primitive_v1.json`](../prompts/primitive_v1.json) reads:

> Motion-planning boundary:
> Move drives toward its fixed gripper target. It does not plan obstacle avoidance or insert clearance waypoints. **Express any required lift, transfer, and descent stages explicitly.** A reachable target is not guaranteed to converge within the primitive budget. No universal safe height, maximum displacement, or guaranteed workspace box is supplied. An API example is not a certified safe task plan.

The bold sentence is an instruction about the intended motion strategy, not a description of the API. The prompts that the frozen runs actually sent (`lmp/call_*/prompt.txt`) all contain it:

| Runs | Configuration | Prompts containing the sentence |
| --- | --- | ---: |
| Revision 5 | `config_frozen_v1`, `v2` | 1,696 / 1,696 |
| Revision 6 | `config_frozen_v3` | 760 / 760 |

A recovery or repair request is the original prompt plus feedback, so every recovery request carried the sentence too.

**Clean parts of the prompt:**

- The task instructions (`INSTRUCTIONS`) and task context state only preconditions, goals and the `tbl` convention.
- The fixed recovery text (`RECOVERY_INSTRUCTIONS`, `recovery_prompt`) only asks for a revision and explains how to read the evidence.
- The Method 2 policy and repair templates, and the condition and output rules, say nothing about motion strategy.
- The rest of each feedback request is measured data. Source comments from earlier rounds (for example `# lift`) are the model's own words, echoed back.

**Statements kept (descriptions of a primitive's own behavior):**

| Text | Where | Why kept |
| --- | --- | --- |
| "Approach horizontally at the current gripper height, open, descend to the block centre, and close" | Pick API entry (`typed_v4`, `typed_v5`) | what Pick itself executes |
| "No automatic lift or transport follows." | Pick API entry | states what Pick does not do |
| "After closing, Pick does not recheck its earlier position criteria and does not automatically lift." | convergence-criteria block | same |
| "This does not lower/place the block before opening." | Release API entry | same |
| "Move drives toward its fixed gripper target. It does not plan obstacle avoidance or insert clearance waypoints." | motion-planning block | states what Move does not do |
| "Approach uses entry block XY/gripper Z; descent retains XY and refreshes block Z after opening" | `primitive_feedback.py` | describes Pick's phases in feedback |
| API examples with a Z offset (e.g. `[0.0, 0.0, 0.08]`) | examples sections | headed "not safe-path prescriptions" |

These say what a primitive does, or does not do. They leave it to the model to decide what program to write, as the paper's API description would. The model can infer that a transfer needs clearance, but no text tells it to add stages. Removing these too would hide real API behavior; they stay unless the user decides otherwise.

**Consequence for earlier results.** Revisions 4.3 to 6 were run with the sentence in place. The Revision 4.3 prompt factorial partly measured its effect, because the motion-planning bit is the fourth bit of the cell code (`p0001`, `p0101` and the other selected cells all have it on). Earlier reports are not edited; this section records that their prompts contained the sentence.

## 2. Change

### 2.1 Prompt package `primitive-prompt-v2`

New file [`prompts/primitive_v2.json`](../prompts/primitive_v2.json): `primitive_v1.json` with exactly one sentence removed from `explain_motion_planning_limits`. The other three blocks, the header and the option names are byte-identical.

| Block | v1 | v2 |
| --- | --- | --- |
| `explain_motion_planning_limits` | … or insert clearance waypoints. Express any required lift, transfer, and descent stages explicitly. A reachable target … | … or insert clearance waypoints. A reachable target … |
| other three blocks, header | – | unchanged |

`primitive_v1.json` is not edited, so v1–v4 render byte-identical prompts.

### 2.2 Selecting the package

- `RunConfig.primitive_prompt_version`: `"primitive-prompt-v1"` (default, v1–v4) or `"primitive-prompt-v2"`.
- [`primitive_prompts.py`](../primitive_prompts.py) loads both packages. `render`, `provenance` and the logged `primitive_prompt_package.json` use the selected version. The four Boolean flags in `PrimitivePromptConfig` are unchanged.
- Prompt provenance (`primitive_prompt_version`, `primitive_prompt_package_sha256`, `primitive_prompt_rendered_sha256`) identifies the package in every result. For v1 it is unchanged.

### 2.3 Frozen configuration `config_frozen_v5`

[`configs/config_frozen_v5.py`](../configs/config_frozen_v5.py) is `config_frozen_v4` plus `primitive_prompt_version: "primitive-prompt-v2"`, for all six tasks and both demonstration arms. Nothing else changes:

| Setting | v4 | v5 |
| --- | --- | --- |
| Primitive prompt cell | `p0101` (convergence, motion-planning limits) | same flags |
| Primitive prompt package | `primitive-prompt-v1` | **`primitive-prompt-v2`** |
| Model, serving, budgets, recovery (`full`, 4 rounds), method `cap_method_2`, `dsl_only`, `typed_v5` | v4 | unchanged |
| Success criterion | `task-spec-v2`, `post_settling_steps: 50` | unchanged |
| Initial settling | 50 steps (default `initial_settling_steps`) | unchanged |
| Demonstration archives and SHA-256 pins | v4 | unchanged |
| Task instructions and block counts | v4 | unchanged |

The prompt sent to the model differs from v4 only by the removed sentence (one line of the API description). The demonstration block is unchanged, so its token count is v4's.

## 3. Rules for later changes

- Keep the CaP prompt an API description: what each primitive does, its parameters, limits and failure behavior. Do not prescribe a task decomposition or a motion strategy (for example lift, transfer, descend; approach heights; waypoint order).
- Information about strategy may reach the model only through the demonstrations (when enabled) and through measured feedback.
- Any such prompt change is a new prompt package and a new frozen version, applied to all six tasks ([CaP agent guide](../AGENTS.md)).

## 4. Implementation (2026-10-26)

- `prompts/primitive_v2.json`, `primitive_prompts.py` (both packages, version selection), `RunConfig.primitive_prompt_version` with validation, runner logging of the selected package.
- `configs/config_frozen_v5.py` and `configs/test_config_frozen_v5.py`:
  - v5 differs from v4 only in `primitive_prompt_version`;
  - the v5 prompt equals the v4 prompt with only the sentence removed, and contains no "lift, transfer, and descent" text;
  - v1–v4 prompt hashes are unchanged (existing prompt-parity tests).
- The CaP agent guide and README point new experiments to v5, and the agent guide states the prompt rule of Section 3.
- No model was called; the change affects only rendered prompt text. No v5 smoke run has been made yet.
