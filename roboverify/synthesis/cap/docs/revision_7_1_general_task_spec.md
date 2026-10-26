# Revision 7.1: general task specification (`config_frozen_v6`)

- **Status:** implemented and run 2026-10-26 (Sections 5 and 6).
- **Goal:** the task given to the model is the task that RoboVerify solves: one program that takes *every* environment satisfying the precondition, with *any* number of blocks, to the postcondition. The prompt must say so, and the task specification must not state a particular number of blocks or goals.
- **Scope:** the task instructions only. The API description, the primitive contracts, the demonstrations, the feedback, the evaluated block counts and all budgets are `config_frozen_v5`'s.

## 1. Finding

The `config_frozen_v5` Pyramid instruction (rendered at the end of every first request, for example [this Revision 6 prompt](../../../runs/cap-revision6/20261026-openrouter-v3-demos/claude-haiku-5.5/pyramid/trials/cap_method_2-dsl_only-001/cap/20261026-164714-3ee7cd7-cap_method_2-recovery-f5d2d4fc/attempts/round_000/cap/20261026-164717-3ee7cd7-cap_method_2-pyramid-41813907/artifacts/trial_001/lmp/call_0001/prompt.txt), lines 1671–1677) reads:

> The input goal handle h0 identifies the head goal of a fixed goal layout of **6 goal locations (layers of 3, 2 and 1 goals**; …)

Grid says "**6 goal locations (3 rows of 2 goals** on the table; …)". Two problems:

1. **Specific counts.** The Grid and Pyramid instructions fix the layout size. A model can satisfy them with straight-line code for six placements, which is not the program RoboVerify synthesizes.
2. **No generality requirement.** No instruction of any task says that the policy must work for an arbitrary number of blocks. The tower instructions (Stack, Unstack, Reverse, ReStack) state no count, but they also do not say that the count is unknown.

## 2. Where the prompt text comes from

| Prompt part | Source | States a count? | Change |
| --- | --- | --- | --- |
| Method template (first lines, "Generate a complete Python robot policy …") | `cap_method_2` templates in `prompts/typed_v5.json` | no | none |
| API description, types, examples | `prompts/typed_v5.json` (via `prompt_version`) | no | none |
| Primitive contract explanations | `prompts/primitive_v2.json` (Revision 7) | no | none |
| Condition-mode rules | `prompts/typed_v5.json` | no | none |
| Input context ("b0 is the input block.", tbl, goal cursors) | `tasks.task_context` | no | none |
| **Task instruction** (`# Instruction: …`) | `INSTRUCTIONS` in the frozen configuration (`config_frozen_v1` towers, `config_frozen_v2` goal tasks, copied unchanged by v3–v5) | **Grid, Pyramid: yes** | **rewritten (v6)** |
| Demonstration block (Revision 6, when on) | `demonstrations.py`, `typed_v5.json` header | shows the recorded scenes: "pyramid, 6 blocks o0..o5", their goals | none (see 4.2) |
| Recovery feedback | `recovery.py`, `primitive_feedback.py` | lists the measured blocks of the evaluation scene | none (measured data) |

The evaluated block counts (`TASKS`: Stack/Unstack/Reverse 4, ReStack 5 with a 3-block tower, Grid 3×2, Pyramid 3+2+1) are run settings, not prompt text, and stay unchanged.

## 3. Proposed prompt change (`task-instructions-v2`)

### 3.1 Generality requirement

Every instruction gets one paragraph, placed after the physical-completion sentence and before the final request. Towers:

> The number of blocks is not given and is not fixed. Write one policy that takes every initial state satisfying the conditions above, with any number of blocks, to a state satisfying the goal. Do not assume or depend on a particular number of blocks.

Goal tasks:

> The size of the goal layout and the number of blocks are not given and are not fixed. Write one policy that takes every initial state satisfying the conditions above, for any layout of this form and as many blocks as goals, to a state satisfying the goal. Do not assume or depend on a particular number of blocks, goals, rows or layers.

The paragraph states the *specification* (its quantification over initial states and block counts), not a strategy: it does not mention loops, cursors or an order of placements. This keeps the Revision 7 rule ([Revision 7, §3](revision_7_update_prompt_and_config.md#3-rules-for-later-changes)).

### 3.2 Layout described by its form

| Task | v5 (`task-instructions-v1`) | v6 (`task-instructions-v2`) |
| --- | --- | --- |
| Grid | "a fixed goal layout of 6 goal locations (3 rows of 2 goals on the table; MoveRight advances within a row and MoveDown moves from a row head to the next row's head)" | "the goal layout. The layout has R rows of C goals each on the table (R >= 1 and C >= 1; neither is given). MoveRight advances within a row and MoveDown moves from a row head to the next row's head. The layout does not change during execution" |
| Pyramid | "a fixed goal layout of 6 goal locations (layers of 3, 2 and 1 goals; h0's row is the bottom layer on the table, and MoveDown moves from a layer's head to the head of the layer above)" | "the goal layout. The layout has n layers (n >= 1, not given) holding n, n-1, ..., 2, 1 goals from the bottom layer up. h0's row is the bottom layer on the table, and MoveDown moves from a layer's head to the head of the layer above. The layout does not change during execution" |

"Fixed" in v5 meant fixed during the trial; v6 says that explicitly, so it cannot be read as "fixed size". All other sentences of the goal instructions (precondition, "as many blocks as goals", goal, physical completion) are v5's. The tower instructions are v5's plus the paragraph of 3.1.

### 3.3 Full v6 Pyramid instruction

```text
Pyramid task from synthesis/cfg/goal_collection.py (goal-occupancy-v1).
The input goal handle h0 identifies the head goal of the goal layout. The layout has n layers (n >= 1, not given) holding n, n-1, ..., 2, 1 goals from the bottom layer up. h0's row is the bottom layer on the table, and MoveDown moves from a layer's head to the head of the layer above. The layout does not change during execution; null is the empty goal. There is no input block.
Initially, every goal is empty and, for all distinct physical blocks x and y, Scattered(x,y) holds.
Blocks are cubes with side length 0.05 m; there are as many blocks as goals.
The goal is that every goal location is occupied: for every goal g, HasBlock(g), i.e. some block centre lies within 0.02 m (3D) of the goal centre.
Physical completion requires every goal location to hold a block at the end; an upper-layer block needs the blocks below it as support.
The size of the goal layout and the number of blocks are not given and are not fixed. Write one policy that takes every initial state satisfying the conditions above, for any layout of this form and as many blocks as goals, to a state satisfying the goal. Do not assume or depend on a particular number of blocks, goals, rows or layers.
Generate and invoke a policy using the provided primitive and predicate APIs.
```

## 4. What this does not change

### 4.1 Evaluation still uses one block count per task

Each trial still runs the generated policy on one scene with the `TASKS` counts. Success therefore measures the policy on that count only; it does not test whether the program generalizes. A program that ignores the instruction and hard-codes six placements still succeeds on a six-goal Pyramid. Measuring generality needs a separate check: execute each final policy, without new model calls, on scenes with other counts (for example Stack 3–6 blocks, Pyramid n = 2–4). That is a possible later revision; it is not part of v6.

### 4.2 Demonstrations show the evaluated count

All 15 demonstrations of a task are recorded at the evaluated count (archives pinned since v3/v4; the runner rejects a trace whose `num_blocks` differs from the run's). The demonstration block therefore lists, for example, "pyramid, 6 blocks o0..o5" and six goals. This is recorded data, not the specification; the instruction now tells the model not to depend on the count. Demonstrations at several counts would need new archives and a new frozen version.

### 4.3 Recovery feedback

Recovery requests contain measured evidence from the evaluation scene (block positions, events). This is measured feedback, which [Revision 7, §3](revision_7_update_prompt_and_config.md#3-rules-for-later-changes) allows. A recovery request repeats the first request, so it carries the v6 instruction too.

### 4.4 Statements about what a primitive does not do

Kept, as decided on 2026-10-26 ("primitive不做什么 is okay to expose"): the "does not lift", "does not lower/place", "does not plan obstacle avoidance" statements of [Revision 7, §1](revision_7_update_prompt_and_config.md#1-finding-prompt-audit-2026-10-26) are API descriptions.

## 5. Implementation

- [`configs/config_frozen_v6.py`](../configs/config_frozen_v6.py): v5 plus `INSTRUCTIONS` of `task-instructions-v2` (`INSTRUCTION_VERSION`). Run configs, model, serving, `TASKS`, archives and demonstration settings are v5's objects.
- [`configs/test_config_frozen_v6.py`](../configs/test_config_frozen_v6.py):
  - run configs (both demonstration arms), model and serving equal v5's;
  - tower instructions equal v5's plus only the generality paragraph;
  - no instruction contains a number other than the physical constants, "3D", the bound 1 and the symbolic "n, n-1, ..., 2, 1";
  - rendered `cap_method_2` prompts (Stack, Pyramid) differ from v5's only by the instruction.
- [`experiments/revision6_experiment.py`](../experiments/revision6_experiment.py): `--frozen v3|v6` (default v3, unchanged for earlier runs). `comparison_plan.json` records `frozen_config`, its source and `instruction_version`.
- `tests/prompt_parity.rendered_prompts` accepts an `instruction`.

v1–v5 are unchanged; their prompt hashes stay the same.

## 6. Experiments

Model `anthropic/claude-haiku-5.5` on OpenRouter, reasoning effort medium, otherwise v6 (method `cap_method_2`, `dsl_only`, full recovery with 4 additional rounds, `task-spec-v2` with 50 settling steps). The OpenRouter key is passed only through `OPENROUTER_API_KEY`.

### 6.1 Smoke (one trial per task)

[`runs/cap-revision7-1/20261026-smoke-v6/report.md`](../../../runs/cap-revision7-1/20261026-smoke-v6/report.md), demonstrations on, scene seed 0, git `a7743cc` (clean).

| Task | Result | Rounds | Provider tokens |
| --- | --- | ---: | ---: |
| Stack | success, round 0 | 0 | 35,591 |
| Unstack | success, round 0 | 0 | 86,229 |
| Reverse | goal not reached after 5 rounds (`max_recovery_rounds`) | 4 | 358,344 |
| ReStack | success, round 0 | 0 | 92,378 |
| Grid | success, round 0 | 0 | 83,502 |
| Pyramid | success, round 0 | 0 | 89,636 |

All six prompts end with the v6 instruction. The Reverse failure is a task outcome: every round executed to the end and failed the settled postcondition. The generated Stack, Grid and Pyramid policies are loops over predicates or goal cursors (`while not forall(...)`; nested `MoveRight`/`MoveDown` loops), not unrolled placements.

### 6.2 Demonstration ablation (same seeds, with and without demonstrations)

- Two arms: demonstrations on and off. Each arm: 6 tasks × 20 trials, scene seeds 0–19 (one per trial, identical in both arms; disjoint from the demonstration seeds 1000–1014). 240 trials.
- Roots: `runs/cap-revision7-1/20261026-demo-ablation-v6/{demos-on,demos-off}/claude-haiku-5.5/<task>/`; combined report at `runs/cap-revision7-1/20261026-demo-ablation-v6/report.md`.
- No failed trial is replaced. Paired comparison per task: trials where only one arm succeeds.
- Launched at git `a7743cc`; `comparison_plan.json` records `git_dirty: true` only because this document was still untracked.

Combined report: [`runs/cap-revision7-1/20261026-demo-ablation-v6/report.md`](../../../runs/cap-revision7-1/20261026-demo-ablation-v6/report.md). Task-spec success (`task-spec-v2`):

| Task | On, round 0 | On, final | Off, round 0 | Off, final | Both | Only on | Only off | Neither | Mean effort tokens, on (+ demo block) | Mean effort tokens, off | Mean recovery rounds on / off |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| Stack | 16/20 | 20/20 | 17/20 | 20/20 | 20 | 0 | 0 | 0 | 16,039 (+36,002) | 17,031 | 0.35 / 0.15 |
| Unstack | 15/20 | 20/20 | 2/20 | 15/20 | 15 | 5 | 0 | 0 | 19,104 (+51,298) | 87,680 | 0.35 / 2.40 |
| Reverse | 0/20 | 10/20 | 1/20 | 15/20 | 5 | 5 | 10 | 0 | 89,484 (+182,131) | 97,962 | 2.85 / 1.90 |
| ReStack | 2/20 | 14/20 | 0/20 | 3/20 | 2 | 12 | 1 | 5 | 115,822 (+285,399) | 126,061 | 2.85 / 2.30 |
| Grid | 18/20 | 20/20 | 18/20 | 20/20 | 20 | 0 | 0 | 0 | 10,900 (+91,912) | 12,596 | 0.20 / 0.10 |
| Pyramid | 20/20 | 20/20 | 20/20 | 20/20 | 20 | 0 | 0 | 0 | 8,167 (+81,503) | 11,988 | 0.00 / 0.00 |
| **All** | **71/120** | **104/120** | **58/120** | **93/120** | 82 | 22 | 11 | 5 | | | |

Failures by cause:

| Arm | Task | Trials | Last attempt |
| --- | --- | ---: | --- |
| on | Reverse | 8 | goal not reached after 5 rounds |
| on | Reverse | 2 | `get` found no object (1); Move nonconvergence (1) |
| on | ReStack | 6 | `get` found no object (2); Pick (1), Release (1), Move (2) nonconvergence |
| off | Unstack | 5 | Move nonconvergence |
| off | Reverse | 5 | round 0 output truncated at the 32,768-token output limit; no rollout, so recovery could not start |
| off | ReStack | 17 | output truncated (10 at round 0, no rollout; 3 in a later round); `get` found no object (1); Pick (1) and Move (2) nonconvergence |

Observations:

- Demonstrations raise round-0 success from 58 to 71 of 120 and final success from 93 to 104. The difference comes from Unstack (round 0: 15 vs 2) and ReStack (final: 14 vs 3). Without demonstrations, 13 Reverse/ReStack trials end because the model's first response exceeds the output limit; with demonstrations none does.
- Reverse is the exception: the off arm succeeds more often (15 vs 10; 10 seeds only off, 5 only on). With demonstrations, Reverse never succeeds at round 0, and 8 of its 10 failures run five complete rounds without reaching the goal.
- Stack, Grid and Pyramid are at or near ceiling in both arms.
- These are single runs per seed; a seed that succeeds in only one arm also reflects sampling variation, not only the demonstrations.
- Cost: the two OpenRouter keys were charged about $3.7 (on) and $2.4 (off) for this experiment.
