# Revision 4.2: primitive nonconvergence

- **Status: not implemented yet; plan under discussion.** This document does not authorize implementation or new experiments.
- **Problem:** generated policies can request poorly constructed motion targets or transport sequences, then fail when Pick, Move, or Release does not converge within its primitive budget.
- **Scope:** clearer primitive contracts and compact, factual primitive-failure feedback. Preserve the existing method workflows, controller behavior, and task-success criteria.
- **Completed prerequisite:** [Revision 4.1 GPU scheduling](revision_4_1_gpu_scheduling.md), including its existing validation. Use it for later parallel experiments; this plan does not change scheduling.
- **Deferred work:** [unimplemented backlog](revision_4_unimplemented.md), including full historical programs, namespace/binding diagnostics, general loop/selection analysis, helper-context propagation, query/goal explanations, and preparation/API retries.

Each `4.x` revision should address one focused problem: define the change, implement it when authorized, run focused tests and a controlled comparison, record the outcome, then select the next problem. A failed primitive may be a symptom of an earlier selection or placement defect; this version must not claim to solve every failure with the same label.

## 1. Primitive contract and failure evidence

### 1.1 What is already present

The [current typed_v3 API footer](../prompts/typed_v3.json) already tells the model:

- Each primitive has at most 50 simulation steps; Pick's phases share that budget.
- Move uses a 0.002 m 3D position tolerance.
- Nonconvergence raises an execution failure; the program does not simply continue to the next primitive.
- One simulation step is one `env.step(action)`, not one wall-clock second. A longer trial deadline does not increase the primitive budget.
- Convergence alone does not certify grasp, collision freedom, physical support, or task completion.

The initial prompt also specifies relative Move offsets, fixed targets sampled at invocation, the possible block–gripper displacement, Pick's lack of an automatic lift, and Release's opening/retreat behavior. The missing contract detail is the explicit success criterion for each Pick/Release phase, rather than a complete absence of motion semantics or limits.

The [failure review](revision_4_recovery_report.md#43-what-each-feedback-supplied-and-what-it-omitted) finds **185 primitive-failure feedback transitions**, all with arguments, targets, residuals, keyframes, and goal clauses. Merely adding those same fields again is not the proposed fix. The improvement is to connect them through target arithmetic, phase information, and relevant action context.

### 1.2 Evidence motivating this focused revision

| Case | Established observation | Relevance and limit |
| --- | --- | --- |
| [Haiku Method 1 trial 4](revision_4_recovery_report.md#451-existing-feedback-can-support-targeted-repair-haiku-method-1-trial-4) | After feedback, the model adds source lift and high transfer before descent; the replacement succeeds from the same initial state | Existing feedback can support targeted repair. Exact contact diagnosis and feedback's causal contribution versus another generation were not isolated |
| [MiniMax Method 1 trial 18](case_study/revision_4_failure_recovery_cases/minimax_method_1_dsl_only_trial_018.md) | Round 0 double-counts reference coordinates, round 1 fixes that conversion, and round 4 reintroduces it | Show reference coordinates + supplied offsets → actual fixed target. Full-history injection remains a separate experiment |
| [MiniMax Method 1 trial 37](../../../runs/cap-recovery-analysis/20261020-history-review/trials/minimax-cap_method_1-037.md) | Absolute values supplied as Move offsets produce target `(2.604309, 1.800790, 1.048942)` m; later correction succeeds | The arithmetic error is observable without claiming a formally proven workspace boundary |
| [MiniMax Baseline trial 6](case_study/revision_4_failure_recovery_cases/minimax_baseline_dsl_only_trial_006.md) | Different Z formulas produce targets 0.333521 m and 0.998942 m, followed by separate Move failures | A repeated nonconvergence label need not represent the same target error or physical cause |
| [MiniMax Baseline trial 48](../../../runs/cap-recovery-analysis/20261020-history-review/trials/minimax-cap_google_baseline-048.md) | Release reference/offset change is followed by success | Distinguish opening from retreat and show the invocation-time reference. No isolated single-edit causal claim |
| [MiniMax Method 1 trial 18, round 3](case_study/revision_4_failure_recovery_cases/minimax_method_1_dsl_only_trial_018.md#6-round-3-base-relative-selection-restored-release-retreat-fails) | Opening completes; retreat fails with 0.013029 m residual | Report the failed phase instead of implying the gripper never opened |

Video observations of the arm stretching toward a distant target are useful diagnostic cues. A claim that the target is outside the reachable workspace needs additional kinematic evidence; nonconvergence and a video alone are not a formal reachability test. This revision does not introduce a guessed workspace box or silently clamp targets.

### 1.3 Candidate changes to discuss and implement separately

| Candidate change | Proposed content | Scope boundary |
| --- | --- | --- |
| Clarify primitive contracts in a new prompt version | Describe the position criterion for Pick's approach/descent and Release's retreat, the separate gripper open/closed criterion, shared per-call budget, and exception behavior | Preserve old prompt versions for comparison. Read thresholds from the actual controller configuration; do not change controllers or assume position convergence proves a grasp |
| Show target derivation in primitive feedback | For Move, record each reference handle/coordinate, supplied offset, resulting gripper target, and final gripper error. For Release, record frozen XY and invocation-time reference Z + retreat offset | Use the action-entry snapshot. Do not replace it with terminal coordinates or invent the intended target |
| Distinguish controller phases and measurements | Identify approach, descend, open/close, move, or retreat; record consumed steps, configured limit, applicable threshold, measured value, and convergence status | Position residual may be unavailable for gripper-state phases; represent the appropriate criterion rather than fabricate a positional error |
| Include compact action context | Show the relevant preceding completed action and measured before/after gripper–block displacement, with source locations and goal outcome | This is a local primitive sequence, not a new full-trajectory, loop-progress, or historical-program feature |

First select one intervention for implementation and testing. For example, evaluate target/phase feedback with the original initial prompt, then evaluate a contract-text revision separately if warranted. Do not combine both changes and attribute the result to one. Exact configuration names and the first implementation slice remain to be agreed; no new flags are implemented by this plan.

Baseline retains its original executor and helper-generation/checking workflow. If shared API text or runtime feedback is evaluated, record it as a versioned experimental input, not a change to the pinned Google implementation. Do not add Baseline preparation repair, a pre-execution DSL gate, or an altered namespace model.

### 1.4 Proposed deterministic evidence record

| Field family | Data source | Interpretation |
| --- | --- | --- |
| Attempt, source, action and phase identity | Existing checkpoint/call/primitive events | Locate the failing action, including an inline helper when recorded |
| References and supplied offsets | Invocation arguments and action-entry coordinate snapshot | Derive each target axis using the actual API semantics |
| Fixed target and terminal state | Controller phase record and corresponding trajectory state | Explain the measured residual against the target that was actually commanded |
| Step and convergence criteria | Resolved controller configuration and result | Explain which threshold was not reached within which limit |
| Gripper–block displacement | Coordinate snapshots before/after the relevant actions | Distinguish gripper targets from block-centre placement intentions; do not assume rigid attachment |
| Relevant preceding action | Recorded action boundary, target and convergence result | Expose the approach/transfer sequence without dumping the whole log |
| Existing task outcomes | Existing physical/logical evaluator results | Retain the distinction between convergence and completion; no new goal-diagnosis feature |

Use deterministic host code, with no extra diagnosis LLM call and no replay of generated helpers or simulation for evidence selection. Keep facts, arithmetic derivations, model-stated hypotheses, and retrospective diagnoses distinguishable. Missing legacy observations are `unavailable`, not fabricated.

Retain current latest-source feedback and earlier short summaries. Full historical code and an edit ledger remain in the backlog, even though trial 18 motivates them. If a target formula is demonstrably doubled, present the operands and evaluated formula; do not automatically rewrite the policy or promise a corrected target will achieve the task.

### 1.5 Compatibility, limits, and logging

- Keep the existing 50-step primitive limit, convergence thresholds, task specification, maximum four recoveries/five rollouts, and 16,384 output-token cap for the first comparison. Increasing controller or model budgets is a separate intervention.
- Every replacement rolls out from the same saved initial state in a fresh execution context. Failed trajectories remain debugging evidence.
- Retain the 8,192 UTF-8-byte evidence bound and current per-round/whole-trial budgets. Compress repeated metadata to fit useful primitive facts instead of appending an unbounded event dump. Count the complete request and reserved output using the existing accounting policy.
- Keep the compatibility profile available, with exact prompt/feedback version provenance. Do not silently truncate source, hide omitted evidence, or change behavior for recovery-off trials when testing a recovery-feedback-only change.
- Preserve condition modes. Numeric diagnostics do not authorize numeric branch/loop/selection conditions in `dsl_only`.
- Save prepared and delivered feedback, source/phase associations, field inclusion/omission decisions, serialized sizes, and actual request payloads. Verify equality through the next policy call.
- No API/preparation retry, helper-feedback propagation, general namespace diagnosis, or general loop/query repair is included here. If the primitive failure originated in such a defect, classify it rather than expanding scope silently.

### 1.6 Focused tests and evaluation

| Validation stage | Required evidence |
| --- | --- |
| Contract consistency | Prompt descriptions and reported criteria agree with the actual controllers; old prompt versions remain reproducible |
| Target arithmetic | Regression fixtures for repeated reference coordinates, invocation-time sampling, gripper–block compensation, and different wrong targets sharing the same failure label |
| Phase attribution | Pick phase failures and Release opening-versus-retreat failures report the appropriate criterion and shared step count |
| Evidence delivery | Deterministic bounded selection; explicit missing/omitted fields; prepared feedback equals the actual next request; source attribution is preserved |
| Compatibility | Unchanged legacy behavior, method-specific checking, limits, condition modes, and exact-state reset; no extra model diagnosis calls |
| Scripted simulator checks | Verify the collector against known executed action targets and outcomes before any live experiment |

For a feedback comparison, branch control and candidate profiles from the **same archived failed initial policy and initial state**, with the same model, effort, task, condition mode, output cap, recovery count, and resource limits. A contract-text comparison should be separate; record whether it evaluates initial generation or only subsequent recovery generation. Do not treat different independently generated failures as matched candidates.

Measure final task success, rounds, model calls/tokens/time, primitive/phase failure counts, verified coordinate-formula defects, and budget/preparation stops. Preserve all failures in the declared denominator and separate physical success, logical postcondition, and DSL compliance. A disappeared exception or a shorter program is not sufficient evidence of successful repair.

Run independent trials in parallel under the implemented GPU scheduler and shared request limits, with the same scheduling policy across profiles. The experiment size and first intervention are not yet fixed. Record the selected change, checks, results, and unresolved limits in this document before moving to the next focused `4.x` version.
