# Revision 4.2: primitive nonconvergence

- **Status: implemented and tested; 220/220 live trials completed.** All four Section 3 candidate changes are configurable. Full-treatment results are Baseline 18/20, Method 1 18/20, and Method 2 20/20; all ablation results and limitations are recorded in [Revision 4.2 report](revision_4_2_report.md).
- **Problem:** generated policies can request poorly constructed motion targets or transport sequences, then fail when Pick, Move, or Release does not converge within its primitive budget.
- **Scope:** four independently configurable additions to the general primitive/API prompt, available during initial generation and recovery. Primitive-failure feedback enrichment remains a separate candidate intervention. Preserve the existing method workflows, controller behavior, and task-success criteria.
- **Completed prerequisite:** [Revision 4.1 GPU scheduling](revision_4_1_gpu_scheduling.md), including its existing validation. Use it for later parallel experiments; this plan does not change scheduling.
- **Deferred work:** [unimplemented backlog](revision_4_unimplemented.md), including full historical programs, namespace/binding diagnostics, general loop/selection analysis, helper-context propagation, query/goal explanations, and preparation/API retries.

Each `4.x` revision should address one focused problem: define the change, implement it when authorized, run focused tests and a controlled comparison, record the outcome, then select the next problem. A failed primitive may be a symptom of an earlier selection or placement defect; this version must not claim to solve every failure with the same label.

## 1. What is already present

The [current typed_v3 API footer](../prompts/typed_v3.json) already tells the model:

- Each primitive has at most 50 simulation steps; Pick's phases share that budget.
- Move uses a 0.002 m 3D position tolerance.
- Nonconvergence raises an execution failure; the program does not simply continue to the next primitive.
- One simulation step is one `env.step(action)`, not one wall-clock second. A longer trial deadline does not increase the primitive budget.
- Convergence alone does not certify grasp, collision freedom, physical support, or task completion.

The initial prompt also specifies relative Move offsets, fixed targets sampled at invocation, the possible block–gripper displacement, Pick's lack of an automatic lift, and Release's opening/retreat behavior. The missing contract detail is the explicit success criterion for each Pick/Release phase, rather than a complete absence of motion semantics or limits.

The [failure review](revision_4_recovery_report.md#43-what-each-feedback-supplied-and-what-it-omitted) finds **185 primitive-failure feedback transitions**, all with arguments, targets, residuals, keyframes, and goal clauses. Merely adding those same fields again is not the proposed fix. The improvement is to connect them through target arithmetic, phase information, and relevant action context.

### 1.1 Verified implementation contracts

The following describes **current runtime behavior**, checked against the [instruction wrappers](../../api/instructions.py), [shared controller](../../api/control.py), and [CaP adapter](../adapter.py). These are controller phase-completion conditions, not formal guarantees of grasp, physical support, collision freedom, or task completion. They are not all stated explicitly in the current model-facing prompt.

| Primitive / phase | Target or operation | Completion criterion |
| --- | --- | --- |
| Pick: approach | Block XY sampled at Pick entry; initial gripper Z | 3D Euclidean gripper-to-target distance ≤ configured position tolerance, currently 0.002 m |
| Pick: open | Command gripper opening after approach completes | Controller open-state predicate is true |
| Pick: descend | Retain the approach target's XY; refresh block Z after opening | 3D distance to this descent target ≤ 0.002 m |
| Pick: close | Command closing after descent completes | Controller open-state predicate is false |
| Move | Fixed target from invocation-time X/Y/Z references plus offsets; command gripper closure during movement | **Only** gripper-to-target 3D distance ≤ 0.002 m; no independent grasp/closed-state test for Move completion |
| Release: open | Freeze target before opening: gripper entry XY and reference entry Z + offset | Controller open-state predicate is true |
| Release: retreat | Move toward the frozen XYZ target after opening completes | 3D distance ≤ 0.002 m, including XY drift; reaching Z alone is insufficient |

The controller's exact open/closed test is:

```text
finger_state_sum = observation[3] + observation[4]
threshold = control.gripper_threshold + control.gripper_tolerance
          = 0.052 + 0.001  (default: 0.053)
open   iff finger_state_sum > threshold
closed iff finger_state_sum <= threshold
```

This is a controller-internal predicate over the two recorded finger-state values, not a detector of a securely grasped or stably placed block. The raw observation indices above document the implementation for maintainers; they do not add an observation API for generated policies. Prompt text may describe the criterion without exposing array indexing, and any numerical values must follow the resolved controller configuration.

**Budget and stage boundaries:**

- Each CaP primitive constructs one controller with the default limit of 50 simulation steps. Its phases share the same counter. The lower-level instruction constructors accept other limits/configurations, but the current CaP adapter uses these defaults.
- A phase whose criterion already holds can complete with zero additional steps. A criterion reached on the 50th step still counts as satisfied; reaching the counter value alone does not cause failure. A later phase that needs additional steps cannot exceed the shared limit.
- Pick proceeds sequentially and stops at the first unsuccessful phase. After its final close-state test, it does **not** recheck the earlier approach/descent position criteria. Earlier completed phases are historical events, not predicates guaranteed to remain true at return.
- Release freezes the retreat target before opening. Later block movement or falling does not update that target. Its retreat controls all three axes to correct XY drift as well as Z error; it does not perform a new final grasp/support test.

**Failure propagation and recorded fields:**

- The lower-level controller returns a Boolean phase/operation result and stores `ControlResult`. Instruction evaluation saves `last_control_result`; it does not itself raise the CaP `PrimitiveFailure` on nonconvergence.
- The CaP adapter checks `last_control_result`, raises `PrimitiveFailure` on nonconvergence, and the runner stops the current rollout. Subsequent policy statements do not execute. Already applied physical effects are not rolled back by the primitive.
- `ControlResult.steps` is cumulative for the primitive and `phase` identifies the last attempted phase. Motion phases provide a position residual; open/close phases set `position_error=None`. The existing result does not contain an exact measured finger-state sum or complete per-phase result history; the implemented collector obtains them from phase events and recorded observations or marks them unavailable. Controller configuration is recorded observationally only when phase diagnostics are enabled.

The existing [controller regression suite](../../api/test_control.py) was run during this source review: **20 tests passed**. It covers controller termination, shared Pick budgets, Release ordering and XYZ drift, and numeric/named instruction parity using deterministic test environments. That source-review check validated the controller implementation; validation of the subsequently implemented Revision 4.2 prompt switches and feedback collector is recorded in Section 6.

## 2. Evidence motivating this focused revision

| Case | Established observation | Relevance and limit |
| --- | --- | --- |
| [Haiku Method 1 trial 4](revision_4_recovery_report.md#451-existing-feedback-can-support-targeted-repair-haiku-method-1-trial-4) | After feedback, the model adds source lift and high transfer before descent; the replacement succeeds from the same initial state | Existing feedback can support targeted repair. Exact contact diagnosis and feedback's causal contribution versus another generation were not isolated |
| [MiniMax Method 1 trial 18](case_study/revision_4_failure_recovery_cases/minimax_method_1_dsl_only_trial_018.md) | Round 0 double-counts reference coordinates, round 1 fixes that conversion, and round 4 reintroduces it | Show reference coordinates + supplied offsets → actual fixed target. Full-history injection remains a separate experiment |
| [MiniMax Method 1 trial 37](../../../runs/cap-recovery-analysis/20261020-history-review/trials/minimax-cap_method_1-037.md) | Absolute values supplied as Move offsets produce target `(2.604309, 1.800790, 1.048942)` m; later correction succeeds | The arithmetic error is observable without claiming a formally proven workspace boundary |
| [MiniMax Baseline trial 6](case_study/revision_4_failure_recovery_cases/minimax_baseline_dsl_only_trial_006.md) | Different Z formulas produce targets 0.333521 m and 0.998942 m, followed by separate Move failures | A repeated nonconvergence label need not represent the same target error or physical cause |
| [MiniMax Baseline trial 48](../../../runs/cap-recovery-analysis/20261020-history-review/trials/minimax-cap_google_baseline-048.md) | Release reference/offset change is followed by success | Distinguish opening from retreat and show the invocation-time reference. No isolated single-edit causal claim |
| [MiniMax Method 1 trial 18, round 3](case_study/revision_4_failure_recovery_cases/minimax_method_1_dsl_only_trial_018.md#6-round-3-base-relative-selection-restored-release-retreat-fails) | Opening completes; retreat fails with 0.013029 m residual | Report the failed phase instead of implying the gripper never opened |

Video observations of the arm stretching toward a distant target are useful diagnostic cues. A claim that the target is outside the reachable workspace needs additional kinematic evidence; nonconvergence and a video alone are not a formal reachability test. This revision does not introduce a guessed workspace box or silently clamp targets.

## 3. Candidate changes to discuss and implement separately

| Candidate change | Proposed content | Scope boundary |
| --- | --- | --- |
| [Add four independent general-prompt options](#31-add-four-independent-general-prompt-options) | Target/offset conversion examples; phase success criteria; partial execution before failure; motion-planning limitations. See Section 3.1.1 | Apply selected blocks to initial and recovery prompts independently of recovery on/off. Preserve existing prompt versions and controller behavior; all-off is the compatibility control |
| [Show target derivation in primitive feedback](#32-show-target-derivation-in-primitive-feedback) | For Move, record each reference handle/coordinate, supplied offset, resulting gripper target, and final gripper error. For Release, record frozen XY and invocation-time reference Z + retreat offset | Use the action-entry snapshot. Do not replace it with terminal coordinates or invent the intended target |
| [Distinguish controller phases and measurements](#33-distinguish-controller-phases-and-measurements) | Identify approach, descend, open/close, move, or retreat; record consumed steps, configured limit, applicable threshold, measured value, and convergence status | Position residual may be unavailable for gripper-state phases; represent the appropriate criterion rather than fabricate a positional error |
| [Include compact action context](#34-include-compact-action-context) | Show the relevant preceding completed action and measured before/after gripper–block displacement, with source locations and goal outcome | This is a local primitive sequence, not a new full-trajectory, loop-progress, or historical-program feature |

The general-prompt design is now specified as four independent Boolean variables in Section 3.1.1. They must not be gated on whether recovery is enabled or whether a failure has already occurred. Keep actual runtime-feedback enrichment separate when evaluating these prompt additions. The configuration names below are implemented RunConfig fields.

Baseline retains its original executor and helper-generation/checking workflow. If shared API text or runtime feedback is evaluated, record it as a versioned experimental input, not a change to the pinned Google implementation. Do not add Baseline preparation repair, a pre-execution DSL gate, or an altered namespace model.

### 3.1 Add four independent general-prompt options

**Purpose:** clarify what the primitive APIs mean before the model generates a policy. This is general knowledge supplied in initial and recovery requests, not a description of an observed failure. Its four Boolean switches control only these additional explanatory text blocks. The three feedback candidates in Sections 3.2–3.4 are separate interventions and are not enabled by these switches.

#### 3.1.1 Four independent general-prompt switches

Each switch controls one additional text block. All four can be enabled or disabled independently, so all **16 combinations** are representable. The compatibility default is **all false**; enabling one must not implicitly enable another. These are general API explanations, not failure-triggered messages.

| Proposed Boolean under `primitive_prompt` | Additional information when true | Compatibility default |
| --- | --- | --- |
| `explain_target_offsets` | Explicit conversion from desired gripper position to reference-relative offsets; measured block–gripper displacement compensation; one numeric correct/incorrect example | `false` |
| `explain_convergence_criteria` | Phase-completion criteria for Pick, Move, and Release; complementary gripper-state tests, 3D positional tests, stage-boundary limitations, and the shared per-call budget | `false` |
| `explain_partial_execution` | A failed primitive may already have moved objects or opened the gripper; failure does not undo those effects; subsequent policy statements do not execute after nonconvergence | `false` |
| `explain_motion_planning_limits` | Move drives toward its fixed target without automatic obstacle avoidance or inserted clearance waypoints; required lift/transfer/descent stages must be explicit; reachability alone does not guarantee convergence within the budget | `false` |

Configuration shape, showing the all-off control:

```json
{
  "primitive_prompt": {
    "explain_target_offsets": false,
    "explain_convergence_criteria": false,
    "explain_partial_execution": false,
    "explain_motion_planning_limits": false
  }
}
```

Set any subset to `true` for an ablation condition. These options are implemented under `RunConfig.primitive_prompt` and accepted in batch JSON `run_config`.

**Content boundaries for the four blocks:**

1. **Target offsets.** State `offset = desired_gripper_position - reference_position`, separately for the selected X/Y/Z references. If targeting a carried block centre, first compute `block_gripper_displacement = block_position - gripper_position` and `desired_gripper_position = desired_block_position - block_gripper_displacement`. A numeric X-axis example should show that reference X=1.30 m and desired gripper X=1.40 m require dx=0.10 m; dx=1.40 m would command X=2.70 m. These are mathematical explanations, not new vector operations or APIs. Current measurements do not guarantee rigid attachment throughout subsequent motion.
2. **Convergence criteria.** Describe the verified phase-completion conditions in Section 1.1. Pick approach/descent, Move, and Release retreat use 3D distance ≤ the configured tolerance (currently 0.002 m); Release also checks XY, not Z alone. Open/close use the complementary gripper-state tests, not position or grasp certification. Pick does not recheck its descent position after closing, and Move completion does not separately certify that the gripper is closed or holding a block. Phases share the per-primitive 50-step budget; zero-step completion and success at the budget boundary are permitted. Render values from the actual configuration without exposing raw observation indexing as a policy API.
3. **Partial execution.** Explain the general possibility of partial effects: Release can satisfy its open-state test and then fail during retreat. The lower-level controller records nonconvergence; the CaP adapter turns it into an exception that aborts the current rollout. Do not imply atomic rollback or successful placement. This block must not invent an observed failure in the initial prompt.
4. **Planning limits.** Explain the absence of automatic route planning and intermediate waypoints. Do not prescribe a universal safe height, maximum displacement, or guessed workspace box. Do not provide a task-specific solution program as an interface example or imply that a particular waypoint sequence always succeeds.

The switches remove or add **only the new explanatory blocks**. Existing typed_v3 semantics such as relative offsets, no automatic Pick lift, the 50-step limit, and the Move tolerance remain in the base prompt when all switches are off. Thus these are ablations of additional explanations, not experiments in which the model has no knowledge of those concepts.

#### 3.1.2 Application across methods and request stages

Use one shared prompt-rendering path for Baseline, Method 1, and Method 2. The same resolved switch values remain fixed throughout a trial, including its recovery rounds.

| Request stage | Selected general-prompt blocks | Actual failure-specific information |
| --- | --- | --- |
| Initial policy generation | Included when enabled, even with recovery disabled | None: no failure has occurred |
| Recovery policy generation | The same selected blocks remain present with the original task/API text | Existing runtime-recovery evidence is appended through its current path |
| Helper generation, when the method requests it | Included with the normal API description, preserving the helper output contract | No new runtime-feedback propagation to helpers in this intervention |
| Existing static repair requests | Included with the normal API description, preserving method-specific repair behavior | Existing static diagnostics remain unchanged |

For example, `explain_partial_execution=true` can tell an initial request that Release **may** open before failing. Only a recovery request with supporting evidence can state that the gripper **did** open in the observed attempt. The switch does not collect, fabricate, or inject a past execution into initial generation.

General API text in helper prompts is separate from the deferred proposal to pass a slice of actual runtime feedback to helper synthesis. Keep role-specific instructions unchanged: policy entry rules must not replace the helper requirement to output a function definition. Keep numeric examples separate from the permission to use numeric conditions; `dsl_only` remains unchanged.

#### 3.1.3 Versioning, provenance, and ablation design

- Preserve existing prompt packages and experiments. Version the new optional text blocks and use a deterministic block order. With all options off, render the pre-change prompt exactly for the selected base version, including role/mode-specific text; do not introduce a new heading or whitespace-only difference.
- Save the base prompt version/hash, supplement version/hash, four resolved Boolean values, enabled block IDs/order, and complete rendered-prompt hash in configuration and per-call provenance. A base package hash alone cannot distinguish these 16 combinations.
- Include the extra general-prompt text in full-request context and token accounting for every affected call. Keep the 16,384 output-token cap and existing budgets fixed initially; record additional input usage and any budget stops. General prompt supplements are not runtime evidence and must not silently consume/truncate the 8,192-byte failure-evidence allocation.
- Possible future finer-grained comparison: all-off plus each single-switch-on profile, with actual runtime-feedback construction unchanged. An all-on profile and leave-one-out ablations can follow; a full 16-condition factorial is optional and requires a separately selected trial budget. Do not infer interactions from single-switch tests alone.
- These options affect initial generation as well as recovery. For end-to-end comparisons, use the same initial scene/task/model settings and report initial-attempt outcomes separately from final recovery outcomes. Different prompts can generate different initial policies, so their later transitions are not matched failed candidates.
- If a separate recovery-only diagnostic is desired, branch from the same archived failed program/state and identify that experiment explicitly as a controlled recovery study. It does not change the agreed general application of these options in ordinary trials.
- The authorized experiment in Section 6 uses all-on as an explicit treatment, not a new compatibility default. It first compares full treatments across methods and then removes one candidate group at a time; individual prompt-switch ablations remain possible but are not part of these 220 trials.

### 3.2 Show target derivation in primitive feedback

**Purpose:** explain the target that the failed action actually commanded. The current feedback already contains arguments and a target, but the model must connect them itself. A compact arithmetic record makes reference/offset errors inspectable without supplying a corrected policy.

#### 3.2.1 Information to collect and present

| Record | What it explains |
| --- | --- |
| Action identity and source location | Which invocation failed, including its checkpoint, call site, and inline helper when available |
| Per-axis reference handle and invocation-time coordinate | Where Move obtains X, Y, and Z; references may be different blocks |
| Supplied offset and resulting fixed target | `target_axis = reference_axis + offset_axis` for the actual numbers |
| Terminal gripper coordinate and error | The measured final displacement from the fixed target, with units and the applicable norm |
| Selected-block/gripper coordinates around the action | The measured relation between a gripper target and a block-centre placement; do not assume rigid attachment |
| Release reference and frozen target | Invocation-time reference Z plus retreat offset, with gripper XY frozen before opening; report the full 3D residual, not only the Z difference |

Use the action-entry snapshot. If a reference block moves later, the current target remains the original fixed target. Do not recompute the failed action's target using terminal coordinates.

#### 3.2.2 Example and diagnostic boundary

A compact record for the initial Move in MiniMax Method 1 trial 18 could show the following rounded values:

```text
X reference: b0, invocation X = 1.301914 m
supplied dx = 1.352395 m
fixed gripper target X = 1.301914 + 1.352395 = 2.654309 m
```

Together with the existing source, this exposes that a support coordinate has been included inside the offset as well as supplied through the reference. Where expression attribution is reliable, include the relevant expression and evaluated operands; otherwise provide the argument values and source location without inventing a derivation. The implemented JSON record retains these operands and the computed/logged target; the example above is a readable rendering of that record. Expression-level causal inference is not performed.

This record does not establish the intended target or prove the actual target is physically unreachable. Do not automatically clamp coordinates, rewrite offsets, or prescribe a replacement motion. The LLM still synthesizes the replacement under the existing method rules.

#### 3.2.3 Activation and validation

This is failure-specific evidence for the next recovery-policy request, not an initial-prompt observation. Enabling `explain_target_offsets` in Section 3.1.1 supplies a general formula/example; it does not activate this collector. The separate `primitive_feedback.target_derivation` Boolean enables it; false retains the legacy feedback control.

Validate mixed X/Y/Z references, moving reference objects, invocation-versus-terminal sampling, Release's frozen XY/Z target, and consistency with the logged controller target. Derivation is deterministic and does not step simulation or call another LLM. Label missing legacy samples explicitly.

### 3.3 Distinguish controller phases and measurements

**Purpose:** distinguish which part of a primitive failed and which criterion was not met. A generic `Pick nonconvergence` or `Release nonconvergence` does not identify whether approach, opening, descent, closing, or retreat was responsible.

#### 3.3.1 Phase-level evidence

| Primitive | Relevant phases | Phase-completion measurement to report |
| --- | --- | --- |
| Pick | Approach → open → descend → close | Motion-phase 3D error versus tolerance; open: finger-state sum > resolved threshold, close: sum ≤ threshold. Do not present the earlier descent error as a final post-close guarantee |
| Move | Move toward the fixed target | Gripper-to-target 3D distance ≤ tolerance; closure is commanded but neither grasp nor a separate closed-state predicate is required by its completion test |
| Release | Open → retreat | Open-state predicate first, then full XYZ positional error against the target frozen before opening; Z-only agreement does not establish convergence |

For the failing invocation, report completed phases, the failing phase, any unstarted phases, cumulative simulation steps consumed, and the shared primitive limit. `ControlResult` stores only the last attempted phase and the cumulative step count; obtain the other phase records from the event/observation sequence rather than assuming the result object contains them. Derive per-phase usage from recorded boundaries, allowing zero-step phases. Pick does not receive a fresh 50-step allowance for every phase; Release opening and retreat also share one allowance. Completion on the final allowed step remains success; nonconvergence requires an unsatisfied criterion.

For a gripper-state phase, `position_error=None` is expected by design. Use its resolved finger-state threshold, comparison operator, measured value, and observation index/time if available. Do not reinterpret `None` as zero error or invent a positional residual. The measured finger-state value is not currently part of `ControlResult`; if valid observations or phase events cannot supply it, plan an observational event addition or emit `unavailable`. Do not infer a successful grasp from the close-state test or a stable placement from the open-state test.

#### 3.3.2 Example and interpretation

MiniMax Method 1 trial 18, round 3, provides a concrete Release example:

```text
primitive: Release
open: completed, 1 simulation step
retreat: did not converge, 49 further simulation steps
total: 50 / 50 steps
retreat position error: 0.013029 m
position tolerance: 0.002 m
```

The phase transition into retreat confirms that the controller's open-state predicate was satisfied. Describe retreat failure after that completed opening phase, without claiming that a held object necessarily detached or was successfully placed. Conversely, completing the open-state test does not prove that the block was stably placed. Record only the phase/state facts supported by the trace.

#### 3.3.3 Activation and validation

This enhancement is enabled by `primitive_feedback.controller_phases`. The `explain_convergence_criteria` and `explain_partial_execution` switches in Section 3.1.1 provide the general rules to initial and recovery prompts; they do not create this per-attempt record.

Test failure at different Pick/Release phases, early convergence, immediate phase completion with zero additional steps, convergence exactly at the step limit, shared-budget exhaustion, complementary open/closed threshold boundaries, and missing measurements. Include a Release case where Z is correct but XY is not, and preserve the distinction between Pick's last positional check and its final close-state check. Verify that instruction-level nonconvergence is converted to an exception by the CaP adapter rather than falsely attributed to the instruction wrapper itself. Phase reporting must not change the controller, create extra simulation steps, or continue the policy after failure.

### 3.4 Include compact action context

**Purpose:** explain the short action sequence leading to the failed primitive. Even a correctly calculated target may follow an unsuitable approach or a disturbed placement. A local sequence can show what happened immediately before the failure without adding the full trajectory or a complete cross-round history.

#### 3.4.1 Deterministic selection of relevant context

| Context | Proposed selection and content |
| --- | --- |
| Failing action | Always retain its source identity, parameters, target, phase, and result |
| Preceding completed action | Prioritize the nearest preceding primitive with a paired end event and `converged=true`, even if its object arguments differ from the failed action |
| Local manipulation sequence | Where identifiable within the context allowance, include the current Pick and subsequent transport/approach actions needed to interpret the failure |
| Before/after geometry | Selected-object/support identities and measured gripper/block positions around the included actions; preserve the measurement time |
| Existing task result | Keep the already recorded physical/logical outcomes; do not infer success from individual converged actions |

**Use actual execution events as the primary source, with AST analysis only as an aid to source attribution.** Static syntax alone cannot establish which branches, loop iterations, or helper invocations ran before the failure.

| Existing event / field | Use in action reconstruction |
| --- | --- |
| `primitive_start.event_id` | Identify one dynamic primitive invocation within the attempt |
| `primitive_end.parent_event_id` | Pair the result with that invocation's start |
| `controller_phase.primitive_event_id` | Associate internal controller phases with the same invocation |
| `primitive_end.converged` | Distinguish a successfully completed call from a nonconverged one |
| `state_index` at action boundaries | Read the corresponding recorded before/after state |
| Arguments, checkpoint and source location | Associate physical object identities and the exact executed source version |

Use `(attempt identity, start event_id)` as the action key. A source line can run repeatedly with different objects and targets, so it is not an invocation ID. A start with no matching end remains incomplete; do not infer successful completion. Use event order to distinguish zero-step actions that share a state index.

**Selection priorities:**

1. Always retain the failing invocation and its required failure evidence.
2. Prioritize the nearest preceding successfully completed primitive, regardless of whether its explicit object arguments match the failed action.
3. Within a bounded recent-action window, add the nearest preceding successful Pick associated with the current local manipulation sequence when identifiable. Treat its operand as a previously selected object, not a certified currently held object. A subsequent Release or another Pick changes this context; do not carry a grasp assumption across those boundaries.
4. Add other nearby actions sharing relevant object/reference identities, in deterministic recency order. References alone are not a complete relevance test: `Pick(candidate)` followed by `Move(support, support, support, offsets)` is related even though the argument handles differ.
5. Serialize the selected records in execution order, with selection reasons, source/state associations, explicit omissions, and uncertain associations. Contextual relevance does not establish that an earlier action caused the failure.

High-level selection logic:

```python
def select_action_context(events, failing_action_id, lookback, evidence_budget):
    actions = pair_primitive_events_within_attempt(events)
    failed = actions[failing_action_id]
    previous = successfully_completed_before(actions, failed.start_event_id)
    candidates = prioritize(
        nearest_predecessor(previous),
        relevant_recent_pick(previous, failed, lookback),
        nearby_actions_with_related_objects(previous, failed, lookback),
    )
    return fit_ordered_action_records(
        required=failed,
        optional=deduplicate(candidates),
        budget=evidence_budget,
        report_omissions=True,
    )
```

AST/source lookup can recover the call expression, surrounding statements, helper definition, and lexical loop location from the recorded checkpoint. Runtime events determine whether it executed, its actual arguments, and which invocation/binding was used. Do not use the latest helper definition to overwrite an earlier execution-time source association or execute the AST to manufacture missing evidence.

The [select_relevant_evidence implementation](../recovery.py) preserves the original path when all feedback flags are false. With `primitive_feedback.action_context=true`, [primitive_feedback.py](../primitive_feedback.py) pairs action events, prioritizes the predecessor/recent Pick/related references, fits compact action records before remaining optional legacy events, and records omitted candidate IDs. It makes no LLM relevance-selection call.

If the selected window starts after the Pick or omits an earlier action, state that explicitly; absence from a truncated excerpt must not be described as proof that an action never occurred. The configurable defaults are `context_lookback=16` recent successful actions and at most `context_actions=4` context candidates before the byte-budget check. Prioritized optional actions may still be omitted under the existing evidence bound, with that omission recorded; the failing action's required evidence must not be silently discarded.

#### 3.4.2 What this adds beyond targets and phases

For the Haiku Method 1 trial 4 failure, the current loop body has `Pick → Move → Release`, with no separate lift before that Move. The successful replacement uses `Pick → source lift → high transfer → descent → Release`. Local source/action context helps the model connect the failure with the transport structure. Feedback should present the observed sequence, not inject that successful replacement as an expert solution.

For MiniMax Method 1 trial 18, a terminal primitive failure can follow earlier movement of the support or selection of an already moved block. Report object identities and position changes in the included local sequence. Full multi-iteration progress, a general diagnosis of the selection condition, and comparisons against all earlier programs remain backlog items.

#### 3.4.3 Activation and validation

This intervention is enabled by `primitive_feedback.action_context`. The `explain_motion_planning_limits` switch supplies a general reminder that the controller does not insert waypoints; it does not inject an observed action sequence or prescribe a path.

Check start/end/phase pairing, attempt isolation, repeated calls from the same source line, branch/helper execution, and nearest-predecessor selection when object arguments differ. Verify that incomplete calls are not labeled successful, a past Pick is not treated as proof of a current grasp, and the same events/budget produce the same selected records. Check that selected actions remain in execution order and use the correct source/binding versions and state snapshots. Handle zero-step actions, interrupted primitives, changing object references, and explicit omissions under the size limit. No extra model calls, simulator steps, automatic waypoint insertion, or helper-runtime-feedback propagation are introduced.

## 4. Implemented configuration and deterministic evidence record

Implemented configuration (all-off compatibility defaults):

```json
{
  "primitive_feedback": {
    "target_derivation": false,
    "controller_phases": false,
    "action_context": false,
    "context_lookback": 16,
    "context_actions": 4
  }
}
```

The general prompt text is stored in [prompts/primitive_v1.json](../prompts/primitive_v1.json);
[primitive_prompts.py](../primitive_prompts.py) handles loading, flag selection,
controller-value substitution, and provenance. Relocating the text after the
completed experiment preserves rendered prompt bytes and supplement hashes.

Prompt supplement version: `primitive-prompt-v1`; diagnostic version: `primitive-feedback-v1`. The optional diagnostics appear under `evidence.primitive_diagnostics` only for an attributable primitive failure. Existing nonprimitive-failure handling is unchanged. Inline helpers receive the general API supplement through their normal method prompt; no new propagation of runtime evidence to helper requests is introduced.

This is the shared record for the three independently configurable feedback candidates in Sections 3.2–3.4. The full treatment enables all four general-prompt blocks plus all three feedback candidates. A candidate-level ablation disables only its corresponding feature group; prompt switches never implicitly enable feedback collectors.

| Field family | Data source | Interpretation |
| --- | --- | --- |
| Attempt, source, action and phase identity | Existing checkpoint/call/primitive events | Locate the failing action, including an inline helper when recorded |
| References and supplied offsets | Invocation arguments and action-entry coordinate snapshot | Derive each target axis using the actual API semantics |
| Fixed target and terminal state | Controller phase record and corresponding trajectory state | Explain the measured residual against the target that was actually commanded |
| Step and phase-completion criteria | Resolved controller configuration, last result, and attributable phase events/observations | Explain the norm or state predicate, comparison boundary, measured value, shared limit, and cumulative/per-phase steps; mark unavailable fields explicitly |
| Gripper–block displacement | Coordinate snapshots before/after the relevant actions | Distinguish gripper targets from block-centre placement intentions; do not assume rigid attachment |
| Relevant preceding action | Recorded action boundary, target and convergence result | Expose the approach/transfer sequence without dumping the whole log |
| Existing task outcomes | Existing physical/logical evaluator results | Retain the distinction between convergence and completion; no new goal-diagnosis feature |

Use deterministic host code, with no extra diagnosis LLM call and no replay of generated helpers or simulation for evidence selection. Keep facts, arithmetic derivations, model-stated hypotheses, and retrospective diagnoses distinguishable. Missing legacy observations are `unavailable`, not fabricated.

Retain current latest-source feedback and earlier short summaries. Full historical code and an edit ledger remain in the backlog, even though trial 18 motivates them. If a target formula is demonstrably doubled, present the operands and evaluated formula; do not automatically rewrite the policy or promise a corrected target will achieve the task.

## 5. Compatibility, limits, and logging

- Keep the existing 50-step primitive limit, convergence thresholds, task specification, maximum four recoveries/five rollouts, and 16,384 output-token cap for the first comparison. Increasing controller or model budgets is a separate intervention.
- Every replacement rolls out from the same saved initial state in a fresh execution context. Failed trajectories remain debugging evidence.
- Retain the 8,192 UTF-8-byte evidence bound and current per-round/whole-trial budgets. Compress repeated metadata to fit useful primitive facts instead of appending an unbounded event dump. Count the complete request and reserved output using the existing accounting policy.
- Keep the all-off compatibility profile available, with exact prompt/feedback version provenance. Enabled general-prompt switches intentionally affect recovery-off and recovery-on initial generation alike. A later recovery-feedback-only intervention must leave initial prompts unchanged. Do not silently truncate source or hide omitted evidence.
- Preserve condition modes. Numeric diagnostics do not authorize numeric branch/loop/selection conditions in `dsl_only`.
- Save prepared and delivered feedback, source/phase associations, field inclusion/omission decisions, serialized sizes, and actual request payloads. Verify equality through the next policy call.
- No API/preparation retry, runtime-failure-feedback propagation to helpers, general namespace diagnosis, or general loop/query repair is included here. If the primitive failure originated in such a defect, classify it rather than expanding scope silently.

## 6. Focused tests and evaluation

| Validation stage | Required evidence |
| --- | --- |
| Contract consistency | Each optional block agrees with Section 1.1 and the actual controller/API semantics, including stage-local guarantees and CaP exception conversion; old prompt versions remain reproducible |
| Independent switches | All 16 combinations render deterministically; each toggle changes only its own block; all-off reproduces the original rendered prompt byte-for-byte |
| Stage/method coverage | The same resolved flags apply to initial/recovery policy, available helper, and static-repair requests across all three methods; recovery-off receives enabled general blocks too; no fabricated initial failure evidence |
| Prompt provenance | Base/supplement versions, four flags, enabled blocks and rendered hashes match the actual request; additional input is accounted for |
| Target arithmetic | Regression fixtures for repeated reference coordinates, invocation-time sampling, gripper–block compensation, and different wrong targets sharing the same failure label |
| Phase attribution | Correct open/closed threshold boundaries, zero-step phases, success on the final allowed step, shared counts, Pick's lack of a final position recheck, and Release XYZ rather than Z-only convergence |
| Evidence delivery | Deterministic bounded selection; explicit missing/omitted fields; prepared feedback equals the actual next request; source attribution is preserved |
| Compatibility | Unchanged legacy behavior, method-specific checking, limits, condition modes, and exact-state reset; no extra model diagnosis calls |
| Scripted simulator checks | Verify the collector against known executed action targets and outcomes before any live experiment |

For the agreed general-prompt comparison, follow Section 3.1.3 and measure initial generation as well as final recovery outcomes. For a separately configured feedback comparison, branch control and candidate profiles from the **same archived failed initial policy and initial state**, with the same model, effort, task, condition mode, output cap, recovery count, and resource limits. Do not treat different independently generated failures as matched candidates.

Measure final task success, rounds, model calls/tokens/time, primitive/phase failure counts, verified coordinate-formula defects, and budget/preparation stops. Preserve all failures in the declared denominator and separate physical success, logical postcondition, and DSL compliance. A disappeared exception or a shorter program is not sufficient evidence of successful repair.

### 6.1 Authorized main comparison and candidate-level ablations

| Cohort | Methods | Feature profile | Trials per method/profile | Total |
| --- | --- | --- | ---: | ---: |
| Full treatment | Baseline, Method 1, Method 2 | All four general-prompt options and all three feedback candidates enabled | 20 | 60 |
| Ablations | Method 1, Method 2 | Remove each of the four Section 3 candidate groups separately | 20 | 160 |
| **Total** | | 11 distinct method/profile settings | | **220** |

The four leave-one-out ablations are:

| Profile | Disabled group | Features retained |
| --- | --- | --- |
| `no_prompt` | Section 3.1: all four general-prompt Booleans false | Target derivation, controller phases, action context |
| `no_target` | Section 3.2: target derivation | All prompt options, controller phases, action context |
| `no_phases` | Section 3.3: controller phases | All prompt options, target derivation, action context |
| `no_context` | Section 3.4: local action context | All prompt options, target derivation, controller phases |

These remove candidate groups, not one of the four general-prompt Booleans at a time. The full Method 1/2 groups in the main comparison are reused as the ablation controls; they are not rerun as additional groups. All-off remains the software compatibility default. Preserve legacy feedback fields in every treatment: disabling an enrichment does not remove information already present in Revision 4.

Use `minimax/minimax-m3.1-flash:free`, medium reasoning, 16,384 output tokens, `typed_v3`, `dsl_only`, Stack with three blocks, and the original full initial snapshot. Recovery is enabled in all 220 trials with at most four repairs/five rollouts. Retain the previous model settings (temperature/top_p omitted, 300-second request timeout, one transport retry), initial 50k-token/12-call/600-second budget, recovery 100k-token/16-call/600-second budget, whole-trial limits, and 8,192-byte evidence cap. Preparation/API handling and Baseline's original execution/checking behavior remain unchanged.

### 6.2 Parallel execution and accounting

Use the latest shared GPU scheduler with explicit allowed UUIDs, verified EGL mapping, and shared slot reservations. The user additionally authorized four credential slots rotated per request. Preserve six-second pacing per credential and apply separate quota-derived credential caps plus a common request ledger/admission reservation; do not interpret rotation as permission to exceed shared provider limits. The original key advertises 1,000 daily free requests and the added keys 50 each at preflight; record actual use and quota changes. Interleave the main and ablation groups in a recorded balanced randomized schedule so they run concurrently. Choose worker limits from available authorized resources, measured worker memory, request latency and provider admission constraints; maximize useful throughput without bypassing quotas or extending per-trial budgets.

Create an immutable 220-job manifest before dispatch. Resume pending jobs without duplicating completed/live trials; keep all failed or incomplete trials in their original groups. Record GPU assignments, queue/active time, calls, tokens, request failures and all profile flags/hashes. Pause admission if quota cannot cover active/new trials, retaining pending identities rather than relabeling them as executed failures.

### 6.3 Outcomes, reports, and plots

Write results to [revision_4_2_report.md](revision_4_2_report.md) and figures under `docs/plots/`, embedding relative image links in the report. Until all runs finish, label tables as partial and display finished/planned counts.

Report separately for each of the 11 settings:

- Initial and final task success / all declared trials; retain the existing physical/logical/normal-completion success definition.
- Initial and final primitive nonconvergence / trials, plus trials with any such failure and nonconverged rollouts / actual rollouts. Separate Pick, Move and Release. Do not hide preparation failures or conflate them with executed failures.
- `nonconvergence_t → nonconvergence_(t+1)` / nonconvergent attempts with an actual next attempt; separate same-primitive recurrence, task success, other execution failure, and preparation failure. Terminal failures with no next attempt are reported separately, not included in that denominator.
- Recovery rounds, calls, total input/output tokens, measured usage gaps, time, and per-trial artifact links. Repeated attempts are not independent samples.

Plot full-treatment success and initial/final nonconvergence across three methods, then full versus each ablation for Method 1/2. Label every denominator and sample size. Trial-level proportion intervals may be shown; do not treat transitions within one trial as independent binomial samples. Compare previous MiniMax results as a separately labeled historical reference, not a paired or contemporaneous control. A successful disappearance of nonconvergence alone is not task success.

Audit the actual delivered prompts/feedback, intended ablation flags, full-state restoration, source attribution, usage, videos and denominators before finalizing the report. Record tests, implementation versions, provider interruptions and remaining limitations. No failed trial may be replaced to improve the reported rate.


### 6.4 Implementation and preflight validation

- Implemented `primitive-prompt-v1` and `primitive-feedback-v1`; all new Boolean options default off. The original prompt package files, vendored Google core, primitive controllers, and DSL checker were not edited.
- Final regression run: **189 passed, one opt-in hardware test skipped** (190 discovered), including the controller suite. The focused feature/report/request-control run passed 24 tests.
- The final scripted GPU preflight covers all 11 method/profile settings: **11/11 recover successfully after a deliberately nonconvergent initial Move**, for 22 rollouts. Ten workers ran concurrently under GPU reservations; sampled peak worker RSS was 945.54 MiB and sampled attributed GPU memory 69.89 MiB.
- The artifact audit found **zero issues** across prompt flags/hashes, delivered request content, prepared/delivered feedback, intended ablation groups, initial-state restoration, token/call totals, and independently decoded video frame counts.
- Four minimal provider compatibility requests (one per credential slot) returned `pass` with the requested MiniMax route/settings. They used 913 reported tokens and provider-reported cost zero, and are excluded from experiment trials.
- The live configuration permits up to 18 workers / 9 per allowed GPU, with 1,024 MiB worker reservations and 4,096 MiB free headroom. Global/per-credential quota reservations may reduce admitted concurrency below that ceiling. No GPU outside the two explicitly configured UUIDs is selected.

Evidence: [final scripted preflight](../../../runs/cap-revision42-validation/20261021-final-preflight/audited_report.md), [artifact audit](../../../runs/cap-revision42-validation/20261021-final-preflight/revision42_audit.json), and [validation logs](../../../runs/cap-revision42-validation/20261021-final-preflight/validation/). The subsequent 220-trial model experiment is complete; preflight/test outcomes remain excluded from its denominator.


### 6.5 Completed experiment and observed limitation

The live batch `20261021-174848-minimax-220` used implementation commit `29d9ebf` and completed all 11 settings at 20 trials each. It produced 207/220 final task successes across the mixed treatments, 446 prepared attempts, 444 actual rollouts, and 493 model requests. Reported usage was 4,378,031 tokens with no missing usage; provider-reported cost was zero. All 207 successful final programs passed the recorded DSL audit. The full experimental and video audit has zero consistency issues.

The full treatment does not dominate all leave-one-out ablations. The clearest descriptive effect is lower initial nonconvergence with the general-prompt additions. Feedback ablations have different independently generated initial failures, and the shared byte cap can alter the amount of other optional evidence retained; a matched failed-policy comparison is needed for stronger causal conclusions.

Baseline full trial 20 stopped after its third rollout because the combined required diagnostic/metadata payload was 8,300 bytes against the fixed 8,192-byte limit, even with no optional action records or supporting events. This is an observed packing limitation, not a reason to erase or replace the trial. It remains a failed outcome; no post-run runtime fix or selective rerun was made. A future change should address redundant metadata/packing separately and preserve reproducible controls.

See [the completed report and plots](revision_4_2_report.md) for group results, per-trial token usage, transition denominators, failure breakdown, source/prompt audits, and artifact links.
