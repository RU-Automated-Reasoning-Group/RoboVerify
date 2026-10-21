# Revision 4: unimplemented backlog

- **Status: deferred; not implemented yet.** Retain these ideas for future small revisions, not for execution as a bundle.
- **Active plan:** [Revision 4.2: primitive nonconvergence](revision_4_2_fix_primitive_noncovergence.md).
- **Completed feature:** [Revision 4.1: GPU scheduling](revision_4_1_gpu_scheduling.md).
- **Evidence:** [Recovery report Section 4](revision_4_recovery_report.md#4-analysis-of-initially-failed-recovery-on-trials).
- **Baseline boundary:** preserve the original executor and generation/checking workflow. No new preparation repair is planned for Baseline; the previously discussed Method 1/2 preparation/API handling is also deferred for now.

A primitive failure may be the last symptom of a different policy defect. For example, [Method 1 trial 18](case_study/revision_4_failure_recovery_cases/minimax_method_1_dsl_only_trial_018.md) selects b0 after changing the query anchor in round 2, and reintroduces an earlier arithmetic defect in round 4. General selection/progress diagnostics and full historical-code injection remain separate backlog items; they are not silently included in Revision 4.2.

## 1. Deferred recovery improvements

**Status: not implemented yet; deferred.** These proposals are retained for later discussion, not as one combined implementation plan. Promote one focused problem into its own `4.x` document when it becomes active. No prompt, collector, recovery trigger, budget, or executor change is authorized by this backlog.

### 1.1 Evidence and previously proposed priorities

The report analyzes **159 initially unsuccessful recovery-on trials: 130 eventually recovered and 29 did not**, with **325 feedback-to-replacement transitions**. Three selected trials never entered their initial rollout. Among the 29 unrecovered trials, 23 exhausted four recovery rounds and six stopped during preparation. These are different units: trial outcomes must not be confused with per-transition outcomes.

All 325 transitions passed feedback/source/request consistency checks. The plan therefore targets the meaning, coverage, and recipients of feedback, rather than assuming existing policy feedback was lost in transport. The following counts and reviewed cases come from [report Sections 4.3–4.7](revision_4_recovery_report.md#43-what-each-feedback-supplied-and-what-it-omitted).

| Priority | Evidence | Gap to address | Planned change |
| --- | --- | --- | --- |
| P0 | Eight of ten NameError feedbacks concern names already assigned at policy top level; two concern missing initialization | The error text does not distinguish namespace visibility from initialization | Explain relevant locals/global/closure visibility and the source of the actual helper binding |
| P0 | 92 budget feedbacks; 61 underlying logs have multiple checked-interpreter loop events, but no feedback contains more than one | No cross-iteration progress summary or explicit exhausted-resource counters | Normalize loop events across methods; report resource/limit, repeated selections, and task-relation progress |
| P1 | 169 feedbacks include older summaries but no older program bodies | Failure and edit history cannot be reconstructed from hashes | Include complete attributable source and bindings, with an immutable edit/outcome ledger |
| P1 | 59 helper calls in 42 replacement rounds, all Baseline, contain no structured runtime-feedback block | A revised policy can delegate to a helper generated without the failure context | Route a bounded relevant context slice to downstream helper requests; audit the actual delivered content |
| P1 | Four no-witness transitions; 34 normal-completion goal failures have empty reason strings despite existing goal evidence | Generic exceptions or clause lists do not explain the failing binding/task deficit | Explain recorded quantified counterexamples and unsatisfied physical/logical goals |
| P1, prerequisite for added context | Feedback-bound median 8,051 of 8,192 UTF-8 bytes | Appending raw events can displace existing evidence | Use compact semantic records and deduplicated metadata; account for the complete request |
| P2, Method 1/2 only | Six preparation stops: five Baseline and one Method 2; none Method 1 | The Method 2 response error produced no executable candidate or new trajectory | Plan bounded preparation handling for Method 1/2 separately from runtime repair. Keep all five Baseline stops as outcomes of its unchanged preparation path |

The priority labels above record the earlier broad review; they do not override the current decision to address one problem per version. Primitive target/phase diagnostics have moved to [Revision 4.2](revision_4_2_fix_primitive_noncovergence.md). Names/bindings, loop/selection progress, historical source, helper propagation, query/goal explanations, and preparation handling remain deferred here.

Repeated failure labels are not sufficient evidence of repeated defects: the report finds 67 transitions with a repeated failure class but only one with the same normalized source bundle. The two final Move failures in Baseline trial 6 request different Z targets. Likewise, ten numeric-literal-only edits with no immediate success do not justify banning numeric tuning. Text feedback is the first proposed intervention; the report does not establish that text can identify every contact or grasp failure.

### 1.2 Shared evidence contract and collection

The collector and summarizer should be deterministic host code, not another LLM diagnosis call. Collect once per failed attempt and expose a shared schema to the existing method-specific synthesis paths. Every record should identify its attempt, source/checkpoint, event or state index, and whether it is an observed fact, deterministic derivation, or unconfirmed hypothesis.

| Evidence family | Environment / recorded-state inputs | Execution-log / source inputs | Planned compact output |
| --- | --- | --- | --- |
| Names and bindings | Opaque generated-program object handles only | Failing scope/line, relevant name presence in locals/globals/closure, caller identity, helper version actually invoked | `uninitialized`, `not_visible_in_referencing_scope`, or `unknown`; caller → bound source identity |
| Loop progress and budget | First/recent observed task relations and selected block identities | Loop source, iteration/query/primitive events, budget counters and configured limits; normalize Baseline Python tracing separately | Exhausted resource, used/limit, observed iteration count, repeated witnesses, first/recent progress and exit status |
| Query witnesses | Predicate values for recorded bindings in that scene | Query source, bound object IDs, available nested quantifier/subpredicate results | Which binding and subexpression falsified the query; mark unavailable subexpression detail explicitly |
| Goal deficits | Actual task evaluator output and logical postcondition evaluation | Normal termination/failure status and task-specification identity | Unsatisfied clauses/physical criteria and their available witnesses; preserve disagreements between evaluators |
| History | Previously recorded attempt states/outcomes only | Complete attributable source units, execution-time helper bindings, source identities and diffs | Chronological source/edit/outcome records, without restoring the failed state for the next rollout |

- Add observational instrumentation where current logs lack needed detail. Legacy artifacts may produce `unavailable` fields; do not invent facts or assume missing events mean no iterations occurred.
- Limit namespace capture to relevant generated-program names and opaque values. Do not serialize host globals, simulator objects, credentials, or arbitrary object representations.
- Derive query explanations from captured truth evaluations or a supported side-effect-free diagnostic evaluator over the saved state. Do not execute arbitrary generated helpers again or advance simulation just to produce feedback.
- Preserve short-circuit semantics. An unevaluated subexpression is not a recorded false value. Identify a proven self-binding contradiction only when supported by predicate semantics and the captured expression.
- Feedback does not relax the configured condition mode: `dsl_only` still prohibits numeric branch/loop/selection conditions; coordinates may still explain failures and parameterize actions.

### 1.3 Diagnostic behavior and reviewed regression cases

| Case from the report | Required diagnostic behavior | Boundary |
| --- | --- | --- |
| Baseline trials 17/18: an assigned name fails inside a lambda | Identify that the assignment populated exec locals while the reference resolves through globals; distinguish this from the two reviewed missing-initialization cases | Preserve pinned Google `exec(code, globals, locals)` behavior. Do not merge namespaces as an incidental recovery change |
| Haiku Baseline trial 5: the called helper differs from the inline definition | Show the caller and the actual bound helper body/version, including the repeated call behavior | A final helper definition must not replace execution-time evidence. Debug history does not alter final-output-only DSL compliance |
| MiniMax Method 1 trial 29: hundreds of primitives precede budget exhaustion; unrolling then leaves the goal false | Summarize repeated selections and goal progress across iterations, with the exhausted resource and limit | Removing the timeout or shortening execution is not successful task repair |
| MiniMax Method 2 trial 26: two rewritten quantifiers fail at `t = s` | Explain the falsifying self-binding and evaluated subexpressions, including reflexive ON* when applicable | Report a demonstrated query defect, not a solver-generated replacement condition |
| Goal-failed attempts with empty reason strings | Produce a factual task-deficit summary from evaluator evidence already available | Keep physical-goal failure and logical clause failure separate; missing evaluator detail stays explicit |

For namespace repair, feedback may state that renaming the variable alone does not fix visibility and explain which names are available in the referencing scope. Any generic prompting advice must be versioned and validated against the real Baseline helper-binding workflow. Do not promise that simply wrapping code in a function avoids all helper-generation issues.

### 1.4 Historical programs and the edit/outcome ledger

Retain the original proposal to include all earlier failed rollout attempts from the **same trial**, with full attributable source and immutable bindings. At recovery round 4, the policy request contains programs from rounds 0–3; round 3 receives the latest detailed evidence.

- Include attempts that entered top-level execution but failed before their first primitive. Candidates rejected before rollout belong to a separate preparation record, not executed history.
- Preserve complete source units, including branches not taken. Do not reconstruct a program from executed statement fragments alone.
- Record attempt/round identity, source/checkpoint IDs, relevant binding versions, failure location, observed failure, and physical/logical/DSL outcomes. Never merge different helper versions into a fictitious single executable.
- Deduplicate identical source bodies in an inline registry. Every source reference required to understand a policy request must resolve to a body in that same request; a path or hash alone is insufficient.
- Keep repeated attempts even when source is deduplicated. Include a deterministic source diff and edit/outcome summary as an aid, not a replacement for complete source.
- Associate changes with the next observed result. For example, record that a removed waypoint was later restored; do not label it a proven causal fix without supporting evidence.
- Do not replay full old trajectories, videos, model reasoning, or previous prompts. Preserve them as artifacts.

Historical rollout states are debugging evidence. **Every replacement policy rolls out from the same saved initial state with a fresh execution context**, never from a failure state.

### 1.5 Independent configuration and prompt routing

All names below are **proposed fields**, not implemented configuration. Keep the legacy combination as the compatibility default and expose each intervention independently for ablation.

| Proposed setting | Compatibility value | Experimental value |
| --- | --- | --- |
| `recovery_history_mode` | `summaries_only`: latest source + earlier failure summaries | `executed_programs`: complete attributable source for every failed executed attempt |
| `recovery_diagnostics_profile` | `legacy_v1`: existing selected events/keyframes/clauses | Earlier proposal `semantic_v2`: deferred broad diagnostics from Sections 1.2–1.3; replace this broad intervention with individually versioned features when selected |
| `recovery_helper_feedback` | `disabled`: existing helper prompt behavior | `relevant_slice`: relevant failure/source/diagnostic context in downstream helper requests |
| `preparation_repair_enabled` | `false`: existing preparation-stop behavior | Method 1/2 only: separate opt-in bounded path described in Section 1.6; excluded from the first runtime-feedback comparison. Baseline does not support enabling it |

Selecting `summaries_only` by itself does not imply the whole prompt is legacy: the diagnostics and helper settings are independent. Recovery off must bypass all new runtime-feedback paths. Existing initial policy/API text, checker behavior, task semantics, and recovery triggers remain unchanged in the first comparison.

Suggested policy request structure:

```text
ORIGINAL TASK, API, CONDITION-MODE RULES, AND POLICY OUTPUT CONTRACT
{existing_initial_prompt}

RECOVERY INSTRUCTIONS
Complete the original task; every replacement starts from the saved initial state.
Use the recorded execution context, not just the exception label.
Compare earlier code and observed outcomes before revising the latest program.
Termination or disappearance of an exception alone does not establish task success.
Return a complete replacement policy under the existing output rules.

SOURCE REGISTRY AND ACTUAL BINDINGS
{latest source; historical source when enabled}

EARLIER ATTEMPTS AND EDIT/OUTCOME LEDGER
{chronological records under the selected history profile}

LATEST FAILURE
{observed outcome and source location}
{names/bindings, loop progress/budget, query/goal diagnostics for the selected future feature}
{remaining relevant evidence within the configured bound}
```

Helper routing should be deterministic: identify the requested helper, its call site, the failed source/dependency chain, and relevant bindings. Supply that slice, the applicable failure facts, and the unchanged helper signature/return/output contract. If attribution is uncertain, include a bounded latest-failure summary marked as such; do not pretend an unrelated helper caused the failure. Include bodies for referenced source units within the slice. The requirement to include *all* prior programs applies to the policy request, not every helper request.

Preserve request roles: a policy response may need top-level invocation; a helper response still emits the requested function definition. Retain the chosen runtime context through any existing local/static repair requests without changing what triggers those repairs. Log policy, helper, static-repair, and runtime-recovery calls separately.

Baseline may generate helpers while executing the current policy. Propagate the existing failed-attempt context to such requests without moving all helper synthesis before rollout or permitting an in-place recovery continuation. If that candidate fails, its replacement still starts at the saved initial state. Method 1 retains recursive checking and Method 2 whole-program checking; Baseline gains no pre-execution DSL gate.

### 1.6 Context budgets and separate preparation handling

**Runtime-feedback budget plan:**

- Retain the **16,384 output-token cap**, maximum four recoveries/five rollouts, and existing per-round/whole-trial limits for the first comparison. More input context is not, by itself, evidence that the output cap should increase.
- Keep the existing **8,192 UTF-8-byte upper bound** for serialized feedback evidence/summaries. Its current configuration name uses “tokens,” but it is not a measured provider token count. Historical-source bodies need a separate explicit allowance; measure archived source sizes before fixing its numerical default.
- Compress repeated event/source metadata before adding context. Reserve space for the failure identity, source associations, applicable diagnostic core, and outcomes; then add representative progress/keyframes/action context. Record inclusion/omission decisions and unavailable fields.
- Do not truncate required code bodies, drop attempts silently, or label a request `executed_programs` after silently falling back to summaries. Missing required history or an oversized required core must produce an explicit context/preparation stop, not another failed rollout.
- Count the full request—including task/API text, source, evidence, helper context, formatting, and reserved output—against model context and existing round/trial reservations. Log the counting method and actual provider usage when returned.
- Save prepared and delivered feedback per call and verify it against the actual request payload. Record profile versions, recipient role, source hashes/bindings, included attempts, sizes, and truncation/omission decisions.

**Preparation handling is a deferred P2 proposal for Method 1/2 only. Provider/API response errors are explicitly set aside for now:**

- **Baseline originality boundary:** do not add preparation-error retry, syntax/format repair, incomplete-response regeneration, or a new DSL gate to Baseline. Preserve its pinned executor, helper-generation workflow, existing transport retry behavior, and preparation-failure stop semantics. A configuration that requests the new preparation-repair feature for Baseline should be rejected explicitly. This decision concerns the new preparation path; it does not disable the existing configurable rollout recovery or expand the scope of the shared feedback proposals above.
- The five Baseline preparation failures remain in the experiment results and are not a target of this intervention. Method 1 had no such final failure in the reviewed cohort; supporting it is a shared-interface extension, not evidence that this sample demonstrated a Method 1 preparation defect.

- Classify syntax/format errors, incomplete responses (`length` versus provider `error`), and transport failures separately. Keep existing transport retries and method-specific static repairs distinguishable; avoid stacking duplicate retry mechanisms.
- Initial preparation failure has no failed trajectory. A preparation failure after an earlier rollout can reference that earlier failure, but must not fabricate a trajectory for the new candidate.
- If later enabled for Method 1/2, distinguish malformed source that can receive a syntax/format diagnostic from an empty provider response that requires another generation request. Preserve the original role contract and explicit preparation call/token/time limits within the existing whole-trial budget. Extra output/call limits are not silently granted.
- A successful preparation repair still precedes a fresh rollout from the initial state. Preparation attempts and their stops are counted separately from the maximum five rollouts; they do not create extra execution opportunities.
- Broader failure routing is a separately configured Method 1/2 intervention, not part of historical-source injection. Do not route Baseline failures into this new handler.
- The preparation retry count, reserved budget, and supported incomplete-response retry policy remain to be selected before implementing that optional path. Leave it disabled in the first runtime-feedback ablation.

#### Method 2 trial 41: what the saved response establishes

The reviewed Method 2 preparation stop occurred at recovery round 1, after an
initial rollout failed on Move nonconvergence. The failure feedback was delivered,
but the replacement request returned no policy source.

| Saved field | Value |
| --- | --- |
| Model / route | `minimax/minimax-m3.1-flash:free` through OpenRouter; recorded provider `Minimax` |
| `finish_reason` / `native_finish_reason` | `error` / `null` |
| Assistant `content` / `refusal` | `null` / `null` |
| Input / output / total tokens | 4,519 / 454 / 4,973; all 454 output tokens are reported as reasoning tokens |
| Output cap | 16,384 tokens; not exhausted by the reported output |
| Logged model phase / request timeout | Approximately 67.21 seconds / 300 seconds; no recorded timeout or budget exhaustion |
| Local exception | `ModelError: Incomplete/empty model response: error` |
| New candidate checking / rollout | Neither started: no replacement source was available |
| Trial stop | `preparation_failure_no_rollout`; no further recovery round |

Evidence: [raw response](../../../runs/cap-revision4-minimax31flash/20261020-152924-free-300/trials/cap_method_2-dsl_only-041-recovery-on/cap/20261020-184857-29473fe-cap_method_2-recovery-1cd26fc8/attempts/round_001/cap/20261020-185106-29473fe-cap_method_2-stack-f6e4d844/artifacts/trial_001/lmp/call_0001/response.json),
[traceback](../../../runs/cap-revision4-minimax31flash/20261020-152924-free-300/trials/cap_method_2-dsl_only-041-recovery-on/cap/20261020-184857-29473fe-cap_method_2-recovery-1cd26fc8/attempts/round_001/cap/20261020-185106-29473fe-cap_method_2-stack-f6e4d844/artifacts/trial_001/execution/traceback.txt),
and [attempt trace](../../../runs/cap-recovery-analysis/20261020-history-review/trials/minimax-cap_method_2-041.md).

The response supplies no detailed provider error code/message. It establishes an
error response with no usable code, not its upstream cause; overload, an internal
service fault, or another explanation cannot be selected from these artifacts.
It is not an observed 16k truncation or a DSL-checker rejection. The existing
transport retry branch handles `TransportError`; this parsed empty response
instead raises `ModelError`, so it did not receive that retry. Method 2's static
repair loop had no candidate to check or repair.

This case motivates considering a bounded, separately recorded regeneration
request for eligible Method 1/2 response errors. It does not justify retrying all
provider errors indiscriminately, increasing the output cap, or changing Baseline.
The eligibility rules and retry budget remain part of the unimplemented P2 plan.

### 1.7 Acceptance checks for future individual revisions

| Stage | Proposed work | Required validation |
| --- | --- | --- |
| Evidence schema and compression | Versioned factual records, missing-data markers, deterministic selection and size accounting | Repeated selection is deterministic; required facts survive near-bound payloads; no silent code/history truncation |
| Deferred diagnostics | Namespace/binding attribution and loop progress, selected as separate features | Reproduce assigned-but-invisible versus uninitialized names; normalize Baseline and checked-interpreter loops without inventing events |
| Historical source | Immutable bundles, deduplicated registry, edit/outcome ledger | Round-4 policy context includes rounds 0–3; later helper changes cannot mutate earlier evidence; all inline references resolve |
| Helper propagation | Shared context API with role-specific slices and preserved repair context | Exact payload checks for policy/helper/static-repair recipients; Baseline execution-time helper generation remains supported |
| Query and goal explanations | Recorded counterexamples and factual task deficits | Self-binding quantifier case is explained; unavailable subterms remain unavailable; early termination with failed postconditions is not accepted as repaired |
| Compatibility and logging | Independent profile selection and request provenance | Legacy defaults/recovery-off unchanged; no new Baseline DSL gate; same initial-state restore and method boundaries; no extra diagnosis model calls |
| Separate preparation path | Later bounded opt-in syntax/format/incomplete-response handling for Method 1/2 | Baseline rejects this option and keeps its existing failures; no fabricated rollout, no duplicate transport/static retries, explicit preparation versus rollout accounting |

Use scripted responses first and retain the report's concrete cases as regression fixtures. Offline reanalysis can verify legacy evidence and source attribution, but cannot recreate observations that were never logged. Additional instrumentation requires controlled scripted rollouts before live experiments. These are planned checks, not tests claimed to have run for Section 1.

### 1.8 Evaluation principles for future revisions

The earlier comparison profiles are retained below as an ablation catalog. Select one mechanism and compare it with its unchanged control when that mechanism becomes active; this table is not a request to implement or run all profiles together. A combined profile is a possible later study only after individual changes are evaluated:

| Profile | History | Diagnostics | Helper runtime context | Preparation repair |
| --- | --- | --- | --- | --- |
| Control | Summaries only | Legacy | Disabled | Disabled |
| History only | Executed programs | Legacy | Disabled | Disabled |
| Diagnostics only | Summaries only | Semantic | Disabled | Disabled |
| Helper context only | Summaries only | Legacy | Relevant slice | Disabled |
| Combined runtime feedback | Executed programs | Semantic | Relevant slice | Disabled |

- For the recovery comparison, start each profile from the **same archived failed initial program, attributable bindings, and saved initial state**. Each fresh replacement rolls out from that initial state. Do not compare profiles starting from independently generated failures as if they were matched.
- Hold model/effort, condition mode, output cap, task, maximum recoveries, and reported resource limits fixed. Stratify by method, model, and observed failure family; the original cohorts were generated by different models and are not a shared-candidate repair benchmark.
- Keep the three initial preparation failures outside the executed-failure recovery cohort and report them separately. Evaluate the optional preparation path later on Method 1/2 only; keep Baseline preparation behavior fixed and retain its failures in all results. Keep later preparation stops in the runtime-feedback outcomes.
- Run independent trials in parallel using the [implemented Revision 4.1 scheduler](revision_4_1_gpu_scheduling.md#1-gpu-scheduling) and shared request limits. Apply the same declared resource policy to all profiles and record actual queue/resource conditions.
- Measure final task success, rounds to success, reviewed defect recurrence, removed/reintroduced edits, repeated source candidates, source/binding attribution, helper context delivery, preparation/context stops, model calls, input/output tokens, and elapsed time. Report physical, logical-postcondition, and DSL outcomes separately.
- Compare costs under the fixed budget first. Any increase needed for larger input contexts is a separate budget ablation. Repeated exception labels and syntactic edit tags are not causal diagnoses.

No backlog implementation or new experiment is authorized by this document. The current recovery behavior and historical experiment results remain unchanged.
