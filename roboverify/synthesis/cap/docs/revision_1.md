# Revision 1: shared observations and condition modes for both CaP methods

Status: implementation plan; runtime changes have not started.

The requested change makes current gripper/block positions readable by generated
programs and gives both `cap_google_baseline` and `cap_method_1` the same two
condition modes. The exact contract is in
[Method1_Configuration.md](Method1_Configuration.md), and the proposed execution
and data flow is in [METHOD-1-FLOW.md](METHOD-1-FLOW.md).

## Current state and target

| Area | Current implementation | Revision 1 target |
| --- | --- | --- |
| Runtime observations exposed to policy | DSL predicate truth and selected object handles | Add current gripper/block XYZ through a shared read-only API. |
| Condition configuration | Both methods advertise DSL-only conditions; method 1 enforces them, baseline has no complete DSL checker | Both methods accept `dsl_only` or `coordinate_predicates`, with `dsl_only` the default. |
| Coordinate-based predicates | Unsupported | LLM-generated pure predicates allowed in `coordinate_predicates`. |
| Method-1 numeric operations | Action-parameter arithmetic; object-only comparisons | Add Position3 component reads; allow general numeric comparisons and counter guards in coordinate_predicates, with DSL-only conditions in dsl_only. |
| LLM feedback | Task/API text, helper signatures, static repair diagnostics | Also include selected mode and coordinate API descriptions; keep actual coordinates runtime-only. |
| Physical execution | Shared Pick/Move/Release controllers | Preserve signatures, controller settings, failure behavior, and scoring. |

All four method/mode combinations have access to the same O_t. The condition
language differs: DSL-only versus DSL plus general numeric predicates, including
coordinates, counters, and numeric computations without a coordinate dependency.
Thus future `dsl_only` retains the current condition restriction while gaining
position-derived action parameters. The
shared API and condition-rule text must also reach the Google baseline. The
baseline must not gain method 1's checker-feedback repair loop merely because
it shares a mode selector.

This revision plans the requested interface and condition-language change.
It does not add per-step LLM replanning, rollout-guided repair, a new controller,
Goal adapters, or symbolic proofs for synthesized coordinate predicates.

## Baseline checking policy

Decision: change baseline prompts and the shared observation API only. Do not
add a complete DSL checker, checked interpreter, or semantic repair loop.
The reason is to keep the baseline as close as possible to Google's original
CaP implementation. Preserve the pinned upstream `LMP -> LMPFGen -> exec_safe`
generation, recursive helper discovery/binding, and Python execution flow.
Keep the vendored core unchanged: add mode-specific prompt construction and the
shared position API in the existing integration/compatibility layer. Do not
replace helper preparation with method 1's validate-before-binding algorithm,
role-guided repair, or checked interpreter. Existing model transport, logging,
and budget integration remains necessary adaptation, not an exact reproduction
of Google's environment.

Both baseline modes retain original recursive helper generation, `exec_safe`,
and Python execution. Shared adapter argument/handle validation, query purity
checks, primitive failure handling, and budgets remain active.

- `dsl_only`: policy and helper prompts require loop/branch/selection conditions
  to use DSL predicates, identity, logic, quantifiers, and their compositions.
  Position reads are described for action-parameter computation only.
- `coordinate_predicates`: policy and helper prompts additionally allow generated
  coordinate predicates and general numeric conditions, including counters.
  Numbers need not originate from coordinates.

These are advertised rules, not enforced guarantees for the baseline. Ordinary
Python may execute a numeric branch that violates its `dsl_only` prompt. It must
not be rejected or repaired by a newly introduced DSL gate. Other existing
runtime checks may still fail a program. Keep baseline checkpoints labeled
`validation=not_run`, `validation_enabled=false`, and
`refinement_enabled=false` in both modes. Method 1 continues enforcing its mode
with deterministic validation and the checked interpreter.

Report results by `(method, condition_mode)` and keep all scheduled failures in
the existing scoring denominator. A baseline success is a physical outcome, not
proof of prompt compliance. Any future post-hoc compliance measurement must be
reported separately and must not alter execution or repair this baseline.

## Work sequence

### 1. Define and resolve the configuration contract

Update [config.py](../config.py), [run_cap.py](../../entry/run_cap.py), and
[runner.py](../runner.py) with the proposed `condition_mode` setting. Reject
invalid mode values. Accept all four method/mode combinations. Resolve the mode
once and pass it to both methods' policy/helper generation, their execution
configuration, applicable validation/repair, and artifacts.

Keep mode-specific rule text separate from the shared action/observation API
text. Record an observation API version and method-1 checker version; record
baseline validation as disabled rather than implying a checker ran. Existing
callers that omit the new setting should use `dsl_only` without silently enabling
coordinate predicates.

Acceptance: configuration defaults/validation and mode propagation are tested;
logs identify the method and actual mode used at every source checkpoint.

| Method | dsl_only | coordinate_predicates |
| --- | --- | --- |
| cap_google_baseline | Position reads; advertised DSL-only conditions | Position reads; DSL plus coordinate/counter/general numeric conditions |
| cap_method_1 | Same observation and condition contract, enforced by method 1 | Same observation and condition contract, enforced by method 1 |

### 2. Add the shared read-only position API

Update [contracts.py](../contracts.py) and [adapter.py](../adapter.py) with
`get_gripper_position()` and `get_block_position(handle)`. Return copied,
immutable finite XYZ triples using the existing observation slices. Preserve
handle validation and keep physical IDs/internal environment objects private.
Position queries must not call `env.step`.

Expose the same getters to baseline Python and method 1's interpreter. Update
`GenerationOnlyAdapter` metadata so generation-only runs can describe the API
while still rejecting physical reads. Keep frozen historical predicates separate
from current position queries. Add coordinate-query events to existing logs.

Acceptance: fixture tests compare each getter to the underlying observation,
check refreshed values after movement and unchanged old copies, and verify
invalid-handle rejection and zero simulation steps for queries.

### 3. Extend method-1 static checking and checked execution together

Update [validation.py](../validation.py) and [runtime.py](../runtime.py).
Introduce Position3 values, literal component reads, numeric type checks, and
condition-source tracking for strict mode.
Preserve restricted syntax; do not enable arbitrary subscripting or imports.

In `dsl_only`, reject direct and indirect coordinate-derived conditions while
allowing coordinate-derived action offsets. In `coordinate_predicates`, permit
pure synthesized numeric predicates, coordinate comparisons, counter guards,
constant comparisons, and composition with DSL predicates. Do not require
numeric expressions in this mode to originate from coordinates. Apply the same rules in if/while, selection
lambdas, helper arguments/returns, and dependencies. Keep action effects out of
all condition evaluations. Runtime must enforce the same selected mode.

Acceptance: paired tests run the same source under both modes, including nested
helpers and value assignments. Positive cases cover DSL-only helpers in both
modes, plus coordinate predicates, counter guards, constant comparisons, and
numeric helper arguments in the extended mode. Negative cases cover numeric
condition bypasses in strict mode, actions in predicates, non-finite values,
invalid indexing, and protected API rebinding. General numeric guards are
positive cases in coordinate_predicates, not a rejection category.

### 4. Thread the mode through both generators and method-1 repair

Update baseline [method.py](../cap_google_baseline/method.py) and
[compatibility.py](../cap_google_baseline/compatibility.py), as well as method-1
[generation.py](../cap_method_1/generation.py),
[lmp.py](../cap_method_1/lmp.py), and all policy/helper/repair prompt templates.
Both baseline policy and helper prompts receive the selected mode and shared
action/position API descriptions. Preserve its upstream recursive helper
generation and disable semantic repair. Method-1 policy/helper/repair prompts
add the applicable role and checker rules. Local helper acceptance remains
distinct from complete root validation.

Repair feedback must describe the applicable mode rule. A generated coordinate
predicate rejected in `dsl_only` should produce a static rule diagnostic; the
extended mode should accept it when its types and effects are valid. Do not
include runtime coordinate results, task success, or controller feedback in
repair requests. No new model calls occur during ordinary coordinate queries.

Acceptance: scripted-model tests inspect policy, helper, and repair prompts;
check correct recursive mode propagation in both methods; verify method-1
complete checking happens before actions; and confirm coordinate queries do not
consume model calls. Test that baseline extended-mode prompts no longer prohibit
numeric conditions and that strict-mode prompts retain their DSL restriction.
Add a regression fixture with a numeric branch under baseline `dsl_only`: it
should still follow upstream execution, confirming that prompt configuration
has not silently introduced a DSL gate. Baseline must never call method 1's
validator or send checker-feedback repair requests in either mode. Retain the
pinned-notebook/core equivalence regression and baseline recursive-generation
tests to verify that changing prompts and API bindings did not replace the
original generation/execution algorithm.

### 5. Update provenance, reports, and documentation

Extend [checkpoints.py](../checkpoints.py) and existing logging/report integration
as needed to retain mode and API/checker versions with source artifacts. Log
coordinate query inputs/results and their execution location using the existing
host-side timeline. Keep reports bounded.

Update README and DSL/integration documentation when implementation lands, clearly
separating current and historical behavior. Update `BASELINE-FLOW.md` when its
shared position API and two prompt modes become available. Keep current status
and planned behavior clearly labeled. Retain the initial prompt-parity test
where applicable, but change its expectations deliberately if mode-specific
instructions make full prompts differ; compare the common API contract instead
of asserting accidental text equality.

Acceptance: link/reference checks pass; artifacts distinguish modes and do not
claim validation ran on the baseline. The two new mode names appear consistently
in configuration, prompts, checkpoints, and reports.

### 6. Validate the integrated paths

Run focused adapter, validation, runtime/method, and runner `unittest` modules
using the existing environment. Follow with the complete CaP test group once
the integrated changes pass. Use a paired scripted-policy simulator diagnostic
from an identical saved start to confirm shared observations, action semantics,
and convergence handling across all four method/mode combinations. Keep Unstack
invocations within 60 seconds.

Document diagnostic outcomes separately from benchmark task successes. A paid
model evaluation or a new multi-trial benchmark is a separate follow-up; this
implementation plan does not assume one has been run.

## Completion criteria

- Both methods support both condition modes, with one resolved mode per trial.
- All four combinations read the same position interface from matched states.
- Strict-mode rules remain DSL-only, including indirect dependencies; distinguish
  advertised baseline rules from enforcement rather than assuming compliance.
- Baseline modes remain prompt-only: no added DSL gate, interpreter, or repair.
- Extended-mode generated predicates can read gripper/block coordinates and combine
  numeric comparisons with existing DSL predicates; counters and other numeric
  conditions are accepted without requiring coordinate-derived values.
- Position reads are fresh, read-only, and local to program execution.
- Primitive/action semantics, independent scoring, and failure accounting remain intact.
- Tests and documentation distinguish interface compliance, execution success, and
  formal verification; no proof support for new predicates is implied.

Implementation should be committed in coherent increments following the work
sequence, with focused checks for each increment. Current changes in this request
are documentation only: the configuration design, revised flow, and this plan.
