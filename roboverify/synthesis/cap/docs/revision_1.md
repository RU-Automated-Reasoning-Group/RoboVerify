# Revision 1: runtime position observations and method-1 condition modes

Status: implementation plan; runtime changes have not started.

The requested change makes current gripper/block positions readable by generated
programs and gives method 1 two condition modes. The exact contract is in
[Method1_Configuration.md](Method1_Configuration.md), and the proposed execution
and data flow is in [METHOD-1-FLOW.md](METHOD-1-FLOW.md).

## Current state and target

| Area | Current implementation | Revision 1 target |
| --- | --- | --- |
| Runtime observations exposed to policy | DSL predicate truth and selected object handles | Add current gripper/block XYZ through a shared read-only API. |
| Method-1 conditions | DSL-only; no mode selector | `dsl_only` or `coordinate_predicates`, with `dsl_only` the default. |
| Coordinate-based predicates | Unsupported | LLM-generated pure predicates allowed in `coordinate_predicates`. |
| Method-1 numeric operations | Action-parameter arithmetic; object-only comparisons | Add Position3 component reads and provenance-checked comparisons where the mode permits. |
| LLM feedback | Task/API text, helper signatures, static repair diagnostics | Also include selected mode and coordinate API descriptions; keep actual coordinates runtime-only. |
| Physical execution | Shared Pick/Move/Release controllers | Preserve signatures, controller settings, failure behavior, and scoring. |

Both method-1 modes have access to the same O_t; only permission to use its
coordinates in conditions differs. Thus future `dsl_only` retains the current
condition restriction while gaining position-derived action parameters. The
shared API should also be exposed to the Google baseline for matched-observation
comparisons, without routing it through method 1's checker or repair loop.

This revision implements the requested interface and condition-language change.
It does not add per-step LLM replanning, rollout-guided repair, a new controller,
Goal adapters, or symbolic proofs for synthesized coordinate predicates.

## Work sequence

### 1. Define and resolve the configuration contract

Update [config.py](../config.py), [run_cap.py](../../entry/run_cap.py), and
[runner.py](../runner.py) with the proposed `condition_mode` setting. Reject
invalid modes and an explicit method-1-only flag on the baseline. Resolve the
mode once and pass it to generation, validation, execution, and artifacts.

Keep mode-specific rule text separate from the shared action/observation API
text. Record an observation API version and checker version. Existing callers
that omit the new setting should use `dsl_only` without silently enabling
coordinate predicates.

Acceptance: configuration defaults/validation and mode propagation are tested;
logs identify the actual mode used at every source checkpoint.

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

### 3. Extend static checking and checked execution together

Update [validation.py](../validation.py) and [runtime.py](../runtime.py).
Introduce Position3 values, literal component reads, and numeric provenance.
Preserve restricted syntax; do not enable arbitrary subscripting or imports.

In `dsl_only`, reject direct and indirect coordinate-derived conditions while
allowing coordinate-derived action offsets. In `coordinate_predicates`, permit
pure synthesized coordinate predicates, direct coordinate comparisons, and
composition with DSL predicates. Apply the same rules in if/while, selection
lambdas, helper arguments/returns, and dependencies. Keep action effects out of
all condition evaluations. Runtime must enforce the same selected mode.

Acceptance: paired tests run the same source under both modes, including nested
helpers and value assignments. Positive cases cover DSL-only helpers in both
modes and coordinate predicates in the extended mode. Negative cases cover
coordinate-condition bypasses in strict mode, actions in predicates, non-finite
values, invalid indexing, protected API rebinding, and unrelated numeric guards.

### 4. Thread the mode through recursive generation and repair

Update [generation.py](../cap_method_1/generation.py),
[lmp.py](../cap_method_1/lmp.py), and all policy/helper/repair prompt templates.
Every source unit receives the same selected mode, role, and shared API contract.
Local helper acceptance remains distinct from complete root validation.

Repair feedback must describe the applicable mode rule. A generated coordinate
predicate rejected in `dsl_only` should produce a static rule diagnostic; the
extended mode should accept it when its types and effects are valid. Do not
include runtime coordinate results, task success, or controller feedback in
repair requests. No new model calls occur during ordinary coordinate queries.

Acceptance: scripted-model tests inspect policy, helper, and repair prompts;
check correct recursive mode propagation; verify complete checking happens before
actions; and confirm coordinate queries do not consume model calls.

### 5. Update provenance, reports, and documentation

Extend [checkpoints.py](../checkpoints.py) and existing logging/report integration
as needed to retain mode and API/checker versions with source artifacts. Log
coordinate query inputs/results and their execution location using the existing
host-side timeline. Keep reports bounded.

Update README and DSL/integration documentation when implementation lands, clearly
separating current and historical behavior. Update `BASELINE-FLOW.md` when its
shared position API becomes available. Retain the initial prompt-parity test
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
and convergence handling across methods/modes. Keep Unstack invocations within
60 seconds.

Document diagnostic outcomes separately from benchmark task successes. A paid
model evaluation or a new multi-trial benchmark is a separate follow-up; this
implementation plan does not assume one has been run.

## Completion criteria

- Both condition modes work end to end with one resolved mode per trial.
- Both modes and the baseline read the same position interface from matched states.
- Strict-mode conditions remain DSL-only, including indirect dependencies.
- Extended-mode generated predicates can read gripper/block coordinates and combine
  checked comparisons with existing DSL predicates.
- Position reads are fresh, read-only, and local to program execution.
- Primitive/action semantics, independent scoring, and failure accounting remain intact.
- Tests and documentation distinguish interface compliance, execution success, and
  formal verification; no proof support for new predicates is implied.

Implementation should be committed in coherent increments following the work
sequence, with focused checks for each increment. Current changes in this request
are documentation only: the configuration design, revised flow, and this plan.
