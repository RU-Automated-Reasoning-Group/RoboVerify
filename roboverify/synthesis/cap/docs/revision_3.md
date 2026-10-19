# Revision 3: shared DSL checker correctness

- **Status:** implemented and tested on October 19, 2026; checker version `shared-contract-v3`. Both reported defects are fixed.
- **Goal:** make the shared checker consistently enforce the existing condition rules without rejecting valid helper compositions.
- **Scope:** loop type/effect propagation and generic helper type analysis. The two condition modes, primitive semantics, prompts, and generation workflows retain their current meaning.
- **References:** [shared checker architecture and rule table](SHARED-DSL-CHECKER.md), [audit results](../../../runs/cap-checker-rule-audit/20261019-161447/results.json), and [audit reproducer](../../../runs/cap-checker-rule-audit/20261019-161447/audit_rules.py).

## 1. Findings and impact

| Issue | Behavior before Revision 3 | Implemented correction | Affected components |
| --- | --- | --- | --- |
| CAP-CHECK-1: loop-carried action provenance | A condition could depend on an action-producing helper through values carried across several loop iterations while complete checking passed | Bounded fixed-point propagation detects the prohibited dependency before execution | Shared checker, Method 1/2 execution gate, baseline post-hoc assessment |
| CAP-CHECK-2: generic helper type merging | A valid object-valued loop passed concrete call-site analysis but failed the additional generic helper pass with `Expected object, got mixed` | Constrained generic parameters preserve compatible types across aliases, returns, comparisons, and loop joins | Shared checker, Method 1/2 generation and execution checks, baseline post-hoc assessment |

- The baseline deliberately has no pre-execution DSL gate. Fixing its shared audit does not add repair or prevent baseline execution.
- The audit checked all 13 table examples in both modes: all 26 combinations matched the documented static and runtime behavior.
- Before implementation, all 41 existing tests in `test_condition_modes`, `test_methods`, `test_revision2`, and `test_prompt_versions` passed. These tests did not cover the two boundary cases.
- The checks used scripted responses and recording-only adapters, with no live model calls or physical simulator runs. Historical experiment results remain unchanged.

## 2. CAP-CHECK-1: propagate loop types and effects to a stable result

### Reproducer

```python
def dirty(x):
    Pick(x)
    return True

a = True
b = True
done = False
stage1 = False
stage2 = False
while not done:
    if a:
        pass
    a = b
    b = dirty(b0)
    done = stage2
    stage2 = stage1
    stage1 = True
```

- On the third iteration, `a` contains a Boolean produced by `dirty`, which executes `Pick`.
- The existing transitive-effect rule rejects a condition that depends on an action-producing helper, even when the result is stored in a variable first.
  - Direct `flag = dirty(b0)` followed by `if flag` is already rejected.
  - Before Revision 3, the reproducer was incorrectly accepted in both modes and completed through both Method 1 and Method 2 with three recorded Pick calls.
  - With Revision 3, both methods reject it with `ACTION_IN_CONDITION` before any action.
  - The Pick calls occur in the loop body, not during evaluation of `if a`. This is a gap in tracking the origin of stored values, not evidence that the direct runtime action-in-condition guard is bypassed.

### Original cause

- [Checker.block](../validation.py) checks the loop body once, merges entry/body environments, and checks the body once more.
- It does not propagate type/effect information until the loop environment stabilizes; effects moving through several variables can arrive after the analyzed iterations.
- [Runtime.condition](../runtime.py) requires a Boolean and blocks actions invoked while evaluating a condition, but stored runtime values do not retain action-provenance metadata.

### Implemented change

- `Checker.loop_flow` computes a fixed point over loop-head environments and generic type constraints.
  - Entry states represent zero iterations; only normal fallthrough and `continue` states feed the next iteration.
  - `break` states exit the loop and bypass its `else`; return states propagate to the caller.
  - Nested loops and branch joins retain separate control-flow outcomes. Unreachable source is still checked but cannot overwrite real exit states.
- `Value` joins retain all type alternatives and action effects. Effects from evaluated helper arguments remain attached to the call result even when the helper ignores that argument.
- Analysis retains the existing shared deadline and adds a deterministic limit of 100,000 analysis operations. Exhaustion returns an analysis `error`, not a source violation, execution approval, or model-repair request.
- The transitive-effect contract is unchanged; runtime action APIs and condition modes are unchanged.

### Acceptance criteria

- Reject the reproducer in both modes before any action executes through Method 1 or Method 2.
- Detect dependencies that require more than two iterations to reach a guard or branch, including longer assignment chains and helper returns.
- Continue accepting pure loop-carried Boolean, object, and mode-permitted numeric values.
- Retain rejection of direct action calls in conditions and the existing Boolean-return requirements.

## 3. CAP-CHECK-2: preserve type constraints during generic helper analysis

### Reproducer

```python
def policy(base):
    top = base
    while exists(lambda x: not ON_star(x, base)):
        if eq(top, base):
            pass
        top = get(lambda x: not ON_star(x, base))

policy(b0)
```

- With `base: ObjectHandle`, `top` is an object handle initially and after every loop iteration.
- Before Revision 3, concrete call-site analysis passed, but complete checking rejected this program in both modes with `TYPE: Expected object, got mixed`.
- With Revision 3, complete checking accepts the program in both modes.
- This is a type-analysis false rejection. The example does not claim termination or task success.

### Original cause

- After analyzing concrete calls, [Checker.check](../validation.py) checks every definition again with unknown parameter types, including definitions already checked at their call sites.
- [merge_values](../validation.py) merges `unknown` and `object` into `mixed` without preserving the parameter constraints established by object-only operations.
- The generic pass then treats this merged value as incompatible with an object argument.

### Implemented change

- Generic parameters use distinct type variables with domains over the supported types, rather than indistinguishable anonymous unknown values.
- `Checker.constrain` narrows compatible domains and links parameters when an operation requires a common type.
  - Aliases, helper returns, comparisons, and loop joins retain the same variable identities and constraints.
  - Concrete object/number conflicts still fail; a compatible generic/object merge no longer becomes a spurious conflict.
- Concrete call-site specialization and generic checking of every definition are both retained, including unused definitions.
- Missing local helper calls remain separate obligations. Complete checks still reject unresolved helpers and incompatible actual arguments.

### Acceptance criteria

- Accept the reproducer in both modes through the complete shared checker.
- Accept equivalent object-valued loops in inline and separately generated helpers.
- Reject callers passing numbers or positions where object handles are required, including incompatible calls to the same helper.
- Continue rejecting forbidden syntax, action effects, and numeric conditions in DSL-only mode inside unused definitions.

## 4. Implementation and validation

| Step | Completed work | Validation |
| --- | --- | --- |
| 1 | Added both audit reproducers as regression tests | Confirm each test distinguishes the current defect from the expected behavior |
| 2 | Corrected generic type constraints and loop joins | Valid object-valued helper cases pass; incompatible concrete types still fail |
| 3 | Implemented stable loop type/effect propagation | Delayed action dependencies fail; pure loops pass; bounded analysis never silently succeeds |
| 4 | Checked integration across all callers | Method 1/2 gate execution consistently; baseline receives the same core verdict post-hoc without behavioral changes |
| 5 | Ran the full CaP suite and focused boundary tests | Preserve condition-mode semantics, prompt-version behavior, helper repair, and artifact capture |
| 6 | Recorded `shared-contract-v3`, source hashes, and analysis limits | Record the implementation version/source hash and any remaining unsupported cases |

- Relevant files: [validation.py](../validation.py), [checker.py](../checker.py), [runtime.py](../runtime.py), [condition-mode tests](../test_condition_modes.py), and [shared-method/audit tests](../test_revision2.py).
- Preserve original source, calls, rollouts, and audit verdicts from completed experiments. Any retrospective assessment with the revised checker must be explicitly versioned and stored separately.
- Do not claim that the fixes change historical success rates without a separately identified evaluation. Static contract compliance remains distinct from physical success and formal task verification.

## 5. Completed validation and remaining scope

- **106 CaP unittest tests passed**, including **18 new Revision 3 tests** in [test_revision3.py](../test_revision3.py).
  - Coverage includes longer loop-carried dependencies, nested loops, branch joins, `break`/`continue`, loop `else`, helper return/argument effects, valid object/position helpers, incompatible callers, unused definitions, and analysis exhaustion.
  - Method 1 and Method 2 reject delayed effects before actions and accept valid object-valued loops in both modes.
  - Baseline still executes without a DSL gate; post-hoc checking detects the delayed effect without changing actions or model calls.
- All **26 rule-table example/mode combinations** retain their expected static/runtime behavior.
- All **four previously rejected successful GPT coordinate policy sources** pass source-only checking under the revised contract.
  - These are saved-source checks, not reconstructions of complete historical baseline binding bundles.
  - Original compliance artifacts, model requests, physical outcomes, and trajectories were preserved; no paid model or physical benchmark rerun was performed.
- Validation artifacts: [suite output and audit summary](../../../runs/cap-revision3-validation/20261019-195250/README.md), [rule checks](../../../runs/cap-revision3-validation/20261019-195250/rule_audit.json), and [historical source rechecks](../../../runs/cap-revision3-validation/20261019-195250/historical_source_rechecks.json).
- The analysis remains conservative and bounded. These fixes do not establish completeness for arbitrary Python, termination, physical safety, or task correctness.
- Revisions [4](revision_4_failure_recover.md), [5](revision_5_add_tasks.md), and [6](revision_6_add_demonstration.md) remain unimplemented; they are outside this change.
