# Revision 3: shared DSL checker correctness

- **Status:** planned; both implementation defects remain open.
- **Goal:** make the shared checker consistently enforce the existing condition rules without rejecting valid helper compositions.
- **Scope:** loop type/effect propagation and generic helper type analysis. The two condition modes, primitive semantics, prompts, and generation workflows retain their current meaning.
- **References:** [shared checker architecture and rule table](SHARED-DSL-CHECKER.md), [audit results](../../../runs/cap-checker-rule-audit/20261019-161447/results.json), and [audit reproducer](../../../runs/cap-checker-rule-audit/20261019-161447/audit_rules.py).

## 1. Findings and impact

| Issue | Current behavior | Required behavior | Affected components |
| --- | --- | --- | --- |
| CAP-CHECK-1: loop-carried action provenance | A condition can depend on an action-producing helper through values carried across several loop iterations, while complete checking still passes | Propagate types and action effects across all possible loop iterations; reject the prohibited condition dependency | Shared checker, Method 1/2 execution gate, baseline post-hoc assessment |
| CAP-CHECK-2: generic helper type merging | A valid object-valued loop passes concrete call-site analysis but fails the additional generic helper pass with `Expected object, got mixed` | Preserve compatible argument/type constraints and accept valid object-handle flows | Shared checker, Method 1/2 generation and execution checks, baseline post-hoc assessment |

- The baseline deliberately has no pre-execution DSL gate. Fixing its shared audit does not add repair or prevent baseline execution.
- The audit checked all 13 table examples in both modes: all 26 combinations matched the documented static and runtime behavior.
- All 41 existing tests in `test_condition_modes`, `test_methods`, `test_revision2`, and `test_prompt_versions` passed. These tests did not cover the two boundary cases.
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
  - The reproducer is incorrectly accepted in both modes and completes through both Method 1 and Method 2 with three recorded Pick calls.
  - The Pick calls occur in the loop body, not during evaluation of `if a`. This is a gap in tracking the origin of stored values, not evidence that the direct runtime action-in-condition guard is bypassed.

### Cause

- [Checker.block](../validation.py) checks the loop body once, merges entry/body environments, and checks the body once more.
- It does not propagate type/effect information until the loop environment stabilizes; effects moving through several variables can arrive after the analyzed iterations.
- [Runtime.condition](../runtime.py) requires a Boolean and blocks actions invoked while evaluating a condition, but stored runtime values do not retain action-provenance metadata.

### Planned change

- Replace the two-pass loop approximation with bounded fixed-point analysis of the loop environment.
  - Merge the entry state with states that can return to the loop head.
  - Propagate value types and action effects together until no information changes.
  - Recheck the guard and body under the stabilized environment.
  - Account for zero iterations, nested loops, branch joins, and `break`/`continue` paths without dropping effects.
- Keep analysis within the shared checker deadline/resource bounds. Exhausted or incomplete analysis must produce an explicit non-pass result, never execution permission.
- Preserve the distinction between executing an action in the body and using an action-derived value as a condition. Retain the existing transitive-effect contract.

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
- Concrete call-site analysis passes, but complete checking rejects this program in both modes with `TYPE: Expected object, got mixed`.
- This is a type-analysis false rejection. The example does not claim termination or task success.

### Cause

- After analyzing concrete calls, [Checker.check](../validation.py) checks every definition again with unknown parameter types, including definitions already checked at their call sites.
- [merge_values](../validation.py) merges `unknown` and `object` into `mixed` without preserving the parameter constraints established by object-only operations.
- The generic pass then treats this merged value as incompatible with an object argument.

### Planned change

- Represent unresolved parameter types separately from genuinely incompatible concrete types.
- Preserve constraints induced by API operations and helper calls during generic analysis and loop joins.
  - An object-only operation constrains its argument to an object handle.
  - Compatible unresolved and concrete types must not become a spurious conflict.
  - Real object/number conflicts must still fail.
- Reconcile generic checks with concrete call-site specialization while retaining checks on unused definitions.
- Do not fix the issue by accepting every unknown/mixed value or skipping checks for uncalled helpers. Unresolved obligations must not authorize complete-program execution.

### Acceptance criteria

- Accept the reproducer in both modes through the complete shared checker.
- Accept equivalent object-valued loops in inline and separately generated helpers.
- Reject callers passing numbers or positions where object handles are required, including incompatible calls to the same helper.
- Continue rejecting forbidden syntax, action effects, and numeric conditions in DSL-only mode inside unused definitions.

## 4. Implementation and validation plan

| Step | Work | Validation |
| --- | --- | --- |
| 1 | Turn both audit reproducers into regression tests | Confirm each test distinguishes the current defect from the expected behavior |
| 2 | Correct generic type constraints and loop joins | Valid object-valued helper cases pass; incompatible concrete types still fail |
| 3 | Implement stable loop type/effect propagation | Delayed action dependencies fail; pure loops pass; bounded analysis never silently succeeds |
| 4 | Check integration across all callers | Method 1/2 gate execution consistently; baseline receives the same core verdict post-hoc without behavioral changes |
| 5 | Run the existing CaP suite and focused boundary tests | Preserve condition-mode semantics, prompt-version behavior, helper repair, and artifact capture |
| 6 | Document the revised checker identity and limits | Record the implementation version/source hash and any remaining unsupported cases |

- Relevant files: [validation.py](../validation.py), [checker.py](../checker.py), [runtime.py](../runtime.py), [condition-mode tests](../test_condition_modes.py), and [shared-method/audit tests](../test_revision2.py).
- Preserve original source, calls, rollouts, and audit verdicts from completed experiments. Any retrospective assessment with the revised checker must be explicitly versioned and stored separately.
- Do not claim that the fixes change historical success rates without a separately identified evaluation. Static contract compliance remains distinct from physical success and formal task verification.
