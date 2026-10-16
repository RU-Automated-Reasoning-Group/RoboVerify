# Method 1 configuration: condition modes and position observations

Status: implemented on 2026-10-16. This document specifies two condition modes over a shared runtime position interface. Both
`cap_method_1` and `cap_google_baseline` must support the same mode selector and
observation API. Method-1 checking and repair remain method-specific. See
[METHOD-1-FLOW.md](METHOD-1-FLOW.md) for the execution flow and
[revision_1.md](revision_1.md) for implementation status and validation evidence.

## Configuration boundary

Both modes let generated programs read the same position observations. The mode
controls whether those values may influence loop/branch conditions and object
selection. Position data is read by the executing program; it is not
implicitly serialized into LLM prompts or sent to the model after every action.

Configuration:

```text
RunConfig.condition_mode = "dsl_only" | "coordinate_predicates"
CLI: --condition-mode dsl_only|coordinate_predicates
Default: dsl_only
```

The selector applies to both `cap_method_1` and `cap_google_baseline`, with `dsl_only` the default for each.
Reject unknown modes, not valid baseline/mode combinations. Record method, mode,
observation API version, and actual validation configuration in every trial and
source checkpoint. A checker version must not imply that checking ran.

The advertised condition rules are identical for the same mode across methods.
The table below specifies method 1's enforced behavior. The current baseline has
no complete DSL checker: its API checks and upstream Python filtering do not
establish compliance. Revision 1 deliberately preserves that baseline behavior:
modes change its prompts and shared observation API, with no added DSL checker or repair. See
[revision_1.md](revision_1.md); do not infer enforcement from mode selection.

| Property | `dsl_only` | `coordinate_predicates` |
| --- | --- | --- |
| Existing task-available DSL predicates | Allowed | Allowed |
| Boolean composition, identity, relational quantifiers | Allowed | Allowed |
| Generated helpers combining existing DSL predicates | Allowed | Allowed |
| Read current gripper/block coordinates | Allowed | Allowed |
| Use coordinates to compute action offsets | Allowed | Allowed |
| Use coordinate comparisons as conditions | Rejected, including indirect uses | Allowed with numeric type checks; no coordinate-origin requirement |
| Counter-based and other numeric conditions | Rejected | Allowed |
| LLM-generated predicates reading coordinates | Rejected when used as conditions | Allowed, including composition with DSL predicates |
| Actions inside conditions or query lambdas | Rejected | Rejected |
| Raw simulator objects, rewards, success flags, images | Unavailable | Unavailable |
| Model repair feedback | Static interface diagnostics | Static interface diagnostics under the extended condition rules |

`dsl_only` preserves the earlier condition restriction while adding position
reads for action parameters. Sharing this observation API
makes the mode comparison about the allowed condition language: DSL-only versus
DSL plus general numeric predicates, including coordinate and counter tests.
The three physical primitives and their meanings are unchanged.

## Runtime observation contract

For a scene with n blocks, expose the position component of state:

```text
O_t = (p_gripper(t), p_1(t), ..., p_n(t))
p_i(t) = (x_i(t), y_i(t), z_i(t))
```

All positions use the simulator world frame and metres. Gripper position means
the existing end-effector XYZ observation used by the shared controller, not
finger joint positions. Block positions are block centres. Velocity, orientation,
gripper opening, desired goals, and reward are outside this observation contract.
Object identity remains separate from position equality.

Runtime API:

```text
get_gripper_position() -> Position3
get_block_position(block) -> Position3
```

`Position3` is a copied, immutable triple of finite numbers. Permit component
access only by literal indices `0`, `1`, and `2`; this is a narrow extension to
method 1's earlier ban on all subscripting. It does not authorize arbitrary container
indexing, attributes, dynamic indices, array libraries, or imports.

Block operands use existing validated object handles (`b0`, assignments, and
query-bound objects). No physical-ID lookup API or coordinate-bearing `tbl` is
introduced. The notation `p_1` through `p_n` denotes the physical domain; it does
not introduce hard-coded `block_1` variables into generated code. Quantifier
lambdas can inspect each bound object's position in `coordinate_predicates` mode.

Each getter reads current state without advancing physics, returns a copy, and
logs its query and result. Saving `p = get_block_position(b)` saves a snapshot:
`p` does not change after a Move. Call the getter again to observe the updated
position. Pure condition evaluation performs no physical actions, so component
reads during a condition refer to the same unchanged simulation state. Any
future asynchronous simulator would need an explicit observation snapshot rule.

The adapter obtains gripper XYZ from `obs[:3]` and block i XYZ from
`obs[10 + 12*i : 13 + 12*i]`. The host retains the ID mapping. Reverse's
`ON_star_zero` continues using frozen trial-entry geometry independently of
these current-state position getters.

## Mode 1: DSL-only conditions

The permitted atoms remain the task-available `ON`, `ON_star`, `Higher`,
`Scattered`, and object identity. `ON_star_zero` is available for Reverse only.
Conditions may use Boolean constants, logical composition, quantifiers, and
pure generated helpers that expand to these operations.

Allowed example using the current predicate interface:

```python
def clear_above(b):
    return forall(lambda other: eq(other, b) or not ON_star(other, b))

if clear_above(b0):
    Pick(b0)
```

This illustrates permitted syntax, not a complete task solution. New helper
names do not imply new primitive predicates: `clear_above` only composes the
existing vocabulary.

Rejected example under this mode:

```python
def above_by_margin(a, b):
    return get_block_position(a)[2] > get_block_position(b)[2] + 0.01

ready = above_by_margin(a, b)
if ready:
    Pick(a)
```

Here `a` and `b` denote previously bound handles. The checker must follow the
value from `above_by_margin` through `ready` to the branch, including dependencies
and helper arguments. A wrapper returning bool cannot hide a coordinate-derived
condition. The same rule applies to while guards, query selection, and conditional
break/continue paths. Coordinate arithmetic is permitted for action parameters,
but cannot determine which branch, loop iteration, or object binding is chosen.

## Mode 2: DSL and synthesized coordinate predicates

The existing DSL remains available. In addition, the LLM may generate pure
predicate functions that read positions and combine checked finite arithmetic,
numeric comparisons, Boolean logic, identity, and existing DSL predicates.
Direct numeric comparisons in guards are also allowed; requiring a named
wrapper would not change their meaning. Despite the mode name
`coordinate_predicates`, numbers need not originate from coordinates: counters,
iteration limits, thresholds, and other local numeric computations are allowed.

Examples, with handles bound by the caller:

```python
def above_by_margin(a, b):
    return get_block_position(a)[2] > get_block_position(b)[2] + 0.01


def gripper_near(b):
    g = get_gripper_position()
    p = get_block_position(b)
    dx = g[0] - p[0]
    dy = g[1] - p[1]
    dz = g[2] - p[2]
    return dx * dx + dy * dy + dz * dz <= 0.0025


def higher_and_separated(a, b):
    return Higher(a, b) and above_by_margin(a, b)
```

These helpers execute locally on current observations. Synthesizing their source
requires model calls; invoking an already prepared helper does not. Thresholds
are model-generated finite constants or checked numeric arguments, recorded in
source artifacts. There is no rollout-based threshold optimization or CEM.

No numeric-origin restriction applies in this mode. The checker verifies types,
supported operations, and condition purity rather than requiring a dependency
on position data. Permit numeric `<`, `<=`, `>`, `>=`, `==`, and `!=`, while
preserving identity semantics for object `==` / `!=`. Local counters may be
updated in program bodies and tested in guards. For example:

```python
def near_within_limit(b, attempts, limit):
    return attempts < limit and gripper_near(b)


attempts = 0
while attempts < 10:
    attempts = attempts + 1
```

These examples illustrate permitted conditions, not a complete task policy.
Method 1 supports finite `+`, `-`, `*`, `/` arithmetic and numeric comparisons
within its declared subset; this mode does not enable arbitrary Python libraries.
Numeric constants and helper arguments are valid comparison operands even when
neither side depends on a coordinate. Apply these rules equally to direct guards,
query lambdas, and nested helpers. Conditions must still return Booleans and
remain free of action side effects throughout their call graph. Object handles
do not become numbers, and the API still exposes no physical IDs or simulator
internals. Existing execution budgets apply even when a program uses its own
counter-based stopping condition.

Value-flow analysis is still required to enforce DSL-only conditions in
`dsl_only`; it must not impose that mode's origin restrictions on
`coordinate_predicates`.

`above_by_margin` is not an alternative implementation of `Higher`; their
semantics differ. Generated functions cannot redefine protected DSL APIs. They
are runtime predicates, not automatically registered symbolic predicates with
Z3 axioms, weakest-precondition rules, or invariant-inference support. Formal
verification of programs using them would require a separate extension.

## Shared generation, execution, and evaluation

Pass the selected mode and position API contract to policy, helper, and repair
prompts, deterministic validation, and the checked interpreter. Resolve the
mode once in the runner. A mismatch between prompt, checker, and runtime modes
must fail rather than execute under a different policy.

Both method-1 modes retain recursive helper generation, local interface-only repair,
complete bundle validation, checked execution, and independent terminal scoring.
Model feedback contains source diagnostics, not observed coordinates, failed
rollout trajectories, controller errors, task-success feedback, or reference
solutions. Actual position values remain runtime data and host-side artifacts.

The baseline receives the same mode-specific condition contract in policy and
helper prompts, but continues original Python execution with
`validation_enabled=false` and `refinement_enabled=false`. Unsupported code may
fail at execution, and some prompt violations may run. No DSL-compliance claim
follows from its selected mode or physical success. Shared adapter checks remain
active; these are not complete loop/branch condition validation.

For comparisons, use identical initial snapshots, model settings, controllers,
position interfaces, and budgets, with mode-specific rule text disclosed.
Report mode, generation/repair costs, validation failures, execution failures,
and physical success separately. Constraint acceptance does not establish task
success or a formal proof. Existing task-adapter gaps and Unstack's 60-second
limit remain in force.

## Implementation and validation

| Area | Required change |
| --- | --- |
| [config.py](../config.py), [run_cap.py](../../entry/run_cap.py), [runner.py](../runner.py) | Resolve, validate, propagate, and record the condition mode for both methods. |
| [contracts.py](../contracts.py), [adapter.py](../adapter.py) | Define position API types/metadata and fresh read-only getters over the existing observation. |
| [generation.py](../cap_method_1/generation.py), [prompt templates](../cap_method_1/prompts/policy.txt) | Carry mode/API rules into policy, helper, and repair requests and recursive preparation. |
| [baseline setup](../cap_google_baseline/method.py), [compatibility.py](../cap_google_baseline/compatibility.py) | Pass the shared position API and selected condition rules to baseline policy/helper prompts; keep validation status explicit. |
| [validation.py](../validation.py) | Add Position3, restricted indexing, mode-sensitive comparisons, and transitive condition checking. Track condition sources for dsl_only; allow general numeric conditions in coordinate_predicates. |
| [runtime.py](../runtime.py) | Interpret the added checked operations and enforce the same selected mode; retain no-action condition guards. |
| [checkpoints.py](../checkpoints.py), [logger.py](../logger.py) | Retain mode/version provenance and coordinate-query events without feeding them into repairs. |

Validation must cover fresh reads after movement, immutable returned positions,
invalid handles, index/type errors, and both modes on the same generated source.
Include indirect coordinate conditions, nested helper dependencies, query
lambdas, action attempts inside predicates, and protected API rebinding. Counter
guards, constant comparisons, and numeric helper arguments must pass in
`coordinate_predicates` and fail as conditions in `dsl_only`. Test
that model requests do not acquire actual runtime coordinate values through
logging or repair. Existing DSL-only cases must retain their behavior.

See [revision_1.md](revision_1.md) for completed implementation and validation.
The new coordinate predicates have no automatic symbolic-proof support.
