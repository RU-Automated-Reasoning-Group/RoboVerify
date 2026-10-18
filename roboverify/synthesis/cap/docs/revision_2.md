# Revision 2 discussion

Status: discussion draft. The requested scope is API/prompt clarity and versioning
(item 1), plus parallel execution of future experiment batches (item 2). The reasoning-enabled,
16k-output Stack experiments have completed using the existing prompts; their
[report](../../../runs/cap-stack-reasoning-comparison/20261018-180533-16k/report.md)
is the reference evidence for this discussion. This document does not change
`API_DESCRIPTION`, runtime interfaces, condition rules, or experimental settings.
Further revision items remain for follow-up discussion.

## 1. Make API types and semantics explicit in the prompt

The proposed direction is to describe **each input's type and meaning**, rather
than rely on the opening sentence that all object operands are opaque handles.
The prompt should also state outputs, effects, coordinate frame/units, and the
important failure cases for each API. This is a prompt-clarity change; it does
not add new primitives or change the two condition modes.

### Motivation from the current interface

[contracts.py](../contracts.py) already states that Move has four arguments and
uses three object references plus an offset vector. Nevertheless, observed
trial failures include six-argument Move calls and passing a position or a
numeric coordinate where an object handle is required. These are distinct
mistakes: correcting argument count alone does not correct operand types.

This observation motivates clearer typing and examples; it does not establish
that prompt changes will eliminate those failures. Controller nonconvergence,
generation limits, and task correctness require separate analysis after the
current experiments. No conclusion about their causes is made here.

### 1.1 Introduce a small type glossary

| Prompt type | Meaning | Source / permitted use |
| --- | --- | --- |
| `ObjectHandle` | Identity of one physical block in the current trial. It is neither a physical ID nor a position. | The input `b0`, identity assignments, and `get(...)` / query bindings. Pass it to primitive and predicate object parameters. |
| `Position3` | Immutable snapshot `(x, y, z)` in world-frame metres. Block positions are block centres; gripper position is the controller's end-effector position. | Returned by position getters. Read components with literal indices 0/1/2. Re-query after motion for a current reading. |
| `Offset3` | Exactly three finite scalar displacements `[dx, dy, dz]`, in metres. | Construct it from constants or permitted arithmetic; use it as Move's fourth argument. It is not an absolute target position. |
| `Scalar` | A finite numeric value. | Action offsets and, in the extended condition mode, numerical predicates and counters. A scalar cannot replace an object handle. |
| `Bool` | A condition result. | DSL predicates, quantifiers, identity/Boolean operations, and mode-permitted numeric conditions. |
| `None` | No policy-visible return value. | Returned by successful primitive calls; it is not a success Boolean or an observation. |

These are explanatory type names in the prompt, not additional constructors or
APIs that generated code may call. `Offset3` names an existing three-element
list/tuple contract, not a new runtime class. The typed signatures below are
reference notation; method 1 still rejects annotations in generated function
definitions. Do not suggest importing or instantiating these types.

### 1.2 Describe every primitive parameter separately

Reference signatures:

```text
Pick(block: ObjectHandle) -> None
Move(x_ref: ObjectHandle, y_ref: ObjectHandle, z_ref: ObjectHandle,
     offset: Offset3) -> None
Release(reference: ObjectHandle, height_offset: Scalar) -> None
```

| API / input | Type | Meaning to state explicitly |
| --- | --- | --- |
| `Pick.block` | ObjectHandle | The block to grasp. The controller approaches horizontally at the current gripper height, opens, descends, and closes the gripper. Pick does not include a subsequent lift or a complete transport. |
| `Move.x_ref` | ObjectHandle | Use this block's current world X to anchor the target X. This argument is the block itself, not its X value. |
| `Move.y_ref` | ObjectHandle | Use this block's current world Y to anchor the target Y. It may differ from x_ref. |
| `Move.z_ref` | ObjectHandle | Use this block's current world Z to anchor the target Z. It may differ from the XY references. |
| `Move.offset` | Offset3 | One vector argument containing dx, dy, dz. The call has exactly four positional arguments, not six. |
| `Release.reference` | ObjectHandle | The block whose invocation-time Z anchors the gripper's retreat target. This does not specify an absolute placement position. |
| `Release.height_offset` | Scalar | Retreat-height offset from the reference block's Z, in metres. It is not an absolute world Z or an extra lowering command. |

Explain Move with an exact equation:

```text
At invocation:
  target_X = X(x_ref) + offset[0]
  target_Y = Y(y_ref) + offset[1]
  target_Z = Z(z_ref) + offset[2]
```

The host reads reference coordinates when the Move starts and drives the gripper
toward that target with the gripper closed. It does not continuously shift the
target as a reference block moves during that primitive. Reading coordinates in
policy code does not change this object-relative action contract.

Release opens the gripper at its current location, then retreats to
`Z(reference at invocation) + height_offset`, preserving the pre-opening XY
target. It does not perform a complete placement sequence. These meanings come
from [move_relative, pick, and release](../../api/control.py), rather than
inferring semantics from the API names.

Each primitive uses the existing shared controller and step limit. A
nonconverged primitive raises an execution failure through the adapter. A
successful return is `None`; convergence alone is not proof of grasp, collision
freedom, or task completion. This information should be stated once in a short
shared effects/returns paragraph, rather than repeated at length for every API.

### 1.3 Give predicates typed domains and precise truth conditions

All binary tower predicates take **two ObjectHandles** and return Bool. They do
not accept Position3 values. The prompt should distinguish identity, reflexive
height/above relations, direct-on relations, and frozen history.

| Predicate | Meaning worth stating in the API reference |
| --- | --- |
| `eq(a, b)` | Object identity, not equal coordinates. Object `==` / `!=` have the same identity meaning. |
| `ON_star(a, b)` | Shared geometric interpretation: `abs(X(a)-X(b)) < 0.025`, `abs(Y(a)-Y(b)) < 0.025`, and `Z(a) >= Z(b)`. Reflexive; this is not itself a physical-support guarantee. |
| `ON(a, b)` | Distinct objects with ON_star(a,b) and no distinct intermediate block c satisfying ON_star(a,c) and ON_star(c,b). Irreflexive. |
| `Higher(a, b)` | `Z(a) >= Z(b) - higher_tolerance`. Non-strict and reflexive. State the actual configured tolerance, rather than only the default 0.001 m. |
| `Scattered(a, b)` | `abs(X(a)-X(b)) >= 0.10 OR abs(Y(a)-Y(b)) >= 0.10`. The OR is important; this is not a Euclidean-distance threshold. |
| `ON_star_zero(a, b)` | ON_star evaluated on frozen trial-entry geometry. Advertise it only for tasks that expose it, currently Reverse. |

These formulas describe the host implementations. In `dsl_only`, generated
conditions must call the allowed DSL predicates or compose them; showing the
formulas is not permission to replace them with numeric comparisons. In
`coordinate_predicates`, generated numerical predicates are allowed, including
counters and comparisons unrelated to coordinates. Neither mode permits
redefining protected DSL APIs. Baseline receives these rules through prompts;
method 1 enforces them with its checker/interpreter.

The geometric facts belong to the shared contract and task configuration.
No sampled scene's coordinates or predicate table should be added to the prompt
as part of this change. If tolerance text is made dynamic, pass the resolved
value consistently to policy/helper/repair prompt assembly for both methods.

### 1.4 Type position getters and selection queries too

```text
get_gripper_position() -> Position3
get_block_position(block: ObjectHandle) -> Position3
exists(lambda objects: condition) -> Bool
forall(lambda objects: condition) -> Bool
get(lambda block: condition) -> ObjectHandle
```

Each lambda parameter binds a physical block handle. `exists` and `forall`
permit one or more parameters; `get` takes exactly one. The syntax above is
schematic: the actual arity is written in the lambda parameter list.

Explain the difference between a Boolean query and selection. `get` returns one
enabled object handle and raises when no witness exists; it does not return
False, None, or a position. The selection order is owned by the host, and the
program should not infer a physical ID from it. Query lambdas must return Bool
and be action-free. Their condition language follows the selected mode.

The position getters are read-only and do not advance physics. A stored
Position3 is a snapshot, not a live reference to the block. There is no generic
`get_observation`, raw `env`, coordinate-bearing `tbl`, or Goal handle in the
current tower API.

### 1.5 Add short interface examples, not benchmark solutions

A few type-focused examples could make the distinction concrete. For instance,
with `u`, `v`, and `w` already bound handles:

```python
Move(u, v, w, [0.01, -0.02, 0.08])
```

This illustrates separate coordinate references and one offset vector. It is
not a prescribed safe waypoint or a Stack policy. A getter example can explain
that `p = get_block_position(u)` makes `p` a Position3 while `u` remains the
ObjectHandle used by Pick, Move, Release, and DSL predicates.

Possible negative examples for an API reference:

| Invalid expression | Explanation |
| --- | --- |
| `Move(u, v, w, 0.0, 0.0, 0.08)` | Six arguments; the offsets must be one three-element vector. |
| `Move(p[0], p[1], p[2], [0.0, 0.0, 0.08])` | The first three arguments are numbers, not object handles. |
| `Move(p, p, p, [0.0, 0.0, 0.08])` | Position3 cannot replace ObjectHandle. |
| `Higher(get_block_position(u), get_block_position(v))` | DSL predicates take handles, not coordinate triples. |
| `Release(u, get_block_position(u)[2])` when intended as an absolute retreat target | The second argument is an offset; the implementation would add it to Z(u). |

The last example is type-correct but semantically wrong for that stated intent;
a type checker alone cannot detect every misunderstanding. Keep this distinction
in the discussion of expected benefits. Negative examples should be clearly
labeled; whether they improve generation more than positive examples alone is
an empirical question.

### 1.6 Suggested prompt organization and evaluation

A compact ordering to discuss is: type glossary; typed API signatures and
parameter meanings; outputs/effects; predicate semantics; short examples; then
the selected condition-mode rules. The shared contract should be reused in
policy and helper prompts for both methods, and in method-1 repair prompts.

Prefer one structured API specification from which descriptions can be rendered,
so arity/type metadata and prompt wording are less likely to diverge. This is a
possible implementation direction, not an approved requirement to redesign the
checker or baseline algorithm. Preserve Google's pinned core and the existing
method distinction.

Before adopting revised wording, check examples against the existing checker
and adapter, and verify identical shared API text across both methods. Then
compare old/new prompt versions with matched models, snapshots, modes, budgets,
and generation settings. Count arity errors, operand-type errors, no-witness
queries, accepted policies, model usage, and physical outcomes separately.
A lower interface-error rate must not be reported as guaranteed task success.

The completed 16k experiments remain the old-prompt evidence. A future
prompt experiment should have an explicit prompt version/hash and its own
report; do not mix changed prompts with those existing results.

### 1.7 Preserve prompt versions and select them through configuration

Requested requirement: retain the existing prompt when introducing the revised
one, and select the prompt version through configuration for later paper
comparisons and controlled variant experiments. Do not overwrite the old prompt
and rely on manually checking out historical commits to reproduce it.

Proposed interface, not yet implemented:

```text
RunConfig.prompt_version = "legacy_v1" | "typed_v2"
CLI: --prompt-version legacy_v1|typed_v2
Compatibility default: legacy_v1
```

| Version | Contents | Status |
| --- | --- | --- |
| `legacy_v1` | The current post-Revision-1 API description, mode rules, and policy/helper/repair templates used by the completed 16k experiments. | Preserve exactly as the reference prompt package. Those runs predate an explicit prompt_version field. |
| `typed_v2` | The revised type glossary, per-input meanings, primitive/predicate descriptions, and whichever interface examples are agreed within item 1. | Discussion draft; not implemented or evaluated. |

Freeze all model-facing components, not only `API_DESCRIPTION`: shared API text,
condition-mode rule text, policy and helper templates for both methods, and the
method-1 repair template. Preserve method-specific prompt assembly behavior,
including whitespace and the upstream baseline's source history/context rules.
For example, the existing baseline retains a final newline from a multiline
task instruction while method 1 strips it; a legacy replay must not silently
normalize this difference.

The configuration axes are independent:

```text
method          = cap_google_baseline | cap_method_1
condition_mode  = dsl_only | coordinate_predicates
prompt_version  = legacy_v1 | typed_v2
```

Changing `prompt_version` changes wording/examples, not the observation API,
allowed condition language, checker behavior, controller, or terminal evaluator.
Both prompt versions must describe the same runtime contract for a selected
condition mode. The baseline still uses prompt-only constraints and the pinned
upstream algorithm; adding version selection must not add a DSL checker or
repair loop to that method.

Resolve a version once per trial, reject unknown versions, and propagate it to
all policy/helper/repair requests. Avoid mixing legacy policy generation with a
new helper or repair template within the same trial. An immutable prompt package
could live under a versioned prompt-resource directory, with the existing
compatibility/generation layers selecting it; the exact storage layout remains
an implementation choice.

Record at least the following in configuration and generation artifacts:

- The selected `prompt_version`, `condition_mode`, method, and prompt-package
  content hash.
- The exact assembled prompt and its hash for every model request, including
  helper generation, repairs, and transport retries.
- References linking source checkpoints to their generating prompt/version and
  model-call ID; retain the separate checker/runtime provenance.

A prompt-package hash identifies the templates. An assembled-request hash also
captures the task/context, generated source, diagnostics, and configured values
inserted into a specific request. Neither hash substitutes for retaining the
actual request text. Existing run artifacts already retain prompt/request text;
versioning should build on that record rather than replace it.

Reproduction checks should compare legacy assembly byte-for-byte with archived
requests under the same task, mode, context, and source/diagnostic inputs. Include
both methods and all request types. Prompt changes should receive a new version
identifier rather than mutating the contents of `legacy_v1`.

For paper comparisons, pair versions on the same initial snapshots, models,
reasoning settings, condition modes, and generation/execution budgets. Report
interface errors, acceptance, repairs, calls/tokens, and physical outcomes per
version. Prompt length is itself a recorded cost; do not hide the extra tokens
introduced by richer descriptions. If later experiments isolate individual
prompt components, define explicit immutable variants and record their manifests
rather than editing a shared template between runs. The particular variants and
any additional Revision 2 work items remain for follow-up discussion.

### Discussion points within item 1

- How much geometric detail should the compact prompt include versus reference
  documentation, particularly the exact ON/ON_star/Scattered formulas?
- Should the prompt include both positive and negative API examples, or start
  with only positive examples plus explicit parameter-type rules?
- Should API text be rendered from shared structured metadata, including the
  actual Higher tolerance, rather than maintained as one free-form string?

These questions concern prompt design only. Item 2 below is the separately
requested batch-execution change; other revision topics remain for discussion.

## 2. Run future experiment batches in parallel

Requested requirement: future multi-trial experiments should run independent
trials concurrently instead of executing the whole batch serially. This is a
planned harness change, not a change to policy generation algorithms. The active
coordinate-predicate comparison keeps its existing schedule: model batches run
concurrently, while trials within each model batch remain sequential.

### 2.1 Scheduling and configuration

Provide a batch-level concurrency option, for example `--max-workers N`, with a
bounded parallel default and `N=1` available for a serial reference or diagnosis.
The exact default worker count should be resolved and recorded before a batch
starts, with regard to simulator/rendering resources and provider request limits.
Do not silently change model, prompt, trial count, or per-trial budgets to fit
more workers.

Create the complete trial manifest before launching requests. A trial identity
includes model, method, condition mode, prompt version, trial index, initial-state
reference/hash, and generation settings. Use a recorded scheduling order that
balances comparison groups. Completion order must not determine trial identity,
which trials enter the report, or which failed trials are replaced.

Parallelize **independent trials**. Do not parallelize or reorder Google's
recursive helper-generation algorithm inside a baseline trial, and do not
replace method 1's existing preparation/repair flow as part of this change.

### 2.2 Process isolation and output ownership

Use separate worker processes rather than sharing live simulator instances
between threads. Each worker owns its environment, object handles, random state,
model session, signal/deadline handling, and recorder. Full initial snapshots
may be shared as immutable files, but every trial restores its own environment.
Preserve the Stack settling/replay rule; loading an archived start must not add
another settling phase.

Assign each trial a distinct output directory before dispatch. Workers must not
write the same status, stdout, timeline, checkpoint, video, or latest-link path.
A parent aggregator owns the batch summary and publishes updates atomically.
Retain the existing per-trial RunLogger/report contract and immutable generated
source checkpoints. Credentials remain process-local environment inputs; do not
put secret values in the trial manifest or artifacts.

### 2.3 Budgets, failures, and reproducibility

Implementation update: the shared runner now has a 600-second wall-clock safety
cap (Unstack remains 60 seconds), an explicit policy simulation-step budget,
first-exhausted-budget reporting, exclusive phase timings, and separate queue
waiting metadata. The requested Batch C rerun uses a two-process artifact-backed
scheduler with a recorded randomized dispatch order. A general configurable
parallel batch CLI remains planned; the existing baseline algorithm and prompts
are preserved. See [resource budgets and timing](../README.md#resource-budgets-and-timing).


Keep the existing per-trial generation/execution budgets unchanged. Start a
trial's wall-clock deadline when its worker begins trial setup, not while the job
is waiting in the batch queue. Report queue delay and active trial time
separately. Unstack still has its 60-second end-to-end trial limit. Keep the same
declared treatment of provider rate limiting and transport retries across groups;
waiting/retrying inside an active trial consumes its existing deadline.

Concurrency is not permission for unbounded retries. A worker timeout, process
crash, primitive failure, generation failure, or logging failure must produce an
explicit outcome or incomplete-trial record. Do not drop it from the scheduled
trial accounting or silently start a replacement. Resume logic must distinguish
finished, interrupted, and never-started trials to avoid duplicate paid requests.

Record requested and actual worker counts, dispatch/start/end times, schedule,
process outcomes, retry counts, and model-call usage. Every trial report must
include `# of tokens used`: cumulative provider-reported input plus output tokens
across initial/helper/repair requests, with incomplete usage marked explicitly.
Reasoning tokens already included in completion usage must not be counted twice.

Scheduling isolation does not guarantee deterministic remote-model responses.
Compare methods/versions under the same declared concurrency and resource policy,
and disclose provider contention or rate-limit failures instead of attributing
them automatically to policy quality.

### 2.4 Validation before adopting the parallel harness

Use scripted model responses and identical snapshots to compare serial and
parallel scheduling without paid requests. Check that trial identities,
restored observations, primitive action sequences, verdicts, and source/video
associations remain correct. Check that logs and checkpoints do not cross trial
boundaries. Include a deliberately failing worker and an interrupted batch to
verify accounting, independent progress, and resume behavior.

After these checks, use the parallel harness for subsequent requested batches
and record its configuration in their reports. This plan does not itself implement a general batch CLI. The timing/budget
support and the isolated-process Batch C rerun described above are implemented;
future reusable scheduling work remains separate. The five-trial 600-second Batch C baseline rerun
completed at 3/5; the [budget/timing record](../README.md#resource-budgets-and-timing)
and linked run artifacts distinguish this pilot from the general scheduler plan.

## References

- [Current API text and mode rules](../contracts.py).
- [Runtime operand validation and queries](../adapter.py).
- [Primitive controller semantics](../../api/control.py).
- [Shared predicate definitions](DSL.md#5-tower-predicates-and-their-exact-definitions).
- [Method-1 mode configuration](Method1_Configuration.md).
- [Revision 1 implementation and evidence](revision_1.md).
