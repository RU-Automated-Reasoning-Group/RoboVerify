# RoboVerify DSL for the Six-Task CaP Comparison

This document describes the paper's language, the current upstream implementation,
and the interface boundaries needed to reconnect CaP.
The [CaP implementation](README.md) now connects the tower block-reference
interface to the original Google baseline and constrained method 1. Goal adapters
and six-task physical evaluation remain incomplete; the limits below still apply.

See [CAP-INTEGRATION.md](CAP-INTEGRATION.md) for the original Google CaP and
proposed RoboVerify pseudocode, highlighted changes, and environment interfaces.

Review date: 2026-10-14. Code baseline: upstream `main` at
`1f55b1b188bde1045b1d1771ed2ba7988d4e2e73`; current working branch: `feature/cap`.
The paper is the repository's [POPL2027.pdf](../../../POPL2027.pdf); all page
references below use PDF page numbers. Implementations and experimental results
from the old `cap-baseline` are not treated as current capabilities here.

The review covers the paper's DSL, synthesis/verification interfaces, six-task
experiments, and predicate/task/program appendices, together with the current
instructions, controllers, program execution/WP, predicate representation and
evaluation, CFG lowering, task factories, resets, success checks, and standalone
verification entry points. The source index is at the end. Actions and Python
features in third-party environment libraries are not automatically part of the
RoboVerify DSL.

## 1. Establish the Six Tasks

Paper §6 (p.36) explicitly lists **Stack, Unstack, Reverse, ReStack, Grid, and
Pyramid**. Insertion Sort, Tower Merge/Split, and other extensions from old
experiments are excluded.

| Paper task | Meaning | Current environment / factory | Current integration limits |
| --- | --- | --- | --- |
| Stack | Stack all scattered blocks on the input base block `b0` | `RoboVerifyStack` / `make_roboverify_stack_env` | Shared task specification, explicit primitive reference program, and standard collection entry point |
| Unstack | Dismantle a single tower and return all blocks to the table | `RoboVerifyUnstack` / `make_roboverify_unstack_env` | Shared symbolic task specification; physical table height must be retained; standard collection CLI does not yet support it |
| Reverse | Reverse the vertical order of all blocks in the original tower | `RoboVerifyReverse` / `make_roboverify_reverse_env` | Environment checks full reversal; the standalone verifier with this name currently uses a dismantling program/specification |
| ReStack | Start with potentially multiple towers, with `b0` possibly inside a tower; finish with all blocks stacked on base `b0` | `RoboVerifyPartialStack` / `make_roboverify_partial_stack_env` | Reset and final environment goal provide a basis for ReStack; the Partial standalone verifier does not implement this task |
| Grid | Place blocks at given grid goal positions | `RoboVerifyGrid` / `make_roboverify_grid_env` | Physical goal layout exists; no Goal DSL execution adapter connects it to the three primitives yet |
| Pyramid | Place blocks at given pyramid goal positions | `RoboVerifyPyramid` / `make_roboverify_pyramid_env` | Currently fixed at six blocks in a `3+2+1` layout; also lacks Goal execution adaptation |

Factories are in [mcmc/synthesis.py](../mcmc/synthesis.py).
`make_roboverify_env` accepts `stack/unstack/reverse/partial_stack/partial`;
Grid/Pyramid require their respective factories. There is currently no `restack`
string entry point. The grid factory defaults to `3×2`, with the block count equal
to rows times columns.

`PartialStack` is more general than adding scattered blocks to a single existing
tower. Its [reset implementation](../environment/cee_us_env/fpp_construction_env.py)
randomly partitions blocks into at least two towers and allows `b0` inside a
tower, providing a scene basis for the paper's ReStack task. An existing
environment does not establish complete integration of the corresponding
program, DSL execution, demonstration collection, and proof.

Reverse requires more than dismantling a tower into scattered blocks. The
environment requires the final tower order to reverse the initial canonical
order, making the original `b0` the top block. Paper §6's wording that `b0`
remains the base conflicts with Table 8's full-reversal formula. The evaluation
specification must be settled before running experiments; see
[discrepancy 40](../../../PAPER-DISCREPANCIES.md#40-six-task-specifications-in-table-8-contain-domain-and-binding-errors).

## 2. Language Layers in the Paper

The core language in Fig.2 and §2.2 (pp.4–5) can be summarized as:

```text
Type      ::= EE | block | other task-declared types
Predicate ::= R(objects) | R*(x,y) | x=y
            | not P | P and P | P or P
            | exists x:Type. P | forall x:Type. P
Action    ::= Pick(b) | Move(u,v,w,delta) | Release(b,h)
Statement ::= Action | S;S | x := get(exists x. P)
            | while (exists x. P) { S }
            | if P then S else S
```

Appendices B/C (pp.45–48) also use `Goal`, goal relations, object assignment,
goal-pointer updates, and input-specified `b0` and goal origin `h0`. `skip`,
assignment, and abstract `put` appear in Tables 1/7. The paper's DSL therefore
extends beyond `ON*` and three actions.

Three layers must be distinguished:

| Layer | Contents | Directly drives the physical environment? |
| --- | --- | --- |
| Primitive implementation | `Pick / Move / Release` | Yes, through shared controllers |
| Program structure and queries | Assignment, object selection, conditions, loops, predicates | Updates bindings or reads state; primitives inside loops/branches perform actions |
| Abstract verification | `Put`, goal marking and relation updates, WP, loop invariants | Requires a concrete implementation satisfying its contract before physical execution |

CaP may generate helper functions composed of primitives. Manually supplied
`stack_one`, `unstack_all`, or `Put` must not be treated as additional physical
capabilities. LLM-generated helper bodies must pass the same checks.

## 3. Object Types, Handles, and Special Objects

| Name | Meaning in the paper / code | Interface requirement |
| --- | --- | --- |
| `EE` / `mu` | End effector; actual state is managed by the controller | Do not expose `env.step` or arbitrary robot-state operations to generated programs |
| `block` / `BoxSort` | Physical block identity, corresponding to a physical ID at runtime | Pass opaque object handles; equality means identical objects, rather than equal coordinates |
| `b0` | Task input block | Must not be inferred again from color, ID order, or geometric extrema |
| `b`, `b_prime`, etc. | Aliases bound through assignment or guard/get selection | Must be bound before use; assignment changes an alias without moving an object |
| `tbl` | Special marker in the tower relation algebra | **Has no XYZ coordinates**; cannot serve as an ordinary Move reference |
| `Goal` / `GoalSort` | Goal-position node for Grid/Pyramid | Not a physical block; cannot be picked and does not participate in block collisions |
| `h0` | Starting node of the goal structure | Task input should provide a Goal handle |
| `null` / `NULL` | End marker for goal traversal | Has no geometry; used only for identity comparisons and the specified goal-relation semantics |

`HighLevelContext` uses `declare` mode for unbounded object reasoning and `enum`
for finite instances. Goal symbols are created only with
`verification_mode="goals"`.

Physical table height belongs to a separate layer from `tbl`:
`table_surface_height` is the table plane's world Z, while `height_offset` is the
resting block-center Z. They must not be interchanged. Placing a block on the
table requires a primitive sequence and geometric references supplied by the
host adapter; arbitrary coordinates must not be assigned to the relational
marker `tbl`.

Current tower environments create `env.symbolic_name_to_box_id`, initially
containing only `b0`, then extend it through assignment/selection. Grid/Pyramid
currently lack the same block-alias interface; Goal binding tables also need
adaptation.

## 4. The Three Reusable Physical Primitives

Implementation: [api/instructions.py](../api/instructions.py) and
[api/control.py](../api/control.py). ID and ByName variants resolve operands
first, then call the same controllers.

| Paper form | Current constructor interface | Exact runtime semantics | Trainable numeric parameters |
| --- | --- | --- | --- |
| `Pick(b)` | `Pick(grab_box_id)` / `PickByName(grab_box_name)` | Approach the block horizontally at the current height, open the gripper, descend, and close the gripper | None |
| `Move(u,v,w,delta)` | `Move(x_id,y_id,z_id,target_offset=[dx,dy,dz])` / `MoveByName(x_name,y_name,z_name,target_offset=...)` | Move to `(X(u)+dx, Y(v)+dy, Z(w)+dz)`; the three references may differ | All three XYZ offsets are registered as Parameter values |
| `Release(b,h)` | `Release(release_box_id,target_z=h)` / `ReleaseByName(release_box_name,target_z=h)` | Open the gripper to release, then retreat to a height defined by block `b`'s Z at invocation plus `h` | One Z offset |

`Move` computes its target from the current observation at each invocation,
rather than permanently using program-initial coordinates. `Release.target_z`
is a **block-relative gripper retreat-height offset**, rather than an absolute
world Z or a separate instruction to place the block at that height. The actual
controller freezes end-effector XY before opening and corrects XYZ during
retreat.

Defaults: a total budget of 50 steps per primitive, 2 mm position tolerance,
proportional gain 20, and uniform XYZ action scaling. Pick's stages share that
50-step budget. Controllers report convergence or budget exhaustion through
`last_control_result`. Convergence establishes that the control stopping
condition was met; it does not independently establish grasp or task success.

`ControlConfig` contains fixed controller settings, rather than CEM variables.
`Parameter(None)` is treated as `0.0` in ordinary runtime vectors, but formal
verification rejects unspecified values when concrete numbers are required.
The proposed CaP interface should require explicit, finite action parameters.
If CEM is used, optimize only registered action parameters, excluding predicate
thresholds, object identities, controller tolerances, and branch structure.

The paper's idealized action contracts and this feedback controller remain
separate layers. Motion proofs do not directly establish all contact, grasp,
arm-geometry, or dynamics behavior of the MuJoCo controller.

## 5. Tower Predicates and Their Exact Definitions

Reuse [predicates/term.py](../predicates/term.py),
[predicates/scene.py](../predicates/scene.py), and [util/on.py](../util/on.py).
Do not implement a separate set of approximate geometric checks in CaP. Block
side length is `L=0.05 m`.

| Name | Type | Current semantics | Notes |
| --- | --- | --- | --- |
| `ON_star(a,b)` / `on*` | block × block → Bool | Physical blocks: `abs(dx)<0.025 and abs(dy)<0.025 and za>=zb` | Reflexive; interpreted geometrically rather than recomputing a support-graph closure each time |
| `ON(a,b)` / relational `on` | block × block → Bool | `a!=b and ON_star(a,b)`, with no intermediate block `c` distinct from `a,b` satisfying `ON_star(a,c) and ON_star(c,b)` | Direct relation in `Term`, distinct from the geometric height-band function below |
| `Higher(a,b)` | block × block → Bool | `za >= zb - tolerance`, default `tolerance=0.001 m` | Non-strict height comparison |
| `Scattered(a,b)` | block × block → Bool | `abs(dx)>=0.10 or abs(dy)>=0.10` | Symmetric and irreflexive; does not test Euclidean distance |
| `ON_star_zero(a,b)` | block × block → Bool | ON* interpretation applied to frozen entry geometry | Historical relation needed by Reverse; temporal scope is described below |
| `eq(a,b)` / `a==b` | Same-type handle × same-type handle → Bool | Object identity equality | `a!=b` is its negation |

### 5.1 Keep the Three Meanings of on Separate

1. **Relational `ON`**: defined through ON* and the absence of intermediate blocks;
   used by predicate search and guards.
2. **Geometric `util.on.on()` / `z3_on()` / `lowlevel_on_direct()`**:
   `abs(dx)<0.025 and abs(dy)<0.025 and 0<=dz<0.075`.
   This function can hold for a physical block paired with itself; it does not
   include relational ON's `a!=b` requirement. Environment tower-success checks
   and some motion checks use this geometric height band.
3. **Paper Fig.12 (p.45)**: states `0.05<=dz<=0.075`.
   These boundaries differ from the current geometric function, and §2.2 uses
   another tolerance expression.

If CaP exposes `on`, it should explicitly mean relational `ON`, with its
semantics recorded in experiment configuration. Do not directly connect the
geometric function with the same name. See
[entry 14](../../../PAPER-DISCREPANCIES.md#14-afr-and-predicate-definitions-are-inconsistent-across-the-paper)
for the existing discrepancy.

`HighLevelContext` has no separate uninterpreted `ON` function;
`to_z3(ON(...))` expands to the ON* formula above. The older Z3 guard evaluator
also does not directly accept calls to a function named `ON`; use an expanded
formula or the `Term` evaluation path.

### 5.2 tbl and Predicate Domains

When a scene includes `tbl`, `ON_star(tbl,tbl)` and `Higher(tbl,tbl)` are true.
Both relations are false between `tbl` and physical blocks. Any Scattered
relation involving `tbl` is false. Consequently, `ON_star(block,tbl)` does not
mean that a block is on the table.

`Scene` quantifiers range over the full `positions` domain; inclusion of `tbl`
depends on scene configuration. If the paper's `forall b:block` is intended to
range only over physical blocks, `tbl` must be explicitly excluded. For example,
pairwise Scattered specifications must exclude both identical objects and
nonphysical markers.

### 5.3 Higher and Frozen Relations

Higher tolerance must match across execution, inference, geometric verification,
and counterexample realization. Use `using_higher_tolerance(value)` for a shared
scope; valid values are finite and in `[0,0.025)` meters. Tolerant comparisons over
arbitrary continuous heights need not be transitive; abstract height axioms still
require a suitable geometric domain.

`Scene.entry_positions` determines `ON_star_zero`. `While.eval` saves
`_guard_entry_positions` at **each loop invocation's entry**, and DemoStore also
retains that invocation's entry. A standalone `Scene` defaults to current
positions when no entry is supplied. Ordinary `Get` does not automatically save
the whole program's initial scene. The name `zero` alone therefore does not
ensure that every call reads the entire task's state zero.

A multistage Reverse adapter must explicitly retain the original tower's
snapshot/historical relations at task scope, rather than freezing the dismantled
scene again for stage two. Symbolic specifications equate
`ON_star_zero == ON_star` only at task entry; a global axiom must not keep them
permanently equal. See
[entry 5](../../../PAPER-DISCREPANCIES.md#5-the-geometric-translator-conflated-initial-and-current-on_star).

## 6. Conditions, Object Selection, and Control Flow

| Construct | Current code | Capabilities / limits |
| --- | --- | --- |
| Sequential execution | `Program.instructions`; `Seq(s1,s2)` for WP | Physical Program executes instructions in order; `Seq` itself has no physical `eval` |
| `x := y` | `Assign(left,right)` | Copies a bound object identity; an unknown RHS raises an error |
| `x := get(exists x. P)` | `Get(var,cond,exists_vars=None,guard_term=None)` | Finds and binds a witness satisfying P; raises `NoGuardWitness` if none exists |
| `while exists vars. P` | `While(instantiated_cond,guard_exists_vars,body,invariant,max_iters=10)` | Queries and binds again each iteration; exits when no witness exists; raises if the budget is exhausted while a witness remains |
| While without new witnesses | `guard_exists_vars=[]` | Constructor permits a pure condition; symbolic Goal examples use this form, which does not establish physical Goal execution |
| `if/else` | Paper Fig.2/Table 1; currently no `If` instruction class | `Program.wp` and current CFG lowering also lack branch-statement support |
| Nested loops | `While` can recurse at runtime | Standard `prepare_program` explicitly rejects nesting; the integrated tower workflow supports only flat loops |

Runtime selects the first satisfying witness, enumerating physical blocks by
ascending ID. Multiple witnesses are allowed. Formal preservation must hold for
every satisfying choice; a uniqueness requirement must not be added.
Counterexample replay can reproduce legal solver-selected choices, without
changing ordinary runtime selection.

The Get WP in paper Table 1 states correctness over all witnesses. The code also
requires witness existence:

```text
exists vars. P  and  forall vars. (P implies Q)
```

This is a settled decision; see
[entry 9](../../../PAPER-DISCREPANCIES.md#9-executable-get-needs-a-witness-existence-obligation).
While's execution limit is a resource budget, rather than a termination proof;
a timeout cannot count as success.

Condition expressions may include `true/false`, identity equality/inequality,
`not/and/or`, implication, and `exists/forall`. Boolean equivalence can be expressed
through implications. Runtime quantifiers use a finite object domain; the
symbolic layer uses Z3 quantifiers. The current `Term` atom registry accepts only
`ON/ON_star/ON_star_zero/Higher/Scattered/eq`. Default `Language` search relations
include `ON/ON_star/Higher/Scattered/eq`, excluding frozen ON* by default.
Invariant-learning and guard vocabularies are configured separately. The absence
of ON from a default invariant vocabulary does not exclude it from the DSL.

## 7. Goal Support Needed for Grid/Pyramid

Paper Appendix B (p.45) and Figs.17/18 (pp.47–48) require the concepts below.
They belong to the six-task DSL scope, but currently have only partial symbolic
implementations.

| Paper construct | Counterpart in the current symbolic model | Status |
| --- | --- | --- |
| `Goal`, `h0`, `NULL` | `GoalSort`, Goal constants, `context.null` | Symbolic types exist; runtime Goal handles resolvable by the three primitives do not |
| `next_c / next_r` | `d_star / r_star` and `dtot/rtot` | Pointer steps use closure and immediate-successor formulas; this is not a direct substitution of names |
| `reach(h0,g)` | `_dr_reach(h0,g) = d_star(h0,l0(g))` | Composite relation in the symbolic Goal model |
| Row start | `l0(g)` | Goal → Goal; not a function selecting a block with maximum X |
| `HasBlock(g)` | `Mark(g)` is closest to its abstract bookkeeping role | Mark is a logical flag, rather than checked physical occupancy |
| `AtGoal(b)` | Absent from current shared `Term`/guards | Used by paper programs; definition, tolerance, and object/goal association need alignment |
| `d := h0`, `r := d` | `GoalAssign` | Symbolic assignment only; `eval` raises |
| `r := r.next_r`, `d := d.next_c` | `MoveRight`, `MoveDown` | Goal-pointer steps for verification only; do not move the arm right/down |
| Abstract update after placement | `MarkGoal` | Rewrites Mark only; does not execute Pick/Move/Release |

[verify_2d_with_learned_invariant.py](../entry/verify_2d_with_learned_invariant.py)
constructs a nested symbolic program that traverses and marks goals. It does not
connect block selection, transport, and actual goal occupancy in a physical
implementation. Executing or verifying Mark-all alone cannot establish physical
completion of Grid/Pyramid.

Existing `d_star_implementation` / `r_star_implementation` are coordinate
interpretation helpers. The former requires equal X/Z and monotone Y; the latter
requires equal Y/Z and monotone X, using exact coordinate equality. They are
absent from the shared tower `Term` atom registry and the `evaluate_z3` predicate
allowlist. Their existence does not establish physical execution of Goal
conditions.

The three primitives currently resolve reference operands only to physical block
IDs. Supporting the paper's `Move(r,r,r,delta)` requires a trusted adapter to
resolve **Goal reference coordinates** while retaining the same Move semantics.
Exposing raw `desired_goal` arrays to the LLM or adding whole-task actions must
not bypass the interface. Release also requires explicit Goal-reference
adaptation; the current controller cannot accept a Goal as though it were a
block.

There is also an evaluation distinction: the paper's `HasBlock(g)` describes
some block occupying a goal, while the current environment measures each block's
distance from its assigned goal, retaining block-to-goal correspondence.
Before implementing occupancy predicates, settle whether the designated block
is required; do not silently switch to arbitrary permutation matching.

## 8. Existing Constructs That Should Not Be Directly Exposed to New CaP

| Construct / data | Current use | Reason |
| --- | --- | --- |
| `PickPlace / PickPlaceByName` | Historical composite controllers | Outside this shared three-primitive interface; standard current program-source checks also reject them |
| `Put(a,b)` / `Put(a,tbl)` | Abstract placement and symbolic WP | `eval` explicitly raises; cannot serve as a single physical action |
| `GoalAssign/MarkGoal/MoveRight/MoveDown` | Goal verification instructions | No physical implementation; adaptation must be added consistently for both methods |
| `Skip(skip_steps=20)` | Repeated observation recording | Does not call `env.step`; cannot mean waiting for 20 physical steps |
| `Top/top` | Shorthand in some paper examples | Removed from the current vocabulary; expand into formulas over existing predicates |
| Dynamic `X_m/Z_m` | Maximum-X/Z object selectors in paper Appendix C | Current primitives resolve ordinary IDs/names only; reviewed shared interfaces have no automatic extrema selectors |
| Raw positions, distances, images, reward/success, controller feedback | Data needed by controllers or host evaluation | Must not provide extra conditions or object-ordering criteria to generated programs |
| Arbitrary `env/sim` objects or library imports | Underlying implementation | Would bypass the experiment's shared primitives/predicates constraint |

Dynamic maximum-coordinate references on paper pp.45–47 also need explicit
handling. The current `_resolve` cannot be assumed to interpret `Z_m`
automatically. If such selectors are permitted later, they must formally become
part of the DSL shared by both methods, distinguishing action references from
condition evaluation. They are not current CaP capabilities.

A clear-top condition does not require restoring `Top`:

```text
forall n:block. (n = candidate or not ON_star(n,candidate))
```

This may be a generated query helper, provided it transparently expands to
allowed predicates and uses the correct object domain.

## 9. Proposed New CaP Boundary (Not Yet Implemented)

This section translates the user's requirements into checkable interface
proposals, preserving the existing semantics above.

| Category | Proposed allowed scope |
| --- | --- |
| Actions | Only `Pick/Move/Release`, with parameters and operands retaining DSL meaning |
| Object operations | Input handles, object-identity assignment, and Get/loop-witness binding defined through relational conditions |
| Tower conditions | `ON/ON_star/Higher/Scattered/eq`; Reverse may use `ON_star_zero` with explicit scope |
| Goal conditions | Goal relations, occupancy/AtGoal, and Goal identity/NULL comparisons after definitions are aligned; mark pending adaptation explicitly |
| Logic | Boolean constants, Boolean combinations, and typed quantifiers |
| Program structure | Sequences, predicate-based if/while, and primitive helper functions; If and nested Goal workflows require runtime support |
| Numbers | Explicit action parameters only; no additional environment observations or algorithmic branch conditions |

The requirement that conditions use only DSL predicates must cover the entire
generated call graph, including:

- Sources of conditions in `if/elif/while`, conditional expressions, and
  `break/continue`.
- List filtering, sort keys, and object selection, without hidden `position.z`,
  distances, ID ordering, or numeric comparisons.
- Helper return values and intermediate variables, without first computing an
  illegal numeric condition and then writing `if result`.
- All model-generated helpers, so their bodies cannot bypass restrictions that
  the top level satisfies.

The host may enumerate objects for quantifiers and enforce loop, primitive, and
time budgets. This does not grant generated programs arbitrary `for range(...)`,
count-based branches, set algorithms, or environment-data access. Even a function
returning bool must be traceable to a legal DSL condition expression. Retain a
typed condition AST, interpret it through existing Scene/Term semantics, and add
Goal support separately.

Numeric restrictions on conditions and action-parameter optimization are
separate concerns. Allowing CEM to adjust Move offsets does not permit optimizing
`if distance<theta`. Enforce restrictions before and during generated-code
execution, rather than relying only on the prompt.

## 10. The Reusable Stack Example

Existing [examples/stack.py](../examples/stack.py) demonstrates a physical program
using the three-primitive interface. The following is a readable structural
version, **not a new CaP API or complete runnable Python**:

```text
b := b0
while exists selected:block.
    selected != b
    and forall n:block. (n = selected or not ON_star(n,selected)):
    Pick(selected)
    Move(selected,selected,b,   [0,0,0.10])
    Move(b0,b0,b,               [0,0,0.10])
    Move(b0,b0,b,               [0,0,0.05])
    Release(selected,0.15)
    b := selected
```

XY references the fixed base block `b0`; Z references the current top `b`.
CaP wrappers must preserve that distinction rather than collapsing all calls to
`Move(target,target,target,...)`. These constants come from the existing reference
program; they do not require putting this complete solution into benchmark
prompts.

A loop with `invariant=None` can execute physically. Symbolic WP/verification
requires a separately supplied inferred invariant. Primitive call traces, loop
heads, normal exits, and binding events can reuse `api/runtime.py`, DemoStore,
and the current recording format. Evaluation/inference data must retain normal
exits and zero-iteration executions.

## 11. Gaps to Address Before Integrating All Six Tasks

The following table lists current gaps, rather than completed implementation:

| Work item | Reusable foundation | Missing work |
| --- | --- | --- |
| Primitive consistency | Shared controllers and Parameter | Restricted CaP call adaptation; host records control results without exposing feedback as extra model guards |
| Condition consistency | Term, Scene, Z3 guard evaluation | Strict type/provenance checks; Goal conditions; If execution semantics |
| Object binding | Assign/Get/While | Goal and null bindings; program-level frozen history; consistent witness selection |
| Task specifications | Stack/Unstack in `cfg/tasks.py`; six environment classes | Correct Reverse/ReStack specifications/reference programs; occupancy and identity conventions for goal tasks |
| Goal transport | Grid/Pyramid goal coordinates | Trusted Goal-reference resolution through the same three primitives, without coordinate queries |
| Demonstrations and initialization | Full-state NPZ, Stack's 50-step preparation and restoration | Valid collection/preparation rules for the other five tasks; unverified historical fixtures must not serve as experts |
| Evaluation | Environment success functions, run logger, recording | Fixed snapshots, independent outcome checks, failure classification, and budget records |

Both `verify_reverse_with_learned_invariant.py` and
`verify_partial_with_learned_invariant.py` currently construct a dismantling loop
with `Put(selected,tbl)` and require no nontrivial ON*, equal physical block
heights, and Scattered at the end. This does not verify the paper's
Reverse/ReStack tasks. These existing implementation gaps are recorded in
[discrepancy 39](../../../PAPER-DISCREPANCIES.md#39-six-task-dsl-coverage-does-not-match-the-named-verification-entry-points);
this review neither fixes nor runs those entry points.

The first evaluation should report physical successes/total trials for all six
tasks, separately recording generation failure, illegal DSL, primitive
nonconvergence, timeout, and final-goal failure. Successful execution, finite-scene
checks, unbounded symbolic proofs, and motion-model proofs must remain distinct.
Unstack end-to-end runs retain the 60-second limit. This interface review produces
no new experimental results and does not reuse numbers from the old nine-task
experiments.

## 12. Reading and Implementation Index

| Source | Material used here |
| --- | --- |
| [Paper Fig.2 / §2.2](../../../POPL2027.pdf#page=4), pp.4–5 | Types, predicates, three primitives, selection, loops, and branches |
| [Paper §3–5](../../../POPL2027.pdf#page=11), pp.11–35 | Guard/structure recovery, abstract/physical layers, WP, and verification scope |
| [Paper §6](../../../POPL2027.pdf#page=36), pp.36–38 | Six tasks, demonstrations, and experiment metrics |
| [Paper Appendix A](../../../POPL2027.pdf#page=43), pp.43–44 | Predicate axioms and Put WP |
| [Paper Appendices B/C](../../../POPL2027.pdf#page=45), pp.45–48 | Task specifications, Goal, historical relations, goal traversal, and dynamic coordinate references |
| [api/instructions.py](../api/instructions.py) | All instruction classes, parameters, loops/Get/aliases, and verification-only instructions |
| [api/control.py](../api/control.py) | Actual control steps and budgets of the three primitives |
| [api/program.py](../api/program.py) | Program execution, parameter registration, WP/VC dispatch, and abstract updates |
| [api/guard_eval.py](../api/guard_eval.py), [api/runtime.py](../api/runtime.py) | Guard evaluation/binding/replay and instruction events |
| [predicates/term.py](../predicates/term.py), [scene.py](../predicates/scene.py), [language.py](../predicates/language.py) | Condition AST, predicate interpretations, and default vocabulary |
| [predicates/guard.py](../predicates/guard.py), [classifier.py](../predicates/classifier.py), [enumerate.py](../predicates/enumerate.py) | Guard/classifier boundaries and quantified search |
| [util/on.py](../util/on.py), [lowlevel_verification_lib.py](../verification_lib/lowlevel_verification_lib.py) | Numeric and geometric Z3 predicates |
| [highlevel_verification_lib.py](../verification_lib/highlevel_verification_lib.py) | Box/Goal types, relation axioms, and goal successors |
| [primitive_motion.py](../verification_lib/primitive_motion.py), [motion documentation](../verification_lib/README.md) | Distinction between idealized primitive models and actual controllers |
| [cfg/tasks.py](../cfg/tasks.py), [lower.py](../cfg/lower.py), [region.py](../cfg/region.py) | Shared task specifications, sequence/loop lowering, and current structure |
| [cfg/program_source.py](../cfg/program_source.py), [program_adapter.py](../cfg/program_adapter.py) | Source checks, named operands, and supported program scope |
| [inference_lib/demo_store.py](../inference_lib/demo_store.py), [inference.py](../inference_lib/inference.py) | Frozen entry, vocabulary configuration, and Goal inference interpretation helpers |
| [mcmc/synthesis.py](../mcmc/synthesis.py) | Six environment factories |
| [fpp_construction_env.py](../environment/cee_us_env/fpp_construction_env.py) | Reset, goal positions, initial aliases, and six-task success checks |
| [examples/stack.py](../examples/stack.py), [collect_demos.py](../entry/collect_demos.py) | Explicit primitive example and current Stack collection scope |
| [Unstack entry point](../entry/verify_unstack_with_learned_invariant.py), [Reverse entry point](../entry/verify_reverse_with_learned_invariant.py), [Partial entry point](../entry/verify_partial_with_learned_invariant.py), [2D entry point](../entry/verify_2d_with_learned_invariant.py) | Actual standalone verification programs and capability gaps |
| [PAPER-DISCREPANCIES.md](../../../PAPER-DISCREPANCIES.md) | Settled semantics, paper corrections, implementation gaps, and model limits |
