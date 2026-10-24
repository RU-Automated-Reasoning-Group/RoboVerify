# Benchmark preconditions and postconditions

All seven benchmarks have symbolic task conditions. Stack, Unstack, Reverse and
Restack already had shared block-predicate formulas in [tasks.py](tasks.py).
Grid, Pyramid and Build3D share the new occupancy contract in
[goal_specs.py](goal_specs.py). Layout shape and traversal are environment/program
properties, not requirements in these pre/postconditions.

These assertions specify the desired behavior of a synthesized program. They do
not provide loop invariants, action transformers, termination arguments or motion
proofs. The [integrated support matrix](../../../README.md#project-status) still
applies: Restack and the three goal tasks do not yet have integrated verification.

## Notation and domain

- B is the set of physical blocks. In contexts containing `tbl`, B excludes it.
  Stack's existing context contains only physical blocks.
- G is the set of goal locations, excluding the separate Goal-sort `null`.
- b0 denotes a designated physical block. For Unstack and Reverse, its physical
  binding comes from the program/pipeline; standalone callers must also assert
  `b0 != tbl`. Restack includes that assertion in its precondition.
- O(x,y) denotes `ON_star(x,y)`: x is on or above y, including x=y.
  O0 is `ON_star_zero`, frozen at program entry.
- H(x,y) denotes `Higher(x,y)`; S(x,y) denotes `Scattered(x,y)`.
  H uses the configured tolerance, 1 mm by default. S is irreflexive, so separation
  conditions apply only to distinct physical blocks.

The formulas below are interpreted under the shared block-domain axioms. They
are relational specifications; workspace bounds, gripper clearance, support and
collision conditions belong to the motion model.

## Tower benchmarks

| Benchmark | Precondition | Postcondition |
| --- | --- | --- |
| Stack | For every distinct x,y in B: not O(x,y) and S(x,y). For every x,y in B: H(x,y). | For every x in B: O(x,b0). |
| Unstack | For every x in B: O(x,b0). | For every x,y in B: O(x,y) implies x=y. |
| Reverse | For every x in B: O(x,b0). At entry, O0(x,y) iff O(x,y), over the complete block-sort domain. | For every x,y in B: O0(x,y) iff O(y,x). |
| Restack | b0 is physical. For every x,y in B: O(x,y) or O(y,x) or S(x,y). | For every x in B: O(x,b0). |

Stack starts with separated blocks at one height level and finishes with every
block in a tower rooted at b0. Its height condition is entry-only.

Unstack starts with one tower rooted at b0 and finishes with no distinct blocks
related by ON*. Its existing postcondition does **not** additionally demand equal
heights or Scattered separation. Those stronger conditions should not be inferred
from this contract.

Reverse must invert every pair's entry order. An unchanged tower, an unstacked
scene or a partially reversed tower fails the postcondition for a nontrivial
input. O0 remains frozen throughout execution; its equality with current O is
an entry assertion, never a global invariant.

Restack permits separated columns, including a tower with b0 at an interior
level and other loose blocks. Its precondition is deliberately more general than
the current reset sampler. It finishes with b0 at the base of the complete tower;
there is no requirement to retain the other blocks' original order.

## Grid, Pyramid and Build3D

All three use the same contract over the supplied goal locations:

```text
Pre:
    G is nonempty
    every goal is initially empty
    every pair of distinct physical blocks is Scattered
    there is one available physical block per goal (|B| = |G|)

Post:
    every goal is occupied by some physical block
```

Block identities are interchangeable. There is no requirement that block i fill
goal i, and no row, layer, shape or traversal condition is added to this contract.
The count equality matches the current benchmark factories and collection
validator; it is stronger than the minimum sufficient-resource condition
`|B| >= |G|` for a variant that permits surplus blocks.

The symbolic vocabulary connects occupancy to the runtime guard predicates:

```text
Occupies(b,g)  means ||position(b) - position(g)|| <= 0.02 m
HasBlock(g)   iff exists b in B: Occupies(b,g)
AtGoal(b)     iff exists g in G: Occupies(b,g)
```

`GoalTaskContext.occupancy_axioms()` excludes `tbl` and `null` from occupancy and
requires that a block occupy at most one goal. The latter follows from the
existing disjoint goal regions. It does not assume every goal is already filled.
With equal finite block/goal counts, final coverage then also implies every block
is used exactly once. Goal coordinates and the object domains stay fixed through
execution; Occupies, AtGoal and HasBlock describe the current state.

The count premise is expressed without choosing a fixed size: two ghost functions
`supply: Goal -> Box` and `slot: Box -> Goal` form a bijection between G and B:

```text
forall g in G: supply(g) in B and slot(supply(g)) = g
forall b in B: slot(b) in G and supply(slot(b)) = b
```

These functions witness resource availability. They neither select runtime
blocks nor constrain the final block-to-goal assignment. They are frozen ghost
symbols. The new assertions impose no finite domain closure; normal benchmark
interpretations are finite. This does not establish termination for arbitrary
first-order models.

## Python API

`symbolic_task_spec(task, context)` in [tasks.py](tasks.py) returns a pair of Z3
formulas for every benchmark. The existing `task_spec(task)` remains the
block-only Term API used by collection and the integrated tower pipeline.

```python
from synthesis.cfg.tasks import symbolic_task_spec
from synthesis.cfg.goal_specs import GoalTaskContext
from synthesis.verification_lib.highlevel_verification_lib import HighLevelContext

# Existing block-only task conditions.
blocks = HighLevelContext(use_tbl=True)
restack_pre, restack_post = symbolic_task_spec("restack", blocks)

# Mixed physical-block/goal conditions; also accepts "pyramid" or "build3d".
goals = GoalTaskContext()
pre, post = symbolic_task_spec("grid", goals)
solver = goals.new_solver()  # Shared block axioms plus occupancy axioms.
solver.add(pre)
```

For a future program proof, check premise consistency separately and establish
`pre => wp(program, post)` with sound action rules and learned loop invariants.
An occupancy-domain assertion after an action must be justified by its symbolic
contract and physical refinement, not simply assumed after an arbitrary move.
Goal-aware inference, goal placement/frame rules and nested-loop integration
remain work for that verification stage. The historical goal-only `Mark` model
is not used as a substitute for physical occupancy.

The existing archive identity and numeric collection conditions are unchanged.
Focused tests check nonvacuity, frozen reversal, the Restack base, sentinel
exclusion, missing goals, permuted occupancy, unseparated inputs and resource
shortages/surpluses using synthetic scenes:

```bash
uv run python -m unittest synthesis.cfg.test_tasks synthesis.cfg.test_goal_specs -v
```
