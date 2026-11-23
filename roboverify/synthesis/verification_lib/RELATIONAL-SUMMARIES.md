# Height and Scattered action summaries

`RelationalSummary` supports an action whose inferred endpoint is
`Higher(x,y)`, `EqualHeight(x,y)` or `Scattered(x,y)`. It is a verification contract for a completed
Pick/Move/Release transfer. Its WP is deliberately conservative: an endpoint
atom leaves most of the moved block's relationships unspecified. The existing
`Put` operation and its predicate rewrites are unchanged.

For example, `Higher(a,b)` permits separate columns and equal heights. It does
not establish `ON_star(a,b)` or `Not(Higher(b,a))`. `Scattered(a,b)` leaves both
height directions open and says nothing about `Scattered(a,c)`. Even relations
between unrelated objects need an explicit frame; they do not follow from the
endpoint atom alone.

## Contract and weakest precondition

Let `m` be the manipulated physical object, `E` the selected endpoint atom, and
`A` the explicitly chosen successor-domain predicate. The mutable relation
state is `(ON_star, Higher, Scattered)` by default. With the
[EqualHeight algebra](../cfg/VERIFICATION.md#equal-height-alternative), replace
Higher by EqualHeight throughout this contract; successor axioms and havoc use
that selected relation. An endpoint from the inactive height vocabulary is
rejected. Equality and frozen `ON_star_zero`
retain their interpretations. Define an admissible transition independently:

```text
Step(s,s') = m is physical
          ∧ A(s')
          ∧ E(s')
          ∧ forall R in {ON_star,Higher,Scattered}, x,y:
              x != m and y != m  =>  R(s',x,y) = R(s,x,y).
```

Every pair incident to `m` may change, including the reverse orientation.
There is no hidden assumption that `Higher` changes only height, or that
`Scattered` changes only XY. A physical placement must separately justify
the frame and successor domain. This contract may admit successors that no
robot can realize; consequently its WP can be stronger than a particular
controller's geometric WP.

For an arbitrary candidate relation valuation `T`, form `patch(s,m,T)`:

| Assertion atom | Rewritten interpretation |
| --- | --- |
| `R(x,y)`, mutable `R`, with `x=m` or `y=m` | `T_R(x,y)` |
| `R(x,y)`, mutable `R`, with both unequal to `m` | `R(s,x,y)` |
| `ON_star_zero(x,y)` | Same frozen entry atom |
| `x=y` | Same object equality |

Apply these rules simultaneously and structurally under truth, falsity,
negation, conjunction, disjunction, universal and existential quantification.
Call this transformation `rewrite(Q,T)`. Implication is Boolean sugar.
With `C(T) = A(patch(s,m,T)) and E(patch(s,m,T))`, the WP is:

```text
WP(summary,Q)(s) = (exists T. C(T))
                and (forall T. C(T) => rewrite(Q,T)).
```

The physical-source guard is also required. Existence prevents an impossible
action, such as `Scattered(a,a)` under irreflexivity, from proving every goal.
Universal quantification makes the choice demonic: every allowed successor
must satisfy `Q`. Thus one helpful placement cannot certify the action.
The whole WP does not distribute through negation; negation commutes only
with the structural rewrite for a fixed candidate interpretation.

The Python implementation uses an equivalent full-successor encoding:
quantify all three successor relations and explicitly constrain the frame.
The Lean `patch_of_frame` lemma proves that every framed successor has a
patch representation. Neither representation bounds the number of blocks
in the semantic rule.

The integrated default `A` is exactly the current-state axioms collected from
`HighLevelContext.add_axiom`, `add_axiom_height` and `add_axiom_scattered`.
Frozen-entry axioms need no new copy. If `use_tbl` is enabled, the source
cannot be `tbl`, and the successor axioms retain table isolation. The direct
`summary_wp(..., admissibility=...)` API can select a different domain; that
changes the contract and requires a matching refinement proof. CFG inference
and motion verification use the default domain together.

## Python and CFG integration

```python
from synthesis.api.relational_summary import RelationalSummary
from synthesis.api.program import wp

# Endpoint order is independent of which operand is moved.
action = RelationalSummary("Higher", "support", "payload", moved="payload")
precondition = wp(action, goal, context)
```

`propose_summaries` recognizes a single `Higher`, `EqualHeight` or `Scattered` edge atom with
named operands, one of which is the held object, at Release. Split transfers
across adjacent CFG blocks remain supported. The moved object is recorded
separately from operand order. Unsupported atom shapes or atoms unrelated to
the held object fail explicitly. Global/conjunctive edge labels retain the
existing transport-target `Put` fallback and its full motion obligations.
An unfinished transfer is never accepted as a completed summary.

`Program.wp`, `Seq`, assignments and CFG lowering consume the new instruction.
Aliases used by motion refinement are frozen at transfer entry, so an
assignment after Release cannot change the endpoint obligation retrospectively.
Symbolic histories include the summary; a later Put cannot re-assume input
alignment merely because no earlier Put appears in that history.

`verify_cfg_motion` retains collision, support, release and consistency checks,
then proves these additional obligations:

- `summary_endpoint`: the physical source is released and the ordered atom holds.
- `summary_frame`: every other object's coordinates remain unchanged, including
  unnamed objects. This is stronger than the required nonmover relation frame.
- `summary_domain_*`: each current-state domain axiom holds at the successor.

The successor axioms are **conclusions**, never endpoint assumptions. Exact
quantified entry premises are reasserted only from the supplied initial
conditions, with a separate consistency check. A true endpoint may still fail:
placing `a` 0.024 m horizontally off `b` can preserve `Higher(a,b)` and direct
support yet break Scattered propagation to a third block 0.11 m from `b`.
The integration regression detects this as `summary_domain_scattered3`.

This adds a reusable verification contract and inference path. It adds no new
controller skill, environment verification result, or goal-program proof support.

## Solver support and limitations

For a finite `EnumSort` context, Python quantifies `3*n*n` Boolean relation
cells and expands object quantifiers exactly over the exhaustive constructors.
This is an exact finite encoding, including nested/shadowed and mixed-sort
binders; it is not sampling. Boolean and array binders are retained with fresh,
capture-free variables. Ground table selections are simplified before solving.
Finite API and CFG tests exercise two- and three-object instances, including
two sequential summaries and subsequent alias assignment.

Unbounded `declare` contexts quantify three Boolean relation arrays. This is
outside the paper's AFR fragment and Z3 may return `unknown` due to quantified
array incompleteness. Nested summaries and larger finite instances can also
be expensive. No completeness or timing guarantee is claimed. Existing
verification treats `unknown` and timeout as unsuccessful; finite success is
not relabeled an unbounded proof.

## Checked theorem scope

[RelationalSummary.lean](../../proofs/wp_soundness/RoboVerify/RelationalSummary.lean)
proves, for an arbitrary physical-object type and arbitrary explicit successor
admissibility predicate:

- `rewrite_correct`: the full assertion rewrite equals evaluation in the patched state.
- `wp_exact` and `wp_weakest`: exact nonvacuous demonic WP for the framed relational contract.
- `no_successor_rejects` and `one_bad_successor_rejects`: neither empty successor
  sets nor a favorable choice can manufacture correctness.
- `implementation_refines`: an implementation satisfying endpoint, frame and
  domain obligations inherits the postcondition. These are explicit hypotheses.
- `sequential_sound` and `sequential_no_fault`: continuation composition preserves
  correctness and enables the second action for every first successor.

Kernel-checked counterexamples show that Higher need not mean ON* or strict
height, and Scattered permits equal heights, either height order, and lack of
scattering from a third object. The table is an explicit `none` object;
the mover has the physical-object type, so it cannot be the table. Assertions
quantify over physical objects and the table, with frozen ON* unchanged.

These are exact theorems about the **stated relational contract**, not an exact
geometric/controller WP. The new module is separate from the existing complete
tower relocation theorem and the existing `Command` language's program proof.
Its generic composition lemmas do not claim a new mechanization of Python's
alias store or loop implementation. There is no checked Python-to-Lean
translation or kernel proof of the current geometric axiom checker.

Build the proof and axiom audit with `lake build` in
`roboverify/proofs/wp_soundness/`; see its [README](../../proofs/wp_soundness/README.md).
Run focused Python regressions from `roboverify/` with the repository's simulator
environment configured:

```bash
uv run python -m unittest synthesis.api.test_relational_summary synthesis.cfg.test_relational_summaries -v
```
