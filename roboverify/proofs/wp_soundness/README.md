# Mechanized WP soundness

This Lean 4.34.1 development formalizes the relational part of the
[WP soundness draft](../../synthesis/verification_lib/WP-SOUNDNESS-DRAFT.md).
It proves exact predicate rewrites, exact guarded primitive WPs, and conditional
program partial correctness. Public `Put(a,b)` requires distinct physical
blocks with **both operands clear before the operation**; it removes the
source's previous support.

The result concerns the complete, separated tower model defined here.
Correspondence with the production Python verifier and physical controllers
remains a separate proof obligation.

## Build and audit

From the repository root:

```bash
cd roboverify/proofs/wp_soundness
lake build
```

The pinned toolchain uses only libraries shipped with Lean; there are no
mathlib or other package dependencies. MuJoCo, Python, and Z3 are unnecessary.
The default build includes all examples and the axiom audit. To re-run the
audit explicitly after building:

```bash
lake env lean RoboVerify/Audit.lean
```

Compilation checks the proof terms with Lean's kernel. Warnings are errors,
so a proof containing `sorry` fails the build.
[Audit.lean](RoboVerify/Audit.lean) additionally checks the transitive axiom
dependencies of every declaration in the `RoboVerify` namespace. It accepts
only Lean's standard `propext`, `Classical.choice`, and `Quot.sound` axioms;
an additional assumed lemma, `sorryAx`, or a `native_decide` axiom is rejected.
The proof uses tactics such as `simp` and `omega`, whose generated proof terms
are kernel-checked. No SMT result is accepted as an oracle.

## The main statements

Start with [Formula.wpPut_exact](RoboVerify/Formula.lean). For every well-formed
scene, frozen entry scene, pair of physical operands, and assertion Q, it proves

```text
eval(s, wpPut(a,b,Q))
  ↔ exists s', PutStep(s,a,b,s') and eval(s',Q).
```

The left side is a syntactically calculated formula. The right side uses
independently defined execution. `PutStep` includes the clear-source,
clear-target, and distinctness checks. Thus an illegal placement has false WP.
The forward implication is soundness; the reverse establishes weakestness
for this deterministic legal operation. The theorem covers negation,
conjunction, disjunction, equality, both quantifiers, the table constant,
and frozen entry ON*.

`wpAttach_exact` separately checks the draft's original additive ON* rule
under its stronger singleton-source guard. `wpSeparate_exact` checks removal
to an isolated column. `wpPut_decomposition` proves that, under the agreed
pre-state guard, direct relocation has the same WP meaning as
`Separate(a); Attach(a,b)`. This is an equality of abstract endpoint behavior;
it does not require a robot to visit the table.

In [Program.lean](RoboVerify/Program.lean), `Action.wp_exact` covers assignment,
placement, and a single-name `Get`. Get's WP requires a witness and correctness
for **every** permitted witness. `Command.loopFree_wp_exact` lifts exactness
to finite loop-free commands; `loopFree_terminates` also establishes a
terminating execution from their WP.

`Command.verified_program` is the program soundness theorem:

```text
valid VCs and valid entry implication
  ⇒ from each well-formed initial state satisfying P:
       no finite execution prefix reaches a modeled fault, and
       every terminating execution ends in a well-formed state satisfying Q.
```

Loops use the supplied invariant. The theorem does not prove termination or
that an arbitrary supplied invariant is weakest. `chooseLoop_vc` handles
one existential guard witness through an explicit Get and requires the
invariant to survive rebinding that witness. It checks the scoping condition
that is easy to leave implicit in a prose argument.

## Model assumptions to review

[Scene.WellFormed](RoboVerify/Model.lean) states four structural properties:
roots are fixed points; roots have level zero; a root and level uniquely
identify a block; and every level below a block has a block in the same chain.
Levels are natural numbers, and different roots represent separated columns.

These hypotheses are assumptions of the theorem, not consequences of the
Python height inequalities. They do **not** assume the WP update equations.
Execution independently changes the moved block's root and level.
`on_iff_reach` proves that the root/level definition of ON* equals the
reflexive transitive closure of direct support. `direct_put` and
`direct_separate` prove that execution changes exactly the intended support
edges. The operations also preserve all four structural properties.

The proof works uniformly over the physical-object type; it does not enumerate
a bounded number of blocks. Finite domains in the draft are included.
The frozen scene need not satisfy structural assumptions for the WP theorem,
because operations never change it; a well-formed frozen entry scene is a
permitted special case.

Quantifiers use Lean functions for binding object values. Program names are
a separate mutable valuation. This avoids variable capture within the
formalization, but does not verify Python's de Bruijn substitutions.

## Paper to Lean correspondence

Names below are inside `RoboVerify`; scene lemmas are inside `Scene`.

| Draft argument | Checked declarations |
| --- | --- |
| Support interpretation and Lemma 1 | `on_iff_reach`, `put_wellFormed`, `separate_wellFormed` in [Model.lean](RoboVerify/Model.lean) |
| Lemmas 2–3 and relocation ON* | `on_attach`, `on_separate`, `on_put` in [Model.lean](RoboVerify/Model.lean) |
| Lemmas 4–6 | `nextHeight_iff`, `bottomHeight_iff`, `higher_put`, `higher_separate`, `scattered_put`, `scattered_separate` in [Effects.lean](RoboVerify/Effects.lean) |
| Lemma 7 and Theorem 1 | `Formula.attachAtom_correct`, `separateAtom_correct`, `rewrite_correct`, `wpAttach_exact`, `wpSeparate_exact` in [Formula.lean](RoboVerify/Formula.lean) |
| Corollary 1 with the agreed initial top guard | `Scene.separate_put`, `Formula.wpPut_exact`, `Formula.wpPut_decomposition` |
| Assignment, Get, composition and Theorem 2 | `Action.wp_exact`, `Command.verified_program`, `Command.chooseLoop_vc` in [Program.lean](RoboVerify/Program.lean) |

## Review examples and remaining correspondence

[Examples.lean](RoboVerify/Examples.lean) proves that the assumptions admit
concrete states and includes a legal move from an existing tower. It also
checks rejection of an occupied target, frozen/current ON* disagreement after
movement, and a quantified postcondition.

Two negative examples make important boundaries explicit. The attachment
formula retains an obsolete support link during general relocation, even
when both operands are clear. A height model satisfying the other three
structural properties but missing an intermediate level invalidates the
quantified Higher identity. These are kernel-checked counterexamples,
not solver sampling.

To review the proof, inspect the definitions of `WellFormed`, `Direct`,
`put`, `PutStep`, `Formula.eval`, `Action.Step`, `Command.Exec`, and
`Command.Fault` first, then compare their contracts with the intended system.
Lean validates the deductions from these definitions; choosing the right
definitions remains a human review task.

The production dispatcher still uses attachment-only ON* and does not encode
all the clear guards. This formalization does not silently fix that mismatch;
see [finding 30](../../../PAPER-DISCREPANCIES.md#30-stack-invariant-vocabulary-and-attachment-semantics).
There is no checked translation from Python ASTs or emitted Z3 formulas to
these Lean formulas, no proof of the solver-axiom inclusion and geometric
interpretation section, and no controller refinement or AFR-closure theorem.
Tuple-valued Get and a direct formalization of the runtime's scoped While
bindings are also outside this first version; the checked witness-loop result
is the explicit single-name lowering described above.
