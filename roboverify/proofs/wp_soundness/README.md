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

[RelationalSummary.lean](RoboVerify/RelationalSummary.lean) additionally proves
exact WPs for nondeterministic actions summarized by Higher or Scattered,
under explicit nonmover frames and successor admissibility. This is a separate
relational contract, not the deterministic complete-tower update above. The
[summary guide](../../synthesis/verification_lib/RELATIONAL-SUMMARIES.md)
gives the full rules, Python integration and solver limits.

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

`Command.branch` extends these soundness, no-fault, and loop-free exactness
theorems to conditionals. `chooseIf_wp` derives the witness-selecting rule:

```text
wp(if get v satisfying G then C else D, Q)
  = [forall v. G(v) => wp(C,Q)[v/name]]
    and [(forall v. not G(v)) => wp(D,Q)].
```

The else branch retains the original name valuation. It is enabled when
**no** eligible witness exists. A separate implication for every ineligible
witness would incorrectly require an unreachable else branch to establish Q.
Standalone Get additionally requires `exists v. G(v)`; an empty Get is a fault.
Kernel-checked negative examples for both points are in
[QuantifierLifting.lean](RoboVerify/QuantifierLifting.lean).

## Shared height witnesses

[WitnessWP.lean](RoboVerify/WitnessWP.lean) implements an alternative exact WP
over the existing formula vocabulary. Its `wpPut_exact` and
`wpSeparate_exact` cover **arbitrary postconditions**, including both
quantifiers and negation, and use the same `PutStep`/`SeparateStep` execution
relations as the original proofs. `wpPut_equiv` establishes equivalence with
the existing guarded Put WP. These are alternatives in Lean; the production
Python transformer has not been replaced.

For physical objects, write the following as formula abbreviations:

```text
Maximum(b) = forall t. Higher(b,t)
NextRep(b,u) = Higher(u,b) and not Higher(b,u)
              and forall t.
                (Higher(t,b) and not Higher(b,t)) => Higher(t,u)
```

`HeightWitness.nextRep_iff` proves that `NextRep(b,u)` means
`level(u) = level(b)+1` in a well-formed scene. `coverage` proves that either
b has maximum height or such a representative exists. This uses tower
completeness, not an extra assumed adjacency axiom or finite enumeration.

The new rule is:

```text
LegalPut(a,b)
and (Maximum(b) => Q[maximum-height atom replacements])
and forall u. NextRep(b,u) => Q[representative atom replacements].
```

For `Higher(a,c)` with c distinct from a, the representative replacement is
`Higher(u,c)`; the maximum-height replacement is `True`. The implementation
also handles operand equalities, table isolation, ON*, Scattered, and frozen
ON*. `putAtom_qf` and `separateAtom_qf` prove the individual replacements are
quantifier-free when the supplied row is. Table placement uses a shared
bottom-height representative. No new relation symbols are introduced.

The complete WP still has quantifiers. The change places universal selectors
in implication antecedents around the **whole** rewritten postcondition,
which is the polarity needed for the AER closure argument.

[Placement.lean](RoboVerify/Placement.lean) now constructs these replacements
in the function-free AER syntax itself. Its concrete `wp : AE V -> AE V`
rewrites every matrix atom, handles physical/table aliases by value, and
rejects illegal sources or occupied/equal block operands. There is no assumed
primitive transformer in this construction. `wpBlock_exact`, `wpTable_exact`,
and `wp_exact` prove equality with the corresponding scene update for every
AER postcondition, including arbitrary input universal/existential prefixes.

[FormulaBridge.lean](RoboVerify/FormulaBridge.lean) embeds AER formulas into the
original assertion language and proves preservation of meaning. The placement
theorems `wpBlock_equivWitness` and `wpTable_equivWitness` use this bridge to
prove equivalence with the concrete `WitnessWP` formulas. `wp_action_exact`
then proves the nonblocking/demonic WP contract directly for the existing
`Action.put` operational semantics. These close the concrete placement-to-AER
translation gap recorded in the first version of this development.

The distinguished table term must denote `none` in the input name valuation.
This is an explicit hypothesis, and `put_preserves_table` proves Put preserves
it. The concrete program language below also preserves it under assignment
and witness binding. No extra root/height axioms were introduced to prove placement.

## Restricted AER/EPR VC theorem

[Fragments.lean](RoboVerify/Fragments.lean) uses a separate deep syntax to
enforce function-free first-order matrices. Terms are variables or free
parameters/constants; there are no object-valued function symbols, arrays,
or quantifiers hidden in matrix atoms. A `Block` is a finite list-shaped tree
of variable slots; it does not enumerate objects. The domain may be infinite.

- `AF` permits Boolean combinations of universal-only or existential-only
  blocks over quantifier-free matrices. This includes single-block AFR
  specifications and explicitly fixes the paper's ambiguous terminology.
- `AE` is a finite `forall* exists*` prefix with a QF matrix (AER).
- `EA` is a finite `exists* forall*` prefix with a QF matrix (EPR).

Checked normalization lemmas preserve meaning for conjunction, disjunction,
AFR-antecedent implication, universal generalization, variable substitution,
and QF atom replacement. `AE.eval_refute` proves that negating AER gives EPR.
`EA.eval_close` existentially closes a finite set of free parameters while
remaining EPR. No object-domain enumeration is needed.

[FragmentVC.lean](RoboVerify/FragmentVC.lean) defines a restricted **normalizing
VC generator**, including tuple-valued Get, witness-selecting if and while,
sequencing, and abstract actions. Specifications and invariants are AF; all
selection matrices G are QF. Every action must supply an AER transformer.
`lift_AF_action` proves how the proposed uniform AFR-preservation assumption
suffices: on a fixed nonempty object domain, the nonblocking demonic WP
commutes with universal quantification of fresh logical parameters. Apply AFR
preservation to the existential suffix, then restore the universal prefix.
This is an explicit hypothesis, not a claim that current Put preserves AFR.
`Rules.sharedPut_exact` instead gives the shared-witness construction directly
in AER, using AF selectors and AER rewritten continuations.

The two main fragment theorems are:

```text
Fragments.Program.vcgen_epr:
  each generated VC has an equivalent EPR negation.

Fragments.Program.vcgen_background_epr:
  for an EPR background Gamma, each Gamma and not VC has an EPR representation.
```

[FragmentSoundness.lean](RoboVerify/FragmentSoundness.lean) interprets that
generator in the existing RoboVerify scene and name-store model, with
independently defined execution and faults. `Fragments.Program.verified_program`
proves that valid generated VCs imply partial correctness and no reachable
modeled fault, **provided the used action transformers are sound and actions
and witness bindings preserve the chosen admissible-state domain**. Atomic
operations now have explicit labels and a transformer interface, so different
operational actions are not identified merely by their transformer functions.
Selection and loop binding are modeled directly, including arbitrary finite
tuples of witnesses.

[BlockProgram.lean](RoboVerify/BlockProgram.lean) instantiates this generator
with concrete assignment and Put operations. A table literal is represented
separately from mutable program variables; assignment and selection cannot
overwrite it. Its `Admissible` domain requires both current and frozen entry
scenes to be well formed and the table literal to denote the relational table.
`Operation.wp_exact` and `Operation.preserves` discharge the atomic obligations,
and `bindings_sound` proves every compiled witness binding preserves that
domain. `BlockProgram.verified_program` consequently has no assumed action
transformer, action-soundness hypothesis, or table-binding hypothesis left to
the caller. Valid VCs imply partial correctness and absence of modeled faults
for this concrete language. `BlockProgram.vcgen_epr` supplies the equivalent
EPR negation for each concrete generated VC.

This establishes the restricted symbolic theorem and the alternative WP's
soundness, not EPR membership of the current Python pipeline. Remaining
connections are explicit: production guard matrices may themselves contain
universals; current/frozen top-existence background axioms are not EPR as
written; relational summaries quantify successor arrays; and no checked
Python-to-Lean translator exists. Concrete placement-to-AER translation and
program integration are now checked. Well-formed scenes are semantic
hypotheses, not a proved EPR encoding of all environment axioms.
EPR decidability follows from the usual fragment theorem; this development
does not mechanize an EPR decision procedure or certify Z3.

The full requested block-system theorem remains open at two explicit points:
the literal AFR-to-AFR action-preservation claim (the placement construction
proves AER closure); and an exact decision/reduction theorem for validity over
the complete block model. The user requires exact model validity: a sound
relational overapproximation with possibly spurious SAT countermodels is not
sufficient. Neither the AER type of the generated formula nor the kernel
axiom audit alone establishes those remaining properties.

## Model assumptions to review

For the new Higher/Scattered contracts, `RelationalSummary.wp_exact` equates
the candidate-relation rewrite with existence of a framed admissible successor
and correctness for **every** such successor. `rewrite_correct` covers the
full assertion language; equality and frozen ON* remain unchanged. The theorem
is uniform over any physical-object type, includes the table as an explicit
object, and allows any explicitly supplied successor-domain predicate. Table
isolation, when required, belongs to that admissibility predicate.
`implementation_refines` requires proofs of the frame, domain and endpoint
for each actual transition. Domain preservation is not inferred from its
endpoint atom. `sequential_sound` and `sequential_no_fault` cover demonic
continuation composition. This module is not yet a constructor in the existing
`Command` theorem below; Python/Lean translation, alias-store correspondence
and geometric refinement remain separate obligations.

Kernel-checked examples establish that Higher permits equal-height separated
blocks and does not imply ON*. Scattered admits unequal heights in either
direction and need not extend to a third block. Impossible successors and one
bad permitted successor explicitly refute the WP. This exactness concerns the
conservative relational contract, not an exact geometric placement WP.

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

The production dispatcher now detaches before attachment and includes the
clear-source/clear-target checks; see
[finding 30](../../../PAPER-DISCREPANCIES.md#30-stack-invariant-vocabulary-and-attachment-semantics).
There is no checked translation from Python ASTs or emitted Z3 formulas to
these Lean formulas, no proof of the solver-axiom inclusion and geometric
interpretation section, and no controller refinement or unconditional
AFR-closure theorem for current actions. Tuple-valued selection is covered by
the restricted generator above; correspondence with the Python runtime's
scoped bindings remains separate from these abstract execution definitions.
