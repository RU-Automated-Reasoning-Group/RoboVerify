# Paper / implementation discrepancies

Remaining paper corrections, modeling limitations, and demonstration issues in
`POPL2027.pdf` ("Component-Based Synthesis from Demonstrations under Local
Specifications") and `roboverify/`. Each entry distinguishes current behavior
from the original finding; an entry here does not necessarily require code work.
Neither the paper nor the code is automatically the ground truth.

**Resolved findings are recorded in [PAPER-RESOLUTIONS.md](PAPER-RESOLUTIONS.md):**

| Original entry | Settled issue |
| --- | --- |
| 5 | Frozen initial and current ON* use separate coordinates. |
| 6 | The plan's endpoint-box equivalence claim is corrected; the shared-`t` collision check is retained. |
| 11 | Multiple guard witnesses are permitted; verification covers every permitted choice. |
| 12 | Root discovery and constructed-tower alignment are checked, under the declared input-tower assumption. |
| 13 (implementation) | Motion verification proves the promised Scattered effects between physical blocks; `tbl` is excluded. |

Entry 13 below retains only the paper's missing block-domain clarification.
Original IDs are stable; do not renumber entries after moving resolved material.
Add new findings with the next unused ID and cite code locations and paper pages.

## Remaining work at a glance

| Entries | Current disposition |
| --- | --- |
| 1, 2, 8, 9, 13, 14 | Paper/formalization clarifications; implemented behavior is recorded in each entry. |
| 3, 10 | Validate or replace demonstrations before making end-to-end learning claims. |
| 4 | Physical-controller refinement remains outside the agreed geometric model. |
| 7 | Document the implemented successor/progress and entry/exit feedback semantics. |
| 15 | Runtime cap mismatch fixed; a total-termination proof is outside scope. |
| 16 | Code fixes complete; update paper rules. General height premises deferred until needed. |
| 17, 18 | Implementation complete; paper edits remain. |

---

## 1. Theorem 5.2 contradicts the paper's own Table 7 (`R_Higher`)

**Status: paper correction.** The theorem's no-new-quantifiers argument conflicts
with the written Table 7. This is distinct from the Higher rule fixes in entry 16;
the current code no longer reproduces the old table verbatim.

Theorem 5.2 claims closure of AFR under WP, justified by rewrites introducing no
quantifiers. Definition 5.1 allows a single quantifier block over a quantifier-free
matrix, but the paper's Higher rule 2 introduces an existential and rule 6 and the
table-placement rule introduce universals. For example, the written rule 2 can
introduce a genuine forall/exists alternation into a universally quantified post.
The claimed justification therefore fails for the paper's own rules.

The corrected implementation removes rule 2's existential but retains fresh
universal auxiliaries in rule 6 and the table case. Their effect on quantifier
prefixes depends on the surrounding formula and polarity; do not infer AFR
closure just because the old existential was removed. See
`rewrite_for_put_for_higher` and `rewrite_for_put_on_tbl_for_Higher` in
`roboverify/synthesis/api/program.py`.

Revise the closure/decidability statement using the corrected rules: either give
an appropriate fragment and proof or qualify the Higher case. No code should be
changed solely to force the original theorem's claim.

## 2. Rollouts from segment-initial states are required but never addressed

**Status: implementation complete in F0; paper omission remains.** Algorithm 5's
PostScore requires rollouts from each assigned segment's start, including starts
inside a demonstration. An observation alone does not contain the full arm,
velocity, control and contact-solver state needed to reproduce that execution.
The old observation setter restores a visualization, not a faithful segment start.

`roboverify/synthesis/cfg/reset.py` now records full simulator state, auxiliary
control/mocap/warm-start arrays and symbolic bindings. It supports direct reset
and deterministic replay; replay remains the default. Tests compare the resulting
state and subsequent action. Observation-only recordings without a replay source
are rejected. Solver-generated scenes are a separate geometric-concretization
path, not recorded states that can be restored.

The paper should state how demonstrations support faithful segment starts and
account for reset/replay in scoring. This is not remaining F0 implementation work.

## 3. The §4 demonstration input was literal data, including truncated datasets

**Status:** the Phase C data path now records real loop-head states; the old
examples survive only as golden fixtures. This does not establish the paper's
end-to-end synthesis or verification claims.

Before Phase C, the four tower inference examples in `inference.py` passed
hand-written dictionaries to `loop_inference`, rather than execution traces.
Their unchanged bodies now live in
`roboverify/synthesis/inference_lib/golden_tower_fixtures.py`:
`run_proposal_example` supplies four empty initial states but two current states;
`run_partial_stack_example` supplies four initial states but five current states.
The `zip` in `compute_dataset` silently truncates these inputs, dropping Partial's
fifth state entirely. These examples therefore cannot support claims about
learning from all observed loop iterations.

Phase C adds explicit aligned snapshots and a tested execution-to-inference
adapter. Real-trace Stack inference completes, but the sampled program fails
verification; observing states and learning a candidate must remain distinct
from proving inductiveness or task success. Fresh experiments are needed for
paper claims involving the complete pipeline.

## 4. Motion proofs depend on a waypoint abstraction, not physical controller dynamics

**Status:** Phase D now checks contract realization and frame preservation in the
explicit geometric model. Full physical-controller refinement remains unproved.

`PickPlaceByName.eval` drives a feedback controller, while
`motion_verification.py` executes idealized waypoints with optional bounded errors.
There is no settling or grasp-failure model. In particular, the existing Stack
body releases at `1.5 * BLOCK_LENGTH`, which is outside the strict direct-on band
at the waypoint endpoint; the new contract query refutes it rather than assuming
that it settles onto the target.

Separately, `_encode_release` in `bmc_lib.py` changes end-effector z but freezes
nominal block positions, while `ReleaseByName.eval` opens the gripper and then
moves the empty arm vertically. The earlier description of lowering a held block
was incorrect. D1 intentionally preserves this legacy default. The new noise
mode perturbs that nominal held-block position; it does not repair the nominal
controller model. BMC results certify a goal in their encoding, not collision
freedom. The motion verifier checks carried-cube and idealized point-gripper
sweeps; it does not certify full-arm or table-plane collision freedom. Hardware-wide claims need an explicit refinement argument and
additional geometry/dynamics, not just a bounded-noise flag.

The earlier BMC frame rule also froze blocks resting on a moved support. D2 removes
that assumption by allowing arbitrary disturbance; MotionVerify refuses to certify
support manipulation without a dynamics model. This deliberately changes the Move
encoding even with noise off. An old regression claiming an ungrasped block could
never move now explicitly excludes contact with the carried support.

## 7. Algorithm 6's invariant progression needs a precise failure state and learner

**Observed during Phase E.** A preservation VC counterexample satisfies the old
invariant at the input of the body. Adding that same state to positive `D_V` cannot
force enlargement; the body successor must violate the invariant. An exit VC
counterexample already satisfies the invariant but violates the postcondition;
weakening the invariant cannot exclude it. Phase E replays the supported symbolic
`Put`/`Assign` body to construct a successor and checks that it is newly uncovered.
It stops explicitly on exit failures instead of cycling on duplicate positives.
This replay is an abstract contract execution, not a MuJoCo trajectory or a proof
about controller settling. The current CFG pipeline integrates this supported
abstract replay; nested-loop synthesis remains outside the agreed flat-loop scope.

The acceptance wording "start at False and always converge" also conflicts with
the decided entry-failure branch: satisfiable entry conditions cannot establish
`False`. Phase E first learns from supplied demonstrations, logging `False` as
iteration zero. With no examples it raises `NeedsResynthesis` as required by the
plan. An establishment failure alone does **not** prove the program is incorrect;
it can also mean that the candidate invariant excludes initial states. The
integrated pipeline reports entry/exit coverage failures as requests for validated
demonstrations and supports resynthesis when those are supplied.

The legacy partition/minimization learner has no checked monotonicity contract.
The standalone CEGIS default learner uses the same Phase C vocabulary and scenes but
retains allowed Boolean truth-table rows under universal quantification. Adding
rows provably weakens this finite-vocabulary formula. Each accepted update also
checks old-invariant implication and has a newly covered positive witness. The
legacy `InvInference` remains selectable, with the same progress checks. This is
an implementation choice driven by the stated monotone-progress requirement, not
an attempt to reproduce Table 3's expressions or iteration count.

## 8. A finite relational countermodel need not describe a physical scene

**Observed during Phase E.** The `Higher` axioms allow two different blocks with
neither `Higher(a,b)` nor `Higher(b,a)`; real heights cannot realize that table.
`Scattered` can likewise differ from the geometry of a canonical ON* drawing.
Consequently, `_extract_direct_on`/`_build_stacks` alone do not close the
counterexample-to-demonstration edge claimed by the plan. The Phase E converter
solves for non-overlapping coordinates that preserve **all four** relation tables,
including separate frozen `ON_star_zero`, preserves aliases and the table marker,
and checks the numeric round trip. Inconsistent or timed-out concretization is
reported explicitly; an altered model is never silently fed to inference. This
implementation path is fixed; the paper still needs to account for geometrically
unrealizable relational countermodels.

## 9. Executable Get needs a witness-existence obligation

**Phase F implementation decision.** The plan describes Get as havoc followed by
assume. That partial-correctness encoding validates a Get with an unsatisfiable
condition vacuously, whereas `api/instructions.py:Get.eval` raises when there is no
witness. `api/program.py:wp` therefore requires both existence and correctness for
all permitted choices: `Exists(x, G) ∧ ForAll(x, G ⇒ Q)`. Tests reject an empty Get.
The paper should distinguish a blocking assume from successful executable binding.

## 10. The historical Unstack oracle does not establish its final task condition

**Observed in Phase F; not silently repaired.** The program preserved from the old
`entry/main.py` now lives in `cfg/demo_sources.py:unstack_oracle`. Its body moves the
chosen top block onto the current block, then advances the current-block binding.
This differs from the `Put(b_prime, tbl)` verification fixture. On the seed-0,
three-block Unstack smoke, the demonstration satisfies the initial tower condition
and reaches the all-unstacked relational predicate transiently, but fails it at
termination. The bounded report records `pre_holds=1`, `post_reached=1`, and
`post_at_end=0` for one demonstration.

Algorithm 5's reach-any-state PostScore is retained, but it cannot substitute for
final-state verification. The pipeline must not label imitation of this oracle a
verified Unstack solution. Reconcile the intended demo source and task before
claiming complete Unstack recovery. Report:
`roboverify/runs/phase-f/cfg/20260920-044337-f261887-integrated-flat-smoke`.

## 13. Paper clarification: Scattered ranges over physical blocks

**Status:** implementation resolved; only the paper's domain restriction needs
clarification. The user confirmed that symbolic table placement promises
Scattered between the placed block and other physical blocks, and motion
verification must establish that promise. The code does both and excludes `tbl`.
See [resolved entry 13](PAPER-RESOLUTIONS.md#13-placement-effects-and-block-only-scattered--implementation-resolved)
for the implementation and passing regressions.

Table 6 (p. 43) isolates the table marker: `Scattered(a, tbl)` is false. Table 7
(p. 44) writes the table-placement Scattered update without explicitly excluding
that marker, which would make it true for `a != tbl` if read over the full sort.
The paper should state the physical-block domain restriction (or include
`m != tbl` and `n != tbl`), matching the settled semantics and current code.
This is a paper notation correction, not an unresolved motion-verification defect.

## 14. AFR and predicate definitions are inconsistent across the paper

**Paper-side; found by direct reading.** Section 5.1 calls an existential followed
by a universal AFR despite Definition 5.1 allowing a single quantifier block. It
also describes the §3 learners as universal-only, contradicting §3.2.1's
existential classifier and §3.3's general guards. This is additional to the
Higher/WP closure contradiction already recorded in discrepancy 1.

Higher uses >= in §2.2/Appendix A but > in Definition 5.4. Direct-on's vertical
bands differ between §2.2, Definition 5.4 and Figure 12. Code uses non-strict
Higher and 0<=dz<1.5L for direct-on. These definitions need reconciliation;
neither blind copying nor a claim of exact agreement is justified. The long
Higher rewrite in Table 7 is also visibly clipped beyond the PDF page boundary.

Definition 5.4 should distinguish whole-state equivalence from preservation of
the verification result. The agreed interpretation is **equisatisfiability of the
universal collision query** after instantiating over the named objects and an
arbitrary collision witness, with all relevant universal axioms/invariants
instantiated over that set. The witness is constrained by those instances; it is
not an unconstrained invalid block. Whole-state equivalence on an arbitrary
larger environment is unnecessary for this argument. State the applicable
fragment and polarity conditions; do not generalize it to arbitrary quantified
formulas or treat finite instantiation alone as a demonstrated code defect.

## 15. Termination is not established by the stated VCs; runtime cap mismatch fixed

**Status:** paper termination claim remains; the runtime mismatch is fixed.
Theorem 5.7 asserts termination
from symbolic invariant and motion obligations without a ranking or progress
premise. Ordinary invariant VCs prove partial correctness, not that the guard
eventually becomes false.

**Runtime mismatch resolved (A2).** Generated loops no longer use the maximum
observed iteration count as an execution cap. An explicitly requested runtime
budget raises `LoopBudgetExceeded` instead of silently taking a normal loop exit;
collectors report an incomplete outcome. This fixes the premature-success issue.
The remaining paper claim is total termination without a ranking/progress
argument; invariant and motion VCs establish partial correctness. See the completed
A2 checklist in
[the active plan](PLAN-popl-alignment.md#audit-remediation--completed).

## 16. The Higher rewrite can disagree with geometric placement

**Code corrections complete under the agreed model. Paper update pending;
general height-premise encoding deferred by user decision.**

- [x] Correct put-on-block Higher rules 2 and 6 and auxiliary-variable capture.
- [ ] **Update the paper:** revise Table 7's Higher rules 2 and 6 to the formulas
  below, state the common-table, uniform-block, exact-support and complete-tower
  assumptions used in their justification, and make the full rule 6 readable.
- **Deferred until necessary:** encode the general supported-height assumptions
  for all blocks, including unnamed objects, in motion verification. Revisit if
  physically inadmissible intermediate-height counterexamples obstruct needed
  verification. This is not an immediate implementation task or a blocker to
  considering the identified entry 16 code bugs fixed.

The two-height premise belongs only to the regression fixture described below;
it is not a production restriction on block heights. Until the deferred work is
needed, motion verification continues using supplied premises and may reject a
valid motion when those premises also admit configurations outside our physical
model. Such a result remains unsuccessful verification; no counterexample is
silently discarded and no failed check is treated as a proof.

Put-on-block rules 2 and 6 were corrected by user decision.
Table 7's original clause for `Put(a,b)` and `Higher(c,a)`, with `c` distinct
from `a,b`, uses `Exists(t, t!=a and Higher(a,t) and Higher(t,b))`.
It ignores c's height: initially level a,b,c make t=b a witness even though
placing a above b makes c lower than a. It also incorrectly admits c=tbl.

The user confirmed the intended physical abstraction: equal-height upright
blocks, one common flat table, exact support, and complete supported towers
without vertical gaps. Center heights therefore lie on one grid spaced by L.
Under these assumptions, after placement `z'(a)=z(b)+L`, so
`Higher'(c,a)` iff `z(c)>=z(b)+L` iff `z(c)>z(b)`.
The implementation now uses **`Higher(c,b) and not Higher(b,c)`** in rule 2.
Both conjuncts are retained: the symbolic axioms do not enforce unconditional
physical-pair comparability, and the positive conjunct excludes the isolated
table marker. This is an intentional correction to the paper, not a claim that
Table 7 already states the correct rule.

Completed regressions prove the rule against arbitrary integer height levels
(independent of the source's old height), preserve table isolation, and accept
the formerly rejected level-source motion.

**Resolved in code — rule 6, `Higher(a,c)`.** Its former universal clause rejected
another object t at c's height above b. For example, c and d can be tops of
separate supported towers at `z(b)+L`. Placing a on b makes a,c,d level, but t=d
falsified the old rule for `Higher(a,c)`. The user-approved correction is:

```text
Higher'(a,c) = Higher(b,c) OR
  (Higher(c,b) AND
   ForAll(t, (Higher(c,t) AND NOT Higher(t,c)) => Higher(b,t)))
```

The antecedent now means strictly lower height, not distinct object identity.
Equal-height peers do not obstruct the comparison. If c is above b by two or
more levels, its complete support chain supplies an intermediate-height witness,
so the universal fails; at exactly one level it holds. Together with the first
disjunct this is equivalent to `z(c)<=z(b)+L` under the agreed assumptions.
The rule 6 auxiliary variable is now fresh. Regressions cover 75 supported-layer
configurations, duplicate-height objects including one named t, table isolation,
renaming a quantified t, and the formerly rejected equal-height tower motion.
That motion fixture explicitly constrains unnamed blocks to its two height
levels; named coordinates alone leave arbitrary intermediate heights possible.
Without those premises the effect check still rejects the fixture. This change
does not install a global supported-height axiom in motion verification.
The paper's long clause is clipped at the PDF's right edge; the complete old
formula reviewed here is the implementation's. The paper still needs updating.

**Resolved in code — auxiliary-variable capture.** The table Higher rewrite
formerly introduced a fixed `Const("t", BoxSort)`, capturing a program object
named t and turning its comparison into `ForAll(t, t!=tbl => Higher(t,t))`.
The shared `synthesis/util/symbols.py` mechanism now allocates auxiliary symbols
and opens/rebuilds existing quantifiers without reconstructing their printed
names. All Put/MarkGoal rewrites, goal-successor helpers and updates, inference
quantifiers, predicate conversion, and verification auxiliaries use the shared
mechanism. CFG-generated names are reserved against free and in-scope names.
Persistent program constants and shared state symbols retain their identities.
Regressions cover table placement against a high block named t, nested shadowing,
free update operands with binder-like names, goal helpers and moves, inference
AST conversion/promotion, and classifier binder collisions. This was an
implementation binding bug, independent of the paper's height formulas.

The review's floating-block/missing-layer examples are outside the agreed model,
not additional discrepancies. In particular, table Higher rules 3 and 4 have
valid geometric interpretations under the supported-height assumptions.
Motion verification continues checking exact ON*, Higher and Scattered effects;
remaining abstraction mismatches are rejected and solver unknown is inconclusive.

## 17. Temporal validation acceptance and the incorrect rejection sentence

**User-adopted interpretation — implemented:** refining an original block with
incoming condition P and outgoing target Q introduces intermediate condition C:

```text
Before: P -> v0 -> Q
After:  P -> v1 --C--> v2 -> Q
```

An edge condition is the source block's exit condition and the destination
block's entry condition, consistent with §2.3's outgoing-edge summaries.
Thus v1 establishes C and v2 establishes Q. Neither incoming P nor intermediate
C must persist until the next condition is reached; P and C, or C and Q, need
not hold simultaneously.

The acceptance comparison is `first(C) <= last(Q)` on **each original unsplit
segment assigned to v0**. Q means the original outgoing target, not incoming P.
Both occurrences must exist. Do not reinterpret this as overlap between the
incoming and outgoing predicates of each child segment.

The recorded segment must also satisfy P at its start and Q at its end, with
first(C) strictly inside. These stronger boundary requirements already imply
the inequality: last(Q) is the original end and first(C) is earlier. The explicit
comparison remains in the implementation to state the intended rule. Equality
passes that comparison alone; it does not override the strict interior-cut rule.
Initial and later blocks use the same checks. At entry, the **precondition**
holds at local time zero; requiring C there would contradict an interior split.

**Code correction complete:** `refine_cfg` scans C and the original outgoing Q
on the original segment. `validate_cfg` checks edge conditions at shared
boundaries, witnesses and adjacency without imposing incoming/outgoing overlap.
Intermediate edges retain first-occurrence cuts, and final targets must hold at
the recorded end. Failed refinements leave the original CFG and demonstration
assignment unchanged. Regressions cover destroyed incoming/intermediate
conditions, entry and offset segments, target reestablishment, missing boundary
conditions, invalid cuts and atomic rejection. Focused validation: 25 tests pass.
Full suite: **215 tests pass in 50.735 seconds**, including simulator tests.

**Paper corrections still required:**

- In §3.5, p. 19, replace "If this holds" with "If this fails" in the rejection
  sentence; Appendix G, p. 57, already has the intended direction.
- Name the original target Q and new intermediate C explicitly. The notation
  `v_phi --psi--> v_psi` does not clearly communicate their roles alongside
  §2.3's outgoing-edge summaries.
- State that first(C)/last(Q) are measured on the original unsplit segment;
  applying the comparison independently to the child segments is different.
- Clarify that the entry-time requirement concerns the precondition, while C
  defines a later interior cut. Record that the stronger boundary checks imply
  the comparison.

These are clarifications adopted with the user, not a claim that the current
paper already specifies this interpretation unambiguously. The previous
incoming/next-condition overlap restriction and pending entry question are
superseded by this decision.

## 18. Primitive motion formulas do not justify the stated Pick/Release claims

**Status: paper changes only within the agreed model.** The code and regression
tests already implement the contact and support semantics below. Physical
simulator/controller refinement is the separate limitation in entry 4, outside
this entry's implementation work.

**Paper, §5.5, PDF pp. 27–29.** Formula (5) excludes only the pre-held object.
Pick starts with no held object and ends at its target block's center, up to
bounded grasp noise. Choose that block as collision witness and t=1: at zero
noise all three coordinate differences are 0 < L. Thus the written formula
reports collision with the target. The claim that a grasp-noise bound below L
prevents this is false; intended contact needs an explicit exception or different
geometry.

The paper's Release leaves a supported block in place and assigns an arbitrary
position to an unsupported block, then moves the empty gripper away. The simulator
opens the gripper before moving vertically. Neither description by itself proves
settling or controller refinement.

**Implemented model:** `verification_lib/primitive_motion.py` treats the empty
gripper as a point and carried blocks as cubes. Pick/Release explicitly exclude
contact with the selected object from their collision queries; other blocks
remain checked. Release must prove support. Unsupported release fails a support
obligation and gives the block an arbitrary falling position, rather than freezing
it in mid-air. `test_primitive_motion.py` covers successful placement, unsupported
release, state propagation and empty-gripper collisions.

**Remaining paper edits:** state the intentional-contact exception and geometry,
remove the incorrect grasp-noise argument, and state the Release support
assumptions and model-only proof scope. See the current
[verification guide](roboverify/synthesis/cfg/VERIFICATION.md).
