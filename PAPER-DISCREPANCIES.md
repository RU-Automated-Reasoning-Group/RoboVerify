# Paper review and decisions

This is the single record of findings from reviewing [POPL2027.pdf](POPL2027.pdf)
against the implementation. It covers §§2–5, Algorithms 1–6, Appendix A/Table 7
and the associated appendix algorithms/proofs; experimental numbers are not
correctness targets. Neither paper nor code automatically wins a disagreement.

Entry numbers 1–23 are stable, including resolved findings. Each entry records
its status, decision/reasoning and remaining action. Add new findings with the
next unused ID. Detailed implementation history and completed audit checklists
remain in Git history; use [README.md](README.md#project-status) for project status
and [the verification guide](roboverify/synthesis/cfg/VERIFICATION.md) for current
model assumptions and APIs.

| Entries | Remaining work |
| --- | --- |
| 1, 2, 7, 8, 9, 13, 14 | Paper/formalization corrections; implementation decisions are recorded below. |
| 3, 10 | Representative demonstration coverage and full end-to-end learning acceptance. |
| 4 | Paper scope correction; physical-controller refinement is outside scope. |
| 5, 6, 11, 12 | Resolved; retain the decisions and alignment proof. |
| 15 | Paper termination claim; runtime cap mismatch is fixed. |
| 16 | Paper rules/assumptions; general solver height premises deferred until needed. |
| 17, 18 | Paper edits only; code fixes complete. |
| 19 | Resolved integration defect: recorded and candidate imitation features. |
| 20 | Shared Stack specification and runtime inference implemented; acceptance remains open. |
| 21 | Switchable ID-first policy implemented; full learning acceptance and policy comparison remain open. |
| 22 | Preserve pending transfers and task-completing continuations across relational CFG cuts and loop exits. |
| 23 | Resolved primitive-controller discrepancy; shared configurable control and execution parity checks. |

## 1. Theorem 5.2 contradicts the paper's own Table 7 (`R_Higher`)

**Status:** paper correction; separate from the implemented Higher fixes in entry 16.

**Decision/reasoning:** Theorem 5.2 claims AFR closure under WP using a
no-new-quantifiers argument. Definition 5.1 permits one quantifier block over a
quantifier-free matrix, but the paper's Table 7 introduces an existential in
Higher rule 2 and universals in rule 6 and the table case. The written rule 2 can
introduce forall/exists alternation into a universal postcondition.

The corrected code removes rule 2's existential but retains fresh universal
auxiliaries. Their effect depends on the surrounding formula and polarity;
removing the old existential does not by itself prove AFR closure. See the
Higher rewrites in `roboverify/synthesis/api/program.py`.

**Remaining action:** revise the closure/decidability statement and proof using
the corrected rules, or qualify the Higher case. Do not change valid code solely
to force the paper's original theorem.

## 2. Rollouts from segment-initial states are required but never addressed

**Status:** implementation complete; paper omission remains.

**Decision/reasoning:** Algorithm 5/§3.4 scores rollouts from each demonstration
segment's start. An observation lacks the full arm, velocity, control and solver
state needed to reproduce that execution. `cfg/reset.py` records full simulator
snapshots, control/mocap/warm-start arrays and bindings. It supports direct reset
and deterministic replay; replay remains the default. Current archives require full snapshots and actions; observation-only
formats and their replay compatibility adapter have been removed. Solver-generated
scenes instead use geometric concretization, not recorded-state restoration.

**Remaining action:** document faithful segment starts and reset/replay costs in
the paper. No remaining reset implementation task.

## 3. The §4 demonstration input was literal data, including truncated datasets

**Status:** trace-based inference implemented; end-to-end learning acceptance open.

**Decision/reasoning:** The old §4 examples used literal dictionaries, some with
unequal initial/current list lengths that `zip` truncated. They remain only in
`inference_lib/golden_tower_fixtures.py` as historical regression fixtures.
`DemoStore` now supplies aligned execution snapshots with per-invocation entry
geometry. Observing states and learning a candidate do not prove inductiveness
or task success. Tests do not require saved demonstration files.

**Remaining action:** use validated task demonstrations for full-pipeline
acceptance and renewed empirical claims. Stack is the current first task (20).
Five primitive Stack recordings at each of three and four blocks, seeds 0–4,
pass initial/final validation; this is demonstration acceptance, not synthesized-program verification.
Do not treat old fixtures or paper numbers as targets.

## 4. Motion proofs depend on a waypoint abstraction, not physical controller dynamics

**Status:** documented model boundary; physical-controller refinement is outside scope.

**Decision/reasoning:** Motion proofs concern idealized waypoints with optional
bounded error, not all executions of the feedback controller. They do not model
settling, grasp failure, full arm/finger geometry or table-plane collisions.
An endpoint outside the placement contract is rejected even if a simulator might
subsequently settle the block onto its target.

The legacy BMC Release changes end-effector z while freezing nominal block
positions; optional release noise perturbs that nominal location. Primitive
Release instead proves support and permits an arbitrary unsupported fall (18).
The simulator opens the gripper before moving the empty arm vertically. BMC goal
proofs and motion collision/contract proofs have distinct encodings and scopes.
Moving an occupied support is not certified by assuming other blocks stay fixed.

**Remaining action:** qualify paper claims to the documented model. A controller
refinement argument and additional geometry/dynamics would be needed for broader
physical claims; that is not immediate implementation work. See the
[motion guide](roboverify/synthesis/verification_lib/README.md).

## 5. The geometric translator conflated initial and current ON_star

**Status:** resolved; no remaining action.

**Decision/reasoning:** `ON_star_zero` denotes frozen invocation-entry geometry;
`ON_star` denotes current geometry. The translator uses separate coordinate
functions. Tasks using the frozen relation equate it with current ON* in the
precondition, never through a global link axiom. Put cannot rewrite the past.
This lets Reverse relate final order to the original order without forcing them
to remain equal.

## 6. Plan correction: an endpoint bounding box is not an equivalent collision check

**Status:** resolved documentation error; no remaining action.

**Decision/reasoning:** §5.5 p. 29 and `encode_collision_at` use one shared
trajectory parameter t across all three coordinates. The proposed enclosing
endpoint-box replacement was never implemented. On diagonal motion, that box
contains space outside the swept path: clearance can prove safety, but overlap
alone cannot establish collision. It is an overapproximation, not an equivalent
query. Retain the shared-t moving-cube model; physical refinement and intended
Pick contact are separate issues (4 and 18).

## 7. Algorithm 6's invariant progression needs a precise failure state and learner

**Status:** feedback semantics implemented; paper clarification remains.

**Decision/reasoning:** A preservation counterexample's input already satisfies
the old invariant. Refinement adds an uncovered **successor**, obtained by
executing the supported abstract body, to force enlargement. This is contract
replay, not a MuJoCo trajectory. An exit counterexample satisfies the invariant
but violates the postcondition; weakening the invariant cannot exclude it.
An establishment failure can mean the invariant misses valid entry states, not
that the program is wrong. Entry/exit coverage failures therefore request
validated demonstrations and support resynthesis when recordings are supplied.

Log False as iteration zero, then bootstrap from supplied traces; do not promise
unconditional convergence from False. Each accepted update checks implication
from the old invariant and newly covered states. Standalone CEGIS defaults to
monotone Boolean rows; the integrated CFG CLI defaults to the legacy learner
with the same explicit progress checks.

**Remaining action:** clarify Algorithm 6's failure-state, successor and progress
semantics. No remaining integration task for the supported flat-loop workflow;
see the [integrated guide](roboverify/synthesis/cfg/VERIFICATION.md).

## 8. A finite relational countermodel need not describe a physical scene

**Status:** conversion implemented; paper clarification remains.

**Decision/reasoning:** Abstract relation tables need not describe physical
geometry. For example, neither Higher(a,b) nor Higher(b,a) can hold in a relational
model although real heights are comparable. `verification_lib/counterexamples.py`
solves for non-overlapping coordinates preserving ON*, frozen ON*, Higher,
Scattered, aliases and table identity, then checks the numeric round trip.
Unrealizable or timed-out models are reported explicitly; they are not silently
changed into usable training scenes.

**Remaining action:** explain how the paper handles geometrically unrealizable
countermodels instead of treating a canonical tower drawing as sufficient.

## 9. Executable Get needs a witness-existence obligation

**Status:** implemented; paper clarification remains.

**Decision/reasoning:** Executable Get fails if no witness exists. A pure
havoc/assume encoding could validate an unsatisfiable binding vacuously. The WP
therefore requires both existence and correctness for every permitted witness:
`Exists(x, G) AND ForAll(x, G => Q)`. See `api/program.py` and `Get.eval` in
`api/instructions.py`.

**Remaining action:** distinguish executable binding from a blocking assume in
the paper's Table 1. The stronger existence obligation is intentional.

## 10. The historical Unstack oracle does not establish its final task condition

**Status:** demonstration validation and full learning acceptance remain open.

**Decision/reasoning:** `cfg/demo_sources.py:unstack_oracle` preserves a historical
program that moves the selected top block onto the current block and advances
its binding. It differs from the Put-to-table verification fixture. The seed-0,
three-block run reached the unstacked predicate transiently but failed it at the
recorded end. Algorithm 5's reach-any-state PostScore cannot replace final-state
correctness. The integrated pipeline rejects invalid demonstrations before search.

**Remaining action:**

1. Reconcile the demonstration source with the intended task or collect new
   recordings; validate both initial and final conditions.
2. Run integrated synthesis on those recordings and recover the complete loop.
3. Verify that same synthesized CFG against the same task specification; record
   exhausted budgets, unsupported candidates and inconclusive proofs explicitly.

Keep Unstack end-to-end runs within the standing **60-second wall-clock limit**.
Learning acceptance is distinct from completed implementation and synthetic
regressions; neither imitation nor the old oracle's structure proves correctness.

## 11. Corrected plan/code restriction: loop guards may have multiple witnesses

**Status:** resolved; no remaining action.

**Decision/reasoning:** The paper's §5.3 arbitrary-witness semantics are appropriate.
A demonstrated selection does not make other valid or indistinguishable choices
wrong. Learning treats demonstrated bindings as positives, unselected bindings at
continuing heads as unlabeled, and **all** bindings at demonstrated exits as
negatives. Runtime selects the first match; no match exits the loop.

Preservation verification covers every guard-satisfying witness, including choices
never demonstrated. The former uniqueness restriction is removed. Standalone Get
still requires witness existence (9).

## 12. Root discovery and tight alignment premises — implemented with an explicit input assumption

**Status:** implemented under the explicit input-tower assumption; no remaining action.

**Decision/reasoning:** §5.5 pp. 29–32 and Appendix J use a tight root-relative
invariant to bound every pair in a tower. Existing towers are assumed to satisfy
that invariant; each constructed placement must prove it. The weaker ON* relation
does not establish the tight premise on its own.

**Root selection.** Seek an in-scope physical name r such that, with
P = wp(remaining symbolic body, postcondition),
`P => ForAll(u, ON*(target,u) => ON*(u,r))`.
Under reflexivity, antisymmetry and satisfiable P, this identifies the bottom
root; a named solution need not exist. The actual entry context must also imply
P. `verification_lib/root_selection.py` and `cfg/verification.py` prove those
obligations using established facts transported through the prefix. Standalone
motion calls use their supplied entry conditions. Unnamed objects are covered by
quantification; no `b0`, first-name or `frame_base` hint substitutes for a proof.
The main text's finite O notation and Appendix J's unrestricted u require the
universal-validity argument, not checking one concrete witness assignment.

**Root-alignment induction.** For members S, root r and F in {X,Y}, define:

```text
Aligned(S,r) := for every a in S, |F(a)-F(r)| <= delta_F
where 2*delta_F <= N_F.
```

1. A singleton tower satisfies Aligned. Pre-existing towers satisfy it by the
   explicit input assumption.
2. Before insertion, assume Aligned(S,r). Preserve existing members and the root,
   or prove their bounds still hold after motion.
3. For the new block x, check only `|F(x)-F(r)| <= delta_F`. This establishes
   Aligned(S union {x},r).
4. For any a,b in the enlarged tower, the triangle inequality gives
   `|F(a)-F(b)| <= |F(a)-F(r)| + |F(b)-F(r)| <= 2*delta_F <= N_F`.
5. The same root-relative invariant is preserved, so induction applies to any
   finite number of insertions. Strict premises yield the strict version.

Thus root-only placement checks suffice; separate new-to-every-old checks are
unnecessary. The lemma does not require pairwise distances <= delta_F. Removing
members preserves the remaining subset's bound if its reference is retained;
changing/moving the reference or merging towers requires re-establishment.

**Implemented obligations.** Strict delta=L/4 and N=L/2 apply in each horizontal
coordinate. `assume_input_alignment` applies only to input geometry for proved
roots, never to a placement's output. Get/Assign may expose another input root
before any Put. `alignment_entry` checks the destination tower; `alignment`
checks the new member after motion, including bounded noise. Frame VCs preserve
all non-manipulated objects, and support checks reject moving occupied supports.
Fresh loop contexts carry the tight geometric invariant. Missing roots,
inconsistent inputs or unknown proofs prevent certification. Vertical support,
collisions and complete relation effects remain separate obligations.

**Why loose pairwise bounds are insufficient.** In units delta=1, N=2, take
bottom-to-top X coordinates `[0,-1,-2,-1,0]`. Adjacent offsets are 1 and old pairwise
distances are at most 2. A new block at +1 is within 1 of the root and old top,
but 3 from the block at -2. This scene violates the tight input premise; it is
not a counterexample to the induction above. It explains why root selection plus
a new-member check cannot replace establishment of that premise.

## 13. Paper clarification: Scattered ranges over physical blocks

**Status:** implementation complete; paper domain clarification remains.

**Decision/reasoning:** After Put(a,tbl), a must be Scattered from every other
physical block. `check_abstract_effects` proves the promised ON*/Higher/Scattered
results from motion geometry, including an arbitrary unnamed object; local
placement success alone cannot bypass those checks. The table-placement rewrite
excludes tbl from both Scattered arguments. Physical table height is separate.

Table 6 (p. 43) isolates tbl, but Table 7 (p. 44) omits that restriction in the
Scattered rewrite. A full-sort reading would contradict isolation.

**Remaining action:** state the physical-block domain or explicitly include
`m != tbl` and `n != tbl`. No further motion-verification fix is required.

## 14. AFR and predicate definitions are inconsistent across the paper

**Status:** paper definitions/formalization need reconciliation.

**Decision/reasoning:** §5.1 calls an existential followed by a universal AFR,
although Definition 5.1 permits a single quantifier block. Its universal-only
learner description also conflicts with §3.2.1's existential classifiers and
§3.3's guards. This is additional to entry 1.

Higher is non-strict in §2.2/Appendix A but strict in Definition 5.4. Direct-on
vertical bands differ across §2.2, Definition 5.4 and Figure 12. Code uses
non-strict Higher and `0 <= dz < 1.5L` for direct-on.

Finite instantiation should preserve the **verification result**, not claim
whole-state equivalence on arbitrary larger environments. The agreed universal
collision-query argument is equisatisfiability after instantiating all relevant
universal axioms/invariants over named objects and an arbitrary collision witness.
That witness is constrained by those instances; it is not an invalid unconstrained
block. Do not generalize the argument to arbitrary quantified formulas, or treat
finite instantiation alone as a demonstrated implementation defect.

**Remaining action:** reconcile predicate/AFR definitions and state the applicable
fragment, polarity conditions and equisatisfiability argument in Definition 5.4.

## 15. Termination is not established by the stated VCs; runtime cap mismatch fixed

**Status:** runtime mismatch fixed; paper termination claim remains.

**Decision/reasoning:** Theorem 5.7 derives termination from invariant and motion
VCs without a ranking/progress premise. Those VCs establish partial correctness.
Generated loops no longer inherit a demo-count cap. An explicit execution budget
raises `LoopBudgetExceeded` while a guard witness remains; collectors report
incomplete execution rather than treating it as a normal guard-false exit.

**Remaining action:** qualify the theorem as partial correctness or supply an
additional termination argument. Total-termination implementation is outside scope.

## 16. The Higher rewrite can disagree with geometric placement

**Status:** code fixes complete; paper update pending; general height premises deferred.

**Decision/reasoning:** The agreed physical abstraction uses uniform upright
blocks of height L, a common flat table, exact support and complete towers without
missing layers. Centers lie on a common L-spaced grid. For Put(a,b), with c distinct
from a,b, the corrected Higher clauses are:

```text
Rule 2: Higher'(c,a) = Higher(c,b) AND NOT Higher(b,c)

Rule 6: Higher'(a,c) = Higher(b,c) OR
          (Higher(c,b) AND
           ForAll(t, (Higher(c,t) AND NOT Higher(t,c)) => Higher(b,t)))
```

Rule 2 follows from `z'(a)=z(b)+L`: on the grid, `z(c)>=z(b)+L` iff
`z(c)>z(b)`. Retain both conjuncts because abstract Higher need not be total and
the positive conjunct excludes tbl. The old existential clause ignored c's height.

Rule 6 tests strictly lower levels, not distinct identity. Equal-height peers do
not obstruct it. If c is at least two levels above b, its complete support chain
supplies an intermediate level, refuting the universal; at one level it holds.
Together with the first disjunct this gives `z(c)<=z(b)+L`. Missing-layer and
floating-block examples fall outside the agreed model. Table Higher rules 3/4
remain valid under these assumptions.

Fresh auxiliary allocation and capture-safe quantifier rewriting are implemented
through `synthesis/util/symbols.py`; a program object named t cannot be captured
by an introduced forall. This binding bug is resolved independently of the height
formulas. See `api/program.py` and the Higher/quantifier-hygiene regressions.

**Remaining action:** update Table 7 rules 2/6, make the clipped rule 6 readable,
and state the physical assumptions used in their justification.

**Deferred by user until needed:** encode general supported-height premises for
all blocks, including unnamed objects, if inadmissible intermediate-height models
obstruct required proofs. The two-height premise is only a regression fixture,
not a production restriction. Until then, motion uses supplied premises and may
reject valid motion when those premises admit nonphysical scenes. Failed/unknown
checks remain unsuccessful verification; no counterexample is silently discarded.

## 17. Temporal validation acceptance and the incorrect rejection sentence

**Status:** implementation complete; paper changes only.

**Decision/reasoning:** Refinement replaces `P -> v0 -> Q` with
`P -> v1 --C--> v2 -> Q`. An edge condition is established by its source and
assumed by its destination (§2.3). Compare `first(C) <= last(Q)` on each
**original unsplit segment**: Q is the original outgoing target, not incoming P.
P holds at the start, first(C) is strictly interior and Q holds at the end.
These boundary checks already imply the inequality. Equality passes the comparison
alone, but cannot override the interior-cut requirement.

Initial and later blocks use the same rule. The entry precondition, not C, holds
at local time zero. Neither P nor C must persist until the next condition; no
overlap is required. `cfg/refine.py` and `cfg/validate.py` retain whole-CFG boundary,
binding and adjacency checks and reject invalid refinement atomically.

**Remaining action:** in §3.5 p. 19 change "If this holds" to "If this fails",
as Appendix G p. 57 already says. Name C/Q clearly, specify the original-segment
domain and clarify the entry precondition. These are user-adopted clarifications,
not a claim that the current paper states them unambiguously.

## 18. Primitive motion formulas do not justify the stated Pick/Release claims

**Status:** implementation complete; paper changes only within the agreed model.

**Decision/reasoning:** §5.5 pp. 27–29 formula (5) excludes only the pre-held
object. Pick begins empty and ends at its target's center plus noise. Taking the
target as collision witness and t=1 at zero noise makes all three differences
0 < L. A small grasp-noise bound therefore does not prevent the written formula
from reporting collision with the selected target.

`verification_lib/primitive_motion.py` explicitly permits Pick/Release contact
with the selected object. It models the empty gripper as a point and payloads as
cubes; other objects remain collision-checked. Release proves support, otherwise
fails an obligation and assigns an arbitrary falling position. The simulator opens
the gripper before moving the empty arm vertically. These semantics do not prove
settling or physical controller refinement (entry 4).

**Remaining action:** state the contact exception, geometry and Release support
assumptions, remove the incorrect grasp-noise argument, and limit the paper's
claim to the documented model. No additional code fix is required for this entry.

## 19. Recorded and candidate CFG trajectories used different imitation features

**Status:** implementation defect fixed during the Stack workflow integration.

**Finding:** Raw demonstration observations retained gripper xyz, finger opening,
and block xyz, but candidate rollouts converted to relational Scenes discarded
the five gripper features. Three-block scoring therefore tried to compare
14-dimensional demonstrations against 9-dimensional candidates and crashed.

**Resolution:** Observation-derived Scenes retain a copy of their source
observation. Both sides use the existing shared comparison indices; no objective
features or distance thresholds were removed or rescaled. Synthetic scenes
without observations retain their geometric feature path. A regression checks
identical feature vectors, and the real full-search smoke now proceeds through
search and refinement to an explicit budget-exhausted outcome.

## 20. Stack entry conditions and invariant data now agree across both modes

**Status:** user-selected workflow implemented; end-to-end acceptance remains open.

**Decision:** Stack starts with unstacked, pairwise-scattered blocks and ends with
all blocks ON* b0. The former integrated True precondition and the standalone
scattered-block precondition were different tasks. Collection and both new modes
now use one specification. The public supplied-program path uses explicit
Pick/Move/Release primitives, not the old PickPlace macro program.

Initial invariants are learned from the actual candidate's recorded continuing
heads and normal guard-false exits after synthesis (or supplied-program loading).
This replaces demonstration-partition inference as the integrated bootstrap.
Guard learning still uses the expert demonstrations, and preservation feedback
still uses checked abstract successor replay (7). Frozen entry geometry and
arbitrary-witness verification are unchanged. Physical repairs cause new runtime
traces and renewed inference/verification; collected observations alone never
prove inductiveness or the physical-controller refinement excluded in entry 4.

**Evidence:** Five real three-block Stack collections, seeds 0–4, satisfy their
initial/final conditions and have 20 FPS videos. Video on/off actions are identical
and observations agree within the existing 1e-8 restoration tolerance. The
five-demo verification smoke collects 15 heads/exits, reaches symbolic checking,
and requests additional demonstrations. The full smoke reaches search/refinement
and exhausts its budget.
Neither is successful verification. Tests use generated data and exercise both
real verifiers and their feedback paths separately from these acceptance runs.

**Remaining action:** supply the requested coverage or improve the candidate as
justified by the failed obligations, recover the full loop through search, and
obtain `verified_model` on that same synthesized candidate.

## 21. Free-object binding is performed before search instead of on the returned candidate

**Status:** user-selected ID-first alternative implemented; the original relational policy remains the default.

**Paper:** Section 2.2 (p. 4) defines object-valued variables, and Fig. 5 (p. 10)
uses bound `b` and `b_prime` in the loop body's physical primitives. Algorithm 5
(p. 17) mutates instructions and operands, then optimizes continuous parameters.
Section 3.4 (p. 17, lines 831–832) describes introducing a typed `Get(True)` for
any object identifier still free in the returned candidate. It does not specify
Python ByName/ID classes or require every search operand to be a numeric ID.

**Existing relational approach:** `entry/synthesize_cfg.py` generates a numeric seed program,
then calls `cfg/bindings.py::close_objects` before `straight_line_synthesize`.
This reuses consistently bound in-scope aliases or inserts fresh `Get` bindings,
and converts physical instructions to ByName variants. `mutate_scoped` then
freezes the Get prefix and samples only names already in scope or introduced by
that prefix. MCMC cannot introduce additional object bindings within that search
attempt. The standalone MCMC CLI instead supplies numeric-ID primitives and an
integer operand pool; it is not the complete relational CFG algorithm.

**Consequence:** ByName instructions represent the paper's variable-based DSL,
but early binding plus a fixed binding prefix is an additional search policy.
It changes the available operands and can change the scored rollouts: a fresh
`Get(True)` need not select the concrete ID that appeared in the random seed.
This is not solely a class-name substitution. Its effect on end-to-end search
acceptance has not been measured, and it is not established as the cause of
existing smoke-run failures.

**Decision and implementation:** The user selected a switchable alternative,
`--synthesis-approach id-first`, while retaining `relational` as the default.
ID-first MCMC uses only numeric physical operands; CFG refinement learns ground
ID predicates without existential binders. After concrete search finishes,
quotienting introduces loop roles from repeated predicates and physical operand
sequences. This is an explicit user-selected search schedule, not a claim that
the paper uniquely prescribes it. Runtime first-witness selection is unchanged.

The user additionally requires a named synthesis output. Before synthesis
returns, residual IDs become fixed entry aliases with preserved identity and
explicit alias facts. Loop variables arise during quotienting; fixed residual
aliases do not claim relational generalization. No numeric primitives or ID
literals are sent to invariant inference or either verifier. Fresh execution and
both verification stages still check the returned candidate, including all
allowed guard witnesses. A numeric rollout alone cannot establish that result.

**Evidence:** Generated tests learn `ON(1, b0)` then `ON(2, 1)` through the actual
classifier, recover carried/rebound loop roles, preserve fixed-base XY and
carried-top Z, and exercise the named boundary through the existing symbolic and
motion verifiers. CLI and resynthesis tests retain the selected policy. The full
suite passes 247 tests. A real five-demo Stack ID-first smoke (seeds 0–4,
base-aligned collection) learns `ON(1, b0)` with no new bindings, then exhausts
its search budget in about 70 seconds. It does not reach verification and is not
learning acceptance.

**Remaining action:** demonstrate complete learning acceptance with representative
recordings and compare the two search policies. The early-Get policy's fixed
prefix remains an explicit limitation of the retained relational variant.

## 22. ID-first continuation exposes placement and loop-exit boundary limits

**Status:** observed implementation limits; diagnostic recorded, synthesis behavior
unchanged. This experiment substitutes known controllers for MCMC, so it is not
end-to-end learning acceptance or formal verification.

**Setup:** Run `synthesis.experiment.id_first_continuation` on the five validated,
base-aligned three-block Stack recordings (seeds 0–4). Predicate enumeration,
segment replay, CFG refinement, quotienting, and full-program simulator execution
are real. Only straight-line search is replaced by supplied numeric fragments
from the demonstration program.

**Findings:**

- The actual classifier first learns `ON(1, b0)`. A complete first-placement
  controller achieves it on all five seeds. With a failed/no-op remaining block,
  the next classifier is `not(Scattered(b0, 2))`. Refinement labels states just
  *before* the final task goal becomes true: `ON(2, 1)` holds in only one of these
  five positive states, so it cannot be an exact separator for this dataset.
  In a three-block task, the second placement completes the goal; it need not
  appear as a separately learned intermediate predicate.
- The earliest `ON(1, b0)` cuts occur at observations 25–30, with block 1 still
  approximately 0.186–0.203 m above b0. First Release completes at observations
  50–65. A controller that assumes the first placement is finished fails on all
  five restored cut states, but succeeds on all five restored post-Release
  states. Relation satisfaction alone does not identify the controller's grasp
  state or the completion of a placement.
- Supplied fragments that respect that pending placement (3 instructions before
  the cut, 7 after it) pass both block targets and produce a successful named
  straight-line program on all five full replays. Quotient is called but makes
  no change: the remaining block begins by completing the preceding transfer,
  and its quantified task goal does not become a second standalone placement
  letter under the current structural recognition.
- In a separate isolated quotient call, two complete five-instruction placement
  fragments provide letters `ON(1, b0), ON(2, 1)`. Anti-unification recovers
  `b = b0`, the rebound `b_prime`, and `b = b_prime`; guard learning returns
  `Scattered(b, b_prime)`. However, the fold is rejected by whole-CFG validation:
  for seed 3 its extracted terminal head is observation 95, while the task goal
  first holds at observation 96. At 95, the numeric XY distances are about
  0.014 m (1 to b0), 0.022 m (2 to 1), and 0.036 m (2 to b0), against the 0.025 m
  ON* XY tolerance. Adjacent ON relations therefore precede all-to-b0 alignment
  in this intermediate simulator state. This is a fact about the concrete
  interpretation, not a counterexample to an abstract transitivity axiom.
- Replaying that **rejected** proposed loop separately succeeds on all five
  seeds, choosing blocks 1 then 2. These replays do not override rejection or
  establish an invariant, arbitrary-witness correctness, or motion-model proof.

**Four-block repeat:** Five newly collected recordings at seeds 0–4 all pass
initial/final task validation and have 20 FPS videos. Run the same diagnostic
with `--num-blocks 4`. The additional block does expose the intended second
milestone, but does not remove the boundary problems:

- Actual refinement learns `ON(1, b0)`, then `ON(2, 1)`. The latter holds in all
  seven positive transition witnesses and none of the 239 negative states.
  There are seven positives because the final task predicate can change truth
  value more than once within a trajectory.
- Complete placement controllers fail from the restored intermediate cuts on
  all five seeds. The normal supplied-controller continuation stops at its
  second block with `budget_exhausted` (zero additional refinement rounds) and
  never calls quotient. Proposed fragments carrying the pending transfer across
  cuts (3, 5, and 7 instructions) achieve the second target on only three seeds;
  this control also stops before quotient.
- The separately labeled isolated quotient receives
  `ON(1, b0), ON(2, 1), ON(3, 2)`. Only the first two were learned by refinement;
  the third is derived by quotient's existing terminal-placement recognition.
  It recovers the same carried/rebound roles and `Scattered(b, b_prime)` guard,
  using 15 extracted continuing heads and five exits. Whole-CFG validation
  rejects the fold: seed 1's reconstructed exit is observation 169, while its
  task goal first holds at 171.
- The original demonstrations execute 3, 6, 8, 3, and 3 iterations. Seeds 1 and
  2 require further placements after the initial `1, 2, 3` sequence; the learner's
  three-iteration extraction does not retain those later continuations. On full
  replay, the rejected learned loop passes only seeds 0, 3, and 4. For seeds 1
  and 2 it selects `1, 2, 3, 0, 3` and finishes with the task goal false, whereas
  the demonstration guard excludes the obstructed base during recovery.
- A supplied numeric straight-line program containing exactly three placements
  passes four of five full replays. After conversion to fixed named operands it
  passes three of five. Numeric and named Release have different physical
  stopping tolerances (23), so these are not equivalent controllers. All failed
  full replays reached the goal transiently but failed the final-state check.

Four blocks provide a useful additional intermediate predicate; they do not
force loop synthesis. Any fixed block count can be unrolled, and these recordings
also expose recovery behavior absent from the simplified three-placement chain.
These results are simulator diagnostics, with MCMC supplied and both formal
verification stages unrun.

**Remaining action:** handle pending physical transfers across relational CFG
cuts and preserve the required task-completing continuation when recovering loop
exits. Do not force the desired classifier, bypass validation, or silently change
ON/ON* semantics to make this diagnostic pass. The prior observation-only unit
fixtures did not expose these intermediate-motion and boundary cases.

## 23. Numeric and named Release use different physical stopping tolerances

**Status:** resolved by shared configurable primitive controllers; the earlier
continuation experiments retain their historical results.

**Finding:** Before the controller redesign, `api/instructions.py::Release.eval`
stopped the empty-gripper retreat
within 0.02 m of its target, while `ReleaseByName.eval` used 0.001 m. The other
retreat logic and target offset were the same. Converting numeric operands to
fixed named aliases therefore changed the number of simulator steps and the
physical rollout, not just operand lookup. The four-block experiment in entry 22
passed four of five complete numeric replays but only three of five named ones.
This comparison exposes a physical semantics difference; neither success rate
establishes either controller's correctness.

**Resolution:** `api/control.py` now owns the shared Pick, Move, and Release
controllers. ID and ByName variants only differ in operand lookup. Immutable
`ControlConfig` settings specify position tolerance, gain, and the gripper-state
threshold/margin. The action helper computes a proportional command without an
unused tolerance argument; the controller owns convergence. Defaults are 10 mm
for Pick, 2 mm for Move and Release, and gain 20. The existing 50-step total
budget per instruction is unchanged. Step exhaustion is recorded explicitly,
and demonstration acceptance rejects any unconverged primitive. Control settings
survive ID-to-name conversion and participate in executable fingerprints.

**Evidence:** Numeric and named programs produce identical MuJoCo actions,
observations, and execution events in the parity regression. Unit tests cover
nondefault control preservation, tolerance-dependent stopping, shared Pick
budgets, vertical retreat after opening, runtime diagnostics, and rejection of
unconverged recordings. The full suite passes 259 tests. A trial at 1 mm with gain 10 stalled in some approach
motions near a 1.5 mm residual; reducing tolerance alone was insufficient.

The Stack example also lowers its transfer waypoint from 0.20 m to 0.10 m above
the current top, avoiding the high configurations where the tighter controller
stalled in the tested scenes. Five new four-block recordings, seeds 0–4, all
complete exactly three iterations with selections 1, 2, 3, satisfy the shared
pre/postconditions, and converge in every primitive within at most 20 steps.
All five have 20 FPS videos; video-on/off actions match exactly and observations
agree within 1e-8. The older recordings that required recovery are not accepted
as the intended three-iteration demonstration baseline. No new synthesis or
formal-verification acceptance is claimed from these controller tests.

**100-seed check:** With the same program and thresholds at commit `a840a54`,
collect seeds 0–99 with four blocks, a three-iteration cap, and no video:

```bash
uv run python -m synthesis.entry.collect_demos \
  --program synthesis.examples.stack:build_program --task stack \
  --num-blocks 4 --num-trajectories 100 --seed-start 0 \
  --max-loop-iterations 3 \
  --output-dir demos/stack/4-blocks-100-trajectories-precise
```

The result is **96/100 accepted**, with failures at seeds **38, 46, 73, 85**.
All 100 initial states satisfy the precondition and all select blocks 1, 2, 3
in their first three iterations. The four failures do not satisfy the final task
condition and still have a guard witness at the iteration cap. Every failure
includes a 50-step exhaustion in the third Pick's approach:

| Seed | Final approach error | Additional unconverged primitive |
| --- | --- | --- |
| 38 | 16.65 mm | None |
| 46 | 13.81 mm | None |
| 73 | 11.37 mm | Third lift: 2.42 mm error after 50 steps |
| 85 | 20.17 mm | Third lift: 3.59 mm error after 50 steps |

The Pick threshold is 10 mm; the lift threshold is 2 mm. Failed approach targets
retain a gripper height of approximately 0.675–0.677 m. Error changes by only
0.03–0.15 mm over the final ten recorded observations, suggesting a difficult
approach configuration. All 1,440 primitives in the 96 successful trials
converge within at most 21 steps. The controlled diagnosis below identifies
self-contact and arm extension at these high approach targets.

The collector correctly rejects the complete requested batch and publishes no
accepted `demonstrations.npz`. All 100 trajectories, including the successes,
are retained under `diagnostics/`; `collection.json` records per-seed verdicts
and `validation-summary.json` records the independent iteration, controller,
pre/postcondition, and final-geometry audit. No seeds were replaced, and no
controller settings or step budgets were changed during this check.

**Physical diagnosis:** Replaying each saved trajectory to the third Pick
reproduces its original 50-step approach failure. `Pick` retains the current
gripper z while approaching the next block's x/y; the previous `Release` has
retreated 0.15 m above the second placed block, leaving z approximately 0.675 m.
The controller therefore attempts a high lateral reach before descending.

For seeds 38, 46, and 85, the simulator reports active contacts between
`robot0:upperarm_roll_link` and both head links. Disabling collision masks on
only the two head geoms in diagnostic environments makes the unchanged targets
converge in 4, 5, and 9 steps, respectively. This isolates head/upper-arm contact
as the obstruction in those three replays. Only diagnostic environments had
their collision masks changed.

Seed 73 has no robot self-contact, and removing head collisions leaves its
failure unchanged. Its elbow angle is approximately -0.00349 radians, with the
shoulder-to-wrist distance at 99.9998% of the two links' combined 0.6735 m length.
At the measured shoulder position and hand orientation, reaching the exact
Cartesian target would require a 0.68487 m shoulder-to-wrist distance. This
identifies a nearly straight-arm reach limitation for that configuration, not
an active joint-limit constraint or a proof of global unreachability.

In a separate replay, changing only the approach target z to 0.60 m makes all
four approaches converge within the unchanged 50-step budget and 10 mm threshold:

| Seed | Lower approach steps | Final approach error |
| --- | --- | --- |
| 38 | 4 | 4.54 mm |
| 46 | 5 | 3.26 mm |
| 73 | 8 | 5.55 mm |
| 85 | 9 | 3.91 mm |

These are isolated approach tests, not accepted full demonstrations or a
validated general clearance policy. Extending the original approach budget to
500 steps also eventually reaches the 10 mm threshold (383, 377, 280, and 133
steps, respectively); the apparent stall is very slow progress, not necessarily
permanent immobility. No production controller, budget, or model was changed.
Diagnostic results and replay scripts are saved beside the collection as
`stall-diagnosis.json`, `stall-isolation.json`, `stall-probe.py`, and
`stall-isolation.py`.

**Stack reset region:** The user clarified that reliable primitive skills are
assumed, with head/arm self-collision outside the experiment's scope, and asked
to sample initial blocks closer to the robot base. Stack reset now calls
`environment/stack_reset.py::sample_stack_xy`: block centers have base-relative
X in [0.54, 0.70] m and Y in [-0.20, 0.20] m, with an additional XY radius bound
of 0.70 m. It retains pairwise Scattered separation, 0.10 m initial-gripper
clearance, resting table height, and the b0 binding. Sampling uses the existing
seeded NumPy stream, restarts crowded layouts, and fails after bounded retries
without expanding the region or returning a partial layout. Other task reset
samplers are unchanged. Saved archives still restore their actual initial
states; new sampling bounds apply when recollecting, not when replaying them.

With the same Stack program, tolerances, 50-step primitive budget, and collision
settings, a fresh collection of **all seeds 0–99 passes 100/100**. Each run
satisfies the precondition and final postcondition, selects blocks 1, 2, 3 in
exactly three iterations, and exits the loop normally. All 1,500 primitives
converge, with a maximum of 22 control steps. All 400 initial block centers lie
within the bound; the largest measured radius is 0.699845 m. No failing seeds
were substituted. The accepted archive and independent audit are saved in
`demos/stack/4-blocks-100-trajectories-near-base/` as `demonstrations.npz` and
`validation-summary.json`. Reproduce with the 100-seed command above, changing
only `--output-dir` to that new directory.

Regression checks cover two-, three-, four-, and six-block layouts across
100 seeds, seeded reproducibility, translated base coordinates, bounded
exhaustion without an out-of-bounds fallback, and actual simulator resets.
The full regression suite passes 263 tests.
The 0.70 m bound holds by construction for every returned layout. The 100-demo
result establishes finite four-block execution evidence, not universal robot
reachability at arbitrary heights, orientations, or block counts.

**500-seed check:** At commit `428fe39`, the same four-block program and
controller settings pass **500/500 seeds (0–499)**. An independent archive audit
confirms every initial precondition and final postcondition, exactly three loop
iterations with selections 1, 2, 3, and normal loop exit. All **7,500 primitive
calls converge**, with a maximum of 22 steps against the unchanged 50-step
budget. All 2,000 initial block centers satisfy the 0.70 m horizontal radius
bound; the largest measured radius is 0.699945 m. No seeds were replaced and no
motion settings were adjusted during the experiment.

```bash
uv run python -m synthesis.entry.collect_demos \
  --program synthesis.examples.stack:build_program --task stack \
  --num-blocks 4 --num-trajectories 500 --seed-start 0 \
  --max-loop-iterations 3 \
  --output-dir demos/stack/4-blocks-500-trajectories-near-base
uv run python -m synthesis.entry.collect_demos \
  --program synthesis.examples.stack:build_program --task stack \
  --num-blocks 4 --seeds 0 38 46 73 85 150 250 350 450 499 \
  --max-loop-iterations 3 --save-video \
  --output-dir demos/stack/4-blocks-near-base-10-videos
```

The ten video seeds were chosen before seeing the 500-seed results and include
the four failures from the former reset region. These separate rendered runs
all pass the same three-iteration task validation. Their actions and primitive
step counts match the corresponding batch trajectories exactly; the maximum
observation difference is 2.33e-10. All ten MP4s decode successfully as 500×500
H.264 at 20 FPS, with frame counts matching one initial frame plus the recorded
control steps. This rendering agreement is scoped to those ten runs.

The 500-run directory contains the accepted `demonstrations.npz`,
`collection.json`, independent `validation-summary.json`, source hashes and
commands in `experiment.json`, and the audit script. The video directory retains
its own accepted archive, `video-validation.json`, and an index of all ten clips.
No implementation changed for this experiment; the existing 263-test result
remains the implementation regression baseline.

**Remaining action:** none for the numeric/named mismatch or requested Stack
reset bound. Full learning and verification acceptance (20–22) must use validated
demonstrations and fresh candidate execution under the chosen controller settings.

## 24. The first Stack placement inherits a transient robot state and a grasp offset

**Status:** the 50-step collection preparation and saved-state replay policy
are implemented. Residual grasp offsets remain; this mitigates the diagnosed
first-placement bias without claiming exact centering. The original issue was
within accepted geometric task tolerance, not a contradiction of the earlier
500-seed task result.

**Finding:** Across all 500 four-block traces in the bounded reset region, the
first placed block (yellow, ID 1) finishes toward the robot relative to b0. The
mean final X offsets from b0 are -12.798 mm for block 1, +2.631 mm for block 2,
and +5.962 mm for block 3. Block 1's range is -16.306 to -9.314 mm; every one of
the 500 offsets points toward the robot when projected onto the base direction.
The upper two blocks differ by 3.331 mm in X on average, explaining why they
appear much more closely aligned with each other.

The bias largely develops during transport. Immediately before the first
Release, the mean gripper X error relative to b0 is only +0.552 mm, but the
carried block is -11.307 mm relative to the gripper (and 16.341 mm above its
tracked site). The block is therefore already about 10.756 mm toward the robot
before opening; release/settling adds roughly another 2 mm. `Move` measures the
gripper site in `observation[:3]`, not the held block's center, and has no grasp
offset compensation. A 2 mm Move tolerance consequently does not imply a 2 mm
block placement error. Current ON/ON_star geometry permits lateral differences
below 25 mm per axis, so the task checker accepts these placements.

**Cause and controlled evidence:** The inherited Fetch setup advances only ten
simulator control steps after commanding the initial robot pose, then saves
`initial_state`. Stack reset restores that same robot state for every seed and
changes the block positions without waiting for robot settling. The saved state
still has gripper linear velocity approximately (-4.14, -0.02, +5.86) mm/s,
angular velocity approximately 0.0305 rad/s about Y, and a 0.704-degree
orientation error. The later picks start after substantially more robot settling.

The following diagnostics restore saved initial simulator states; they do not
change the production implementation or the primitive budgets:

- Replaying the original first placement reproduces its recorded observations
  within 1e-8 on seeds 0, 38, 73, and 499.
- Holding the initial gripper position for 50 control steps before that placement
  changes the release-time X offsets as follows. These are isolated first-placement
  experiments, not complete-program acceptance results.

| Seed | Original first-placement offset | After diagnostic settling |
| --- | --- | --- |
| 0 | -12.844 mm | +1.586 mm |
| 38 | -11.913 mm | +1.302 mm |
| 73 | -12.610 mm | +1.392 mm |
| 499 | -15.619 mm | +0.373 mm |

- On seeds 0 and 499, restoring only the blocks' original positions/velocities
  after settling preserves the improvement. Restoring the robot's original
  joint positions/velocities reinstates the original bias to approximately
  0.00012 mm. This isolates the robot state from object settling and elapsed time.
- Picking block 2 first instead gives that block an offset of -14.583 mm (seed 0)
  or -12.863 mm (seed 499): the effect follows the first pick, not its color/ID.
- Opening the gripper before the first approach does not remove the bias.
  Tightening Pick to 2 mm reduces it only partially (roughly 5–6 mm remains).
  Sending an absolute orientation target alone also does not remove it. Merely
  zeroing initial robot velocities worsens the tested first-placement X errors
  to approximately 55–65 mm, despite reported primitive convergence; a consistent
  settled state is needed rather than a velocity-only reset. The
  evidence identifies the initial robot-state transient and uncorrected held-block
  displacement; it does not isolate a single contact/force parameter as the cause.

Data and replay scripts are saved with the 500-seed collection:
`placement-offset-audit.json`, `grasp-isolation.json`,
`first-placement-transient.json`, and `reset-transient-metrics.json`, with their
corresponding Python scripts. The 500-trace audit measures final geometry and
instruction boundaries; the diagnostic scripts restore simulator snapshots and
record their interventions separately from the accepted demonstrations.

**Rendered full-run follow-up:** Seeds 0, 38, 73, and 499 were restored from
those same saved initial states, held at the initial gripper position for 50
control steps, and then executed through the unchanged complete Stack program.
All four pass pre/postcondition validation, finish exactly three iterations with
selections 1, 2, 3, and converge in all primitives within at most 21 steps.
Final yellow-block X offsets from b0, after all three placements, are:

| Seed | Original full run | Full run after settling |
| --- | --- | --- |
| 0 | -12.516 mm | +1.720 mm |
| 38 | -11.600 mm | +1.327 mm |
| 73 | -12.357 mm | +1.872 mm |
| 499 | -15.234 mm | +0.435 mm |

`demos/stack/4-blocks-settled-50-steps-videos/` contains four full videos that
include the settling prefix, four side-by-side comparisons, final-frame
previews, a video index, and `collection.json`. All eight MP4s decode at 20 FPS.
The full videos show the 50 settling steps during the first 2.5 seconds of
playback; the comparisons align DSL program starts and hold the final frames.
The accepted program executions are saved as `demonstrations.npz`; the separate
`settling-prefixes.npz` retains their full-state diagnostic prefixes. The saved
`generate_videos.py` reproduces the intervention and media generation. These
four full executions extend the earlier isolated first-placement evidence;
they do not establish a new default or broad seed acceptance after settling.
Production motion and reset code remain unchanged.

**Ten-step follow-up:** The same four saved initial states and unchanged Stack
program were rerun with only 10 holding steps before execution. All four pass
pre/postcondition validation, finish exactly three iterations, and converge in
all 60 primitive calls (at most 21 steps per call). Final yellow-block X offsets
from b0 are:

| Seed | 10 settling steps | 50 settling steps |
| --- | --- | --- |
| 0 | -5.945 mm | +1.720 mm |
| 38 | -6.221 mm | +1.327 mm |
| 73 | -6.553 mm | +1.872 mm |
| 499 | -7.034 mm | +0.435 mm |

Ten steps roughly halve the original bias, but leave 5.9–7.0 mm toward the
robot. The corresponding 50-step runs have only 0.4–1.9 mm of X displacement.
This is evidence from four seeds, not broad acceptance of either intervention.

The [video index](roboverify/demos/stack/4-blocks-settled-10-steps-videos/README.md)
links four full 10-step videos, four original-versus-10 comparisons, and four
10-versus-50 comparisons. All 12 public MP4s decode at 20 FPS. Full videos include
the 0.5-second settling prefix; comparisons align program starts and hold final
frames. `settling-comparison.json` records the measurements and video metadata;
`demonstrations.npz` saves accepted program executions, and the separately saved
`settling-prefixes.npz` was checked to contain exactly 10 actions per seed.
`generate_videos.py` and `compare_settling_durations.py` reproduce the experiment
and comparisons. Production motion and reset code remain unchanged.

**Saved settled-state replay:** The user selected 50 settling steps and asked
whether collection can omit the settling prefix while preserving its benefit
when a different environment restores the initial state. For seeds 0, 38, 73,
and 499, the full snapshot after exactly 50 holding actions was saved alone in
`initial-states.npz`: S50, the 51st state when counting the reset state as the
first. Each snapshot was checked against both its saved prefix endpoint and the
first snapshot of the earlier 50-step program execution; serialization and
reload preserve every saved field exactly.

A separate Python process constructed a fresh environment per seed, first
resetting with a different seed (original seed plus 10000), then restored S50
using the existing full-state restore API. It ran the unchanged DSL immediately,
without replaying any settling actions. All four executions pass pre/postcondition
validation, select blocks 1, 2, 3, and finish exactly three iterations. All 60
primitive calls converge, with at most 21 steps per call. Compared with the
earlier settled executions, actions and observation-to-action indices match
exactly; the maximum observation difference is 4.657e-10 and the maximum saved
simulator-state difference is 2.457e-10. Final observed block coordinates match
exactly, including yellow's X offsets of +1.720, +1.327, +1.872, and +0.435 mm.
Restoration does not reintroduce the original large first-placement bias in
these four cases.

The [replay video index](roboverify/demos/stack/4-blocks-settled-50-steps-restored/README.md)
links four fresh-environment execution videos and four comparisons against the
earlier settled runs. All eight MP4s decode at 20 FPS; the execution videos and
accepted `demonstrations.npz` start at S50 and contain zero settling actions.
`result.json` records the comparisons, while `check_saved_start.py --prepare`
and `--replay` reproduce the separate-process experiment. This establishes the
saved-state workflow for the tested cases using full snapshots, including robot
positions/velocities, controls, mocap, and solver state; saving observations
alone is insufficient. The production collector/reset default is unchanged.

**Adopted implementation:** Fresh Stack collection now holds the reset gripper
position with the gripper open for exactly 50 control steps before invoking the
DSL recorder. S50 becomes state zero, including all simulator snapshot fields.
Preparation actions/states never enter the archive, loop-entry geometry, or
20 FPS video. Per-trajectory `initialization` metadata records whether execution
started from a fresh reset (50 preparation steps) or a supplied snapshot (zero).
The existing trajectory deadline also bounds preparation. Restoring a supplied
snapshot bypasses both reset and settling, including after motion repair.

The integrated CFG search, segment replay, and candidate tracing already restored
saved snapshots. Inspection found a related defect in standalone MCMC: its CLI
loaded archive observations but scored candidates by regenerating environments
from seeds. Both MCMC implementations now accept a seed-to-snapshot map, threaded
through CEM, scoring, and candidate videos. The standalone CLI supplies each
selected demonstration's first snapshot and records that initialization source
in run configuration. Missing requested snapshots are errors, never a seed-reset
fallback. This changes the physical evaluation start, not the search objective.

**Validation:** All 268 unittests pass, including fresh collection with exactly
50 excluded steps, saved-state replay in a new simulator, both segment reset
modes, verification candidate restarts, MCMC archive/seed selection, video/action
consistency, and original/instrumented MCMC parity with and without snapshots.
The normal collection CLI passes seeds 0, 38, 73, and 499 in three iterations
with all 60 primitive calls converged (maximum 21 steps). Its
[new archive and video index](roboverify/demos/stack/4-blocks-4-trajectories-settled-default/README.md)
retain only program execution: 124, 120, 116, and 114 actions, with exactly one
more frame per 20 FPS video. Fresh-environment replay from that archive matches
all actions exactly and observations within 4.657e-10; yellow's final X offsets
are +1.720, +1.327, +1.872, and +0.435 mm. This is four-seed acceptance for the
adopted default, not a replacement 500-seed result.

**Remaining action:** evaluate held-block feedback or grasp-offset compensation
if tighter centering is required. Settling does not establish perfect primitive
skills or arbitrary-seed acceptance. Recollect older archives to adopt settled
starts; they continue to restore their own saved states without modification.
End-to-end learning acceptance remains open.
