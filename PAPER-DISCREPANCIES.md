# Paper review and decisions

This is the single record of findings from reviewing [POPL2027.pdf](POPL2027.pdf)
against the implementation. It covers §§2–5, Algorithms 1–6, Appendix A/Table 7
and the associated appendix algorithms/proofs; experimental numbers are not
correctness targets. Neither paper nor code automatically wins a disagreement.

Entry numbers 1–34 are stable, including resolved findings. Each entry records
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
| 16 | Paper rules/assumptions; opt-in height consequences implemented for Stack (27). |
| 17, 18 | Paper edits only; code fixes complete. |
| 19 | Resolved integration defect: recorded and candidate imitation features. |
| 20 | Supplied Stack verification passes with the intended learner (32); full synthesis acceptance remains open. |
| 21 | Switchable ID-first policy implemented; full learning acceptance and policy comparison remain open. |
| 22 | Placement-cut and loop-exit limits identified; full loop recovery remains open. |
| 23 | Resolved primitive-controller discrepancy; shared configurable control and execution parity checks. |
| 24 | Settled collection starts and saved-state replay implemented; residual grasp offsets and full learning acceptance remain. |
| 25 | Resolved motion translation defect: normalize negative quantifiers before weakening premises. |
| 26 | Resolved primitive-model discrepancy: Pick follows horizontal approach and vertical descent. |
| 27 | Motion-model fixes implemented; supplied Stack passes with intended inference, equal-height entry and Higher tolerance (32). |
| 28 | Resolved imitation-sampling defect: candidate boundary callbacks no longer add scoring samples. |
| 29 | Intended symbolic inference enforced; alternate learner selection removed. |
| 30 | Vocabulary/attachment limits and exact-height failure diagnosed; the default tolerance removes the spurious bootstrap clause (32). |
| 31 | Equal initial heights added; intended-learner verification passes. |
| 32 | Configurable Higher tolerance aligns saved height comparisons with ideal levels; Scattered differences are diagnosed in 33. |
| 33 | Scattered mismatch diagnosed; further work deferred by user decision because of its low observed frequency. |
| 34 | Section 6.2 fixed-program experiment implemented for Stack; initial-state witness generation and iteration counts made explicit. |

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

**Status:** trace-based inference implemented; supplied Stack verification passes (32), while full synthesis acceptance remains open.

**Decision/reasoning:** The old §4 examples used literal dictionaries, some with
unequal initial/current list lengths that `zip` truncated. They remain only in
`inference_lib/golden_tower_fixtures.py` as historical regression fixtures.
`DemoStore` now supplies aligned execution snapshots with per-invocation entry
geometry. Observing states and learning a candidate do not prove inductiveness
or task success. Tests do not require saved demonstration files.

**Remaining action:** use validated task demonstrations for full-pipeline
acceptance and renewed empirical claims. Stack is the current first task (20).
Collection and saved-state replay do not establish synthesized-program verification.
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
from the old invariant and newly covered states. Standalone CEGIS and both
integrated pipeline modes use `InvInference` → `inference.loop_inference`, with
no alternate learner selection (29). This is the intended algorithm. Monotonicity
is checked on each update rather than assumed from the learner.

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
its binding. It differs from the Put-to-table verification fixture. Algorithm 5's
reach-any-state PostScore cannot replace final-state correctness. The integrated
pipeline requires validated archives and rejects invalid demonstrations before
search; the legacy oracle is not a fallback input.

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
non-strict abstract Higher and `0 <= dz < 1.5L` for direct-on. Its concrete Higher
comparison includes the configurable tolerance documented in entry 32.

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

**Status:** code fixes and opt-in height consequences implemented; paper update pending.

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

**Implemented when required by Stack verification (27):** `--supported-towers`
adds quantified height consequences for all blocks, including unnamed objects:
roots rest at the common table height, all blocks are above it, and distinct
ON*-related blocks differ by at least L in height. These weaker consequences
allow gaps; they suffice for the checked Stack body but are not a complete
encoding of the supported-tower model or a general proof of all Higher rules.
Their preservation is checked at loop boundaries. The two-height premise remains
only a regression fixture. Failed/unknown checks are unsuccessful verification;
no counterexample is silently discarded.

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
identical feature vectors.

## 20. Stack entry conditions and invariant data now agree across both modes

**Status:** supplied Stack verification passes with the intended learner, equal-height entry and Higher tolerance (32); full synthesis acceptance remains open.

**Decision:** Stack starts with unstacked, pairwise-scattered blocks at one height
level (31–32) and ends with all blocks ON* b0. The former integrated True
precondition and the standalone scattered-block precondition were different
tasks. Collection and both pipeline modes use one specification. The public
supplied-program path uses explicit Pick/Move/Release primitives.

Initial invariants are learned from the actual candidate's recorded continuing
heads and normal guard-false exits after synthesis (or supplied-program loading).
This replaces demonstration-partition inference as the integrated bootstrap.
Guard learning still uses the expert demonstrations, and preservation feedback
still uses checked abstract successor replay (7). Frozen entry geometry and
arbitrary-witness verification are unchanged. Physical repairs cause new runtime
traces and renewed inference/verification; collected observations alone never
prove inductiveness or the physical-controller refinement excluded in entry 4.

**Verification scope:** the intended partition learner produces a six-clause
bootstrap invariant that passes both proof stages with the 1 mm Higher tolerance
and explicit supported-tower motion premises (32). The abstract body is
`Put(b_prime, b); Assign(b, b_prime)`. This supplied-program result establishes
partial correctness within that model; it does not establish search acceptance.

**Remaining action:** recover the full loop through search and obtain
`verified_model` on that synthesized candidate with validated settled starts.
The supplied-program invariant supports both proof stages (32).

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
acceptance remains an open comparison.

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
motion verifiers. CLI and resynthesis tests retain the selected policy.

**Remaining action:** demonstrate complete learning acceptance with representative
recordings and compare the two search policies. The early-Get policy's fixed
prefix remains an explicit limitation of the retained relational variant.

## 22. ID-first continuation exposes placement and loop-exit boundary limits

**Status:** continuation and quotient acceptance remain open under the current
collection policy.

**Decision/reasoning:** A geometric relation can become true during a placement,
before the primitive finishes. Refinement cuts must preserve the actual grasp
and controller state at that boundary; a complete placement fragment cannot be
assumed to work from every intermediate state. Extracted loop exits must satisfy
the final task condition, not merely reach it later in the original recording.

`synthesis.experiment.id_first_continuation` replaces MCMC with supplied placement
candidates while retaining simulator execution, predicate refinement and loop
quotienting. It distinguishes automatic continuation, isolated quotient calls,
and replays of rejected candidates. These diagnostics do not establish full
learning acceptance or formal verification. See the
[continuation workflow](roboverify/synthesis/cfg/VERIFICATION.md#manual-continuation-experiment).

**Boundary diagnosis:** complete supplied placements can replay successfully when
concatenated yet fail to resume from the earliest relational cuts, which can occur
before lowering/release finishes. Splitting at those cuts can create fragments
with different instruction shapes that the current quotient cannot fold.
An isolated quotient may also produce a loop that replays successfully while its
extracted demonstration exit precedes the task goal. Such a fold must be rejected.
Neither whole-demo success nor a fitted guard establishes valid segment boundaries
or inductiveness.

**Remaining action:** recover physically compatible repeated fragments and valid
loop exits through search, then verify the returned candidate. Do not move a cut
or accept a rejected fold solely because the complete demonstration later succeeds.

## 23. Numeric and named Release use different physical stopping tolerances

**Status:** resolved by shared configurable primitive controllers.

**Finding:** Numeric and named Release previously used different stopping
criteria. Changing operand representation must preserve the physical rollout.

**Resolution:** `api/control.py` owns shared Pick, Move and Release controllers.
ID and ByName variants differ only in operand lookup. Immutable `ControlConfig`
settings specify position tolerance, gain and gripper-state thresholds. The
action helper computes proportional commands; the controller owns convergence.
Defaults are 10 mm for Pick, 2 mm for Move and Release, and gain 20. Each
instruction retains its 50-step total budget. Exhaustion is recorded explicitly,
and collection rejects any unconverged primitive. Settings survive ID-to-name
conversion and participate in executable fingerprints.

The Stack example transfers at 0.10 m above the current top, then lowers to
0.05 m. Its reset sampler uses base-relative X in [0.54, 0.70] m, Y in
[-0.20, 0.20] m, and horizontal radius at most 0.70 m. It preserves pairwise
Scattered separation and initial gripper clearance of 0.10 m, uses seeded NumPy
sampling, and fails explicitly after bounded retries. The experiment assumes
reliable primitive skills; head/arm self-collision is outside its evaluation
scope. Demonstrations must still complete and satisfy the task conditions.

**Checks:** Generated tests cover numeric/named execution parity, per-instruction
budgets, controller metadata, archive rejection of unconverged traces, sampler
bounds and seeded reproducibility. Entry 24 specifies the settled-state
collection and replay policy.

**Remaining action:** none for the controller mismatch or reset bounds.
End-to-end learning acceptance is separate.

## 24. The first Stack placement inherits a transient robot state and a grasp offset

**Status:** 50-step collection preparation and saved-state replay implemented.

**Decision/reasoning:** Reset restores a robot state with a remaining transient.
The first grasp can inherit a systematic offset. Move controls the gripper site,
not the held block's center, so its convergence tolerance does not imply exact
block centering. Settling addresses the initial transient; it is not grasp-offset
compensation or a physical-controller correctness proof.

Fresh Stack collection holds the reset gripper position with the gripper open
for exactly 50 control steps before invoking the DSL recorder. S50 becomes state
zero, including all simulator snapshot fields. Preparation actions/states do not
enter the demonstration, loop-entry geometry or 20 FPS video. Per-trajectory
`initialization` metadata distinguishes fresh reset (50 preparation steps) from
supplied snapshot (zero). The trajectory deadline also bounds preparation.
Restoring a supplied snapshot bypasses reset and settling, including after repair.

Integrated CFG search, segment replay and candidate tracing restore saved
snapshots. Both standalone MCMC implementations also accept a seed-to-snapshot
map through CEM, scoring and candidate videos. The CLI supplies each selected
demonstration's first snapshot and records that initialization source in run
configuration. Missing requested snapshots are errors, with no seed-reset
fallback. This changes the physical start, not the search objective.

**Remaining action:** evaluate held-block feedback or grasp-offset compensation
if exact centering becomes a requirement. Full synthesis acceptance remains open;
supplied-program symbolic and motion verification passes in the model of entry 32.

## 25. Motion translation rejected the normal guard-false loop exit

**Status:** resolved; mixed-polarity quantifiers remain explicitly unsupported.

**Finding:** The integrated verifier constructs a fresh motion state with
`Not(Exists(witness, guard))` at a loop exit. The assumption translator rejected
all negative quantifiers, so a supplied program could pass symbolic verification
and crash before producing a motion verdict.

**Resolution:** Dualize negative quantifiers before applying the existing
positive-polarity rules: `not forall` becomes `exists not`, and `not exists`
becomes `forall not`. Universal finite instantiation still weakens premises;
existential witnesses remain fresh and may denote unnamed blocks. Quantifiers
under Boolean equality retain their mixed-polarity rejection. The integrated
verifier reports unsupported translations explicitly instead of crashing.
Regression checks cover unnamed counterexample witnesses, named universal
instances and the actual nested Stack guard-false condition.

## 26. The motion model gave Pick a diagonal path absent from its controller

**Status:** resolved within the existing idealized waypoint model.

**Finding:** `PrimitiveController.pick` first moves horizontally above its target
at the current arm height, then descends vertically. The motion verifier encoded
one diagonal segment to the block center. This could invent obstacle intersections
and did not check the path the primitive actually specifies.

**Resolution:** Check both approach and descent segments, retaining explicit
contact permission only for the selected object. Each segment has a distinct
obligation label. Regression scenes distinguish a diagonal-only intersection
from a real low-approach collision. This aligns waypoint semantics; it does not
claim a proof of MuJoCo feedback dynamics.


## 27. Stack motion needed geometric loop invariants

**Status:** geometric verifier fixes implemented; supplied-program verification passes with intended inference, equal-height entry and Higher tolerance (32).

**Diagnosis:** fresh motion states at loop boundaries discarded facts established
by prior iterations. An arbitrary arm could begin inside the tower; loose ON*
geometry admitted floating blocks and fractional height gaps; tight XY alignment
alone admitted offsets that invalidate the exact Scattered WP rule when placement
uses b0 for XY and b for Z. For example, with L=0.05, an unrelated block exactly
0.10 m from b0 can be less than 0.10 m from a slightly offset b. Placing at b0 then
changes its Scattered relation relative to the one inherited from b. This is a
real counterexample to those premises, not evidence that the supplied noiseless
program creates such a state.

**Resolution:** opt-in `--supported-towers` supplies the height consequences in
entry 16. Alongside the learned relational invariant, the CFG verifier checks:

- the empty arm is at least L/2 above every block center;
- ON*-related blocks have identical XY coordinates;
- the supported-height consequences continue to hold.

Each candidate geometric invariant has an entry and a preservation obligation,
quantified over unnamed as well as named objects. Exact columns and clearance are
not assumed at program entry. Stack's scattered entry establishes exact columns
trivially; its ideal root-aligned placement preserves them. Fresh loop-head and
exit contexts carry these facts only as part of a result requiring all their
obligations to pass. Noise or an offset placement can invalidate preservation;
neither is silently treated as exact. Symbolic task pre/postconditions and the
supplied physical program are unchanged.

**Inference (29):** the partition-based implementation in `inference.py` is the
intended symbolic algorithm. All production symbolic inference calls `InvInference`;
its failures must be diagnosed without substituting another learner. Entries 31–32
record the initial-height premise and Higher tolerance used by the successful
supplied-program proof. The default bootstrap passes without refinement.
Geometric regressions use an explicitly labeled structural invariant fixture;
they test solver obligations and make no inference-acceptance claim. The geometric
invariants above remain checked templates, separate from relational learning.

**Nonvacuity:** quantified consistency can time out even when all violation
queries are unsatisfiable. A bounded-domain SAT witness now establishes premise
consistency; every Box quantifier is expanded over that domain with explicit
closure. Finite UNSAT/UNKNOWN proves nothing and falls back to the unbounded
solver. Safety queries retain the unbounded domain. Regression tests reject an
unsafe arm, low transfer and offset placement, and ensure finite consistency
checking cannot restrict safety queries.

**Scalability and diagnostics:** archive members are decompressed once per
trajectory, and predicate evaluation caches syntax rather than state values.
CFG artifacts record structure and segment indices without recursively printing
trajectory payloads. Per-obligation artifacts retain proof scope and countermodels;
reports recognize `verified_model` as successful and show both proof stages.
The Put text formatter now follows its constructor order, `Put(upper, base)`;
the former reversed display did not change the stored operands or WP semantics.
See the [workflow](roboverify/synthesis/cfg/VERIFICATION.md#provided-stack-verification)
for the diagnostic command and current acceptance scope. Full synthesis,
termination and physical-controller refinement are separate claims.


## 28. CFG imitation scoring counted instruction-boundary callbacks as observations

**Status:** resolved implementation defect; full synthesis acceptance remains open.

**Finding:** `execute_current` reports appended observations and instruction-end
states to its callback. The latter retain alias changes made by `Assign` and `Get`
without a simulator step. `segment_rollout` treated every callback as an imitation
sample, whereas demonstration collection records only the initial observation and
observations appended by the program. Even an identical execution therefore had
extra endpoint samples, changing the density and its KL/MMD score. Deduplicating
observation values would also be wrong: a program can record unchanged states.

**Resolution:** the rollout retains boundary scenes for PostScore and predicate
checks, alongside the execution's original observation sequence. Imitation features
use only that sequence. Features, distance scales, convergence thresholds and
physical execution are unchanged. Synthetic coverage retains repeated observations
and binding-only states; a freshly generated settled Stack replay checks complete
feature-sequence parity with collection, without saved demonstration fixtures.


## 29. Symbolic inference must use the intended partition-based algorithm

**Status:** user decision implemented; alternate learner selection removed.

**Decision:** `InvInference` → `inference.loop_inference` is the intended symbolic
invariant inference algorithm, not a legacy fallback. Candidate bootstrap,
recovered-loop inference, integrated preservation refinement and standalone
symbolic CEGIS all use it. The `--learner` flag and Python learner-selection
parameters are removed; passing them is an error. No unsuccessful inference
attempt falls back to observed Boolean patterns.

`MonotoneInvariantLearner` remains an independent utility in
`inference_lib/observed_patterns.py` for other analyses. Symbolic verification
workflows do not import or select it. Coverage, explicit implication/progress
checks, nonvacuity and increasing-size/unbounded verification remain in force.
Routing standalone tests through the intended algorithm exposed an empty-binder
adapter bug: a ground clause is now returned directly instead of constructing
Z3's invalid `ForAll([])`. Feature selection and formula learning are unchanged.

**Acceptance correction:** the previous Stack success with the alternate learner
does not validate the intended inference workflow. Supplied-program and full
synthesis acceptance require both stages to pass using `InvInference`. The current
supplied-program result uses that algorithm with equal-height entry and Higher
tolerance (32); full synthesis remains separate. Geometric unit tests use a
declared structural invariant to retain independent collision, clearance and
alignment coverage; this fixture is never injected into production
inference. Regression tests check the removed CLI/API options and that bootstrap
and counterexample refinement invoke the intended algorithm.


## 30. Stack invariant vocabulary and attachment semantics

**Status:** vocabulary and exact-height failures diagnosed. The corrected entry
premise and default Higher tolerance yield a bootstrap invariant that passes both
verification stages (32). Full synthesis and the attachment/replay model issue
remain separate.

**Finding:** the partition learner selects a minimum separating feature subset
for each predicate partition. Enlarging its vocabulary need not strengthen the
result: a different separator can omit useful correlations. With ON*, Scattered
and equality, Stack inference can retain only pairwise comparability-or-separation,
a clear b, and `ON*(b,x) => ON*(x,b0)`. That invariant permits a second tower.

**Preservation diagnosis:** let a = b = b0 be alone, with c = b_prime above d in
another, separated tower. The invariant and guard hold. The attachment WP uses
`ON_new(x,y) = ON_old(x,y) or (ON_old(x,c) and ON_old(b,y))`: it adds c above a but
retains c above d. After `b := c`, the clause `ON*(b,x) => ON*(x,b0)` demands d above
a and fails. Geometric replay instead detaches c from d and preserves the invariant,
so it cannot supply the failing successor required for invariant weakening. This
explains `no_progress`; a predicate-boundary conversion error can independently
interrupt realization of such a model. Neither result is successful refinement.
The attachment rule and geometric replay remain unchanged in this task.

**Symbolic-only configuration:** `--invariant-relations ON_star equality` uses the
same `InvInference` algorithm and actual candidate heads and normal exits. Its
learned invariant, under the relational axioms, says that every lower member of
a strict ON* pair belongs to the b0 tower, b is clear, and everything beneath b
is on b0. Thus there is one possible nontrivial tower, rooted at b0 and topped by b;
every outside block is a singleton. It excludes the second-tower counterexample.
Establishment, preservation and exit pass both finite checks and unbounded proof.
No handwritten invariant, alternate learner or changed program is used. The
shared entry premise now also records equal initial heights (31). The vocabulary
remains an explicit option, not a new default.
A synthetic-loop-state regression learns the formula and checks these obligations
for sizes 2–6 and an uninterpreted domain; it does not inject a fixture invariant.

**Other failure and remaining actions:** the default vocabulary also includes
Higher, which inferred observed facts such as `forall x: Higher(b,x)` that the
former height-free Stack precondition did not imply. Entry 31 adds the missing
reset assumption and resolves establishment for the diagnosed invariant. The
abstract Higher axioms still admit incomparable pairs in general, but the new
entry premise excludes them, as well as unequal initial heights.

**Exact-comparison diagnosis (`--higher-tolerance 0`):** the learned clause
`Higher(y,x) and Higher(b0,x) => ON*(b,x) or Higher(b0,y)` fails preservation from
three equal-height singletons. After placing one above b0, choose x as the remaining
singleton and y as the new top: both height premises hold and both conclusions
are false. The full learned invariant and guard hold before that placement.
The recorded simulator states can satisfy this clause because b0 moves slightly
downward after stacking, making `Higher(b0,x)` false for untouched blocks. Ideal
symbolic placement preserves b0's height. This is a learned dependency on a small
simulator displacement, distinct from the attachment/replay mismatch above.
The equal-height entry premise added in entry 31 does not repair this preservation
failure on its own. Entry 32 changes the concrete Higher interpretation to
ignore small height differences; learner and placement rules remain unchanged.

The ON*/equality invariant omits separation facts needed by motion collision
checks. The full vocabulary with the default Higher tolerance retains sufficient
separation for both proof stages without refinement (32). The mismatch between
attachment WP and replay outside source-singleton states remains a model limitation.
The [workflow](roboverify/synthesis/cfg/VERIFICATION.md#provided-stack-verification)
contains the current reproduction command and invariant interpretation.


## 31. Stack reset's equal-height assumption belongs in the task precondition

**Status:** user decision implemented; supplied-program symbolic and motion verification pass with the intended learner.

**Decision:** add `forall x,y. Higher(x,y)` to the existing unstacked and
pairwise-Scattered precondition. Higher denotes non-strict height ordering, so
quantification over both ordered pairs requires every block to start at the same
height. Entry 32 subsequently adds a configurable tolerance to the concrete
comparison, interpreting small physical deviations as one level. This premise
holds at the archived, settled beginning of the Stack demonstrations and is
validated by collection and by both integrated pipeline modes. It is an entry
condition, not a global axiom or an invariant imposed on subsequent states.
The postcondition remains `forall x. ON*(x,b0)`.

The standalone Stack verification API now obtains its conditions from the same
`task_spec` function instead of maintaining a duplicate precondition. Archives
collected before this precondition change store the former task identity;
the integrated CLI rejects that mismatch and requires recollection. No
archive-format conversion or relaxed validation is introduced.

**Verification effect:** the equal-height premise establishes the initial height
facts required by the intended learner. It does not by itself remove the
exact-comparison preservation failure diagnosed in entry 30. With the shared
1 mm Higher tolerance (32), the full default vocabulary produces a six-clause
bootstrap invariant that passes unbounded symbolic verification and the noiseless
motion checks under the explicit supported-tower model. The physical program and
learner are unchanged. The [workflow](roboverify/synthesis/cfg/VERIFICATION.md#provided-stack-verification)
contains the current command and learned clauses.

Regressions reject starts whose height spread exceeds the configured tolerance,
even when the final stacking goal holds. They check that entry establishes both
height bounds on b/b0 without vacuity and retain unequal heights in final towers.
Full synthesis acceptance, termination and physical-controller refinement remain
separate claims.


## 32. Higher tolerance for contact-induced height differences

**Status:** user decision implemented; saved Higher tables match ideal Stack
levels, and supplied-program symbolic/motion verification passes without refinement.

**Decision:** use `Higher(x,y) := z(x) >= z(y) - tolerance` for geometry, with a
1 mm default and a configurable threshold below half the 50 mm block length.
Zero retains the exact comparison. Runtime guards, predicate search, invariant
learning, low-level Z3 translation and counterexample realization share the
setting; abstract axioms and placement WP rules are unchanged. The collector
and integrated pipeline expose `--higher-tolerance` and record the chosen value.
Saved coordinates and simulator contacts are unchanged.

**Scope:** tolerance is a reading of approximately discrete height levels.
It is not a transitive order for every continuous scene: heights 0, 0.75 mm and
1.5 mm form a counterexample at 1 mm tolerance. The abstract ordering assumptions
therefore still require an appropriate geometric domain. Exact agreement on
saved loop states is evidence about those states, not universal simulator
refinement. Motion-noise and controller tolerances remain separate.

**Validation:** the saved-state comparison reconstructs ideal heights from the
recorded placement sequence, independently of measured Z coordinates, and checks
all ordered pairs at continuing heads and normal exits. Higher, ON*, frozen ON*
and equality agree on the checked archive; separate Scattered differences occur
when horizontal placement error crosses its 0.10 m separation boundary. The
five candidate executions used for supplied-program inference agree for all five
predicates. No Scattered definition or archive is changed to hide those cases.
Detailed experiment counts belong in generated run artifacts.

With the new default, the intended learner produces a six-clause bootstrap
invariant and both verification stages pass without counterexample refinement.
The [current reproduction command](roboverify/synthesis/cfg/VERIFICATION.md#provided-stack-verification)
and [comparison interface](roboverify/synthesis/inference_lib/README.md#higher-height-tolerance)
describe the supported workflow. Full synthesis and physical-controller
refinement remain separate claims.


## 33. Scattered's sharp XY boundary exposes held-block placement error

**Status:** diagnosed simulator/model mismatch; further work deferred at the
user's request because of its low observed frequency. Focus on other project
directions first. Numeric evaluation and the low-level Z3 predicate agree.
The limitation remains unresolved; retain the investigation and reproduction
artifacts, with no change to predicates, controllers or reset sampling.

**Observation:** saved Stack heads can contain a placed tower block that is not
Scattered from an untouched block, although ideal placement at b0 would keep the
pair Scattered. The diagnosed cases occur at intermediate heads; initial layouts
and final towers agree with the ideal Scattered table. Successful collection and
primitive convergence do not require every intermediate relation to match ideal
placement.

**Cause:** Scattered requires `abs(dx) >= 2L or abs(dy) >= 2L`, with L=0.05 m.
Height is irrelevant. Reset accepts layouts arbitrarily close to that boundary
without an additional placement-error margin. Move controls the gripper site;
its convergence test does not measure the held block's error or compensate its
changing offset from the gripper. Archived instruction boundaries show that the
held block can remain millimetres off-center even when the gripper is much closer
to its target. Most diagnosed crossings already exist before Release; opening
and retreat can add enough displacement to cross the remaining small margins.
The offset changes during lifting and transfer, including in the gripper frame,
so it cannot be treated solely as an initial Pick-centering error or wrist rotation.
The evidence does not isolate the individual contributions of contact compliance,
friction, jaw motion and block rotation.

**Checks:** align only the recorded tower blocks' XY with either initial or current
b0, leaving untouched blocks as recorded: every diagnosed mismatch disappears.
Freezing untouched blocks to their initial XY changes nothing. Restoring the
saved simulator snapshots and reading full-precision site coordinates preserves
all the mismatches; archive rounding is far smaller than the threshold deficits.
The independent tower-membership oracle agrees with the ideal coordinate table.
Detailed pair counts, margins, instruction-boundary decomposition and replayable
analysis scripts belong in the generated `runs/scattered-analysis/` artifacts.

**Verification implication:** these states violate the clause
`Higher(x,y) => Scattered(x,y) or ON_star(x,y)` in the invariant learned from the
five supplied-program executions. The noiseless supported-tower proof assumes
and preserves exact XY columns (27), which exclude these recorded scenes; it
therefore does not establish refinement of the physical controller. The final
stacking postcondition can still hold because ON* permits horizontal offsets.

**Deferred follow-up:** if this direction is revisited, decide how the reliable
primitive abstraction should be realized, for example through block-center
feedback and validated placement/error
margins. A smaller gripper stopping tolerance alone does not bound the held-block
offset. Horizontal separations are continuous, unlike the separated height levels
motivating Higher tolerance; a changed Scattered threshold would change the
separation contract and needs its own justification.

## 34. Section 6.2 does not explain how induction countermodels become initial environments

**Status:** standalone Stack experiment implemented; paper clarification remains.

**Decision/reasoning:** Section 6.2 starts from False and no demonstrations,
requests a smallest counterexample satisfying the environment initial conditions,
and executes the correct program. A preservation countermodel instead satisfies
the invariant and guard at an intermediate state; it need not be reachable from
any legal initial environment. These are distinct objects (see entries 7 and 8).

The standalone `synthesis.entry.learn_invariant` checks the unbounded symbolic
obligations, then directly searches increasing block counts for a valid initial
environment and an execution reaching a selected failed VC. No separate finite
inductiveness check minimizes unreachable countermodels. Establishment targets
the first head outside I; preservation targets I-and-guard followed by a successful
body outside I, retaining the negated VC. Earlier heads do not assume I.

The original ordered generic-coverage query has been replaced by explicit
unrolling with fresh symbolic guard witnesses and shared placement WP rules.
Every enabled binding is eligible, and assignments carry witnesses into later
iterations. Solver choices are saved and physically replayed with guard checks;
ordinary executions retain their lowest-ID policy. A complete trajectory must
reproduce the selected failure at the predicted iteration before supplying data.
Merely encountering some other uncovered state is insufficient. No induction
countermodel or abstract successor is automatically inserted into the dataset.
Minimality is scoped to the initial domain and execution bound; UNKNOWN stops
search, and absent bounded witnesses do not prove unrestricted unreachability.

Generated Stack scenes use the current reset workspace and height assumptions,
settle for 50 steps, and save a full snapshot. Only complete, converged physical
executions satisfying the pre/post transition supply learning states. Normal
exits are included, including the one-block zero-iteration case. This is necessary
for initialization from False; a body-entry-only dataset would remain empty.
The intended partition learner and explicit coverage/enlargement checks are used.
An exit failure requiring strengthening, or failed induction without a reachable
missing state within the search bounds, stops with an explicit diagnosis.

Symbolic-only mode is the Section 6.2 default. Optional motion verification runs
after symbolic success and does not repair the supplied program. The final
symbolic result requires an unbounded proof; a bounded search cutoff is never
reported as success. The new runner leaves the existing demonstration-based
pipeline and abstract-successor refinement API unchanged. Future environments
provide an adapter; this does not imply all paper benchmarks are implemented.

**Remaining action:** clarify the paper's witness construction, normal-exit
sampling, and counting convention. Report verification attempts (including False
and the successful final check), accepted counterexample executions, and learner
updates separately. Paper table counts and exact formula spellings are not
correctness targets. See the [experiment guide](roboverify/synthesis/experiment/invariant_learning/README.md).
