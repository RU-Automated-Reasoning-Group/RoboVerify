# Paper review and decisions

This is the single record of findings from reviewing [POPL2027.pdf](POPL2027.pdf)
against the implementation. It covers §§2–5, Algorithms 1–6, Appendix A/Table 7
and the associated appendix algorithms/proofs; experimental numbers are not
correctness targets. Neither paper nor code automatically wins a disagreement.

Entry numbers 1–21 are stable, including resolved findings. Each entry records
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
| 21 | Decide and document the object-operand search space and timing of Get insertion. |

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
Five three-block primitive Stack recordings at seeds 0–4 pass initial/final
validation; this is demonstration acceptance, not synthesized-program verification.
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

**Status:** paper/code discrepancy identified; operand-search policy remains unresolved.

**Paper:** Section 2.2 (p. 4) defines object-valued variables, and Fig. 5 (p. 10)
uses bound `b` and `b_prime` in the loop body's physical primitives. Algorithm 5
(p. 17) mutates instructions and operands, then optimizes continuous parameters.
Section 3.4 (p. 17, lines 831–832) describes introducing a typed `Get(True)` for
any object identifier still free in the returned candidate. It does not specify
Python ByName/ID classes or require every search operand to be a numeric ID.

**Implementation:** `entry/synthesize_cfg.py` generates a numeric seed program,
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

**Remaining action:** choose and document the intended operand domain and binding
schedule. If binding is moved after candidate selection, rescore the resulting
executable and retain verification of every allowed witness; a successful numeric
rollout does not establish the behavior of an arbitrary Get binding. If early
binding is retained, explicitly justify the fixed prefix or support searching
bindings as well. Neither paper wording nor current code alone settles the choice.
