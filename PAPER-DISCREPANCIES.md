# Paper review and decisions

This is the review record for [POPL2027.pdf](POPL2027.pdf): §§2–5,
Algorithms 1–6, Appendix A/Table 7, associated proofs, and the Section 6.2
experiment. Neither paper nor code automatically wins a disagreement, and paper
experiment numbers are not correctness targets.

Entry IDs **1–39 and their headings are stable**. Retain decisions, proof
arguments and unresolved limits here; implementation history belongs in Git.
See [project status](README.md#project-status) and the
[verification guide](roboverify/synthesis/cfg/VERIFICATION.md) for current workflows.

## What can be done for the paper now

The following edits can be drafted from the settled decisions without new runs.
Implementation fixes listed below are already complete unless marked open.

| Paper area | Edit now | Entries |
| --- | --- | --- |
| Table 1 and guard semantics | Give executable Get both witness existence and correctness for every allowed choice; permit multiple loop-guard witnesses. | 9, 11 |
| §3.4–3.5 / Algorithm 5 | Explain full-state segment restoration and reset/replay costs. Change “If this holds” to “If this fails” in temporal rejection; define C/Q on the original unsplit segment. Describe early binding and ID-first as implementation search policies. | 2, 17, 21 |
| §4 / Algorithm 6 | Specify execution-derived heads and normal exits, the intended partition learner, successor-based preservation feedback, and checked enlargement. Explain why entry and exit failures require different treatment and why countermodels may be unrealizable. | 3, 7, 8, 20, 29 |
| Table 7 / Appendix A | Replace Higher rules 2/6 with the formulas in entry 16 and state their supported-grid assumptions. Restrict Scattered arguments to physical blocks and keep frozen entry ON* separate from current ON*. | 5, 13, 16 |
| §5.5 / motion claims | State waypoint geometry, intentional Pick/Release contact, Release support, input alignment, and checked geometric loop invariants. Bound the claim to that model; controller refinement is unproved. | 4, 12, 18, 26, 27 |
| §6.2 | Explain bounded initial-state witness search, replay of solver guard choices, physical reproduction of the selected VC failure, normal-exit sampling, and separate counts for verification attempts, executions and learner updates. | 34 |

**Formal claims need correction or additional proof.** Theorem 5.2's
no-new-quantifiers argument does not establish AFR closure; reconcile the AFR
definitions and prove the applicable fragment or qualify the Higher case
(1, 14, 16). Reconcile strict/non-strict Higher and direct-on definitions, and
limit finite-instantiation claims to the supported equisatisfiability argument
(14). Theorem 5.7 should state partial correctness unless a separate termination
argument is supplied (15).

**Full synthesis acceptance is resolved**, as confirmed by the user.

**Recorded supplied-program evidence:** the 2026-10-01 PyEDA recheck passed with
five fresh four-block demonstrations and current controllers. All 12 symbolic
checks (sizes 2–4 and unbounded) and all 63 noiseless motion checks were valid,
with equal-height entry, 1 mm Higher tolerance and explicit supported-tower
premises. The bootstrap needed no refinement or repair (29). The earlier SymPy
proof predates the combined controller update, whose paired simulation evidence
is recorded in entry 37.

**Other empirical work:** compare search policies before claiming an advantage
(21); run and report the Section 6.2 experiment before giving new benchmark counts
(34). The four historical demonstration archives remain deleted; the fresh PyEDA
collection is documented in the verification guide. Physical refinement and the Scattered mismatch remain open;
Scattered follow-up is deferred (4, 33); redundant motion checks are consolidated
with fixed-scene coverage preserved (35).

## 1. Theorem 5.2 contradicts the paper's own Table 7 (`R_Higher`)

**Paper correction.** Definition 5.1 permits one quantifier block over a
quantifier-free matrix, but Table 7 introduces an existential in Higher rule 2
and universals in rule 6 and the table case. Rule 2 can introduce forall/exists
alternation into a universal postcondition, contradicting the theorem's
no-new-quantifiers argument.

Corrected code removes rule 2's existential but retains fresh universal
auxiliaries (16); their effect depends on surrounding formula and polarity.
Revise the closure/decidability claim and proof or qualify Higher. Do not change
valid code merely to force the original theorem.

## 2. Rollouts from segment-initial states are required but never addressed

**Implemented; paper omission.** Algorithm 5/§3.4 needs rollouts from each
demonstration segment's start. Observations omit arm, velocity, control and solver
state. Current archives and `cfg/reset.py` retain full snapshots, actions and
bindings; direct reset and deterministic replay are supported, with replay the
default. Observation-only archives are no longer supported. Solver-generated
scenes use geometric concretization instead.

Document faithful segment starts and reset/replay costs; no reset fix remains.

## 3. The §4 demonstration input was literal data, including truncated datasets

**Resolved: trace inference and full synthesis acceptance.** Historical
literal dictionaries included unequal lists silently truncated by `zip`; they
remain only as regression fixtures. `DemoStore` supplies aligned execution
snapshots and frozen per-invocation entry geometry.

Historical fixtures, collected states and a learned formula alone do not establish
inductiveness or task success.

## 4. Motion proofs depend on a waypoint abstraction, not physical controller dynamics

**Model limit; paper scope correction.** Proofs cover idealized waypoints with
optional bounded error, excluding settling, grasp failure, full arm/finger
geometry and table-plane collisions. An endpoint outside its placement contract
fails even if subsequent simulation could settle it correctly.

Legacy BMC Release freezes nominal block positions while moving the end effector
(optional noise perturbs that location). Primitive motion Release instead proves
support or permits an arbitrary unsupported fall (18). Moving an occupied support
cannot be certified by assuming other blocks remain fixed.

Qualify the paper accordingly. Physical tracking, attachment and dynamics
refinement is outside current scope; see the
[motion guide](roboverify/synthesis/verification_lib/README.md).

## 5. The geometric translator conflated initial and current ON_star

**Resolved.** `ON_star_zero` denotes frozen invocation-entry geometry and
`ON_star` current geometry, using separate coordinate functions. Tasks equate
them in the precondition, never through a global link axiom. Put cannot rewrite
the past; Reverse can therefore relate final order to original order.

## 6. Plan correction: an endpoint bounding box is not an equivalent collision check

**Resolved documentation error.** Retain §5.5's shared trajectory parameter t
across XYZ. An enclosing endpoint box contains space outside a diagonal swept
path: clearance proves safety, but overlap does not establish collision.
The proposed replacement was never implemented.

## 7. Algorithm 6's invariant progression needs a precise failure state and learner

**Implemented; paper clarification.** A preservation counterexample's input
already satisfies I. Refinement adds an uncovered **successor** from checked
abstract-body replay to force enlargement; this is contract replay, not a MuJoCo
trajectory. Weakening cannot exclude an exit state in I that violates the
postcondition. Establishment failure may instead mean missing valid entry states.
Entry/exit coverage failures request validated demonstrations and can trigger
resynthesis when recordings are available.

Record False as iteration zero, then bootstrap from supplied traces. Every
accepted update checks old-invariant implication and coverage of added states;
the partition learner's monotonicity and convergence are not assumed.
All symbolic workflows use `InvInference → inference.loop_inference` (29).

Clarify failure-state and progress semantics. The separate no-demonstration
Section 6.2 experiment learns only from validated physical executions (34).

## 8. A finite relational countermodel need not describe a physical scene

**Conversion implemented; paper clarification.** Abstract Higher can leave two
blocks incomparable, unlike real heights. `verification_lib/counterexamples.py`
solves for non-overlapping geometry preserving ON*, frozen ON*, Higher, Scattered,
aliases and table identity, then checks the numeric round trip.

Explain that unrealizable or timed-out models are reported explicitly, never
silently converted into training scenes. A canonical tower drawing is insufficient.

## 9. Executable Get needs a witness-existence obligation

**Implemented; paper clarification.** Get fails if no witness exists; a
havoc/assume encoding can validate an impossible binding vacuously. Its WP is
`Exists(x, G) AND ForAll(x, G => Q)`. Distinguish executable binding from a
blocking assume in Table 1.

## 10. The historical Unstack oracle does not establish its final task condition

**Historical oracle limitation.** The historical
`cfg/demo_sources.py:unstack_oracle` moves the selected top onto the current block,
unlike the Put-to-table verification fixture. Reach-any-state PostScore cannot
replace final-state correctness. The pipeline rejects invalid demonstrations and
does not use the legacy oracle as fallback.

A separate user-supplied candidate was tested on 2026-10-15 with dynamically
recomputed maximum-X placement and maximum-Z retreat, ignoring its supplied
invariant. On five seeds each, only 1/5 two-block, 0/5 three-block and 0/5
four-block executions passed demonstration validation. Logical final-state
success was 5/5, 4/5 and 4/5 respectively, but every rejected trial had an
unconverged primitive. The transport instruction adds another 0.15 m above the
already lifted block; exact-state replays of four failed moves still failed
with 500-step budgets. Ideal top-removal tests terminate for sizes 2–10, while
the one-block guard loops forever. This is a distinct candidate and empirical
controller diagnosis, not a correction to the historical oracle or a new formal
verification result. See the [program, semantics, counts and saved traces](roboverify/synthesis/cfg/VERIFICATION.md#user-supplied-unstack-groundtruth-diagnostic).

**Right-to-left revision, 2026-10-15.** Initializing the tower on the higher-Y
side of the base in a bounded forward workspace and placing blocks at successive
`min(Y)-0.15 m` targets passes all 15 trials (five seeds each at 2–4 blocks).
All task pre/postconditions hold, every primitive converges, and each execution
takes N−1 iterations. The current revision holds the gripper Z reached by the lift
throughout the sideways transfer. Repeating all 15 trials from exactly the same
initial scenes still passes, with at most 1.221 mm gripper height deviation across
30 transfers (2.481 mm for the carried block). The preceding lateral version also
passed but retained an unwanted extra 0.15 m rise. Its initial workspace and
direction changed together, so those effects were not separated. The diagnostic
runner and videos are recorded
[in the verification guide](roboverify/synthesis/cfg/VERIFICATION.md#right-to-left-unstack-revision).
This establishes empirical validity for the tested scenes, not a formal proof;
production extrema support and Unstack collector integration remain separate work.

**Environment consolidation, 2026-10-23.** The right-side tower reset is now the
only production Unstack reset. The lateral runner uses that shared factory,
with no instance-specific reset override. Reverse was unchanged by that
consolidation; its subsequent lateral reset/program is recorded in entry 39.
Archived simulator states still restore exactly. This promotes the environment
layout, without changing the experimental program or its verification scope.

Keep end-to-end Unstack invocations within **60 seconds**; exhausted budgets and
inconclusive proofs are unsuccessful results.

## 11. Corrected plan/code restriction: loop guards may have multiple witnesses

**Resolved.** The paper's arbitrary-witness semantics are appropriate. Demonstrated
bindings are positives; unselected bindings at continuing heads are unlabeled;
all bindings at demonstrated exits are negatives. Runtime selects the first match
in ascending physical ID order and exits if none exists.

Preservation covers every satisfying choice, including undemonstrated choices.
No uniqueness restriction remains. Standalone Get still requires existence (9).

## 12. Root discovery and tight alignment premises — implemented with an explicit input assumption

**Implemented under an explicit input-tower assumption.** ON* alone does not
establish the tight root-relative alignment required by §5.5 and Appendix J.

For `P = wp(remaining symbolic body, postcondition)`, select an in-scope physical
root r by proving `P => ForAll(u, ON*(target,u) => ON*(u,r))`. Reflexivity,
antisymmetry and satisfiable P identify the bottom root; entry must also imply P.
A named root need not exist. Quantification covers unnamed objects; a name hint
or one concrete witness assignment is insufficient.

For each horizontal coordinate F, the alignment induction is:

1. Require `|F(a)-F(r)| <= delta_F` for every existing member a, with
   `2*delta_F <= N_F`. Singletons satisfy this; existing towers require the input
   assumption.
2. Preserve the root and old members' bounds and prove the same bound for new x.
3. The triangle inequality gives every pair distance at most
   `2*delta_F <= N_F`; the root-relative invariant survives further insertions.

Code uses strict delta=L/4 and N=L/2. `alignment_entry` checks the destination,
`alignment` checks the new member including noise, frame VCs preserve other
objects, and support checks reject moving occupied supports. Input alignment
applies only to proved input roots, never placement outputs. Missing roots,
inconsistent inputs and unknown proofs prevent certification. Removing members
preserves bounds if the reference stays; moving/changing it or merging towers
requires re-establishment.

Loose pairwise bounds do not suffice: with delta=1, N=2 and bottom-to-top X
coordinates `[0,-1,-2,-1,0]`, a new block at +1 is close to root and top but
distance 3 from -2. This violates the tight input premise, not the induction.
Vertical support, collision and relation effects remain separate obligations.

## 13. Paper clarification: Scattered ranges over physical blocks

**Implemented; paper domain correction.** Put(a,tbl) must establish Scattered
from every other physical block, including unnamed objects. Table 6 isolates tbl,
but Table 7 omits that restriction. State the physical-block domain or add
`m != tbl` and `n != tbl` to its Scattered rewrite. Physical table height is
separate; motion checks already prove the complete abstract effects.

## 14. AFR and predicate definitions are inconsistent across the paper

**Paper formalization open.** §5.1 calls an existential followed by a universal
AFR, unlike Definition 5.1's single block. Its universal-only learner description
also conflicts with existential classifiers (§3.2.1) and guards (§3.3).

Higher is non-strict in §2.2/Appendix A but strict in Definition 5.4. Direct-on
vertical bands differ across §2.2, Definition 5.4 and Figure 12. Code uses
non-strict abstract Higher and `0 <= dz < 1.5L` for direct-on; concrete Higher
has the tolerance in entry 32.

Reconcile these definitions. Finite instantiation should preserve the verification
result through equisatisfiability for the applicable universal collision queries:
instantiate all relevant axioms/invariants over named objects and an arbitrary
collision witness, which is constrained by those instances. State the fragment
and polarity conditions. This does not prove whole-state equivalence in larger
environments or apply to arbitrary quantified formulas; finite instantiation
alone is not an established implementation defect.

## 15. Termination is not established by the stated VCs; runtime cap mismatch fixed

**Runtime fixed; paper claim open.** Theorem 5.7 lacks a ranking/progress premise;
invariant and motion VCs establish partial correctness. State that scope or add
a termination argument.

Generated loops no longer inherit demonstration-count caps. An explicit budget
raises `LoopBudgetExceeded` if a guard witness remains, and collection reports
incomplete execution rather than normal exit. Total-termination implementation
is outside scope.

## 16. The Higher rewrite can disagree with geometric placement

**Code fixed; paper rules/assumptions pending.** The agreed abstraction uses
uniform upright blocks of height L, a common flat table, exact support, and
complete towers without missing layers on a common L-spaced height grid.
For Put(a,b), with c distinct from a,b:

```text
Rule 2: Higher'(c,a) = Higher(c,b) AND NOT Higher(b,c)

Rule 6: Higher'(a,c) = Higher(b,c) OR
          (Higher(c,b) AND
           ForAll(t, (Higher(c,t) AND NOT Higher(t,c)) => Higher(b,t)))
```

Rule 2 expresses `z(c) >= z(b)+L` as `z(c) > z(b)` on the grid. Keep both
conjuncts: abstract Higher need not be total, and the positive one excludes tbl.
The old existential ignored c's height. Rule 6 tests strictly lower levels, not
distinct identities; equal-height peers do not obstruct it. A complete support
chain provides an intermediate level when c is at least two levels above b,
refuting the universal. Together the disjuncts express `z(c) <= z(b)+L`.
Table rules 3/4 remain valid under these assumptions.

Update Table 7 rules 2/6, make clipped rule 6 readable, and state the assumptions.
Fresh auxiliary allocation and capture-safe quantifier rewriting are fixed
separately; a program object named t must never be captured.

Opt-in `--supported-towers` adds weaker consequences: roots at table height,
all blocks above it, and distinct ON*-related blocks separated vertically by
at least L, with preservation checked at loop boundaries. These permit gaps
and suffice for the checked Stack body (27); they are not a full encoding or
general proof of all Higher rules. Failed/unknown checks remain unsuccessful.

The [Lean formalization](roboverify/proofs/wp_soundness/README.md) now proves
these Higher identities under an explicit complete-chain model and includes a
checked missing-level counterexample. It does not derive completeness from the
weaker motion premises or verify the Python-to-Lean translation.

## 17. Temporal validation acceptance and the incorrect rejection sentence

**Code complete; paper edits only.** For refinement
`P -> v1 --C--> v2 -> Q`, compare `first(C) <= last(Q)` on each original
unsplit segment. Q is the outgoing target, not incoming P. P holds initially,
C's first occurrence is strictly interior, and Q holds at the end; these checks
already imply the inequality. Equality alone cannot override the interior cut.
P and C need not persist, and no overlap is required.

In §3.5 p. 19 replace “If this holds” with “If this fails,” as Appendix G does.
Define C/Q, the original-segment domain and the entry precondition explicitly.
Whole-CFG boundary, binding and adjacency checks remain in force.

## 18. Primitive motion formulas do not justify the stated Pick/Release claims

**Code complete; paper edits only within the model.** Formula (5) excludes only
the pre-held object. Pick starts empty and ends at its target: choosing that
target as collision witness at t=1 and zero noise gives differences 0 < L.
Small grasp noise cannot justify the written collision claim.

State intentional contact with the selected object, the point model for an
empty gripper, cube payloads, and checks against other objects. Release proves
support or fails and permits an arbitrary falling position; simulation opens
before arm retreat. Remove the grasp-noise argument and avoid claims about
settling or controller refinement (4).

## 19. Recorded and candidate CFG trajectories used different imitation features

**Resolved implementation defect.** Candidate Scenes dropped the five gripper
features retained by demonstrations, causing a 14-versus-9-dimensional comparison
for three blocks. Observation-derived Scenes now retain the source observation;
both sides use shared indices. Synthetic scenes retain their geometric features.
The objective and distance scale are unchanged.

## 20. Stack entry conditions and invariant data now agree across both modes

**Resolved: shared Stack conditions, inference data and full synthesis acceptance.**
Collection, standalone verification and both pipeline modes share one task:
unstacked, pairwise-Scattered blocks at one height level, ending with all blocks ON* b0.

Integrated invariants come from the actual candidate's continuing heads and
normal exits; physical repairs require fresh traces. Expert demonstrations remain
guard-learning/search targets, and preservation feedback uses checked abstract
successors (7). The supplied primitive program's abstract body is
`Put(b_prime,b); Assign(b,b_prime)`.

The intended learner's six-clause bootstrap passed both proof stages under
entries 27 and 31–32. That archived supplied-program proof predates the combined
controller update (37).

## 21. Free-object binding is performed before search instead of on the returned candidate

**Switchable policies implemented; comparison open.** §3.4 introduces typed
Get(True) for identifiers still free in a returned candidate. The default
`relational` implementation instead closes objects before search, freezes the
Get prefix, and mutates only in-scope operands. This changes available bindings
and potentially rollouts; a fresh Get(True) need not select the numeric seed's
object. ByName classes themselves are consistent with the paper's variable DSL.

The user-selected `id-first` alternative searches/refines numerically, then
quotients relational repetitions and searches folded bodies with ByName operands
over all extracted iterations. Matching numeric code is only an optional seed.
Residual IDs become fixed entry aliases; these preserve identity without claiming
relational generalization. Inference and both verifiers receive named programs.

Describe these as implementation policies, not the paper's uniquely prescribed
schedule. Compare policies before making comparative performance claims. The
default policy's fixed binding prefix remains a limit.

## 22. ID-first continuation exposes placement and loop-exit boundary limits

**Physical-shape restriction removed; boundary limitations documented.**
Algorithm 4 matches relational encodings, not instruction lists. Different
instruction counts or classes no longer reject a fold; an aligned physical
fragment can seed subsequent named body search.

Relational conditions can become true before lowering/release finishes.
Complete placement fragments may therefore fail from the extracted physical
heads despite successful concatenated replay. Exits must satisfy their outgoing
condition at the extracted boundary; later whole-demo success is insufficient.
For a loop with a continuation, validation additionally requires the exit to be
the first recorded state satisfying the negated guard. That temporal restriction
can reject otherwise matching endpoints.

Quotienting compares CFG fragments; it does not split repetitions inside one
already successful straight-line block. The
[continuation diagnostics](roboverify/synthesis/cfg/VERIFICATION.md#manual-continuation-experiment)
use supplied candidates to diagnose these boundary and granularity limitations.

Matching relational encodings still requires compatible physical heads and exits.
Do not move cuts or accept rejected folds solely because a complete demonstration
succeeds.

## 23. Numeric and named Release use different physical stopping tolerances

**Resolved.** ID/ByName primitives share `api/control.py` and differ only in
operand lookup. Immutable settings survive conversion and enter executable
fingerprints. Current defaults are 2 mm Pick/Move/Release tolerance, gain 20 and
50-step instruction budgets; collection rejects exhaustion.

Stack reset is bounded to base-relative X in [0.54, 0.70] m, Y in [-0.20, 0.20] m
and XY radius at most 0.70 m, retaining Scattered separation and 0.10 m initial
gripper clearance. Bounded retries fail explicitly. Settling and later controller
decisions are in entries 24 and 36–37; no mismatch/reset fix remains.

## 24. The first Stack placement inherits a transient robot state and a grasp offset

**Settling and replay implemented; grasp-offset limitation remains.** Fresh Stack
collection holds the initial gripper position open for 50 steps before recording.
That full settled snapshot becomes state zero for actions, frozen geometry and
video. Preparation is excluded and remains within the trajectory deadline.

Supplied snapshots bypass reset and further settling. CFG execution and both
standalone MCMC implementations restore archived starts through scoring, CEM and
video; missing requested snapshots are errors, not seed-reset fallbacks.

Settling addresses the initial transient. Move controls the gripper site, so
convergence does not imply exact held-block centering. Evaluate block feedback or
offset compensation only if that stronger requirement is pursued (33).

## 25. Motion translation rejected the normal guard-false loop exit

**Resolved; mixed-polarity quantifiers unsupported.** Negative quantifiers are
dualized before translation: not-forall becomes exists-not, and not-exists
becomes forall-not. This supports the normal guard-false exit. Universal finite
instantiation still weakens premises; fresh existential witnesses may be unnamed.
Quantifiers under Boolean equality remain unsupported and are reported explicitly
rather than crashing.

## 26. The motion model gave Pick a diagonal path absent from its controller

**Resolved within the waypoint model.** Pick is checked as a horizontal approach
above the target followed by vertical descent, with separate obligations.
A single diagonal segment could invent collisions and miss actual path hazards.
Contact is permitted only with the selected object. This aligns specified
waypoints without proving feedback dynamics.

## 27. Stack motion needed geometric loop invariants

**Verifier fixes implemented; supplied verification passed.** Fresh loop contexts
lost arm clearance; loose ON* admitted unsupported heights; tight but nonzero XY
offsets could invalidate Scattered inheritance when placement used b0's XY and
b's Z. At Scattered's sharp 0.10 m boundary, offset b and b0 need not have the
same separation from a third block.

Alongside the learned invariant, opt-in supported-tower verification checks:

- empty-arm clearance of at least L/2 above every block center;
- identical XY coordinates for ON*-related blocks;
- the supported-height consequences from entry 16.

Entry and preservation obligations cover named and unnamed objects. Scattered
entry establishes exact columns trivially; ideal root-aligned placement preserves
them. Noise/offset placements can fail preservation. These are checked geometric
templates, separate from the intended relational learner.

**Translated-premises-only ablation, 2026-10-01.** Removing all extra motion
premises, tight input alignment, the initial arm pose, and exact quantified
premise reassertions leaves the learned Higher invariant unchanged and all 12
symbolic checks valid. Motion then refutes three empty-gripper approach checks
and the input-alignment check; all carried-block collision and predicate-effect
checks pass. Six auxiliary geometric conclusions also fail as diagnostics.
Validated finite scenes satisfy the complete learned invariant and guard while
exhibiting an unsafe gripper approach or an excessive tower offset. A Higher-free
structural control additionally loses horizontal carried-block collision safety.
These are model-query failures, not new simulator failures. The other vocabulary
controls also fail motion; one ON*-only query is unknown at its time budget.
See the [full premise ablation](roboverify/synthesis/cfg/VERIFICATION.md#removing-all-extra-motion-premises).

The ordinary supported-tower proof is not quantifier-free: it supplies quantified
geometric facts and exact relational premises. Removing those premises makes
every captured motion premise quantifier-free, but the exact Higher-effect and
auxiliary geometric conclusions still contain quantifiers, as do separate
abstract root-selection proofs. A fully quantifier-free motion pipeline remains
unestablished; this diagnostic does not change production verification.

A finite-domain SAT witness can establish consistency of quantified premises.
Finite UNSAT/UNKNOWN instead falls back to the unbounded solver; safety queries
remain unbounded. This optimization cannot turn vacuous premises or bounded
safety checks into certification. See the
[proof configuration](roboverify/synthesis/cfg/VERIFICATION.md#provided-stack-verification).

## 28. CFG imitation scoring counted instruction-boundary callbacks as observations

**Resolved.** Callbacks include binding-only instruction boundaries as well as
recorded observations. Counting all callbacks added samples absent from the
demonstrations and changed KL/MMD scores. Rollouts now use the original observation
sequence for imitation, retaining boundary scenes for PostScore/predicate checks.
Equal-valued observations remain valid samples; deduplication would be incorrect.

## 29. Symbolic inference must use the intended partition-based algorithm

**User decision implemented.** All production symbolic inference uses
`InvInference → inference.loop_inference`; learner-selection flags/APIs and
fallbacks are removed. The independent observed-pattern utility is not selected
by verification workflows. Coverage, enlargement and nonvacuity checks remain.
The optional `--invariant-minimizer pyeda` uses Espresso instead of SymPy for
Boolean minimization within this same algorithm. Both preserve all truth-table
values, including the existing completion policy for unobserved rows; this is
not an alternate learner. SymPy remains the default.

The 2026-10-01 recheck used PyEDA with five fresh four-block Stack demonstrations,
learned from 20 candidate runtime states, and returned `verified_model`: all 12
symbolic and 63 noiseless motion checks passed without refinement or repair.
See the [reproducible configuration and run](roboverify/synthesis/cfg/VERIFICATION.md#provided-stack-verification).

Earlier success with an alternate learner did not validate intended inference.
The current supplied-program result does (31–32); structural geometric test
fixtures are not injected into production learning.

## 30. Stack invariant vocabulary and attachment semantics

**Default bootstrap passes; attachment/replay model limit remains.** A larger
vocabulary need not strengthen the partition learner's minimum separating
features. ON*/Scattered/equality can admit a second tower.

For a singleton a=b=b0 and c=b_prime above d in another tower, attachment WP
adds c above a while retaining c above d:
`ON_new(x,y) = ON_old(x,y) or (ON_old(x,c) and ON_old(b,y))`.
After b:=c, `ON*(b,x) => ON*(x,b0)` fails for d. Geometric replay detaches c
from d instead, so it cannot provide that failing successor for enlargement.
The resulting no-progress diagnosis is not successful refinement. This mismatch
outside source-singleton states remains unresolved.

The agreed public Put contract requires both operands to be distinct, clear
physical blocks before execution. The [Lean formalization](roboverify/proofs/wp_soundness/README.md)
proves the guarded general-relocation WP and its deletion/attachment
decomposition under that contract, and checks a counterexample to using the
additive rule on a supported source. The production dispatcher is unchanged;
implementing and establishing correspondence to the proved rule remains open.

**Higher-removal ablation, 2026-10-01.** On the five current four-block
demonstrations, ON*/Scattered/equality learned the clear-top, root and
column-separation clauses but omitted the single-tower restriction. Removing
Higher from both inference and verification reproduced a three-block preservation
counterexample and five refuted motion checks; ON*/equality alone passed symbolic
checks but lacked XY separation for three lift-collision checks. A diagnostic
invariant adding `forall x,y. ON*(x,y) => x=y or ON*(y,b0)` to the three learned
clauses passed all 12 symbolic and all 62 remaining motion checks without any
Higher axioms, entry clause or effect obligation. The common-table supported-height
premises remained; removing those too refuted transport collision and arm-clearance
checks. Higher is therefore not indispensable for this Stack proof, but the current
learner did not recover the necessary replacement in that vocabulary. The structural
control is explicitly supplied, not a new learner or production fallback; the
attachment rule remains unchanged. See the [ablation results and reproduction](roboverify/synthesis/cfg/VERIFICATION.md#higher-removal-ablation).

ON*/equality alone can learn a one-tower invariant and pass symbolic verification,
but lacks motion separation facts. Under exact height comparison, the full
vocabulary can instead learn a clause reliant on slight downward displacement
of b0 after placement:
`Higher(y,x) and Higher(b0,x) => ON*(b,x) or Higher(b0,y)`.
Ideal placement violates it by taking x as a remaining singleton and y as the
new top. Equal-height entry fixes establishment, not that preservation failure;
the shared tolerance removes the spurious clause (31–32).

## 31. Stack reset's equal-height assumption belongs in the task precondition

**User decision implemented.** Add `forall x,y. Higher(x,y)` to unstacked,
pairwise-Scattered entry. Non-strict ordering over both pairs requires one
height level, allowing the concrete tolerance in entry 32. This is an entry
condition, not a global axiom or a later-state invariant. The goal remains
`forall x. ON*(x,b0)`.

Collection and verification share this task specification. Older task identities
are rejected and require recollection. Equal-height entry establishes the learned
initial height facts; with 1 mm tolerance the intended bootstrap passes both
proof stages. Final towers can have unequal heights.

## 32. Higher tolerance for contact-induced height differences

**User decision implemented; supplied bootstrap passes.** Geometry uses
`Higher(x,y) := z(x) >= z(y) - tolerance`, default 1 mm, configurable in
[0, 0.025) m; zero restores exact comparison. Runtime, inference, predicate search,
geometric translation and counterexample realization share the setting.
Abstract axioms, WP rules and saved coordinates are unchanged.

Tolerance represents approximately discrete levels, not a transitive order over
arbitrary continuous scenes: heights 0, 0.75 and 1.5 mm give a counterexample at
1 mm tolerance. The abstract order still requires a suitable separated-level
domain. Controller tolerances and motion noise are separate.

Saved-state comparisons independently reconstruct ideal heights from placements.
Higher, ON*, frozen ON* and equality matched the checked archive; Scattered had
separate differences (33). The five candidate executions used for the supplied
proof matched all five predicates. The intended six-clause bootstrap passed
symbolic and noiseless supported-tower motion checks without refinement.
See the [tolerance guide](roboverify/synthesis/inference_lib/README.md#higher-height-tolerance);
these checks do not establish universal simulator refinement.

## 33. Scattered's sharp XY boundary exposes held-block placement error

**Unresolved model mismatch; deferred by user decision because of its low observed
frequency.** Scattered requires `abs(dx) >= 2L or abs(dy) >= 2L`, with L=0.05 m
and no height condition.
Initial layouts can lie arbitrarily close to that threshold. Millimetre-scale
held-block offsets can cross it even when the gripper converges; most diagnosed
crossings precede Release, which can add displacement.

Aligning only tower blocks' XY with b0 removes the diagnosed mismatches.
Freezing untouched blocks does not; full-precision replay excludes archive
rounding. Numeric and Z3 predicates agree. Intermediate heads can disagree with
ideal placement while initial/final relations and the final stacking goal hold.
Detailed analysis remains in `runs/scattered-analysis/` artifacts.

These states violate `Higher(x,y) => Scattered(x,y) or ON_star(x,y)`; the exact
columns in the noiseless proof exclude them. Tighter gripper stopping alone does
not bound attachment error. If revisited, assess block feedback and justified
placement margins; changing Scattered's threshold changes its contract and needs
separate justification. No predicate or sampler change is currently requested.

## 34. Section 6.2 does not explain how induction countermodels become initial environments

**Standalone Stack experiment implemented; paper clarification remains.**
An induction countermodel is an intermediate state, not necessarily reachable
from a legal initial environment. Starting from False and no data,
`synthesis.entry.learn_invariant` checks unbounded VCs, then searches increasing
block counts for a valid initial state and bounded execution reaching a selected
failed VC. Earlier heads do not assume I. There is no separate minimization of
unreachable finite induction countermodels.

Establishment targets the first head outside I. Preservation targets an I-and-guard
head followed by a successful body outside I, retaining the negated VC. Shared
placement WP and fresh guard witnesses permit every enabled binding. Solver
choices are replayed and checked physically; ordinary execution uses lowest-ID
selection. Minimality is relative to the initial domain and execution bound;
UNKNOWN stops, and bounded absence does not establish unreachability.

Generated scenes settle for 50 steps and save full starts. Only complete,
converged executions satisfying pre/postconditions and reproducing the selected
failure at the predicted iteration supply data. Include normal exits, especially
zero-iteration exits needed when starting from False. No abstract successor is
injected. The intended learner must cover cumulative states and strictly enlarge I.
Exit failures needing strengthening and missing reachable witnesses stop explicitly.

Describe these semantics and count verification attempts (including False and
the final check), accepted executions and learner updates separately. Final
success requires an unbounded symbolic proof; optional motion checks run afterward
on the unchanged program. Stack support does not imply all paper benchmarks.
See the [experiment guide](roboverify/synthesis/experiment/invariant_learning/README.md).

## 35. Named motion checks overlap the arbitrary-object witness

**Redundant VCs consolidated at user request.** An unrestricted arbitrary
block `sym` may alias any named block, so its collision/Move-support obligation
covers separate named cases. But the concrete-scene API can fix sym's position;
then it is not an unrestricted witness and removing named checks loses coverage.

Ordinary proofs now generate one `_sym` VC per collision sweep or Move-support
check. When explicit current/frozen positions or alias bindings restrict `sym`,
one `_scene` query combines violations for all represented obstacles. Operand
coordinates, alias updates, contact exclusions, disturbance semantics and
placement/effect obligations are retained. This covers fixed-scene tests and
motion-counterexample penalties during CEGIS/CFG repair; those penalties do not
replace unrestricted verification or execute the simulator.

The Higher-retained supported Stack recheck passes 12 symbolic and 36 motion
checks (formerly 63). With extra motion premises removed, 28 motion checks pass,
the input-alignment and single gripper-approach check fail, and six auxiliary
geometry diagnostics fail. That run's symbolic exit check is unknown; the
supported run proves the same byte-identical invariant. See the
[recheck evidence](roboverify/synthesis/cfg/VERIFICATION.md#consolidating-redundant-obstacle-checks).

## 36. Component-wise saturation bends the physical gripper path

**Uniform XYZ scaling adopted.** Component clipping of proportional XYZ actions
changed directions for unequal multi-axis errors. The shared helper and active
CEE-US Fetch backend now divide XYZ by `max(1, max(abs(XYZ)))`, independently of
the finger command; repeated scaling is idempotent. Finger bounds remain, and
executable fingerprints include `uniform-xyz-v1`.

In 100 paired four-block Stack starts, both variants completed valid executions.
Uniform scaling eliminated measured command-direction distortion and reduced
approach/transfer maximum sampled bend to 2.52 mm; Release outliers remained.
Those historical results used 10 mm Pick, head contacts and Z-only Release,
superseded by entry 37. Detailed phase distributions and plots remain in
`runs/controller-paths/` and `runs/controller-drift/` artifacts and Git history.

This supports the controller choice for the sampled program, not exact
straightness or certified tracking/attachment bounds (4).

## 37. Release drift and Pick stopping tolerance after uniform scaling

**Approved controller/model changes implemented.** Release freezes pre-opening
XY and corrects XYZ during retreat, stopping on 3D error. Pick now defaults to
2 mm in numeric/named forms; gain 20, uniform scaling and 50-step budgets remain.
The active CEE-US Fetch model disables both head collision masks, retaining
appearance and mass. Physical head/arm clearance is not proved.

The former Z-only Release accumulated lateral tracking error; upper-arm/head
contact caused the largest outliers. Paired interventions and a Pick tolerance
sweep motivated the changes. Tighter Pick stopping did not materially improve
payload alignment.

Combined validation covered **200 paired starts / 400 complete four-block
programs**, all valid with converged primitives; updated maximum budget use was
24/50 steps. Maximum sampled Release XY deviation fell from 11.816 to 1.463 mm;
all-phase gripper deviation reached 2.500 mm. These are 40 ms boundary samples,
not certified bounds. Final tower alignment did not materially improve.
The recorded 350-test regression suite passed.

See the [controller measurement report](roboverify/synthesis/experiment/CONTROLLER-PATHS.md)
for metric definitions, full results, commands and artifacts. The four local
demonstration archives were deleted; reports/videos remain. No replacement
collection or fresh demonstration-consuming proof run accompanied this update.
Physical refinement still requires justified tracking and attachment bounds.

The 2026-10-23 [Reverse follow-up](roboverify/synthesis/experiment/TOWER-WORKSPACE.md)
reproduced an 11.18–12.30 mm bottom-two-block offset: the first transferred block
and later blocks acquire different gripper-relative offsets. Five paired
executions with diagnostic block XY feedback reduced this to 1.35–1.49 mm at
the same fixed destination; tighter Pick tolerance and block XYZ feedback each
failed their unchanged step budgets. These remain experiments, not production
controller changes or a physical-refinement proof. The same investigation
validated 200 complete Unstack/Reverse executions in a larger source rectangle,
while confirming initial gripper overlap for some centerline starts. The user
adopted base-relative X in [0.54, 0.70] m and Y in [0.15, 0.25] m for both tasks;
all 30 fresh production-factory executions at 2–4 blocks passed. The user
explicitly deferred payload-aware placement because it is a stronger primitive
and requires reviewing motion contracts and verification assumptions. Historical
Unstack motion results retain their original, narrower input region.

## 38. Unsat-core minimization can consume the budget after proving a VC

**Performance defect addressed; proof obligations preserved.** Symbolic solvers
previously enabled `smt.core.minimize` unconditionally. A saved Stack exit VC
returned UNSAT in 4–8 ms, but retrieving a minimized core consumed the entire
1 s or 5 s budget in a stage-timing diagnostic. Without minimization, retrieval
took less than 0.1 ms; premise consistency still returned SAT. Whole-VC times of
approximately 300 s therefore do not establish that proof search needed 300 s.

Minimization now defaults off, with explicit CLI on/off switches and a scoped
Python option. Core extraction still supports the vacuity shortcut, and a core
containing the negated conclusion still requires a separate premise check.
Valid, invalid, vacuous and unknown remain distinct; no assumption was removed.

Ten concurrent fresh-process Higher-retained experiments, each with a 300-second
symbolic query limit, passed all 12 symbolic VCs: 120 valid results and no
unknowns. The invariant was byte-identical across runs. Unbounded exit VCs took
0.013–1.583 s (median 0.022 s). The translated-only motion failures remain.
This is evidence of improved runtime and repeatability, not a guarantee of
deciding every quantified VC or an isolation of every earlier unknown's cause.
See [configuration and evidence](roboverify/synthesis/cfg/VERIFICATION.md#unsat-core-minimization).

## 39. The historical Reverse verifier encodes Unstack rather than reversal

**Historical implementation discrepancy; obsolete verifier removed in the collection audit (42).**
The former `entry/verify_reverse_with_learned_invariant.py` used `Put(b_prime,tbl)` and an
unstacked/equal-height/scattered postcondition. It does not establish a tower
whose order is the reverse of frozen entry ON*. That historical entry point
also explicitly reported unsupported motion. It was removed rather than left
as a misleading verifier for the new Reverse program.

At the user's request, Reverse now resets its source tower 0.54–0.70 m ahead of
the robot base and 0.15–0.25 m toward higher Y, sharing Unstack's expanded source
workspace. `examples/reverse.py` starts
with `b <- tbl` and handles all N transfers in one loop, including the first
block and the original base last. After each placement it assigns `b <- b_prime`.
Ordinary Move parameters choose the destination relative to the physical table
frame; the groundtruth supplies (+0.45, -0.15) m XY offsets, exposed for later
optimization. The physical table reference uses Z = surface minus half a block,
so one +0.05 m placement offset works for both table and block supports.
This controller reference does not change the isolated relational tbl marker.
Missing table geometry fails explicitly, and the motion verifier reports these
new table-coordinate operands as unsupported. The existing 50-step budgets and
2 mm tolerances remain.

`cfg.tasks.task_spec("reverse")` requires the final ON* relation to equal the
transpose of frozen entry ON* on physical blocks. The shared collector now
supports this task. All 15 fresh single-loop executions passed on seeds 0–4 at
2–4 blocks, including primitive convergence, N loop iterations, normal exit and
final-state validation. All 54 focused controller, task execution, collection,
replay and motion tests passed. Table-relative semantics enter fingerprints for
programs using table references; earlier first-block-outside-loop archives remain
historical artifacts.
This is empirical evidence, not a symbolic/motion proof or controller refinement.
Connecting the correct Reverse abstraction, invariant learning and motion
contracts to a verifier remains open. See the
[program, collection and evidence](roboverify/synthesis/inference_lib/README.md#collect-reverse-demonstrations).

## 40. Grid and Pyramid execution and verification have separate support

**Environment, runtime and collection gaps addressed; synthesis and verification deferred.**
The 2026-10-23 review compared Fig. 10, Appendix B/Table 8 and Figs. 17–18 in
[POPL2027.pdf](POPL2027.pdf) with the implementation. The user clarified that
only goals have traversal pointers, then scoped implementation to environments
and execution of the supplied program. Physical blocks have no traversal links.

Grid takes `grid_rows` and `grid_cols` (default 3 by 2). Pyramid now takes
`bottom_row_blocks=k`, creates layers k,k−1,...,1 and allocates k(k+1)/2 physical
blocks and goals: 3 gives six blocks; 4 gives ten. Positive integer dimensions
are required. Layouts that exceed the nominal workspace or cannot fit separated
source blocks fail explicitly. Configuration is not a reachability guarantee
for arbitrary sizes; the execution checks below cover bottom rows 1–4.

The shared [reference program](roboverify/synthesis/examples/goal_layout.py)
implements the identical Grid/Pyramid programs in Figs. 17–18 (which the paper
labels synthesized programs):

```text
d <- h0
while d != null:
    r <- d
    while r != null and exists physical block b: not AtGoal(b):
        Move(b, b, r, (0, 0, 0.15))
        Pick(b)
        Move(b, b, r, (0, 0, 0.15))
        Move(r, r, r, (0, 0, 0.15))
        Move(r, r, r, (0, 0, 0))
        Release(r, 0.15)
        r <- r.next_r
    d <- d.next_c
```

`d` and `r` here are goal cursors. The immutable goal graph exposes `r` within
rows and `d` between row heads, corresponding to `next_r` and `next_c` in the
paper. Other goals have no `d` successor. Pyramid starts with its bottom row,
then advances physically upward in Z. Grid rows lie in XY. Move's operands
supply X, Y and Z respectively; goal coordinates denote desired block centers,
so placement uses zero additional Z offset.

`GoalAssign`, `MoveRight` and `MoveDown` now execute through `Program.eval` and
its ordinary nested `While` interpreter. Goal references/bindings are separate
from physical block IDs. Named Move/Release resolve either kind of coordinate
reference; Pick continues to require a physical block. Runtime Z3 guards
support Goal equality/null and geometric AtGoal/HasBlock. Block selection
retains the existing lowest-ID policy. The controllers, 2 mm tolerances,
50-step primitive budgets and six-motion sequence remain unchanged; the
payload-feedback skill deferred in entry 37 was not adopted.

Shared 20 mm occupancy semantics replace the former block-i-to-goal-i success
check: `AtGoal(b)` means b occupies some goal, and `HasBlock(g)` means some
block occupies g. Goal regions must be disjoint. Reward, success and runtime
guards use these semantics. Goal markers remain noncolliding visual references.
Reset clears runtime aliases, and simulator state restoration reinstalls goal
coordinates and markers from the saved goal vector.

**Execution defects found and addressed:**

- Robot action encoding formerly assumed one mocap body and failed on the
  additional goal-marker mocaps. Actions now contain entries for every mocap,
  with robot commands applied only to the robot's mocap. Markers stay fixed.
- Grid's former 65 mm spacing let the fingers displace already placed blocks
  when approaching later rows. Grid now uses 100 mm spacing. Pyramid retains
  65 mm horizontal spacing and 50 mm layer height.
- Far source positions prevented a ten-block Pyramid approach from converging
  at the higher goal-relative clearance. The source region is now centered
  100 mm closer to the base in X than the initial gripper, on the lower-Y side.
  The scattered fallback uses 105 mm separation without per-block jitter,
  checks its bounds and clearance, and refuses invalid layouts. The previous
  independently jittered fallback could violate Scattered.

**Run the supplied programs**, from `roboverify/` with the simulator environment
in AGENTS.md configured:

```bash
uv run python -m synthesis.entry.execute_goal_program --task grid --save-video
uv run python -m synthesis.entry.execute_goal_program --task pyramid --bottom-row-blocks 3 --save-video
uv run python -m synthesis.entry.execute_goal_program --task pyramid --bottom-row-blocks 4 --save-video
```

Use `--grid-rows 2 --grid-cols 3` to change the Grid. The Python factories are
`make_roboverify_grid_env(grid_rows=..., grid_cols=...)` and
`make_roboverify_pyramid_env(bottom_row_blocks=...)`; both examples expose
`build_program`. The execution entry point settles the initial scene for
50 steps, then runs the DSL program, rejects unconverged primitives, checks
each placement and preservation of earlier occupied goals, and validates final
coverage. `--seed`, `--max-loop-iterations` and `--timeout-seconds` control runs.
Reports and optional MP4s live under `runs/goal-execution/<run>/`; the detailed
`artifacts/execution.json` records actual placements, goal coordinates and loop
exits. These are execution reports, not synthesis demonstration archives.

**Validation:** 63 focused tests passed across goal runtime/layout execution,
existing controllers, guard replay, quantifier hygiene, collection, snapshot
replay and tower task conditions. Simulator regression executions passed for
default Grid and bottom-row-3/bottom-row-4 Pyramids on seeds 0 and 1. Additional
seed-2 executions passed for a 2×3 Grid and bottom-row-1/bottom-row-2 Pyramids.
Recorded seed-2 executions passed for default Grid and the ten-block Pyramid:

- [Grid video](roboverify/runs/goal-execution/20261023-121327-54185cf-grid-seed2/artifacts/execution.mp4)
- [4+3+2+1 Pyramid video](roboverify/runs/goal-execution/20261023-121208-54185cf-pyramid-seed2/artifacts/execution.mp4)

**Groundtruth collection added at the user's request.** The shared
`collect_demos` entry point now supports Grid/Pyramid and selects their provided
factories by default. Layout options determine block counts. The loader opts into
nested goal programs only for collection, retaining the synthesis/verification
restriction. Goal fingerprints include cursor operations, Z3 guards, goal-center
reference semantics and occupancy tolerance. Existing tower fingerprints remain
unchanged.

Archives extend the current version-2 format with a per-trace immutable goal
layout and per-snapshot goal bindings. Nested events retain cursor bindings,
invocation IDs and frozen entry indices. Snapshot restoration and action replay
restore both sorts of bindings, including logical event changes between actions;
wrong row layouts fail explicitly. The numeric validator checks scattered/empty
entry conditions, converged primitives, actual released-payload occupancy,
preservation of previously occupied goals, and final coverage. Release's operand
is a height reference and is not treated as a placement-XY specification.
Existing tower archives remain readable. The block-only inference adapter rejects
goal archives explicitly.

All 15 requested batch demonstrations passed on seeds 0–4, with videos:

- `demos/grid/3x2-6-blocks-5-trajectories/`
- `demos/pyramid/3-bottom-row-6-blocks-5-trajectories/`
- `demos/pyramid/4-bottom-row-10-blocks-5-trajectories/`

Maximum final goal errors were 9.57, 8.51 and 10.16 mm respectively; no primitive
used more than 21 of its 50 steps. The requested 3×3 Grid also passed on seed 0:
nine transfers, 54 converged primitives, at most 9.05 mm final goal error and
18 primitive steps. Its [video](roboverify/demos/grid/3x3-9-blocks-1-trajectories/videos/seed_0000.mp4)
and `demonstrations.npz` are in `demos/grid/3x3-9-blocks-1-trajectories/`.
All 77 focused collection/replay, runtime/controller, task and existing inference
tests passed. See the [collection commands and archive contract](roboverify/synthesis/inference_lib/README.md#collect-grid-and-pyramid-demonstrations).

**Deferred scope:** `entry/verify_2d_with_learned_invariant.py` remains a separate
symbolic mark-all example using synthetic states. It neither selects blocks nor
certifies the executable above. The synthesis predicate language, partition
learner inputs, nested quotient recovery and motion contracts have not gained
goal-task integration. In particular, Pyramid support
across two lower blocks is not modeled by the single-column tower contracts.
The canonical goal diagram still draws vertical links in every column, unlike
the row-head-only nested axioms; this illustration mismatch is unresolved.

**Paper/model clarifications retained for future verification:** Table 8's Grid
row quantifies `g` but writes `reach(h0,x)`; its Grid/Pyramid Scattered clauses
omit the distinct-block restriction. The displayed `reach` definition omits
closure stars: immediate-successor composition cannot describe all visited
goals. Specify closure-inclusive row traversal, including h0. Verification also
needs a sufficient-block premise; otherwise the inner guard can stop with goals
unfilled. The existing partial-correctness VCs alone do not establish termination
(15). No synthesis or formal verification result is claimed for these executions.

## 41. Restack needs a terminating first loop and an executable lateral layout

**Paper corrections and execution support; synthesis/verification not claimed.**
The 2026-10-23 review compared Fig. 10d, Appendix B/Table 8 and Fig. 16 with the
existing `RoboVerifyPartialStack` environment. That historical environment
randomly partitioned blocks over the whole workspace; it did not provide the
requested lateral layout or a collector-ready Restack program.

**Paper issues:**

- Fig. 16's first guard selects a clear block ON* b0. Because ON* is reflexive,
  b0 continues to satisfy the guard after it has been placed on the table.
  Literal execution therefore keeps moving b0 and never reaches the second loop.
- Table 8 requires distinct ON*-related blocks to be Scattered, contradicting
  the intended initial towers. The executable collection contract instead
  requires each pair of physical blocks to be comparable by ON* or Scattered;
  its postcondition is that every physical block is ON* b0. The isolated tbl
  marker is excluded from these physical quantifiers.
- The printed invariant I2 contains a free `x`. No printed invariant is supplied
  to the executor and no verification result is inferred from the paper.

**Executable adaptation:** [`examples/restack.py`](roboverify/synthesis/examples/restack.py)
retains the two flat loops and the six primitive transfers of Fig. 16. A cursor
`p <- tbl` records each newly parked block. The first guard additionally excludes
`p`, and each iteration ends with `p <- b_prime`. This transfers the blocks above
b0 and then b0 exactly once; when p becomes b0, the guard becomes false.
The second phase starts `b <- b0` and repeatedly transfers a clear physical
block other than b onto the destination, then assigns `b <- b_prime`.
It can dismantle a remaining tower below b0 while rebuilding on the new base.

For the requested geometry, first-phase placement uses
`Move(tbl, p, tbl, (+0.45, -0.12, +0.05))`: table-relative X fixes a column,
previous placement Y advances a separated temporary row, and table Z supplies
the resting height. The first p is tbl. The last temporary placement is b0 and
becomes the new tower base. Stack placement uses b0 for XY and b for support Z.
The paper's dynamic Xm/Zm references are replaced with these ordinary reference
operands and fixed table-relative clearance `N*0.05 + 0.10` m above its support
reference. The table-placement +0.05 m follows the existing support convention;
the paper's zero offset would put the target below the surface under that
convention. These are explicit waypoint adaptations, not a literal transcription.
All parameters remain trainable. Controllers, tolerances and instruction budgets
are unchanged; payload-aware feedback remains deferred.

`RoboVerifyRestack` and `make_roboverify_restack_env` supply one randomized source
tower with b0 at a random interior level (at the top for a two-block tower), plus
loose blocks. Total count defaults to four and initial height to max(2, N-1).
The source tower uses base-relative X [0.54, 0.70] m and Y [0.15, 0.25] m;
loose blocks may extend closer in X to 0.40 m. Every source remains on the
higher-Y side. Distinct columns have at least 105 mm separation along an axis,
with 100 mm clearance from the initial gripper. Bounded sampling rejects layouts
that cannot fit; this region accommodates at most three source columns.
There is no 2–4-block restriction in the program, but arbitrary counts/heights
are not a reachability guarantee. The historical PartialStack environment, factory, standalone verifier and
inference option were subsequently removed at the user's request; Restack is
the only current task for this behavior.

`collect_demos --task restack` defaults to the new reference program, accepts
`--initial-tower-blocks`, settles for 50 steps, and records both loop traces,
aliases, actions and full simulator snapshots. Each accepted trajectory must
finish with converged primitives and satisfy its pre/postcondition. Saved starts
and action replay restore exactly without settling again. This uses the shared
collector and ordinary Program/While execution, without a runtime monkey patch.

**Validation:** six Restack regression tests and 36 shared tests passed. Tests
cover bounded/randomized resets, both phases and termination, the designated
base postcondition, snapshot/action replay, and rejected controller or budget
failures. Four-block, three-block-source runs passed on seeds 0–4. Five-block
runs passed on seeds 0–4 for both a three-block source plus two loose blocks and
a four-block source plus one loose block, including both interior b0 levels.
All 15 recorded executions completed a physical tower on b0, with every primitive
converged (maximum 18/50 steps). The largest final XY distance from b0 was
15.48 mm. This measures ordinary grasp/placement offset, not feedback correction.

Archives, per-seed reports and videos are under:

- `roboverify/demos/restack/4-blocks-5-trajectories/`
- `roboverify/demos/restack/5-blocks-3-tower-5-trajectories/`
- `roboverify/demos/restack/5-blocks-4-tower-5-trajectories/`

The aggregate diagnostic is `roboverify/demos/restack/validation.json`.
These are empirical execution results, not a symbolic or motion proof.

## 42. Cross-environment collection coverage is broader than synthesis and verification

**Shared collection gaps addressed; proof boundaries retained.** The 2026-10-23
audit found that Unstack still required an experiment-local execution wrapper,
while the shared collector supported the other five paper tasks. The existing
lateral program is now `examples/unstack.py`. The controller resolves minimum-Y,
maximum-Z and instruction-entry gripper coordinates through typed observation
references. A Move freezes its target before its servo loop; this preserves the
original horizontal transfer. Two- and four-block seed-0 executions were compared
against saved pre-migration executions: every action and observation was identical.
No payload-feedback skill, tolerance change or relaxed step limit was introduced.
The reference semantics enter fingerprints. Collection explicitly opts in, while
the normal synthesis/verification loader rejects these unencoded references.

`collect_demos --task TASK` now selects a provided groundtruth for all six tasks.
The old `unstack_lateral` module retains only compatibility imports and a CLI
forwarder; its separate runner and monkey-patched controllers were removed.
The user explicitly retired PartialStack, so its environment, factory, reset
helpers, standalone verifier, inference option and unused golden fixture were
removed. The figure helper was migrated to Restack. The obsolete Reverse
verifier was also removed: it encoded Unstack rather than the current reversal
specification (entry 39). Useful controller, workspace, invariant and search
diagnostics remain, with their purpose and limitations listed in the
[experiment inventory](roboverify/synthesis/experiment/README.md).

**Pyramid defect found by seed expansion:** the original bottom-row-4 run failed
on seed 12 at the highest-goal approach to the last source block. At the far,
lower-Y corner, the approach used all 50 steps and remained 5.714 mm from its
target, above the unchanged 2 mm tolerance; the following Pick also exhausted
its budget. The source layout's center is now 140 mm closer in X than the initial
gripper, instead of 100 mm. Grid remains at 100 mm. The goal geometry and supplied
program are unchanged. All original seeds 0–14 passed after this 40 mm workspace
correction. The original failed archive and a replay video remain under
`demos/pipeline-audit-20261023/pyramid/`; the replay differs from recorded
observations by at most 2.33e-10. A seed-12 regression covers this approach.

**105 accepted demonstrations:** 15 seeds each for four-block Stack, Unstack
and Reverse, five-block Restack with initial tower heights three and four,
3×3 Grid, and bottom-row-4 Pyramid. The reusable `synthesis.entry.audit_demos`
checks the exact requested seed set, current task/program identities, validation,
snapshot restoration, full action replay, physical final success and success
after 50 additional settling steps. Every final replay error was zero; every
primitive converged, with a maximum 24 of 50 steps across all tasks. Unstack
collection and audit invocations each ran under a 55-second external timeout.
The aggregate manifest is `demos/pipeline-audit-20261023/summary.json`.

The 82 focused tests passed across reference semantics, archive auditing,
collection/replay, controllers, task removal, inference entry points, goal
layouts, Pyramid seed 12, Restack and Reverse. These are execution results and
regressions. No new symbolic or motion verification result, and no full synthesis
run, is claimed by this audit. The integrated CLI still exposes Stack/Unstack;
current Unstack runtime references, Reverse/Restack integration and goal-aware
synthesis/verification remain open. Historical proof evidence retains its
recorded executable, premises and workspace scope.
