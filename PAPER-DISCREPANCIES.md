# Paper / implementation discrepancies

Places where `POPL2027.pdf` ("Component-Based Synthesis from Demonstrations under Local
Specifications") and the code in `roboverify/` disagree, or where the paper is internally
inconsistent.

These are recorded rather than acted on: the code is being brought to a sound state first,
and the paper will be revised afterwards to incorporate new experiments. Neither side is
automatically the ground truth — each entry says which one looks wrong and why.

Add an entry whenever a discrepancy is found. Cite `file:line` so a claim can be rechecked
without re-deriving it.

---

## 1. Theorem 5.2 contradicts the paper's own Table 7 (`R_Higher`)

**Status:** paper-side. The implementation is faithful; the paper needs the fix.

Theorem 5.2 (Closure under wp) claims that for every abstract action `put(b',b)` /
`put(b',tbl)` and every AFR formula `Q`, `wp(a, Q)` is AFR. Its stated justification is that
the rewrite operators work

> "without introducing any quantifier … each operator only ever combines existing
> quantifier-free atoms with ∧, ∨, and substitution of the action's named objects."

But the paper's own Table 7 defines `R_Higher` for `put(b',b)` with **eight disjuncts, two of
which introduce a fresh quantified variable `t`**:

- disjunct 2: `m≠b' ∧ m≠b ∧ n=b' ∧ ∃t. t≠n ∧ Higher(n,t) ∧ Higher(t,b)`
- disjunct 6: `m=b' ∧ n≠b' ∧ n≠b ∧ (Higher(b,n) ∨ ∀t. …)`

and `R_Higher` for `put(b',tbl)` likewise introduces `∀t. t ≠ tbl ⇒ Higher(t,n)`.

Definition 5.1 admits AFR only as QFR, or a single quantifier block over a **quantifier-free**
matrix. Rewriting a `Higher` atom inside `Q = ∀u,v. ψ` therefore yields `∀u,v. ψ'` whose
matrix contains `∃t` — not AFR. It also creates genuine `∀∃` alternation, which is what
Theorem 5.3 exists to characterise. So the theorem's **statement** fails, not merely its
proof sketch.

The implementation reproduces Table 7 disjunct for disjunct:

- `rewrite_for_put_for_higher` — `roboverify/synthesis/api/program.py:779-811`
  (`Exists([t], …)` at `:785`, `ForAll([t], …)` at `:798`)
- `rewrite_for_put_on_tbl_for_Higher` — `roboverify/synthesis/api/program.py:670`

Scope is narrow: `R_⟨on*⟩` is purely propositional and `R_Scattered` is five quantifier-free
disjuncts, so `Higher` is the only offender, and `:670`, `:785`, `:798` are the only
quantifier introductions anywhere in the wp path.

**Repair options for the paper**, in rough order of least disruption:

1. Weaken Theorem 5.2 to hold modulo the `Higher` rewrite, and handle `Higher` under
   Theorem 5.3's conditional-decidability treatment.
2. Widen Definition 5.1's AFR to permit a non-alternating inner quantifier block, and check
   that Theorem 5.3's polarity argument still goes through.
3. Replace Table 7's `R_Higher` with a genuinely quantifier-free rule, if one exists — this
   would also change the implementation.

Worth settling before the AFR fragment checker is written, since it determines what the
checker should flag.

---

## 2. Rollouts from segment-initial states are required but never addressed

**Status:** paper-side omission; the code needs work regardless.

Algorithm 5 defines

> `PostScore(π, D_v, φ) = Pr_{s₀∼D_v}[π rolled out from s₀ reaches a state s ⊨ φ]`

annotated "fraction of rollouts from demonstration-initial states reaching φ" (§3.4 repeats
it). For a loop body, `D_V(v_body)` comes from `ExtractIterations`, so those segments begin at
loop-entry states. Every block must therefore be scored by rolling out from *its own* segment
start.

The paper never says how the simulator gets into that state. Its only uses of the word
"reset" concern symbolic state in §5 ("no reset occurs between consecutive blocks", "the loop
resets to a fresh σ*"). The cost is treated as free.

On the implementation side, `set_state_from_observation` for the Fetch environment
(`roboverify/synthesis/environment/cee_us_env/fpp_construction_env.py:850`) describes itself
in its first line as "a dummy function to only visualize the object dynamics": it restores
block poses, hard-codes the robot's 16-dim state to a fixed home pose, and zeroes velocities.
The held-object state is not represented at all. This is a consequence of the observation
format — `agent_dim = 10` is Cartesian end-effector data with no arm joint configuration — so
reconstructing from the observation would need IK.

A revision should either state the assumption (demonstrations are replayed, or full simulator
state is recorded) or drop the claim that scoring happens from segment-initial states.


---

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


---

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
nominal block positions, although the physical Release controller lowers a held
block. D1 intentionally preserves this legacy default. The new noise mode perturbs
that nominal held-block position; it does not repair the nominal controller model.
BMC results certify a goal in their encoding, not collision freedom. The separate
motion verifier checks block-block sweeps; neither certifies arm or table-plane
collision freedom. Hardware-wide claims need an explicit refinement argument and
additional geometry/dynamics, not just a bounded-noise flag.

The earlier BMC frame rule also froze blocks resting on a moved support. D2 removes
that assumption by allowing arbitrary disturbance; MotionVerify refuses to certify
support manipulation without a dynamics model. This deliberately changes the Move
encoding even with noise off. An old regression claiming an ungrasped block could
never move now explicitly excludes contact with the carried support.

## 5. The geometric translator conflated initial and current ON_star

**Status:** fixed in Phase D. The old `_translate_expr` mapped both `ON_star` and
`ON_star_zero` to the current coordinates. That could strengthen or contradict a
Reverse invariant that compares two different configurations. The translator now
uses separate frozen `X0/Y0/Z0` functions for `ON_star_zero`, with a regression in
which an initial on-relation holds and the current one does not. This follows the
already-settled initial-state semantics; it introduces no global link axiom.

## 6. Plan correction: an endpoint bounding box is not an equivalent collision check

**Status:** documentation corrected; the proposed fallback was never implemented.
This entry identifies a plan error, not an error in the paper's shared-parameter
collision formula or the code implementing that formula.

- **Paper:** §5.5, p. 29 defines collision using one segment parameter `t` shared
  across X, Y and Z. All three overlap conditions must hold at the same position
  along the straight trajectory.
- **Code:** `encode_collision_at` in
  `roboverify/synthesis/verification_lib/lowlevel_verification_lib.py` retains that
  shared-`t` query for the moving-cube model. No endpoint-box fallback is used.
- **Plan:** D2 originally proposed replacing the query with one axis-aligned box
  enclosing the entire movement and incorrectly treated this as equivalent. The
  original paragraph is now corrected; the Phase D spike did not require a fallback.

For diagonal motion, the enclosing box contains space outside the swept path.
An obstacle there can overlap the box without colliding with the moving cube.
A correctly enclosing box can therefore prove clearance when it is clear, but
overlap alone cannot establish a collision. The retained query is exact for its
straight-moving cube model; this does not establish physical-controller behavior
or resolve the separate Pick-contact issue in discrepancy 18.

## 7. Algorithm 6's invariant progression needs a precise failure state and learner

**Observed during Phase E.** A preservation VC counterexample satisfies the old
invariant at the input of the body. Adding that same state to positive `D_V` cannot
force enlargement; the body successor must violate the invariant. An exit VC
counterexample already satisfies the invariant but violates the postcondition;
weakening the invariant cannot exclude it. Phase E replays the supported symbolic
`Put`/`Assign` body to construct a successor and checks that it is newly uncovered.
It stops explicitly on exit failures instead of cycling on duplicate positives.
This replay is an abstract contract execution, not a MuJoCo trajectory or a proof
about controller settling. Nested loops and CFG trace propagation await Phase F.

The acceptance wording "start at False and always converge" also conflicts with
the decided entry-failure branch: satisfiable entry conditions cannot establish
`False`. Phase E first learns from supplied demonstrations, logging `False` as
iteration zero. With no examples it raises `NeedsResynthesis` as required by the
plan. An establishment failure alone does **not** prove the program is incorrect;
it can also mean that the candidate invariant excludes initial states. The existing
branch decision remains in effect until the program-synthesis integration exists.

The legacy partition/minimization learner has no checked monotonicity contract.
The new default CEGIS learner uses the same Phase C vocabulary and scenes but
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
reported explicitly; an altered model is never silently fed to inference.


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

## 11. Corrected plan/code restriction: loop guards may have multiple witnesses

**Status:** uniqueness restriction removed following the user's semantic correction.
The paper's arbitrary-witness semantics are appropriate here; this entry records
an incorrect plan/implementation restriction, not a paper error.

The paper's §5.3 requires correctness for every witness satisfying the existential
guard. Choosing one object in a demonstration does not make other choices wrong;
symmetric objects can be indistinguishable and equally valid. First-match runtime
selection is permitted because it chooses one of the witnesses covered by the proof.

Previously, guard learning labeled all unselected bindings negative, generated loops
required uniqueness, runtime raised on multiple matches, and a `guard_unique` VC
rejected such guards. Those restrictions could reject valid programs and are removed:

- Learning uses demonstrated bindings as positives. Unselected bindings at a
  continuing head are unlabeled; every binding at a demonstrated exit is negative.
- Runtime selects the first matching binding. No match exits a loop normally;
  standalone Get still requires a witness (entry 9).
- The preservation VC retains an arbitrary guard witness. Searching for a
  refutation can select any matching witness, including one never demonstrated;
  successful verification therefore covers all permitted choices.

Regressions cover indistinguishable alternatives, multi-variable witnesses,
witness-free exits, execution with multiple matches, acceptance when all choices
preserve the invariant, and rejection when an additional permitted choice breaks it.
This corrects the uniqueness restriction; matching demonstrations alone still does
not prove that a learned guard or its body is correct.


## 12. Alignment certification is absent from the implemented motion checks

**Direct algorithm audit, 2026-09-20; unresolved code/plan gap.** Section 5.5,
equations (6)/(7), and Appendix J require root-relative alignment to justify a
bounded geometric interpretation of transitive reachability. Phase D2 specified
local direct-on and frame checks but omitted this obligation. The current motion
checker has no root search or alignment VC.

A synthetic clear scene passes MotionVerify with root x=0, top x=.024 and a new
block placed at x=.048 (block length .05). Both neighboring ON* relations hold,
but ON*(new,root) is false under the implementation's .025 horizontal bound.
Thus local placement plus frame/collision checks does not establish the symbolic
transitive effect. Reconcile root and pairwise bounds before adding the missing
obligation; do not silently substitute the paper's inconsistent thresholds.
See A1 in `AUDIT-popl-alignment.md`.

## 13. Placement summaries assume relational effects their physical contract does not check

**Direct algorithm audit; unresolved bridge and paper specification issue.**
Table 7 rewrites Scattered after Put(a,tbl) to true for every distinct object.
MotionVerify's table contract checks release and table height. In a concrete scene
with a block .075 m away, MotionVerify passes while geometric Scattered is false
(the threshold is .1); the symbolic WP for Scattered(a,b0) is merely a!=b0.
The checker must establish the complete agreed abstract effect, not only height.

There is also an internal paper contradiction: Table 7 makes Scattered(a,tbl)
true for a!=tbl, while Table 6's isolation axiom makes it false. The implementation
currently retains both that table-placement rewrite and table isolation. The
standing decision to isolate tbl is unchanged; this newly identified rewrite
conflict requires correction rather than reopening that decision.

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

Definition 5.4 claims finite universal instantiation is exactly as strong as the
invariant on arbitrary environments. That equivalence is not true in general;
the sound direction of abstraction and quantifier polarity must be justified.

## 15. Termination is not established by the stated VCs; capped execution differs

**Paper claim plus implementation mismatch.** Theorem 5.7 asserts termination
from symbolic invariant and motion obligations without a ranking or progress
premise. Ordinary invariant VCs prove partial correctness, not that the guard
eventually becomes false.

Independently, generated LoopRegions use the maximum observed iteration count
as While.max_iters. Runtime silently breaks at that cap while the symbolic exit
VC assumes the guard is false. This can stop a learned loop prematurely on more
objects. Budget exhaustion must be an explicit incomplete outcome or part of the
verified semantics; it is not a substitute for the paper's missing termination
argument. See A2 in `AUDIT-popl-alignment.md`.


## 16. The Higher rewrite can disagree with geometric placement

**Found while closing audit A1.** Table 7's put-on-block Higher disjunct for
m distinct from source/target and n=source uses Exists(t, t!=n and Higher(n,t)
and Higher(t,target)); it does not depend on m's height. With source, target,
and an unrelated object initially at equal height, the target itself witnesses
that existential. The rewrite predicts the unrelated object is at least as high
as the placed source, although the source has just been raised above the target.

The implementation matches this paper rule, so the standing decision not to
speculatively replace Higher's WP remains intact. Motion verification now checks
ON*, Higher and Scattered outcomes against the actual rewritten formulas and
rejects this mismatch. This can reject physically reasonable controllers whose
claimed abstraction is wrong; that is an explicit abstraction failure, not a
motion proof. Exact quantifiers/premises are retained for these equivalence checks;
solver unknown remains inconclusive. The regression preserves this counterexample.

Audit A1 also adds a root-alignment check with radius L/4, consistent with the
existing pairwise ON* bound L/2, and fixes Scattered's table-isolation rewrite.
The two original geometric audit probes are now rejected. This does not prove
physical settling, arm collision avoidance, or total loop termination.

## 17. Temporal validation when milestones persist

Algorithm 2's last-old-true/first-new-true ordering rejects useful consecutive
milestones whenever the older relation remains true after the next one is
established (for example, placing another block on an existing tower). The CFG
implementation instead requires strictly advancing first establishment times,
starting from the previously assigned cut. It validates every segment's entry,
exit, Get witness and adjacency across the complete CFG before committing a
split or fold. Persistent truth is allowed; zero progress and mismatched cuts
are rejected. Synthetic tests cover both cases. This is an explicit resolution
of the temporal ambiguity, not a claim to implement that literal paper formula.

## 18. Primitive motion formulas do not justify the stated Pick/Release claims

**Direct rereading of §5.5, PDF pp. 27–29.** Formula (5) excludes only the
pre-held object from collision checks. Pick starts with no held object and ends
at its target block's center (up to grasp noise). Taking the collision witness
to be that target and t=1 satisfies all three strict L bounds at zero noise.
The claim that a grasp-noise bound below L avoids this self-collision is false
under the written formula. Intended grasp contact needs an explicit exception
or a different gripper geometry; it cannot be silently counted as collision-free.

Release leaves a supported block in place and havocs an unsupported block's
position, while moving the arm to a release-height offset. This is not the
simulator controller's lowering trajectory (discrepancy 4). A primitive verifier
must state which semantics it checks, preserve unsupported/falling outcomes,
and cannot transfer such a result to hardware without a controller refinement
argument. The current audit work must not silently replace either model.

Appendix G p. 57 also reverses the wording of Validate's rejection test relative
to §3.5 p. 19 ("If this fails" versus "If this holds"). The explicit temporal
validation decision is recorded in discrepancy 17.

### Remediation status after the direct paper rereading

Audit A1 closes discrepancies 12–13's missing checks: placement must satisfy the
complete ON*/Higher/Scattered effect and consistent root alignment. The earlier
passing counterexamples are now regressions that must be rejected. Discrepancy
15's silent loop-cap exit is fixed; termination still is not proved.

For discrepancy 18, the new primitive model explicitly excludes intended
Pick/Release contact with the selected object, models the empty gripper as a
point, and checks supported Release versus an arbitrary unsupported fall. This
is a stated modeling resolution, not a proof of the simulator's controller.
Inspection of `ReleaseByName.eval` shows that it opens the gripper first and then
moves the empty arm vertically; the earlier wording in discrepancy 4 about it
lowering a held block was inaccurate. Simulator settling after opening and the
primitive model's support assumption still need a physical refinement argument.

Algorithm 6 now operates on the actual synthesized CFG. It preserves the
existing documented abstract counterexample-replay decision (discrepancy 7),
checks strict learning progress, and surfaces entry/exit coverage failures as
requests for validated demonstrations. It does not assume that a failing
establishment VC proves the program itself is wrong. Motion repair can change
instruction structure while retaining the complete symbolic program. Success is
reported as `verified_model`, expressly excluding a total-correctness or hardware
claim. See `roboverify/synthesis/cfg/VERIFICATION.md`.
