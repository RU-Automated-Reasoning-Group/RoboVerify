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

Definition 5.4 claims finite universal instantiation is exactly as strong as the
invariant on arbitrary environments. That equivalence is not true in general;
the sound direction of abstraction and quantifier polarity must be justified.

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
argument; invariant and motion VCs establish partial correctness. See A2 in
`PLAN-popl-alignment.md` and the historical audit.


## 16. The Higher rewrite can disagree with geometric placement

**Code corrections implemented; the paper still needs updating.**
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
position, while moving the arm to a release-height offset. The simulator opens
the gripper and then moves the empty arm vertically; its settling dynamics still
require a refinement argument (discrepancy 4). A primitive verifier must state which semantics it checks, preserve unsupported/falling outcomes,
and cannot transfer such a result to hardware without a controller refinement
argument. The current audit work must not silently replace either model.

Appendix G p. 57 also reverses the wording of Validate's rejection test relative
to §3.5 p. 19 ("If this fails" versus "If this holds"). The explicit temporal
validation decision is recorded in discrepancy 17.

### Remediation status after the direct paper rereading

Audit A1 added the complete ON*/Higher/Scattered effect and designated-reference
alignment checks, rejecting the original discrepancy 12–13 counterexamples.
Follow-up implementation now proves root discovery and preserves tight alignment
under the user-declared input-tower assumption
([resolved entry 12](PAPER-RESOLUTIONS.md#12-root-discovery-and-tight-alignment-premises--implemented-with-an-explicit-input-assumption)). The earlier completion
claim preceded those obligations; the new implementation closes them. Discrepancy 15's
silent loop-cap exit is fixed; termination still is not proved.

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
