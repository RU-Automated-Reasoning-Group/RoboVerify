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

## 6. The planned swept-AABB fallback is conservative, not equivalent

**Status:** plan correction; no fallback was needed in the Phase D spike.

`encode_collision` uses one shared segment parameter across all axes. A per-axis
endpoint bounding box can contain points that are never close to the segment at
any single parameter value (notably on diagonal paths). Replacing the swept-cube
query with this bounding box can safely overapproximate collisions, but can also
introduce spurious counterexamples. The exact bilinear query is retained. This is
a limitation of the plan's proposed fallback, not a reason to silently change the
collision predicate or report a bounding-box result as exact.
