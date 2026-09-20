# Align RoboVerify with the POPL submission

> **Status — read this first.**
>
> Branch: `phase-d-motion-obligations`; Phases A–C are integrated into `main`.
> This plan is being executed top to bottom, Phase A first. Progress:
>
> **Phase A: all seven items done.** Listed in plan order; commits landed in a
> different order.
>
> - [x] **A.1 / A.2** Motion verification now reports what it examined:
>       `lowlevel_verification` fails closed when no loop was checked, and
>       `start_verification`'s blanket skip became a three-way inert /
>       motion-unhandled / unknown classification. Commit `97a196d`.
> - [x] **A.3** `pdb.set_trace()` → `InferenceDataError` / `SeparationInfeasible`,
>       plus a message on `Parameter.update`'s bare `ValueError`. Commit `3d6635f`.
> - [x] **A.4** Solver timeouts added so `unknown` is reachable rather than a hang;
>       `check_tautology` keeps the clause instead of asserting, and
>       `highlevel_verification` now distinguishes *refuted* from *inconclusive*.
>       Commit `9c77f09`. **Deviation from the plan as written:** A.4 specified a
>       tri-state return for `check_tautology`, but both call sites test
>       `if not check_tautology(...)`, so `False` already means "keep the clause" —
>       the conservative answer. A third value would have changed both callers
>       without changing behaviour, so the boolean contract was kept and the
>       inconclusive case surfaced as a warning instead.
> - [x] **A.5** RNG isolation — `preserved_global_rng` in `synthesis/mcmc/synthesis.py`.
>       The frozen-CEM-directions bug was confirmed empirically before fixing
>       (four iterations drew the identical perturbation matrix) and parity still
>       passes. Commit `afd04a6`.
> - [x] **A.6** `POPL2027.pdf` committed at the repo root. Commit `c2f0744`.
> - [x] **A.7** `PAPER-DISCREPANCIES.md` created and seeded with the Theorem 5.2 /
>       Table 7 contradiction and the segment-reset omission. Commit `c2f0744`.
>
> **Phases A–D are complete. Phase E is next.**
> Phases A–C are included in `main`.
> Validation is recorded below; Unstack is subject to a 60-second wall-clock cap.
>
> - [x] Explicit numeric table marker and predicate isolation (`b972637`).
> - [x] Symbolic table isolation, Scattered boundary agreement, and direct-on
>       predicate (`c3e285b`).
> - [x] Existential translation (`ba45fbc`): fresh witnesses replace the planned
>       finite disjunction, which would unsoundly restrict the witness domain.
> - [x] Remove Top (`7740d0c`) and pin Reverse's initial relation (`7fbd433`).
> - [x] Predicate/vocabulary regression tests, table identity in equality,
>       collision-free witness names, and Boolean/finite-sort equality translation
>       (`f6ea593`). Boolean equivalence gives quantifiers mixed polarity and is
>       refused rather than unsoundly expanded.
> - [x] Physical table surface height on all four tower environments, separate
>       from block-center height and observation packing (`f972e3c`).
> - [x] Move xyz parameters and independent parameter-slot tests (`c6f96e1`).
> - [x] All **49 unittest tests pass**, including BMC, run logging, MCMC parity,
>       table geometry in all four tower tasks, and Phase B regression tests.
>       `bash format.sh` completed; unrelated formatting changes were restored.
> - [x] Stack infinite and finite (4 blocks): both retain `hl_ok: False`,
>       `ll_ok: True`, with the same VC 0/1 refutations documented below.
> - [x] MCMC smoke comparison: all runs complete; results and budgets below.
> - [x] Phase C trace store, adapter, and runtime callbacks (`e492688`). Snapshots
>       retain every physical block, current bindings, and per-invocation entry geometry.
> - [x] Phase C entry-point migration: all four tower verifiers now require explicit
>       `--demo-store` input; literals are test-only golden fixtures. A Stack collector
>       and JSON persistence supply a runnable execution-to-inference workflow.
> - [x] All **60 unittest tests pass** (49 existing + 11 Phase C), including golden
>       Stack invariant equivalence, entry-point wiring, snapshot isolation, and
>       three loop-head rows from a real three-iteration MuJoCo rollout. Formatting
>       completed; unrelated formatter changes were restored.
> - [x] Real-trace CLI smoke: seed 0, four blocks, three iterations. Collection and
>       inference complete. Rollout success is **False**; finite Stack verification
>       reports **`hl_ok: False`, `ll_ok: False`** (VC 0/1 refutations and a tube_0
>       counterexample). This uses a newly learned invariant from actual execution,
>       not the old literal dataset; do not present it as a verified Stack program.
>       Unstack was not rerun for Phase C and retains the one-minute cap below.
> - [x] Phase D1 (`b4bb3c8`): opt-in bounded grasp/move/release noise, explicit
>       BMC verdict mode and counterexamples. Noise remains off by default.
> - [x] Phase D2: explicit placement and frame contracts, updated-state swept
>       collision checks, structured counterexamples, bounded solver queries, and
>       tower CLI flags. Supported blocks are no longer assumed frozen. Frozen
>       `ON_star_zero` geometry is distinct from the current loop-head geometry.
> - [x] All **81 unittest tests pass**, including MCMC parity, contract success and
>       refutation, support disturbance, noise bounds, and unknown/inconsistent
>       handling. The legacy untouched-block BMC test now explicitly excludes
>       support contact, as required by D2's intentional frame correction.
> - [x] Phase D noisy Stack end-to-end smoke completes: `hl_ok: False`,
>       `ll_ok: False`, with `collision_0_sym` and `contract` refutations. Reverse
>       and Partial explicitly report unsupported motion checking (no lowered
>       physical programs). Unstack was not rerun; its one-minute cap still applies.
> - [x] Phase E timing gate recorded: exact noisy collision query **0.0032 s**;
>       full MotionVerify **0.122 s noiseless / 0.107 s noisy**, 34/38 checks on the
>       concrete existing Stack body. Both refute its release endpoint contract.
>       The exact swept-cube encoding is retained; no AABB fallback was needed.
>       These are fixture timings, not a universal bound or hardware proof.
> - [ ] Phases E–F: not started.
>
> **Pre-existing failure, not a regression.**
> `verify_stack_with_learned_invariant` reports two high-level VC failures
> (`[FAIL] VC 0 check 2 returned sat`, same for VC 1). Confirmed identical before
> and after the A.1/A.2 change by stashing and re-running, so it predates this
> work. Phase E's counterexample-guided loop is what is meant to address it —
> do not treat it as damage from Phase A.
>
> **Unstack validation remains inconclusive.** The final corrected code was run
> with `timeout 60s` and stopped with exit 124 during high-level verification;
> that run did not reach the motion checks. Per the user's standing instruction,
> **cap Unstack end-to-end runs at one minute** rather than waiting on a stalled
> solver. An earlier Phase B run reached the existing 300-second solver timeout
> on VC 2 check 2; the Phase A comparison (`ede20b5`) completed with
> `hl_ok: True`, `ll_ok: False` (inconsistent low-level initial conditions).
> Thus Unstack verdict parity is **not established**, and must not be reported
> as passing. Its solver/invariant behavior remains follow-up work; Phase B's
> finite-sort translation and predicates have direct regression coverage.
>
> **Phase B optimizer budget check (2026-09-20 UTC).**
> Same seed 0, four blocks, two demo seeds, 20 MCMC iterations, two CEM iterations,
> and `cem_init_std=0.1`; BMC and goal-feature reward disabled. Higher cost is better.
>
> | Parameters | CEM N/K | Best cost | Elapsed |
> |---|---:|---:|---:|
> | Phase A z-only (`ede20b5`) | 16/4 | -0.163374 | 53s |
> | Phase B xyz | 16/4 | -0.182996 | 68s |
> | Phase B xyz | 48/12 | -0.156418 | 76s |
>
> The larger vector still optimizes: both xyz runs improve from -0.303462,
> with nonzero CEM improvement in 8/9 evaluations. The larger sample budget
> recovers the short-run baseline score here. Defaults remain 16/4 and std 0.1:
> this one seed is a smoke check, not evidence for a universal tuning change or
> task success (success stayed zero). Use 48/12 as a measured comparison budget
> when evaluating longer xyz runs. Read the three recorded runs with
> `uv run python -m synthesis.experiment.report --glob 'runs/phase-b/*' --table`
> from `roboverify/`; run slugs are `phase-a-z-only`, `phase-b-xyz`, and
> `phase-b-xyz-n48-k12`.
>
> **Setup:** see `AGENTS.md`. Two environment variables are needed, not one.
>
> Conflicts 1–7 near the end of this document are **all adjudicated** — do not
> reopen them. Conflict 1 is paper-side with no code action and is recorded in
> `PAPER-DISCREPANCIES.md`.

## Context

`POPL2027.pdf` ("Component-Based Synthesis from Demonstrations under Local
Specifications") describes this repository, but a section-by-section comparison shows the
code and the paper have drifted apart in two different ways.

**The synthesis half (paper §3, Algorithms 1–5) does not exist.** There is no relational
CFG, no `RefineCFG`, no `Quotient`, no `LearnClassifier`, no `Validate` — none of these
identifiers appear anywhere in the repo. Loops, loop guards and loop-carried updates are
*hand-written* in every entry script: [verify_stack_with_learned_invariant.py:51](roboverify/synthesis/entry/verify_stack_with_learned_invariant.py:51)
literally types out the paper's Eq. 2 guard, and `synthesis/entry/main.py` does a
hard-coded 2-way demo split that never recurses and never builds a program.

**The verification half exists but has soundness holes that let it report success without
checking anything.** These are the more urgent problem, and are the subject of this plan:

- `Program.lowlevel_verification` examines **only** `While` instructions
  ([program.py:1171](roboverify/synthesis/api/program.py:1171)), and `ok` starts `True`, so
  a program whose motion obligations were never looked at returns the same verdict as one
  that genuinely passed.
- `start_verification` `continue`s past anything that is not a `PickPlaceByName`
  ([lowlevel_verification_lib.py:359](roboverify/synthesis/verification_lib/lowlevel_verification_lib.py:359)).
  Correct for `Assign`; silently unsound for `Pick`/`Move`/`Release`, which do sweep
  through space.
- **BMC has no noise variables and no collision constraints.** `bmc_verify` proves only
  that one idealized noiseless trajectory reaches the goal. The paper's §5.5 makes
  `η_g, η_m, η_r` free NRA variables precisely so the checks cover all executions within
  the hardware noise bounds. *(Scoped by decision: noise becomes an **opt-in** mode, off by
  default — see D1. The missing collision reasoning is not deferred.)*
- **Contract realization is never checked**, which leaves the two-level proof open
  independently of what the paper claims. The symbolic level treats each block as a black
  box *assumed* to establish its relational summary (`wp(π(μ,b',b)[on(b',b)], ·)` rewrites
  as if the placement succeeded). Nothing anywhere discharges that assumption: no check
  verifies a synthesized body actually establishes `on(b',b)`, nor that tower blocks are
  undisturbed. So every high-level "verified" verdict is conditional on an obligation the
  codebase never states. Only a form of collision-freedom exists.
- Invariant inference is a faithful implementation of Eq. 3 but is fed **hand-written state
  dicts** ([inference.py:2322](roboverify/synthesis/inference_lib/inference.py:2322)),
  not loop-head states from `D_V` — which is §4's headline claim.
- `pdb.set_trace()` sits in library paths (9 sites in `inference.py`, 1 in
  `instructions.py`), which hard-blocks any unattended CEGIS loop.

**Intended outcome:** a verification layer that reports what it actually checked, closes
the contract-realization and frame-preservation gaps, learns invariants from real
execution traces instead of literals, and runs the counterexample-guided loop to a
fixpoint — with noise-bounded checking available as an opt-in mode. Then a synthesis
pipeline that derives the CFG, loops, guards and loop-carried updates from demonstrations,
so the entry scripts stop hand-writing the programs they verify.

**Sequencing:** Phases A–E (verification) land first by choice of milestone — they are
independent of the CFG work and the verifier currently reports success without checking.
Phase F (synthesis, §3 and Algorithms 1–5) follows. F1's IR also supplies the per-block
relational summaries Phase E's motion CEGIS loop needs, so if the two are worked in
parallel, F1 is the stage to pull forward.

**The paper is an artifact under test, not a specification.** It was written by the same
people as the code, may describe intended rather than implemented behavior, and its
theorems and tables may be wrong. This is not hypothetical: Theorem 5.2 asserts the wp
rewrite operators introduce no quantifier, yet `rewrite_for_put_for_higher` introduces a
fresh `t` under both `Exists` ([program.py:785](roboverify/synthesis/api/program.py:785))
and `ForAll` ([:798](roboverify/synthesis/api/program.py:798)). So no item below is
justified by "the paper says so" alone — each stands on an independent argument about
soundness, internal consistency, or a demonstrable defect, and where paper and code
disagree, **neither side automatically wins**; the divergence goes in the conflicts table
for you to adjudicate. Published numbers are never regression targets.

**Scope: the four block-tower tasks** — Stack, Unstack, Reverse, ReStack/partial. Grid and
Pyramid are **out of scope**, which also excludes `verify_2d_with_learned_invariant.py` and
the entire goals-mode vocabulary it runs on (`GoalSort`, `Mark`, `d_star`, `r_star`, `l0`,
and the `GoalAssign`/`MarkGoal`/`MoveRight`/`MoveDown` instructions), plus `loop_inference_2d`
and the `run_2d_*_example` drivers. Leave that code in place, untouched and untargeted. Two
consequences worth stating rather than discovering later:

- The geometric translator raises `NotImplementedError` on every goals-mode predicate
  ([lowlevel_verification_lib.py:468](roboverify/synthesis/verification_lib/lowlevel_verification_lib.py:468)),
  so motion-level verification was never possible for those tasks anyway. Descoping makes
  an existing limitation explicit instead of silent.
- **Every in-scope task is a flat single loop.** `verify_2d` was the only nested-loop
  program, so F7 needs no Kleene-star matching — see F7.

**Working agreement:** topic branch off `main`; one commit per stage, never one squashed
commit. Stage explicit paths — the repo has unrelated untracked files (`demos/`,
`plot.py`, `create_env_figure.py`).

---

## Phase A — Fail closed (small; no dependencies)

Nothing else is trustworthy until these stop reporting false success.

1. **`lowlevel_verification` cannot distinguish "passed" from "never checked"** —
   [program.py:1171](roboverify/synthesis/api/program.py:1171). The body is
   `for inst in self.instructions: if isinstance(inst, While): ...`, so **loops are the
   only thing it ever examines**. Straight-line instructions are never matched — including
   ones outside the loop in a program that has one. `ok` starts `True` and is only
   falsified by a loop check, so a program with zero examined obligations returns the same
   value as one that genuinely passed.

   *Not* a case for raising: a straight-line program is a legitimate input with real motion
   obligations (§5.5 is per **basic block**, and Algorithm 6's `MotionVerify` returns
   counterexamples `(v, μ_k)` indexed by block, not by loop).

   **Fix in two parts.** Now: return `MotionVerificationResult(ok, checked_blocks,
   skipped)` so callers can treat `checked_blocks == 0` as *not verified* rather than
   verified; update `verify_stack`/`verify_unstack`, which call it (`verify_reverse` /
   `verify_partial` have it commented out). Later: extend coverage to non-loop blocks.
   That is blocked on a geometric context — `start_verification(conditions, body,
   constants)` needs `conditions` to say where the *other* blocks are (Definition 5.4), and
   for a loop that comes from invariant + guard. A straight-line block needs its entry
   condition, and `lowlevel_verification`'s signature has no precondition parameter. So
   full per-block coverage lands with the CFG's per-block summaries (Phase F); Phase A only
   stops the false "verified".
2. **The instruction skip is unbounded** — [lowlevel_verification_lib.py:359](roboverify/synthesis/verification_lib/lowlevel_verification_lib.py:359).
   `if not isinstance(instruction, PickPlaceByName): print(...); continue`.

   This must *not* become a blanket raise: the lowered loop body in
   [verify_stack_with_learned_invariant.py:97](roboverify/synthesis/entry/verify_stack_with_learned_invariant.py:97)
   appends `Assign("b", "b_prime")`, so the branch fires on every run today, correctly —
   `Assign` has no motion semantics.

   Replace the catch-all with an explicit three-way classification:
   - **inert** (`Assign`, `Skip`) — skip silently, they carry no geometry;
   - **motion, unhandled** (`Pick`, `Move`, `Release`, `PickPlace`) — fail closed; these
     *do* sweep through space and are currently passed over in silence;
   - **unknown type** — raise `UnsupportedMotionInstruction`.
3. **`pdb.set_trace()` → typed exceptions** — `inference.py:366, 400, 419, 545, 571, 586,
   668, 963, 973` and [instructions.py:88](roboverify/synthesis/api/instructions.py:88).
   `inference.py:668` is the worst: `learn_from_partition` drops into a debugger when the
   Eq. 3 optimization is UNSAT, which is a *normal* outcome once CEGIS adds a conflicting
   row. Introduce `InferenceDataError` / `SeparationInfeasible`.
4. **Solver `unknown` handling** — `check_tautology` does `assert False` on `unknown`
   ([inference.py:889](roboverify/synthesis/inference_lib/inference.py:889)); return a
   tri-state and keep the clause. `highlevel_verification` lumps `unknown` in with `sat`
   ([program.py:1116](roboverify/synthesis/api/program.py:1116)) and then builds a
   counterexample from a `None` model.

5. **Isolate the RNG streams — this is a correctness bug, not just hygiene.**
   `set_np_seed` ([synthesis.py:689](roboverify/synthesis/mcmc/synthesis.py:689)) sets the
   *global* numpy and Python RNG, and `rollout_demos`
   ([synthesis.py:1058](roboverify/synthesis/mcmc/synthesis.py:1058)) calls it once per
   rollout, from inside the objective.

   *What works:* rollouts are seeded per demo before `make_roboverify_env`, so each policy
   rollout starts from its corresponding demonstration's initial state and re-evaluating a
   candidate is deterministic. The KL/MMD comparison is well-defined. **Keep this property —
   the objective is meaningless without it.**

   *What breaks:* in `cem_optimize` ([cem.py:6](roboverify/synthesis/mcmc/cem.py:6)) the
   parent process draws `samples = np.random.randn(N, dim) * sigma + mu`, then at the end of
   each iteration calls `f(mu)` **in the parent**, which runs rollouts and resets the global
   RNG to a state fixed by `rollout_seeds[-1]`. So every iteration begins from the same RNG
   state and draws the **identical `N × dim` perturbation matrix**. CEM still moves because
   `mu` and `sigma` change, but it only ever probes the same N directions — 16 frozen
   directions at the default `cem_N=16`. This plausibly explains the "cem delta mean ~0 /
   many zero-delta iters" symptom already documented in `CLAUDE.md`.

   *Predicted signature, and how to tell it from real convergence:* with `Z` fixed, the same
   elite rows keep winning, so `sigma_new = sigma · std(Z_elite)` shrinks geometrically
   (the std of K=4 fixed normals is typically < 1). Once sigma collapses, every candidate is
   essentially `mu`, scores coincide and the CEM step stops improving — i.e. real gain for an
   iteration or two, then flat. That is the `cem delta mean ~0 / many zero-delta iters`
   symptom in `CLAUDE.md`'s diagnostics table. Genuine convergence produces the same delta
   but a *gradual* sigma decay, so `cem_sigma_norm` — already recorded per iteration by
   `CEMStats` in [experiment/mcmc/cem.py](roboverify/synthesis/experiment/mcmc/cem.py) —
   distinguishes them.

   *Confirm first, two lines:* print `np.random.randn(2)` at the top of each CEM iteration
   and check whether it repeats. This is a hypothesis until that check runs.

   *Fix:* give rollouts their own `np.random.Generator`, seeded per rollout, instead of
   mutating global state. Rollouts stay reproducible; CEM's stream is untouched. Then record
   the top-level seed in `config.json` (which already captures git sha, argv and versions)
   and make "re-run iteration N from its recorded seed" supported — Phase E and F5 nest
   CEGIS → MCMC → CEM → MuJoCo and are undebuggable otherwise.

   *Consequence for existing results:* every MCMC run to date optimized parameters with
   frozen CEM directions, so prior runs are not a clean baseline for before/after comparison
   once this is fixed. Re-run any numbers that matter rather than comparing across the fix.
6. **Commit `POPL2027.pdf`** at the repo root, where `CLAUDE.md` says notes live. It is the
   reference every discrepancy note cites, so it belongs in the repo rather than in someone's
   Downloads. Note for later: PDFs do not delta-compress, so each revision adds a full ~2 MB
   copy to history. Fine for a handful; if revisions become frequent, commit the LaTeX source
   too, which deltas well and makes the discrepancies reviewable as diffs.
7. **Record the paper discrepancies in the repo** (docs, not code). Create
   `PAPER-DISCREPANCIES.md` at the repo root — per `CLAUDE.md` the root is where notes
   live — capturing each place the submission and the implementation disagree, so they can
   be worked once the code is settled. Seed it with conflict 1 (Theorem 5.2 vs Table 7's
   `R_Higher`, with the `:670`/`:785`/`:798` citations and the three repair options), and
   add an entry whenever a later phase turns one up. Cross-reference from the conflicts
   table so neither drifts.

**Test:** new `synthesis/verification_lib/test_solver_robustness.py` — force an
unseparable S/U partition and a 1 ms solver timeout; assert typed exceptions, no hang.
Re-run `uv run python -m unittest synthesis.verification_lib.test_bmc_lib -v`.

---

## Phase B — Predicate, vocabulary and parameter alignment (medium; needs A)

Sized medium because the table rework touches `on.py`, the geometric translator and
`inference.py`, and records a surface height in `synthesis/environment/` for Phase D. Per
`CLAUDE.md`, the `10 + 12*i : 10 + 12*i + 3` box-position slice convention recurs across
`instructions.py`, `on.py` and the `While` guard evaluator — add the table surface as a
separate attribute, never a slot in that packing.

The invariant is *learned* under `synthesis/util/on.py` semantics and *checked* under
`lowlevel_verification_lib.py` semantics, and the two disagree:

| predicate | `on.py` | `lowlevel_verification_lib.py` |
|---|---|---|
| `higher` | `0 <= z1-z2 and z1 >= 0 and z2 >= 0` ([on.py:119](roboverify/synthesis/util/on.py:119)) | `Z(b1) >= Z(b2)` ([:135](roboverify/synthesis/verification_lib/lowlevel_verification_lib.py:135)) |
| `scattered` | `>= 2*L` and `z1,z2 >= 0` ([on.py:130](roboverify/synthesis/util/on.py:130)) | `> 2*L`, strict ([:139](roboverify/synthesis/verification_lib/lowlevel_verification_lib.py:139)) |

**What the axioms require.** `tbl` is deliberately a null/bottom marker, isolated in all
three relations ([highlevel_verification_lib.py:431](roboverify/synthesis/verification_lib/highlevel_verification_lib.py:431),
`:184`, `:210`):

```
on_tbl             ∀x. (ON*(x,tbl) ∨ ON*(tbl,x)) → x = tbl
higher_tbl         ∀x. (Higher(x,tbl) ∨ Higher(tbl,x)) → x = tbl
scattered_not_tbl  ∀x. ¬Scattered(x, tbl)
```

"On the table" means `on*`-**isolated**, not `on*(b, tbl)` — consistent with
`rewrite_for_put_on_tbl_for_ON_star` ([program.py:583](roboverify/synthesis/api/program.py:583)),
which *detaches* `b'` by dropping the `on*` pairs routed through it rather than asserting a
new edge. So the numeric and symbolic sides do agree today; the problem is *how*.

**Root cause: the isolation is accidental, achieved three different ways.** `on*` is false
for the table only because the sentinel `[-100,-100,-100]`
([inference.py:1846](roboverify/synthesis/inference_lib/inference.py:1846), `:1858`) lies
far outside the `BLOCK_LENGTH/2` xy tolerance; `Higher` and `Scattered` are false only
because of the sign of `z`. Both satisfy the axioms by coincidence. Change `BLOCK_LENGTH`,
or let a block drift below the table plane, and the semantics shift silently.

**Approach: keep `tbl` a sort element, and delete its fake position rather than making it
real.** The relational level needs no table geometry at all — it needs an explicit
`is_table` test in all three numeric implementations, each stating the axiom it enforces.
This leaves untouched everything that assumes `tbl` is a sort element: Table 7's
`put(b', tbl)` rules, the axioms, `Ω_Inv` atom generation, and the finite-expansion
translator. (Removing `tbl` from the object universe entirely would delete the pervasive
`m != tbl` side conditions, but would rework the wp action forms, the predicate vocabulary
and every spec — a later direction, not this round.)

- Replace the `[-100,-100,-100]` literals and the three `z >= 0.0` tests in
  `synthesis/util/on.py` with an explicit table marker and `is_table` test. As written, any
  block legitimately below the table plane is silently dropped from `Higher` and
  `Scattered`.
- Give `on_star_implementation` ([on.py:89](roboverify/synthesis/util/on.py:89)) the same
  explicit exclusion; today it enforces `on_tbl` only by coordinate distance.
- **A real table attribute belongs at the motion level, not the relational one.** The
  physical surface height genuinely matters for release height and collision in Phase D;
  record it in the environment for that, and keep it out of the relational predicates. The
  pseudo-block currently conflates these two distinct needs.
- On the symbolic side, exclude the `tbl` **constant** explicitly in `lowlevel_higher` /
  `lowlevel_scattered`, matching the existing `higher_tbl` and `scattered_not_tbl` axioms
  ([highlevel_verification_lib.py:184](roboverify/synthesis/verification_lib/highlevel_verification_lib.py:184),
  `:210`). Then the two sides agree by construction rather than by duplicated arithmetic.
- Six commented-out `if block1_name == "tbl" or block2_name == "tbl":` guards in
  `inference.py` (`:368, 402, 421, 547, 573, 588`) are an abandoned attempt at exactly
  this; resolve them rather than leaving them.
- Match strictness (`>=` vs `>`) between `scattered_implementation` and
  `lowlevel_scattered` once the table handling is settled.
- Rename `lowlevel_on` → `lowlevel_on_star` (it correctly mirrors
  `on.on_star_implementation`) and drop the vestigial single-conjunct `Or` at
  [:126](roboverify/synthesis/verification_lib/lowlevel_verification_lib.py:126).
- **Add `lowlevel_on_direct`** mirroring `on.z3_on` ([on.py:70](roboverify/synthesis/util/on.py:70))
  with the `0 <= Δz < 1.5L` band. Phase D's contract-realization obligation cannot be
  stated without it.
- Add an `Exists` case to `_translate_expr` ([:431](roboverify/synthesis/verification_lib/lowlevel_verification_lib.py:431))
  as the finite-disjunction dual of the existing `ForAll` expansion; today an existential
  clause falls through to a generic rebuild and is silently mistranslated.
- **Delete `Top`** (conflict 5): the `Function` declaration at
  [highlevel_verification_lib.py:92](roboverify/synthesis/verification_lib/highlevel_verification_lib.py:92),
  the `Ω_Inv` branch at [inference.py:341](roboverify/synthesis/inference_lib/inference.py:341),
  the `spec_to_expr` decl entry at
  [:741](roboverify/synthesis/verification_lib/highlevel_verification_lib.py:741), and the
  translator case. It is debug shorthand; removing it eliminates the wp hole outright.
  `ON_star_zero` stays exactly as it is — see conflict 6.
- **Pin `ON_star_zero` to the initial state in the precondition** (conflict 6): for every
  task whose spec or invariant vocabulary includes `ON_star_zero`, conjoin
  `∀x,y. ON_star_zero(x,y) ↔ ON_star(x,y)` into `φ_pre`. This belongs in the precondition,
  **not** the axioms — `put` never rewrites `ON_star_zero`, so the two relations correctly
  diverge as execution proceeds, which is what Reverse relies on. Start with
  `verify_reverse_with_learned_invariant.py:105`, whose current `precondition` does not
  mention it.
- **Unfreeze `Move`'s x and y offsets** (conflict 7): drop the `target_offset[2:]` slice in
  `Move.register_trainable_parameter` and `update_trainable_parameter`
  ([instructions.py:372](roboverify/synthesis/api/instructions.py:372)) so CEM optimizes all
  three components, as Algorithm 5 specifies. Independent of everything else in this phase
  and can land on its own. The CEM vector grows 3× per `Move`, so re-check `cem_N`/`cem_K`
  and `cem_init_std`; `--smoke` is the cheap way to confirm the optimizer still converges
  before committing to full-length runs.

**Test:** property test asserting `lowlevel_*` and `on.*_implementation` agree on random
block configurations; a test asserting no `Top` symbol survives in any built vocabulary;
a test asserting `register_trainable_parameter` returns 3 slots per `Move`, and a `--smoke`
MCMC run to confirm convergence is not degraded.

---

## Phase C — Invariant inference from real traces (medium; needs A)

§4's central claim. The machinery is already a strong match — **reuse unchanged**:
`compute_S_U:606`, `learn_from_partition:623` (Eq. 3 verbatim, with `sel_i` /
`opt.minimize(z3.Sum(sel))`), `construct_truth_table_and_extract_expression_for_phi:671`,
`check_tautology:863`, `check_redundancy:1783`, `loop_inference:1512`.

**New** `synthesis/inference_lib/demo_store.py`:
- `LoopHeadState` carrying positions for *every* environment object (Definition 5.4 needs
  the ones `π` never references), the loop-entry state, and the constants mapping.
- `DemoStore` (= `D_V`), `to_inference_inputs(...)` emitting the three lists
  `compute_dataset:260` zips over, and `InvInference(D_V, loop_id, vocab, context)`.

**The bridge:** `While.eval` ([instructions.py:1062](roboverify/synthesis/api/instructions.py:1062))
gains an optional `on_loop_head` callback fired after `_find_and_bind_guard_exists`
returns `True` — at that point `traj[-1]` is the loop-head observation and
`env.symbolic_name_to_box_id` is already the constants mapping. Positions via
`on_util.get_block_pos`.

The literal `run_proposal_example` / `run_unstack_example` / `run_reverse_example` /
`run_partial_stack_example` bodies become **golden test fixtures**, not entry points.

**Test:** adapter output reproduces the `run_proposal_example` literals exactly; the
invariant learned from adapter output is z3-equivalent to the golden one; a 3-iteration
rollout yields 3 loop-head rows, not 1.

**Completed implementation notes:**
- `Program.eval`, `Program.eval_from_observation`, and `run_program_rollouts` forward
  an optional `on_loop_head` callback; `store.add` records a copied `LoopHeadState`.
  Program loop IDs are instruction paths (`"1"` for the existing tower loop).
- `DemoStore.save/load` persists plain JSON, with symbolic constant names resolved
  against the requested inference context by `to_inference_inputs`. Empty datasets
  and missing bindings fail explicitly. Core inference algorithms remain unchanged.
- **Fixture clarification:** the legacy Stack example supplies four empty entry
  dictionaries but only two current states/bindings; `compute_dataset` consumes two
  rows through `zip`. Tests reproduce those current states/bindings exactly and
  prove learned-invariant equivalence with Z3. New traces retain complete, aligned
  entry states; Stack's vocabulary does not read `ON_star_zero`. The legacy Partial
  fixture similarly has four entry dictionaries for five states and silently drops
  the fifth. These historical fixtures remain unchanged as evidence, not live input.
- All four tower entry points take explicit trace input. The bundled collection CLI
  executes the existing physical Stack program; other tasks can record their own
  executable programs through the same callback. This stage does not synthesize
  missing task programs or repair the existing Reverse/Partial verification templates.
- Workflow and callback contract: `roboverify/synthesis/inference_lib/README.md`.
  Simulator test: `synthesis.experiment.test_loop_traces`; adapter/golden tests:
  `synthesis.inference_lib.test_demo_store`. The full suite has 60 passing tests.

---

## Phase D — Motion obligations under noise (large; needs B, C)

### D1 — Free NRA noise in BMC — **opt-in, off by default**
`bmc_lib.py` has zero occurrences of `noise|perturb|epsilon`.

Add `NoiseSpec(eps_grasp, eps_move, eps_release)` as an **optional** parameter defaulting
to `None`. With `None`, the encoding stays exactly as it is today — `_encode_pick:352`,
`_move_target:372` and `_encode_release:425` keep their exact equalities, the transition
relation stays deterministic, and every existing `test_bmc_lib` case passes bit-for-bit.
Passing a `NoiseSpec` adds per-step free reals bounded `-ε <= η <= ε` at those three sites.
Surface it as a flag on the entry points, defaulting off.

This keeps NRA out of the default path, which also means the Phase D spike below stops
being a schedule risk: if the noisy query turns out to be slow or `unknown`, only the
opt-in mode is affected and everything else proceeds.

**Document the consequence rather than leaving it implicit:** with noise off,
`bmc_verify` proves that *one idealized noiseless trajectory* reaches the goal, which is
strictly weaker than the paper's §5.5 claim that the checks cover all executions within
the hardware noise bounds. Have the result object carry which mode it ran in, so a
"verified" verdict states its own strength. That is the difference between a known
limitation and a silent overclaim — and it is what makes the default acceptable.

### D2 — Contract realization + frame preservation
New `synthesis/verification_lib/motion_verification.py`. Reuse the consistency check
(`:345-352`) and swept-tube collision check (`:358-414`) verbatim; **add**:
- `check_contract_realization` — symbolically execute the body's `PickPlaceByName` chain,
  then require `invariant ∧ guard ∧ transitions ∧ ¬lowlevel_on_direct(b',b)` UNSAT.
- `check_frame_preservation` — for the Skolem block `sym` and each named constant,
  require `lowlevel_on_star(c, b0)_pre ∧ moved(c)` UNSAT.
- `MotionCounterexample(block_v, mu_k)` — the concrete failing positions, i.e. `Pen(v)`'s
  payload.

Also fix the frame bug at `_encode_move:408`: `If(held, <tracks EE>, _frame_block(...))`
freezes a block *resting on* the carried block, so towers pass through each other.

**Risk + spike (only needed before enabling D1's opt-in mode).** `encode_collision`
([:149](roboverify/synthesis/verification_lib/lowlevel_verification_lib.py:149)) is
already bilinear in the tube parameter `t`; adding free `η` multiplies free variables and
may push Z3 to `unknown`. With noise off by default this no longer blocks D2, the
counterexamples or the CEGIS loop — it only gates the opt-in mode, so the spike can be
deferred until you actually want paper-strict checking. Take the concrete 3-instruction
body at
[verify_stack_with_learned_invariant.py:70](roboverify/synthesis/entry/verify_stack_with_learned_invariant.py:70),
hand-build one collision query with `η ∈ [-0.005, 0.005]³` added to `end_pos`, and time
`s.check()`. **Pre-planned fallback:** the swept AABB of a segment is the AABB of its
endpoints, so `encode_collision` can be rewritten quantifier-free and *linear* as a
per-axis `Abs(X(a) - midpoint) < L + halfspan` test — a single-function change that drops
the query into LRA while keeping `η` free.

**Test:** extend `test_bmc_lib.py` with `test_default_encoding_unchanged` (no `NoiseSpec`
⇒ identical constraints to today) and, marked as the opt-in path,
`test_verify_fails_when_eps_exceeds_on_tolerance`. New `test_motion_verification.py`: a
scene where the contract holds; one where the release offset is too large so contract
realization fails and the returned `μ_k` is the breaking configuration; one where a tower
block is displaced.

**Gate before Phase E: time one `MotionVerify` call.** The motion CEGIS loop as designed
re-synthesizes a block per counterexample, which assumes `MotionVerify` is cheap relative to
`StraightLineSynthesize`. If it is not, the loop must batch counterexamples — accumulate all
of `CEx` and re-synthesize once — and that is a structural change, so make it before the
loop is built rather than after. Record the number; it sets the iteration budget Phase E can
afford.

**Completed implementation and scope refinements:**
- `NoiseSpec` is accepted by all BMC APIs; `bmc_verify` now returns a
  bool-compatible result with `status`, `mode`, model, and trace symbols. Solve
  mode remains existential, including noise, and is not robust synthesis.
- Tower CLI flags: `--motion-noise GRASP MOVE RELEASE` (metres, omitted by default)
  and `--motion-timeout-ms`. Unstack also requires the physical
  `--table-surface-height`; relational `tbl` still has no coordinates.
- `Program.lowlevel_verification` takes explicit per-loop `MotionContract`s and
  returns per-obligation results. Unknown, inconsistent premises, missing
  contracts, missing lowered programs, and uncovered motion all fail closed.
- **Frame clarification:** preserve the original tower's *other* objects, excluding
  the declared manipulated source and its aliases. Otherwise every legitimate
  Unstack step would violate its own frame condition simply by moving its source.
  Unknown support dynamics is overapproximated by unconstrained displaced
  positions and rejected by a separate support obligation.
- **Default-encoding clarification:** D1 preserved all three legacy transitions
  exactly. D2 deliberately weakens Move's frame assumption for supported objects;
  exact backward identity there would retain the bug this phase must fix. The
  regression checks exact Pick/Release formulas and equivalent Move formulas on
  unsupported scenes, plus a displaced-support counterexample.
- **Remaining abstraction limits:** BMC goal checking retains the old nominal
  Release convention (block positions frozen while EE z changes). MotionVerify
  uses idealized waypoints and no settling model; its collision obligations cover
  blocks, not arm or physical table-plane geometry. Do not describe either as
  verification of all physical controller executions. These limits, the old
  `ON_star_zero` translation defect, and the non-equivalence of the proposed AABB
  fallback are recorded in `PAPER-DISCREPANCIES.md`.
- Reproduce the timing gate with
  `uv run python -m synthesis.entry.benchmark_motion_verification` from
  `roboverify/`. Final measured exact bilinear spike: SAT in 0.0032 s with
  `eta` bounded by 0.005 m. Full body: 0.122 s noiseless, 0.107 s with all three
  bounds 0.005 m. The existing release offset of `1.5 * L` fails the strict
  direct-on band; no example offsets were retuned to make validation pass.
- **Phase E budget implication:** this fixture does not require batching to make
  motion checking affordable; begin with per-counterexample refinement and retain
  elapsed-time reporting. Generalized conditions can still time out, so `unknown`
  must not trigger refinement as if it were a concrete counterexample. Measure
  real synthesis/verification ratios before selecting longer-run budgets.
- API usage and assumptions: `roboverify/synthesis/verification_lib/README.md`.
  Full suite: 81 tests pass. Required formatter completed; unrelated edits restored.

---

## Phase E — SymbolicVerify and the Algorithm 6 loop (medium; needs C, D)

1. **Label the VCs.** `VC_gen` ([program.py:987](roboverify/synthesis/api/program.py:987))
   returns anonymous `Implies` nodes. Return `VC(kind: establish|preserve|exit|body,
   loop_id, expr)` — CEGIS must distinguish "loop-VC counterexample → extend `D_V`" from
   "entry-condition failure → ask the user for demonstrations". The current establishment
   handling is *correct* (`wp(While,Q) = invariant` folds it into the top-level implication);
   just label it. Likewise the preservation/exit asymmetry (`instantiated_cond` vs
   `Not(cond)`) is right for an ∃-guard — document it so nobody "fixes" it.

   **Three-valued verdict with core-based vacuity detection.** Each VC discharges to
   `VALID` / `INVALID(model)` / `VACUOUS` rather than a boolean, where **`VACUOUS` means
   `axioms ∧ P` is itself unsat** — no state satisfies both the domain axioms and the
   premise, so `Implies(P,Q)` holds trivially whatever `Q` is.

   *Why this is not academic:* with `I = False` (the RQ2 starting point), preservation
   `Implies(And(cond, False), wp(body, False))` and exit `Implies(And(Not(cond), False), Q)`
   both have unsat premises and pass vacuously. Only establishment `Implies(φ_pre, False)`
   genuinely fails, since `φ_pre` is satisfiable. So without this check, two of the three
   loop VCs report success for an identically-false invariant.

   Vacuity is *semantic* (unsatisfiability of `axioms ∧ P`), so it cannot be read off
   the VC syntactically — but
   it can be read off the **unsat core of the validity query**, avoiding the second solver
   call in the common case:

   - Set `unsat_core=True` and track `P` as `premise` and `¬Q` as `neg_conclusion`. The
     axioms are already tracked by name via `assert_and_track`
     ([highlevel_verification_lib.py:154](roboverify/synthesis/verification_lib/highlevel_verification_lib.py:154) onward);
     `check_satisfiable` currently adds the formula with a plain `s.add(formula)` and never
     configures or reads cores.
   - Run `axioms ∧ P ∧ ¬Q`. If `sat` → `INVALID` with the model. If `unsat` and the core
     **omits** `neg_conclusion` → the contradiction lies entirely in `axioms ∧ P` →
     `VACUOUS`.
   - **One-sided test:** Z3 cores are not guaranteed minimal, so a core *containing*
     `neg_conclusion` is inconclusive, not proof of non-vacuity. Fall back to the explicit
     `axioms ∧ P` query only in that case — exact, and usually one query instead of two.
   - Make core minimization explicit and local. `smt.core.minimize` is currently set as a
     **global** z3 option at [inference.py:14](roboverify/synthesis/inference_lib/inference.py:14),
     so whether a `highlevel_verification_lib` solver gets it depends on import order.
     Set it per-solver where cores are actually read.
2. **`symbolic_verify.py`** — returns `(ok, failed_vc_kind, model, loop_head_state)`.
   `_extract_direct_on:534` + `_build_stacks:555` already turn a model into a tower; add
   `stacks_to_positions` producing exactly the dict Phase C's adapter consumes. That closes
   the `s0 → D_V` edge. Select the **smallest** counterexample by iterating
   `num_blocks = 2, 3, 4, …` in `mode="enum"` — smaller models are cheaper to re-execute
   and yield more targeted loop-head states, independent of the paper also doing this.
3. **`cegis.py` + `synthesis/entry/verified_synthesis.py`** — Algorithm 6's two loops.
   **Phase E can only ship two of the three branches.** Algorithm 6's symbolic loop forks on
   the failing VC kind: a *loop*-VC counterexample extends `D_V` and re-learns (implementable
   now), while an *entry-condition* failure calls `P, G ← Synthesize(M, L, G, D_V)` — which is
   F6. Until F6 lands, that branch records the counterexample and raises
   `NeedsResynthesis(s0)` rather than silently treating it as an invariant problem; the two
   are diagnosed differently and conflating them would send the loop after the wrong fix. The
   motion loop is fully implementable now.
   `Pen` store plus objective (8)'s penalty term added to the score at
   [synthesis.py:1416](roboverify/synthesis/mcmc/synthesis.py:1416), threaded through
   `optimize_program:651`. Assert block relational summaries are unchanged across motion
   re-synthesis — that assertion is what justifies not re-running `SymbolicVerify`.
   Reuse `RunLogger` ([run_logger.py:101](roboverify/synthesis/experiment/run_logger.py:101));
   emitting `(iteration, phase, n_clauses, invariant_sexpr)` records the actual invariant
   progression, which is what makes it comparable to the paper's Tables 2 and 3.

---

## Phase F — Synthesis half (§3, Algorithms 1–5)

None of this exists: no `CFG`, `RefineCFG`, `Quotient`, `LearnClassifier`, `Validate`,
`PostScore` or `ExtractIterations` identifier appears anywhere in the repo.

**Two IRs, one lowering.** The CFG is the *search* IR; `api/program.Program` stays the
*verification/execution* IR. `Program` is a flat fixed-length `List[Instruction]`
([program.py:928](roboverify/synthesis/api/program.py:928)) whose contract is `eval` against
MuJoCo and `VC_gen`/`wp` for Z3; pushing `D_V` into it would force every verification entry
point to learn about demonstration segments. Two new packages:

| package | contents |
|---|---|
| `synthesis/cfg/` | `graph.py` (`RelationalCFG`), `demos.py` (`DemoSegment`, `DemoAssignment`), `region.py` (`BlockRegion`/`LoopRegion`), `lower.py`, `validate.py`, `scope.py`, `refine.py`, `kleene.py`, `quotient.py`, `execute.py`, `synthesize.py` |
| `synthesis/predicates/` | `atoms.py`, `language.py`, `scene.py`, `enumerate.py`, `classifier.py`, `guard.py` |

**Dependency graph.** Note that `Quotient = identity` is a sound degenerate implementation
(it simply never finds loops), so **F6 ships an end-to-end runnable pipeline with the
hardest stage stubbed**, and F7 slots in behind a flag. Do not let Algorithm 2's textual
ordering dictate build order.

```
F3a (term ADT) ──┬── F1 (IR + lowering) ──┬── F2 (Validate) ──┐
                 │                         │                   ├── F4 (LearnClassifier + RefineCFG) ──┬── F6 (Alg 2) ── F7 (Quotient)
                 └── F3b (enumerator) ─────┴───────────────────┘                                      │
F0 (segment reset) ──────────────────────── F5 (StraightLineSynthesize) ───────────────────────────────┘
```

**Out of scope: branches.** §3.6 branch refinement and the `If` statement form are omitted —
no current environment needs them. That drops `BranchRegion`, the `If` instruction and its
`wp`/`VC_aux` cases. The CFG stays single-entry chains and loops, which also keeps
`scope(v)`'s must-reach analysis simpler (no join points where a variable is introduced on
only one arm). Revisit if a task ever needs a guarded alternative.

**Land F3's term ADT first.** CFG edge labels, loop guards and Kleene letters are all terms,
so `synthesis/predicates/term.py` plus `to_z3` is the foundation everything else names — see
F3. It is a small module and splitting it out ahead of `cfg/graph.py` avoids having to
retype edge labels later.

### F0 — Faithful segment-start reset (medium; prerequisite for F5)

**The algorithm needs something the environment cannot currently do.** Algorithm 5's
`PostScore` is `Pr_{s₀∼D_v}[π rolled out from s₀ reaches s ⊨ φ]` — "fraction of rollouts
from demonstration-initial states reaching φ" — and the loop body's `D_V(v_body)` comes from
`ExtractIterations`, whose segments begin at loop-entry states. So every block must be
scored by rolling out from *its own* segment start.

What exists: `Program.eval_from_observation` ([program.py:953](roboverify/synthesis/api/program.py:953))
→ `set_state_from_observation`. For the Fetch env
([fpp_construction_env.py:850](roboverify/synthesis/environment/cee_us_env/fpp_construction_env.py:850))
that function is labeled in its own first line as *"a dummy function to only visualize the
object dynamics"*. It restores block poses, **hard-codes the robot's 16-dim state to a fixed
literal home pose**, and zeroes all velocities. The held-object state `h ∈ O ∪ {⊥}` is not
represented at all.

Note the stub's limitation is a consequence of going *through the observation*:
`agent_dim = 10` is Cartesian end-effector xyz, finger joints and velocities, carrying no
arm joint configuration, while MuJoCo needs 16 robot values — reconstruction would need IK.
The fix is to stop reconstructing and start recording.

**Target: direct state restore. `O(1)`, exact, no IK.** `get_GT_state()` /
`set_GT_state()` already exist ([mujoco.py:27](roboverify/synthesis/environment/cee_us_env/mujoco.py:27))
as `sim.get_state().flatten()` and `set_state_from_flattened()` + `forward()`, covering
`time`, `qpos`, `qvel`, `act`. Extend demonstration collection to record the GT state per
timestep alongside the observation, have `DemoSegment` reference the state at `t_start`, and
reset by restoring it.

**Implement both modes; default to replay, switch to reset once validated.**

- **Mode A — `reset` (the goal).** Restore the recorded GT state. Constant time; this sits
  in the innermost scoring loop, so it is the only mode that is actually affordable at MCMC
  scale.
- **Mode B — `replay` (reference oracle).** Re-execute the demonstration deterministically
  from its true initial state to `t_start`. Slow, but needs no new recording, works with the
  existing `demos/` files, and is exact by construction.
- **The differential test is what earns the switch:** for every segment, `reset` and
  `replay` must land in the same simulator state within tolerance. This is also the cheapest
  way to settle the one genuine uncertainty — `MjSimState.flatten()` covers `time/qpos/qvel/act`
  but not the contact solver's warm-start, and `set_GT_state` calls `sim.forward()` to
  recompute derived quantities. For a quasi-static block domain that is almost certainly
  exact, but measure it rather than assume it.

**Three consumers, three paths — do not conflate them:**

| caller | state source | mechanism |
|---|---|---|
| block scoring (F5) | recorded demo | Mode A, falling back to B for legacy demos |
| visualization | observation | keep `set_state_from_observation`, renamed to say it is approximate |
| CEGIS counterexample (Phase E) | SMT model — **no recorded state exists** | construct a scene from block positions; neither A nor B applies |

That third row is easy to miss: Phase E's `s₀` comes from a solver, not a demonstration, so
it needs a scene *constructor*, not a state *restorer*. That is where a fixed-up
observation-based setter still earns its keep.

- `main.py`'s stage-2 path (`checkpoint_states = [split["part1"][-1] ...]` →
  `rollout_demos_from_initial_states`) already routes through the stub, so existing stage-2
  results silently start from a home-pose arm. Note it when F6 replaces that driver.
- Add to `PAPER-DISCREPANCIES.md`: the paper requires rollouts from segment-initial states
  but never addresses how the simulator reaches them; its only uses of "reset" concern
  symbolic state in §5.

### F1 — CFG + `D_V` IR and lowering (medium)

`DemoSegment` is the load-bearing type: `demo_idx`, **absolute** `t_start`/`t_end`, `states`,
`bindings: dict[str,int]`, `parent`. Absolute indices are mandatory — Validate (§3.5)
compares `i_s(d,ψ)` against `i_f(d,φ)` across *different* blocks' segment sets, and
`split_demo_at_feature` ([decision_tree.py:102](roboverify/synthesis/mcmc/decision_tree.py:102))
returns bare `demo[:split_idx+1]` / `demo[split_idx+1:]` slices, so after a second split the
absolute timestep is unrecoverable. `bindings` records which object was bound on the
segment — that *is* LoopGuardSynthesis's positive example `(s, b_s)`.

**Lowering folds a recorded region tree; it does not recover structure.** Quotient is the
only constructor of non-linear shapes and produces single-entry reducible regions by
construction, so store the derivation in `Node.region` and walk it. `LoopRegion` lowers to
exactly the shape hand-written at
[verify_stack_with_learned_invariant.py:48](roboverify/synthesis/entry/verify_stack_with_learned_invariant.py:48).

**One new instruction** in `synthesis/api/instructions.py`: `Get(var, cond, exists_vars)`, a
one-shot binder — `While._find_and_bind_guard_exists` with the loop stripped off. Algorithm 5
needs it to close over object identifiers left free in a synthesized block
(`v := get(∃v : Λ(o). True)`). Extract `_eval_z3_guard` / `_find_and_bind_guard_exists`
([instructions.py:902–1060](roboverify/synthesis/api/instructions.py:902)) into
`synthesis/api/guard_eval.py` so `Get` and `While` share one evaluator, with `While`
delegating so existing behavior is unchanged. `Get` needs `wp` and `VC_aux` cases —
`wp(get, Q) = ∀v. G(v) ⇒ Q` per Table 1, structurally a havoc-plus-assume on a fresh name.

**Test — the whole point of the stage:** hand-build the unstack CFG, lower it, assert
structural equality with the hand-written `Program` at
[verify_unstack_with_learned_invariant.py:80](roboverify/synthesis/entry/verify_unstack_with_learned_invariant.py:80),
then assert `highlevel_verification` returns the same verdict for both. That proves the IR
is expressive enough and that verification is unaffected.

### F2 — `Validate` §3.5 (small)

Pure function over absolute indices: reject a split when `i_s(d,ψ) ≤ i_f(d,φ)`, or
`i_s(d,ψ) = 0` when φ is `φ_pre`. Reuse the first-true scan at
[decision_tree.py:94](roboverify/synthesis/mcmc/decision_tree.py:94). No ML, no MuJoCo, no
solving — fully unit-testable with synthetic segments.

### F3 — Predicate language and enumerator (medium; no prerequisites)

**Architectural decision: own the term representation; lower to Z3 only at the solver
boundary.** `synthesis/predicates/term.py` defines the canonical formula ADT — named
binders, structural equality, hash-consed — and `to_z3(term, context)` is the single
conversion, called where the solver is called. Nothing ever parses Z3 back into internal
form; that reverse direction is the one that hurts, and it is not needed.

This works because **Quotient never has to see a Z3 expression.** The Kleene letters are
edge labels and loop guards, both produced by the enumerator; invariants stay Z3 but
Quotient does not match on invariants. Consequences:

- *Anti-unification* (F7) becomes a ~30-line recursion on an ADT — same constructor and
  arity, recurse; otherwise mint a fresh template variable, memoized so a pair maps
  consistently across both substitutions. On Z3 ASTs it means fighting `decl()`,
  `children()`, `is_var`/`get_var_index` and manual quantifier re-wrapping.
- *Alpha-equivalence* disappears: with canonical renaming, alpha-equivalent is structurally
  equal. Z3's de Bruijn indices are correct but require tracking binder depth during
  traversal — exactly where both existing attempts in this repo went partial
  (`While._eval_z3_guard`, and the `rewrite_for_put_*` binder inconsistency).
- **`_eval_z3_guard` can be deleted** ([instructions.py:902–1008](roboverify/synthesis/api/instructions.py:902),
  109 lines): the guard's runtime interpreter becomes the enumerator's own evaluator, since
  `topdown.py`'s nodes already carry `evaluate_specific`. One representation, one evaluator.

**The cost, which must be paid explicitly:** two semantics — the internal evaluator and the
Z3 lowering — can drift. That is the same failure already present as `on.py` vs
`lowlevel_verification_lib.py` (Phase B). Pin it with a differential test: over random
terms and random scenes, internal evaluation must agree with evaluating `to_z3(term)` under
the axioms. Also separate `Term` from `SearchNode` — `topdown.py`'s classes are search
objects with holes (`expand()`, `is_complete()`), not a clean term ADT.

Salvage `synthesis/topdown/`, which already has the grammar
(`Exists|ForAll|Not|And|Or|ON_star|Equal`, [topdown.py:105–333](roboverify/synthesis/topdown/topdown.py:105)).
Three defects block reuse:

1. **`# return program` is commented out** ([topdown.py:391](roboverify/synthesis/topdown/topdown.py:391)) — the search prints its answer and returns `None`.
2. **Quantifier depth is enforced by counting `"∀"`/`"∃"` characters in the printed form** (`:395`) while every node's `quantifier_count()` sits dead.
3. **No observation→scene adapter.** Input is `{"target", "all_box", "constants", "result"}` built by hand at `:406/:546/:734`. `synthesis/predicates/scene.py:scene_from_obs`, over `util/on.py:get_block_pos`, is the single most important missing connector in the repo.

Beyond the defects, the unbounded BFS frontier (`deque.popleft`, a printed line per
candidate) is not the paper's "bottom-up over increasing depth and m". Replace with
iterative deepening over `(depth, m)` so "first separator" means "smallest" and budget
exhaustion is a clean failure. Two modes on one enumerator: **LearnClassifier** (forced
outermost ∃, variables from `scope(v)`) and **LoopGuardSynthesis** (full first-order
vocabulary including ∀, as §3.3 specifies).

Delete `synthesis/topdown/dsl.py` (276 lines, near-duplicate, zero importers) and
`synthesis/entry/topdown.py`. Keep the three hardcoded datasets as the regression corpus.

**Cheap spike, ten minutes:** fix the `return` and *time* `topdown_synthesize` on
`run_unstack_hardcoded_dataset()`. It runs once per CFG block per Synthesize round; if it
takes minutes on hand-written 4-box data, the iterative-deepening rewrite is load-bearing
rather than cosmetic. This repriced F3, F4 and F7 at once.

### F4 — `scope`, `LearnClassifier`, `RefineCFG` (medium)

`scope(v)` is the standard all-paths must-reach fixed point: a loop guard's bound variable
is in scope inside the body but not at the exit, since the exit path bypasses the guard
edge. With branches out of scope the CFG is chains and loops only, so there are no join
points and scope accumulates monotonically along each chain — keep the `∩`-at-joins
formulation anyway, so adding branches later is a no-op here.

`refine_cfg` implements Algorithm 3 with the partitioning **in the same procedure** — the
lockstep is the point, do not split it into two passes.

**Positive/negative sets change materially.** `compute_positive_set`
([decision_tree.py:235](roboverify/synthesis/mcmc/decision_tree.py:235)) takes exactly one
state per demo; the paper needs every transition witness
`P_s = {s_t | s_t ⊭ φ ∧ s_{t+1} ⊨ φ}`. `compute_negative_set` takes every state of a
*random* program's trajectory; `N_s` must come from executing `G` in `M` (`cfg/execute.py`).

**`decision_tree.py` splits three ways.** *Retire* `learn_features`, `compute_ON_features`,
`compute_features`, `compute_positive_set`, `compute_negative_set`, `check_feature` and the
`sklearn`/`matplotlib` dependency — the hypothesis class is a single ground `ON(i,j)` atom
at `max_depth=1`, which cannot express `∃x̄. φ_QF`, cannot scale past a fixed block count,
and writes `decision_tree{idx}.png` to CWD unconditionally (`:67`; ten such files sit in the
repo root now). *Promote* `ON_feature` → `predicates/atoms.py:GroundON`, keeping `.b1`,
`.b2`, `.reward`, since `bmc_goal_from_on_feature` and `goal_feature_reward_at_execution_end`
both consume it. *Move* the split helpers (`:94–177`) into `cfg/refine.py`, re-emitting
`DemoSegment` with absolute indices and pushing the mp4 side effects out through a callback
so `refine.py` needs no `synthesis.mcmc.synthesis` import (currently a lazy one at `:143`).

### F5 — `StraightLineSynthesize`, Algorithm 5 (large)

Per conflicts 2 and 3: KL behind a `TrajectoryDistance` protocol, the ε-pool, `PostScore`
filter-then-rank, annealed `T_k`. Today's MCMC has `T ≡ 1`
([synthesis.py:580](roboverify/synthesis/mcmc/synthesis.py:580)), no pool, and a weighted-sum
objective. `PostScore` reuses `rollout_demos_from_initial_states`
([synthesis.py:1132](roboverify/synthesis/mcmc/synthesis.py:1132)), which already does
exactly this and is currently used only for stage-2 negatives.

**Depends on Phase A.5.** The frozen-CEM-directions bug lives in the code F5 extends, so fix
the RNG isolation before tuning anything here — otherwise annealing, the ε-pool and
`PostScore` all get evaluated against a crippled inner optimizer and the measurements mean
nothing.

**Gate before wiring: time one `StraightLineSynthesize` call with `PostScore` enabled.**
`PostScore` costs a rollout batch per pool member and runs on every convergence check, so it
lands inside the innermost loop. Measure it against the current objective-only cost with
`--smoke` before committing to the pool size; if the ratio is bad, cap the pool harder or
score `PostScore` only at convergence rather than maintaining it continuously.

**Refactor before extending.** `synthesis/experiment/mcmc/search.py` duplicates `MCMC`,
`score_candidate_program`, `optimize_program` and `Runner`, so every change must land twice
or `test_mcmc_parity.py` breaks. First extract the shared decision points —
`acceptance_probability(delta, T)`, `CandidatePool`, the objective — into `synthesis/mcmc/`,
have both copies call them, and re-pin parity at the shared-helper level. Each new behavior
lands as an optional argument defaulting to current behavior.

### F6 — Algorithm 2 driver and a real entry point (medium)

`cfg/synthesize.py` implements Algorithm 2 literally, with `Quotient` stubbed to identity.
`synthesis/entry/synthesize_cfg.py` reuses the experiment contract, recording CFG moments
(`refine`, `validate_reject`, `quotient_loop_found`) alongside per-block search progress.
`log_event` already rate-limits per kind, so no change to `run_logger.py` is needed for the
records themselves.

**Check nesting before relying on it.** The natural design is one `RunLogger` run-dir per
`StraightLineSynthesize` call under a top-level CFG run, but `_start_stdout_capture`
([run_logger.py:209](roboverify/synthesis/experiment/run_logger.py:209)) does `os.dup2` onto
fds 1 and 2 — process-global — and every logger registers its own `atexit` hook. Strict LIFO
nesting likely works, since the inner logger saves and restores whatever the outer installed,
but it has never been exercised. Either add a nesting test first, or use a single top-level
logger with a `block_id` field on each metrics record — the simpler option, and it keeps
`report.py`'s aggregate view intact.

**Rewrite `synthesis/entry/main.py`** — 426 lines, one `if __name__` block, no functions, a
hard-coded non-recursive 2-way split, stage-2 feature learning with no stage-2 MCMC, and
dead code from `:217` referencing a nonexistent `program.PickP` at `:290`. Replace with a
thin shim. **Preserve** the hand-written unstack `While` demo generator (`:16–69`) as
`cfg/demo_sources.py:unstack_oracle()` — it is the ground-truth program the pipeline is
meant to re-derive, so it belongs as a named fixture.

### F7 — `Quotient`, Algorithm 4, flat case only (medium)

**Descoping Grid and Pyramid removes most of this stage's difficulty.** Every in-scope task
is a flat single loop, so there is no nested-loop consumer for the Kleene-star machinery.
Build the paper's flat case, which it describes directly: "each sub-sequence is a single
basic block, and matching reduces to checking whether two atomic edge labels share a
predicate template" — `on(b₁,b₀)` and `on(b₂,b₁)` both matching `on(x',x)`.

**In scope:** `encode(region)` for the non-starred fragment (`⟦π_φ⟧ = φ` with get-bound
arguments marked, `⟦S₁;S₂⟧ = ⟦S₁⟧·⟦S₂⟧`, assignments erased); `alpha_equal` and
`anti_unify` over F3's term type; Phase 1's scan for the first adjacent equal-length pair
matching a template, shorter `l` first; deriving init `x ← σ₁(x)` and update
`x ← σ₁⁻¹(σ₂(x))`; `extract_iterations`; `loop_guard_synthesis`.

Keep the length-`l` scan even though the block-tower loops have `l = 1` — a repeated unit
of two or more blocks is still flat and costs nothing extra.

**Deferred with Grid/Pyramid:** the starred production
`⟦while (∃x.g(x)) do S⟧ = (g(x̲)·⟦S⟧)* · ¬(∃x.g(x))`, matching where starred subterms match
only starred subterms of identical shape, and Phase 2's recursion into loop bodies. Shape
`encode`'s return type as a word over letters that *could* carry a star, so adding nesting
later is an extension rather than a rewrite, but do not build star matching now.

**What remains genuinely hard**, and why F3 matters: the matching step is still
second-order — solve for an unknown template plus two substitutions, not check an equality.
Over Z3 ASTs that would be serious (alpha-equivalence over de Bruijn binders,
`ExprRef.__eq__` returning a constraint, two partial attempts at exactly this already in the
repo). **F3's canonical term ADT removes that**, leaving anti-unification as a recursion
over a datatype. The one subtlety that survives:

- `x ← σ₁⁻¹(σ₂(x))` requires `σ₁` injective on the relevant variables, and its **undefined
  case carries meaning**: `σ₂(b') = b₂` is unmapped by `σ₁`, and that undefinedness is
  precisely the signal that `b'` is not loop-carried and must be rebound by the guard each
  iteration. The composition must detect partiality and branch on it. There is no worked
  example of this case anywhere in the code.

**Spike (~half a day, no pipeline needed).** Hand-write the *unrolled* 3-iteration unstack
CFG as a fixture, with edge labels lifted from the guard at `main.py:22`. Implement only
`encode_fragment`, `alpha_equal`, and `anti_unify` over F3's term type. Check three things:
the three iterations yield equal-length encodings; anti-unifying adjacent pairs gives a
template whose σ₁/σ₂ differ exactly in the `b`/`b'` binding; and `σ₁⁻¹ ∘ σ₂` reproduces the
`Assign("b","b_prime")` at `main.py:54`.

**Two details that will bite.** `While.max_iters` defaults to 10 and silently truncates
([instructions.py:851](roboverify/synthesis/api/instructions.py:851)) — derive it from
`max(len(iterations))`. And `_find_and_bind_guard_exists` returns the *first* satisfying
binding in block-id order, while LoopGuardSynthesis learns from one positive per state;
if `G` admits several witnesses, runtime picks by index while the demo picked by something
else. Make witness uniqueness a hard constraint in guard synthesis, not merely a set of
negative examples, and warn when only a weaker guard is found.

### Sizes

| stage | size | prereqs |
|---|---|---|
| F0 Segment-start reset | medium | — |
| F1 IR + lowering + `Get` | medium | F3a |
| F2 Validate | small | F1 |
| F3 Predicate language | medium | — |
| F4 scope + LearnClassifier + RefineCFG | medium | F1, F2, F3 |
| F5 StraightLineSynthesize | large | F0, F1 |
| F6 Alg 2 driver + entry point | medium | F1, F2, F4, F5 |
| F7 Quotient (flat case) | medium | F1, F3, F4 |

---

## Conflicts

Adjudicated case by case. Each is verified in the code, not taken from the paper.
**All seven are settled.** 2–7 carry code actions, listed in the phases above. Item 1
carries **no code action** — the implementation is faithful; what needs fixing is the
paper, and the choice there determines what the AFR fragment checker should flag.

| # | Conflict | Status / recommendation |
|---|---|---|
| 1 | **Theorem 5.2 contradicts the paper's own Table 7 — the code is not at fault.** `rewrite_for_put_for_higher` ([program.py:779–811](roboverify/synthesis/api/program.py:779)) reproduces Table 7's `R_Higher` for `put(b',b)` **disjunct for disjunct, all 8**, including the `∃t` in disjunct 2 and the `∀t` in disjunct 6; `rewrite_for_put_on_tbl_for_Higher` ([:670](roboverify/synthesis/api/program.py:670)) likewise matches the `∀t` in Table 7's tbl case. | **Paper-side; no code change.** The inconsistency is internal to the paper, and it defeats the theorem's *statement*, not just its proof sketch. Definition 5.1 admits AFR only as QFR or a single quantifier block over a **quantifier-free** matrix. Theorem 5.2's justification claims the operators work "without introducing any quantifier… only ever combin[ing] existing quantifier-free atoms with ∧, ∨, and substitution" — but rewriting a `Higher` atom inside `Q = ∀u,v. ψ` yields `∀u,v. ψ'` whose matrix now contains `∃t`, so the result is **not AFR**, and the `∀∃` alternation it creates is precisely what Theorem 5.3 exists to characterize. Scope is narrow: `R_⟨on*⟩` is purely propositional and `R_Scattered` is five quantifier-free disjuncts, so `Higher` is the only offender and `:670`, `:785`, `:798` are the only quantifier introductions in the wp path. Options for the paper: weaken Theorem 5.2 to "AFR modulo the `Higher` rewrite", widen Definition 5.1's AFR to permit a non-alternating inner block, or fold the `Higher` case into Theorem 5.3. **Recorded in `PAPER-DISCREPANCIES.md` (Phase A.5) and deferred until the code is settled**, but worth deciding before the fragment checker is written, since it determines what the checker should flag. |
| 2 | **KL (paper) vs MMD (code).** `Runner.__call__` uses `-MMD²_RBF` ([synthesis.py:1404](roboverify/synthesis/mcmc/synthesis.py:1404)); the KL estimators at `cost_func.py:58,82` are dead code. | **DECIDED — use the paper's design where needed.** Revive `kl_divergence_kde`/`kl_divergence_gmm` behind a `TrajectoryDistance` protocol; MMD stays selectable. KL is required wherever the paper's algorithm depends on its *scale* — the `KL < θ` convergence test and the ε-pool band (conflict 3) — because MMD supplies no comparable threshold. Implementation note, not an objection: both estimators are sampling-based (`n_samples=10000`) and would sit inside the CEM inner loop that already drives MuJoCo, so cache the demo-side density across CEM iterations (`τ_D` is fixed per block) and cut `n_samples` for the inner loop, keeping the full estimate for the outer accept/convergence test. Measure before tuning. |
| 3 | **Weighted-sum goal reward vs pool + PostScore.** Code: `score = -MMD + w·goal_reward`, `w=1.0`. Paper: distance alone, then `argmax_{ρ∈pool} PostScore`. | **DECIDED — implement the paper's.** These are genuinely different algorithms: the weighted sum lets a candidate trade imitation fidelity against goal achievement, which filter-then-rank exists to prevent. MCMC optimizes distance alone; goal satisfaction enters only as `argmax_{ρ∈pool} PostScore` among candidates already within ε of the best. Keep the weighted sum reachable via `goal_feature_reward_weight=0.0` for A/B comparison. Cap the pool (≈10) and score `PostScore` on demo seeds only, since each pool member costs a rollout batch. (Phase F.) |
| 4 | **The `axioms ∧ premise` sat check** ([program.py:1105](roboverify/synthesis/api/program.py:1105)) exists to rule out vacuous discharge — if the premise contradicts the axioms, `Implies(P,Q)` is valid having proved nothing. Correct and worth keeping. The defect is that it collapses into the same boolean `ok` as a genuine refutation, and yields `model1 = None`, so CEGIS has no `s0` to learn from and the iteration is unrecoverable. | **Keep the check; make the verdict three-valued** (`VALID` / `INVALID(model)` / `VACUOUS`) — see Phase E.1. Vacuity needs its own branch because it is a spec/invariant bug with no counterexample trace to learn from, so the CEGIS response differs. **It can usually be detected from the validity query itself**, without a second solver call, via the unsat core. |
| 5 | **`Top`** — declared ([highlevel_verification_lib.py:92](roboverify/synthesis/verification_lib/highlevel_verification_lib.py:92)), selectable into `Ω_Inv` ([inference.py:341](roboverify/synthesis/inference_lib/inference.py:341)), never axiomatized, never rewritten by wp, raises `NotImplementedError` in the geometric translator. | **DECIDED — delete it.** It is debug shorthand, not part of the predicate vocabulary. Remove the `Function` declaration, the `Ω_Inv` branch, the `spec_to_expr` decl entry ([highlevel_verification_lib.py:741](roboverify/synthesis/verification_lib/highlevel_verification_lib.py:741)) and the translator case. Cleaner than the macro-expansion I first proposed, and removes the wp hole outright rather than papering over it. |
| 6 | **`ON_star_zero`** is axiomatized as a clone of on1–on5 with nothing linking it to `ON_star`. | **DECIDED — keep it, and keep it unlinked.** It denotes the environment's configuration *at program entry*, which is what specs over the initial state need (Reverse must relate the final tower to the original order). It enters via the invariant vocabulary — `run_reverse_example` passes `[ON_star, ON_star_zero, equality]` ([inference.py:2505](roboverify/synthesis/inference_lib/inference.py:2505)) — and is evaluated against `state_zero` during learning ([inference.py:376](roboverify/synthesis/inference_lib/inference.py:376)). Two corrections to my earlier note: passing through `wp(Put, ·)` unchanged is right *because `put` cannot alter the initial state*; and **the link axiom I suggested is unsound** — for Reverse the final order is the inverse of the initial, so `ON_star_zero(x,y) → ON_star(x,y)` is false by construction. The codebase already knows this: those exact axioms sit commented out at `inference.py:1611, 1725, 1736, 1745`. The separate on_zero_1–5 clone is correct, since `ON_star_zero` is a genuine reflexive-transitive closure over its own frozen state. **Also decided:** for every environment whose spec or invariant uses `ON_star_zero`, the **precondition must state that the two relations coincide on all pairs at program entry** — `∀x,y. ON_star_zero(x,y) ↔ ON_star(x,y)` conjoined into `φ_pre`. Without it `ON_star_zero` is unconstrained relative to the actual initial configuration, so `φ_pre ⇒ I[b:=b0]` fails spuriously whenever `I` mentions it. Putting it in `φ_pre` rather than in the axioms is what keeps it entry-time-only: `put` never rewrites `ON_star_zero`, so the two relations correctly diverge as the program runs — which is exactly what Reverse depends on. Add it to the Reverse entry script and any other task whose vocabulary includes `ON_star_zero` (`run_reverse_example` passes it at [inference.py:2505](roboverify/synthesis/inference_lib/inference.py:2505)). |
| 7 | **`Move` trains only `z`** — `register_trainable_parameter`/`update_trainable_parameter` slice `target_offset[2:]` ([instructions.py:372](roboverify/synthesis/api/instructions.py:372), commit `cca6e27`), freezing x and y. | **DECIDED — use the paper's format: unfreeze x and y.** Algorithm 5 optimizes the mutated program's continuous parameters, so CEM gets all three components. Small, self-contained change to the two methods; can land independently of everything else. Expect the CEM vector to grow 3× per `Move`, so re-check `cem_N`/`cem_K` budgets. This also removes a source of *spurious* `StraightLineSynthesize` failures in Phase F — with x/y pinned a `Move` can only travel vertically over a block, making many block postconditions unreachable and driving refinement splits the paper would not produce. |

---

## Verification

After each phase:

```bash
cd roboverify && unset LD_PRELOAD
uv run python -m unittest synthesis.verification_lib.test_bmc_lib -v
uv run python -m unittest synthesis.experiment.test_run_logger -v
bash format.sh
```

End-to-end, the existing entry points are the regression oracle — their verdicts must not
change in Phases A–B, and must become *stricter* (never looser) in D–E:

```bash
uv run python -m synthesis.entry.verify_stack_with_learned_invariant
uv run python -m synthesis.entry.verify_stack_with_learned_invariant --verification-mode finite --num-blocks 4
uv run python -m synthesis.entry.verify_unstack_with_learned_invariant
```

For Phase E, the **acceptance criterion is a property, not a number**: starting from
`I = False`, each CEGIS iteration must strictly enlarge the set of loop-head states the
invariant covers (monotone progress in the finite space `SpecInv`), and the loop must
terminate with an invariant that discharges every VC. That is checkable without reference
to the paper.

The paper's Table 2 (4–7 iterations per environment) and Table 3 (the Stack progression
`False → ∀u,v.u⟨on*⟩v → spurious 3-conjunct → Fig. 6`) are recorded as a **comparison
datapoint, not a target**. A different iteration count is a fact to report and
investigate, not a defect to engineer away — it may equally indicate the table is wrong.
Log the actual progression so the two can be compared, and do not tune toward the
published numbers.

For Phase F, the acceptance criterion is likewise structural rather than numeric: running
`synthesize_cfg` on the unstack demos must **re-derive** a CFG that lowers to a program
verifying against the same spec the hand-written one does. `cfg/demo_sources.py:unstack_oracle()`
is the ground truth to compare against — the pipeline should rediscover the loop, guard and
`b ← b'` update that `main.py:16–69` currently hard-codes. Partial credit is meaningful and
should be reported: F6 with `Quotient` stubbed will produce a correct *acyclic* CFG for a
fixed block count, and only F7 generalizes it to a loop.

New tests are `unittest`, colocated as `test_*.py` inside their package, matching
`synthesis/verification_lib/test_bmc_lib.py`. Add `synthesis/cfg/` and
`synthesis/predicates/` to `format.sh`, which globs explicit paths and will otherwise
silently skip them.
