# Synthesis and verification algorithm audit

Remediation tracking: the A1–A11 and D1–D3 checklist in
[PLAN-popl-alignment.md](PLAN-popl-alignment.md) records fixes and validation.
This document preserves the findings at the audited baseline.

Audit date: 2026-09-20. Reviewed implementation: `132cfee`, branch
`phase-f-synthesis`. The working tree initially contained only the three previously
known unrelated untracked paths.

**The paper's complete synthesis/verification algorithm is not implemented yet.
Correct demonstrations alone will not close the gaps.** There are both missing
connections and reproducible correctness defects. The earlier statement that only
learning success remained uncertain was too strong.

This review compares the local `POPL2027.pdf`, the active implementation plan,
and the synthesis, inference, execution, and verification code. It uses the
algorithmic material in §§2–5, Algorithms 1–6, Appendix A, and the corresponding
appendix algorithms/proofs (7–10). Experimental results, reported learned programs,
and historical demonstrations are not correctness targets. Algorithm 4 and Table 7
were also inspected as rendered PDF pages; Table 7's long Higher formula is visibly
clipped at the right page boundary, so exact agreement with its missing tail cannot
be established from this PDF.

Existing tests: **133 passed in 25.822 seconds** during this audit. Five additional
synthetic probes below reproduced uncovered problems without saved demonstrations.
Passing the suite establishes regression coverage, not full algorithm conformance.
No implementation fixes were made during this review.

## Findings requiring code or integration work

### A1. Motion success does not establish all effects assumed by symbolic Put — high priority

Paper: §5.5, pp. 28–32, especially alignment equations (6)/(7), Theorem 5.6;
Appendix J, pp. 58–59; Table 7, p. 44.

Code: `roboverify/synthesis/verification_lib/motion_verification.py:393` checks
released/direct-on or table height; `:424` checks the original tower's frame.
`roboverify/synthesis/api/program.py:691` rewrites Scattered after table placement.
`roboverify/synthesis/verification_lib/lowlevel_verification_lib.py:239` interprets
ON* using a fixed horizontal tolerance.

Two independent synthetic examples expose the missing bridge:

- **No root-alignment obligation.** With block length 0.05, root x=0, current top
  x=0.024, and a new block placed at x=0.048, the new block is locally on the top,
  and the top is on the root. All current motion checks pass in the concrete clear
  scene, but geometric ON*(new, root) is false because 0.048 exceeds 0.025.
  Symbolic reachability would infer that relation by transitivity. There is no
  root selection, root-relative placement obligation, or separate chain-bound
  certification corresponding to equations (6)/(7).
- **Table placement does not establish separation.** A released block at table
  height, 0.075 m horizontally from another block, passes MotionVerify. Numeric
  Scattered is false (threshold 0.1), yet `wp(Put(a,tbl), Scattered(a,b0))`
  simplifies to `a != b0`. The symbolic action assumes an effect the checked
  physical contract does not establish. Frame preservation does not supply it.

These are successes of the isolated motion checker, not demonstrations that the
current CLI has certified a complete incorrect task. They show why composing
its verdict with the symbolic action requires additional obligations.

The plan's D2 explicitly requested direct placement and tower frame checks but
omitted the paper's alignment certification and full relation-effect agreement.
This is a **plan coverage gap as well as an implementation gap**. Do not merely
copy the paper's tolerance values: reconcile direct-on, ON*, root bounds, and
all relations consumed by the WP rules, then check the agreed action abstraction.

### A2. Executable loops stop at a demonstration-derived cap absent from their proof — high priority

Paper: §2.2, p. 5 and §3.3, pp. 15–16: continue while a guard witness exists;
the intended generalization is to more objects than in the demonstrations.

Code: `roboverify/synthesis/cfg/region.py:25` sets max_iters to the maximum recorded
iteration count; `cfg/lower.py:31` passes that cap into While.
`api/instructions.py:946` breaks when the cap is exceeded, even if the guard is
still true. `api/program.py:1007` generates preservation/exit VCs for the ordinary
guard, with no obligation for termination at the cap.

F7 explicitly requested the maximum observed count, so this implementation follows
the plan; the plan failed to reconcile that bound with the proof semantics.

A loop recovered from three iterations therefore executes at most three iterations
on a larger scene. The runtime can leave the loop without the negated guard that
its exit proof assumes. A resource limit must surface as incomplete execution,
or verification must explicitly include its exit behavior. An unbounded symbolic
verdict does not justify this capped runtime.

Separately, the paper's total-termination claim needs a progress argument; ordinary
invariant VCs alone establish partial correctness (see paper issues below).

### A3. Anti-unification can capture a pre-existing program name — confirmed bug

Paper: §3.3, pp. 14–16: a template and substitutions must reconstruct the matched
fragments; template variables must be fresh.

Code: `roboverify/synthesis/cfg/kleene.py:41` creates p0, p1, ... without excluding
existing free names. Anti-unifying `ON(a,p0)` and `ON(b,p0)` returns `ON(p0,p0)`.
Substituting either recorded mapping fails to reconstruct the original predicate.
This corrupts the relational template. The earlier predicate-enumerator freshness
fix does not cover this separate naming path.

### A4. Later iterations need not obey the inferred carried update — confirmed bug

Paper: Definition 3.2, p. 16 requires the same adjacent-substitution composition
throughout the merged sequence.

Code: `roboverify/synthesis/cfg/quotient.py:58` extends a repetition using only
`match_template`; `:149` derives carried bindings from the first two substitutions.

The sequence `ON(b1,b0), ON(b2,b1), ON(b4,b3)` is accepted as three repetitions.
The first two infer “current <- selected”; the third requires current=b3 rather
than b2. The check should stop the repetition or reject the inconsistent extension.
The probe establishes a matching defect; a subsequent guard check may reject some
such graphs, but it does not replace the missing structural condition.

### A5. Flat loop recovery implements only part of Algorithm 4

Paper: Algorithm 4, p. 15; extraction explanation p. 16 and pp. 55–56.
Plan: F7 explicitly includes the flat case and ExtractIterations.

Code: `roboverify/synthesis/cfg/quotient.py:70`, `:128`, `:250`, `:323`;
`cfg/synthesize.py:37`; `entry/synthesize_cfg.py:140`.

- ExtractIterations groups already-partitioned node segments. It never scans
  the remaining demonstration for the next template-satisfying state and updates
  bindings to discover additional iterations. Its fixed count comes from matched
  CFG labels, and the stored count is identical for every demonstration.
  The paper explicitly recovers all demonstrated iterations, including more than
  the pair that triggered loop detection.
- Folding requires existing BlockRegions in the first repeated unit. RefineCFG
  creates unresolved nodes, and Algorithm 2 calls quotient before realization.
  The paper's structural quotient can create an unknown loop body before those
  blocks have been synthesized; the current fold cannot always do so.
- A quotient call performs at most one collapse. The driver calls it once per
  refinement round and can return immediately after successful realization.
  Algorithm 4's repeat-to-fixed-point at the current level is absent.
- Failed realization inside a LoopRegion returns the outer loop as the failed
  node; the driver invokes ordinary top-level RefineCFG on that node. There is
  no body CFG refinement path retaining the discovered loop structure.

These limitations concern flat loops and their basic blocks. They are separate
from the deliberately deferred nested/starred quotient machinery.

### A6. Algorithm 6 does not operate on the program produced by Algorithm 2

Paper: Algorithm 6, p. 34 (duplicate Algorithm 10, p. 59), §5.7.
Plan: E3 explicitly defers the NeedsResynthesis branch until F6.

Code: `roboverify/synthesis/entry/synthesize_cfg.py:217` retains `symbolic=None`
for arbitrary physical candidates and `:323` always records
`formal_verification="not_run"`. `cfg/lower.py:15` correctly refuses symbolic
lowering without a relational summary. `entry/verified_synthesis.py:33` instead
builds the existing Stack fixture; `:79` records NeedsResynthesis and exits.

Missing: attach/check supported symbolic summaries for synthesized blocks; pass
that same CFG/program, task specification, bindings and demonstrations into
symbolic and motion verification; route structural counterexamples and added
user demonstrations back to that CFG; retain per-block penalties during repair.

The task specifications also need to be shared: `entry/synthesize_cfg.py:46` uses
True as the Stack precondition, while `entry/verified_synthesis.py:47` requires
unstacked, scattered blocks. Connecting the CLIs without reconciling this would
verify a different input domain from the synthesis request.

The motion CEGIS library currently preserves instruction structure/operands and
repairs offsets (`verification_lib/cegis.py:415`, `:459`). The paper re-invokes
straight-line instruction synthesis with the penalty objective. The plan recorded
this restriction as temporary Phase E scope. Phase F has not closed it.

### A7. Motion verification does not cover the synthesis instruction language or general CFG positions

Paper: §5.5, pp. 26–31: symbolic execution of Pick/Move/Release and forward
propagation through consecutive blocks, loops, and branches.

Code: `roboverify/synthesis/verification_lib/motion_verification.py:254` supports
PickPlaceByName waypoints, Assign, and Skip; primitive Pick/Move/Release and named
variants are unsupported. `api/program.py:1207` verifies loop bodies, rejecting
uncovered straight-line motion before or after loops. F's search produces exactly
the primitive controllers this motion checker cannot consume.

The rejection is sound and explicit; it is still a missing algorithmic connection.
There is no general Block/Seq context propagation for synthesized CFG blocks.
BMC's separate primitive encoding checks bounded goals and is not a substitute
for motion collision/frame/contract verification. Noise is opt-in, and controller,
arm, table-plane and settling limitations remain as documented in discrepancy 4.
Branches are intentionally outside the current plan; ordinary straight-line
blocks and the selected synthesis primitives are not removed by that decision.

### A8. Concrete object IDs are not generally closed into executable bindings

Paper: §3.4, p. 18: introduce Get bindings for object identifiers in returned
straight-line programs. Plan F1 explicitly describes this consumer of Get.

Code: `roboverify/synthesis/entry/synthesize_cfg.py:166` searches numeric object
IDs. `cfg/lower.py:47` emits Get only for bindings already on refinement edges.
`cfg/quotient.py:260` generalizes some numeric operands during folding, but there
is no general closing/generalization pass for a successful straight-line block.

The named mutation path at `entry/synthesize_cfg.py:179` also draws names from
all demo bindings, rather than the block's proven runtime scope. Demo-only names
such as o0 can enter candidates even though only b0 and emitted bindings are
available in an ordinary execution. Scoring from recorded segments can mask this.

### A9. Predicate search is narrower than the paper's declared class

Paper: §3.2.1, p. 13 explicitly allows m=0 fresh existential variables.

Code: `roboverify/synthesis/predicates/enumerate.py:62` starts classifier counts at
one. With max_variables=0, a ground Higher(a,b) separator exists but the classifier
examines zero candidates and returns no_separator. At larger budgets a vacuous
existential can sometimes express the same predicate, but consumes extra depth
and can introduce an unnecessary binding.

`predicates/language.py:6` also omits direct ON from the default relation list,
although ON is implemented and is the paper's principal refinement example.
This should be an explicit vocabulary choice rather than an assumed reproduction
of that example. Bounded depth, candidates and time are sensible implementation
limits, already anticipated by the plan.

### A10. Validate covers a proposed split, not the paper's whole-CFG check

Paper: §3.5, p. 19 and p. 57 require checks on edges across both adjacent blocks'
assigned segments. Plan F1 emphasizes absolute indices for that cross-block use.

Code: `roboverify/synthesis/cfg/refine.py:59` checks first/last indices only inside
the currently failed node's segments; `cfg/validate.py` operates on those maps.
There is no traversal validating all relevant edges and neighboring assignments,
or a corresponding check after quotienting. Atomic mutation and absolute indices
are implemented correctly as foundations, but are not that entire validation step.

The paper's inequality wording is itself ambiguous about pass versus rejection;
resolve its temporal meaning before extending validation. This is a coverage
finding from inspection, not a reproduced incorrect acceptance.

### A11. The active invariant path differs from §4, and F narrows its inputs further

Paper: §4, pp. 20–22 and Appendix F describe feature-subset optimization, Boolean
minimization, and sufficient/necessary clauses for each target predicate.

That learner exists in `roboverify/synthesis/inference_lib/inference.py:606`
(feature selection) and `:1130` onward (partition-based inference), accessible
through DemoStore/InvInference. It is not absent from the repository.

The default E learner (`verification_lib/cegis.py:48`) instead builds the strongest
universal formula from observed truth-table rows. This is an intentional,
documented monotonicity choice, not the same learning algorithm. F's
`entry/synthesize_cfg.py:237` hardwires that learner to ON*/equality with two
variables and records iteration starts only. Higher, Scattered, frozen relations,
and terminal loop-head rows are not supplied through this path. Those distinctions
matter for motion reasoning and exit invariants even with correct demonstrations.

## Parts that are present, and deliberate differences

| Paper component | Current implementation and qualification |
|---|---|
| Relational CFG, scoped bindings, recursive splitting | Present as structured chain/loop IR, absolute demo segments, Get, must-scope analysis, and all transition-witness positives. Limitations A5/A8/A10 remain. |
| Algorithm 5's main search | MCMC mutation, CEM, annealed acceptance, cached KL estimates, epsilon pool and PostScore ranking are present. A capped pool and PostScore=1 acceptance are recorded plan deviations. Reach-any-state PostScore still is not final-state correctness. |
| Faithful segment starts | Snapshot/replay and frozen entry geometry are implemented; this fills a paper omission rather than matching an experimental number. |
| Symbolic WP/VC generation | Skip/sequence/Assign/Put/Get/While, establishment/preservation/exit obligations are present. This is not a blanket proof that every rewrite implements physical motion (A1). |
| Verification verdicts | Valid/invalid/vacuous/unknown, finite countermodel search, optional unbounded checks, timeouts and relation-preserving concretization are implemented. Finite success is not promoted to unbounded success. |
| Counterexample feedback | Symbolic successor refinement and accumulated motion penalties exist for supported cases. The full synthesis feedback connection is missing (A6). |
| Guard learning | Global exact separation, exit negatives and a uniqueness VC are stronger/different than the paper's per-state conjunction with True fallback. Recorded in the plan/discrepancy 11. |
| Get | Code requires witness existence as well as correctness for every permitted choice; the paper's Table 1 omits existence despite runtime failure without a witness. Recorded discrepancy 9. |
| Branches and nested loops | Deliberately deferred by the plan, as are Grid/Pyramid. This audit does not reopen those decisions or count them as accidentally unfinished F work. |

## Paper issues that must not be copied into code

Existing discrepancies 1–11 remain relevant. Additional issues from direct reading:

- §5.1 p. 23 calls an existential followed by a universal AFR despite Definition
  5.1 admitting a single quantifier block. It also says the §3 predicate learners
  search only universal templates, contradicting §3.2.1's existential class and
  §3.3's general guard search. The AFR/decidability claims need correction;
  current solver timeouts/unknown handling are appropriate.
- Theorem 5.7 p. 32 asserts termination from invariant and motion VCs without
  a ranking/progress premise. Those premises alone support partial correctness.
  A silently exhausted runtime iteration cap does not fill this proof gap.
- Predicate definitions disagree: Higher is non-strict in §2.2 and Appendix A
  but strict in Definition 5.4; direct-on thresholds differ between §2.2,
  Definition 5.4 and Figure 12. Code uses non-strict Higher and direct-on vertical
  gap 0 <= dz < 1.5L. There is no single consistent paper definition to copy.
- Table 7's table-placement Scattered rewrite makes the moved block scattered
  from every distinct object, including tbl, while Table 6 prohibits Scattered
  with tbl. Code retains that contradiction in this rewrite. The non-table
  geometry example in A1 shows a separate missing physical separation obligation.
- Definition 5.4's finite instantiation is not generally equivalent to the
  universal invariant over arbitrary additional objects. Its use must be justified
  as an appropriately directed abstraction, with polarity handled, rather than
  accepted as the paper's claimed exact equivalence.

## Recommended next work

1. Close verification soundness gaps first: reconcile the complete abstract action
   effects with checked geometry (including alignment/separation/table isolation)
   and make loop-budget exits consistent with proof semantics.
2. Fix the independently reproducible template freshness and carried-update bugs;
   implement flat iteration re-extraction, unresolved-body folding, and body refinement.
3. Complete block binding/summary lowering and motion support for the synthesized
   primitives; connect the resulting artifact to Algorithm 6 and counterexample repair.
4. Reconcile validation, vocabulary and invariant inputs with the supported algorithm.
5. Then evaluate full learning with validated/new demonstrations. New demos are not
   needed to fix or test steps 1–4 structurally.

Removing saved-demo dependencies from all existing tests remains a **proposed next
step awaiting the user's consideration**, as previously requested. This audit did
not make that change. The existing parity test's use of saved trajectories does
not invalidate these independent probes.

## Reproduced probe results

Executed from `roboverify/` with the AGENTS.md simulator environment and `uv run
python`. All objects/scenes were constructed directly; no trajectory files loaded.

| Probe | Result |
|---|---|
| Match ON(b1,b0), ON(b2,b1), ON(b4,b3) | Three substitutions accepted; second-to-third carry consistency false. |
| Anti-unify ON(a,p0), ON(b,p0) | Template ON(p0,p0); both substitution round trips false. |
| Ground Higher(a,b), max_depth=1/max_variables=0 | Expected labels [True, False]; classifier no_separator, examined=0. |
| Place at x=.048 above top x=.024 and root x=0 | MotionVerify true; ON*(top,root)=true, ON*(new,top)=true, ON*(new,root)=false. |
| Place on table .075 m from b0 | MotionVerify true; geometric Scattered(a,b0)=false; symbolic WP is a!=b0. |

For the last two probes, the arbitrary collision witness was fixed at (2,2,0)
(or table-center height for the second scene). They demonstrate concrete admitted
geometries, not an unbounded proof of a particular tower program. The alignment
body used three PickPlaceByName waypoints: lift the source by .2; move over the
top at x offset .024 and z offset .2; descend to z offset .05 and release. The
table example used an already separated source at z=.025, table surface 0, and
a zero-offset release. These choices isolate the absent obligations.

Full-suite command:

```bash
cd roboverify
unset LD_PRELOAD
export LD_LIBRARY_PATH="$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia"
uv run python -m unittest discover -s synthesis -p 'test_*.py'
```
