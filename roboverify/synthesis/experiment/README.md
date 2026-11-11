# Maintained experiments and diagnostics

This package contains runners needed for current research results and reusable
diagnostics. Temporary investigation scripts belong outside the repository;
historical reproducibility alone is not a reason to retain a runner.

## Current tools

| Tools | Purpose |
| --- | --- |
| `mcmc/`, `config.py` | Instrumented MCMC, enumeration or contract proposals; see the [search guide](../mcmc/README.md). |
| `search_comparison/` | Frozen-budget beam versus matched Metropolis comparison, separate legacy CFG reference, and fresh full-loop evaluation. |
| `run_logger.py`, `report.py` | Logging used by the main pipeline and bounded inspection of run artifacts. |
| `invariant_learning/` | Fixed-program counterexample learning; see its [guide](invariant_learning/README.md). |
| `ablate_higher.py`, `motion_premise_ablation.py` | Current vocabulary and motion-premise ablations; see the [verification guide](../cfg/VERIFICATION.md#higher-removal-ablation). |
| `compare_stack_heights.py` | Inspect predicate discrepancies in saved traces; see the [height comparison guide](../inference_lib/README.md#higher-height-tolerance). |

Tests here cover these maintained tools. Environment regressions live in
`synthesis/environment/test_*.py`; collection, replay and task-program
regressions live in `synthesis/cfg/test_*.py`.

Routine demonstrations use `synthesis.entry.collect_demos --task TASK`, with a
provided groundtruth for Stack, Unstack, Reverse, Restack, Grid and Pyramid.
`synthesis.entry.audit_demos` checks saved archives, replays their actions and
checks final physical success after settling. See the
[105-demonstration audit](../inference_lib/README.md#cross-environment-audit)
and [pipeline support matrix](../../../README.md#project-status).
Execution and collection support do not imply synthesis or formal verification.

## End-to-end synthesis from full demonstrations

The [2026-11-10 end-to-end study](../../runs/contract-end-to-end/20261110/report/end-to-end-synthesis.html)
starts from five complete, unsegmented four-block demonstrations per task.
They were collected with an independent direct-action controller, as a proxy
for good demonstrations from another source. They are not human teleoperation.
The learner receives observations, exact reset snapshots, observable holding
state, and initial `b0`/table bindings. It receives no teacher executable,
instruction/loop events, transfer roles, waypoints, supplied body or loop scaffold.
Holding state is derived from restored finger contacts and aperture; unknown
holding states fail closed. See the [input audit](../../runs/contract-end-to-end/20261110/data/demo-input-audit.json).

Contract-guided beam search replaces the straight-line proposal stage; existing
BMC screening, numeric scoring, predicate refinement and CFG quotienting remain.
`--mode synthesize` skips invariant bootstrap and final verification. Search
acceptance is followed by separate complete-program simulation on fresh starts.

| Experiment | Result | Scope |
| --- | --- | --- |
| Stack, five training demonstrations | Complete 17-primitive program, 89 s, three refinements | Four blocks, four CFG segments, no learned loop |
| Frozen Stack candidate, four blocks | **15/15** fresh seeds 100–114 | Global task goal, every controller converged, 50 final settling steps |
| Same executions, independent physics audit | **15/15** at 5 mm frame tolerance | Contacts, grasp/lift/release, whole action stream and settling |
| Same candidate, three blocks | Unsupported on all three attempted starts | Saved physical alias 3 is absent; no execution |
| Same candidate, five blocks | **0/3** fresh seeds 100–102 | Fifth block remains unfinished |
| Unstack, five full training demonstrations | No complete program | Three short segments of the first transfer synthesized; release boundary unresolved |

The primary successful Stack run used MMD (`theta=epsilon=0.1`) with **CEM
disabled**: observation-derived float proposals were sufficient. Its contract
beam was 12, maximum body length 5, proposal budget 5 s, and three candidates
could be checked per block. All five demonstrations were used, with exact
snapshot restoration and single-threaded numerical libraries. The
[original run](../../runs/contract-end-to-end/20261110/cfg/20261110-212211-4cc93db-stack-independent-mmd-reset/config.json)
records the full command and source revision `4cc93db`. Subsequent fixes change
the search trajectory; the frozen executable, rather than a new search run, is
the object evaluated here. A later CEM-enabled run did not complete and does
not support a claim that CEM improves end-to-end synthesis.

The recorded runtime used Python 3.10.22, NumPy 1.26.0 and the existing Gym
0.21 compatibility installation at `/tmp/lus-runtime`. In addition to the
repository's MuJoCo environment variables, experiment processes set
`PYTHONPATH=/tmp/lus-runtime:$PWD` and
`OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1`. Use the saved run
configurations and source manifests when comparing timings or reproducing a
historical search; the package defaults were not changed for this study.
The [runtime manifest](../../runs/contract-end-to-end/20261110/runtime.json)
records the compatibility-source hashes; the
[study summary](../../runs/contract-end-to-end/20261110/study-summary.json)
indexes the frozen evidence without treating bounded failures as impossibility
proofs.

The maximum nonmover drift was **2.757 mm**; only **1/15** passes the stricter
1 mm diagnostic. Exact action replay reproduced every recorded observation,
and the independent audit covered 2,536 actions and 50,810 physics/boundary
samples. This is empirical evidence under the existing simulator/contact model,
not a motion proof. See the [fresh-run results](../../runs/contract-end-to-end/20261110/evaluation/stack-mmd-unrolled/summary.json)
and [physical audit](../../runs/contract-end-to-end/20261110/evaluation/stack-mmd-unrolled/physical-audit/summary.json).

The accepted Stack candidate is preserved with exact float/control values,
initial aliases and executable fingerprint:

- [Executable JSON](../../runs/contract-end-to-end/20261110/programs/stack-four-block/program.json)
- [Readable program with offsets](../../runs/contract-end-to-end/20261110/programs/stack-four-block/program.txt)
- [CFG](../../runs/contract-end-to-end/20261110/programs/stack-four-block/cfg.json)
- [Fifteen complete candidate execution traces](../../runs/contract-end-to-end/20261110/programs/stack-four-block/evaluation-traces.npz)
- [Provenance](../../runs/contract-end-to-end/20261110/programs/stack-four-block/provenance.json)
- [Video of the exact synthesized program](../../runs/contract-end-to-end/20261110/report/media/stack-four-block-seed100.mp4)

### Unstack: release boundaries still prevent completion

The main five-demo feasibility run used the documented 600 s allowance. Its
settings were ID-first contract search, beam 12, maximum body length 6,
5 s proposal searches, two candidate checks per block, exact snapshot reset,
and one CEM round with four samples and two elites. Ordered arc distance used
`theta=0.03`, `epsilon=0.005`, and `cem_std=0.004`, all in metres. These thresholds
are not comparable to the Stack MMD thresholds. The first run stopped after
about 276 s; the corrected CEM-retention run stopped after 254.5 s when predicate
refinement exhausted its local budget. Neither produced a full executable or
a fresh whole-program simulation result. Partial bodies and exact configurations
remain under [Unstack diagnostics](../../runs/contract-end-to-end/20261110/unstack-diagnostics/experiment-index.json).
The [final evidence summary](../../runs/contract-end-to-end/20261110/unstack-diagnostics/unstack-final-summary.json)
includes the exact run paths, contact observations and 18 labeled partial-body
artifacts. These fragments are not full Unstack programs.

Three consecutive segments reached their geometric postconditions on all five
demonstrations. The next `Higher(b0,3)` milestone lies inside release: its recorded
endpoint is empty-handed. The best Move-only prefix closely matches the segment
(1.03 mm mean arc distance), but remains holding. Two other CEM trials reach the
height predicate on all five demonstrations and remain within the near-best
imitation margin, yet also remain holding. They cannot be substituted for the
recorded endpoint of the next segment. No refinement-only flag was bypassed.

The nominal Release transition clears heldness without changing block height;
the simulator lets the released payload settle onto the table. Millimetre-scale
observed grasp offsets are material to the 1 mm `Higher` threshold. This is a
remaining boundary/model mismatch, not evidence that all synthesis approaches
are incapable of Unstack. A continuation needs a release/settling contract or
phase-aware partition that accounts for the actual terminal hand and payload
state, followed by complete-program simulation.

The investigation made two general corrections without weakening acceptance:
observed-held Move/BMC transitions preserve the payload's observed displacement
from the gripper, and contract search retains CEM's evaluated numeric variants
in the bounded epsilon pool using their executable fingerprints. CEM still
optimizes imitation; the final pool is ranked by PostScore. The latter fixes a
real sample-discard issue, but did not resolve this release boundary. MCMC and
enumeration retain their previous pool defaults.

### Why the original milestones prevented loop recovery

With the quotient defects repaired, the repeated Stack labels recover
`b=b0; while exists b_prime. Scattered(b0,b_prime): ...; b=b_prime`, with
body postcondition `ON(b_prime,b)`. However, the first geometric `ON` state
occurs while the moved block is still grasped. Across the five demonstrations,
the resulting 15 body entries contain five empty hands and ten hands holding
the previous destination. The shared nominal grammar allows Pick only when
all entries are empty and Release only when all are holding. Move preserves
heldness and cannot change blocks in the empty entries. Thus neither Pick nor
Release is available at any depth, while the body goal remains false in those
five examples. More beam/CEM budget cannot solve this partition.

The [certificate](../../runs/contract-end-to-end/20261110/loop-body-holding-obstruction.json)
records the extracted states, exact holding-state closure, four inductive Z3
checks and probes of the implemented scalar/batched transitions. It establishes
an obstruction for this nominal grammar and partition, not for every physical
controller or conditional language. The next design needs observable
phase-consistent milestones (placement and a released hand), or an inferred
prologue, steady-state loop and release epilogue. See
[discrepancy 43](../../../PAPER-DISCREPANCIES.md#43-geometric-loop-milestones-can-mix-incompatible-gripper-phases).

An [observable release diagnostic](../../runs/contract-end-to-end/20261110/stack-released-milestone-diagnostic.json)
keeps the learned `ON` milestone but waits for a known empty hand. It recovers
three iterations in each demonstration, with all 15 heads and endpoints empty;
the folded CFG validates and the unchanged guard learner returns
`Scattered(b,b_prime)`. This removes the mixed-hand obstruction for those
segments. Segmentation feasibility alone does not establish body synthesis or
whole-loop success.

Two experimental continuations then searched a shared body from all 15 aligned
segments, retaining only the automatically learned CFG/guard/bindings. Obsolete
physical realizations were cleared; no fixed object aliases or supplied body
entered the grammar. The final [beam-12 run](../../runs/contract-end-to-end/20261110/released-head-continuation-beam12/result-summary.json)
reached depth 6 in 8.88 s of proposal search (84.3 s total), checked eight
BMC-feasible prefixes, and found no nominal complete candidate. Four prefixes
achieved `ON(b_prime,b)` at the final state on all 15 simulations, but all four
ended with closed fingers; none matched the observed empty-hand endpoint.
The best MMD was 0.003145. Thus refinement-only status here does not mean the
geometric relation failed. Neither a complete body nor its remaining tail was
accepted. This is a bounded beam result, not an impossibility theorem. Entry
probes additionally show that the first empty-hand sample can still place the
gripper only about 6 mm above the destination: empty-handedness alone does not
align the retreat phase.

### Completed retreat recovers a loop, with remaining execution failures

A [follow-up experiment](../../runs/contract-retreat-boundaries/20261110/protocol.json)
continued the automatically learned CFG using the same five full independent
demonstrations. It delayed placement boundaries until observed support, an empty
hand, stable blocks, clearance above the scene and low gripper speed after an
upward retreat. It used observations and contact-derived holding status, with
no teacher program, action labels or supplied loop body. All 15 extracted starts
then admitted the nominal Pick contract; the earlier empty-hand cuts admitted
only the five first iterations.

Contract search found a five-primitive shared body and an accepted empty tail,
with initialization `b=b0`, guard `exists b_prime. Scattered(b,b_prime)` and update
`b=b_prime`. The [exact executable](../../runs/contract-retreat-boundaries/20261110/programs/stack-retreat-loop/program.json)
passed the immediate goal check on all five whole training executions; a later
50-step settling audit passed three of five. Its floats were not CEM-optimized.
The search accepted a body in 29.6 seconds. Strict temporal and centering tests
used to select demonstration cuts were reported separately from executable
acceptance, which required the terminal relation, supported placement, empty
hand, retreat clearance and converged controllers; this distinction was
[recorded before fresh evaluation](../../runs/contract-retreat-boundaries/20261110/protocol-amendment-endpoint-audit.json).

The frozen loop's [fresh evaluation](../../runs/contract-retreat-boundaries/20261110/evaluation/stack-loop/summary.json)
and independent [contact/frame audit](../../runs/contract-retreat-boundaries/20261110/evaluation/stack-loop/physical-audit/summary.json)
gave the following results on seeds 200–214:

| Blocks | Task goal, convergence, 50-step settling and 5 mm physical audit |
| --- | --- |
| 3 | 15/15 |
| 4 | 8/15 |
| 5 | 0/15 |

All controllers converged, but this was not a reliable larger-tower controller.
The [diagnosis](../../runs/contract-retreat-boundaries/20261110/evaluation/failure-diagnosis/summary.json)
identifies accumulated placement error: the learned descent takes its X
reference from the held payload. Moving the gripper to the payload's current X
adds the existing payload/gripper offset to the placement error. Local support
can hold while alignment with the original base fails. In three cases, drift
also moved the current loop reference close enough to an untouched block that
the 100 mm Scattered guard became false prematurely. Five-block runs exhibited
substantial tower instability. These are execution/abstraction problems after
recovering a loop, distinct from the original mixed-hand obstruction.

The unchanged loop separately passed all 12 symbolic checks, including an
unbounded proof with a learned invariant and no repairs, in 38.4 seconds.
[Verification evidence](../../runs/contract-retreat-boundaries/20261110/verification/summary.json)
records the old-WP/domain assumptions. Motion refinement and termination were
not proved; the symbolic result does not certify the failed physical runs.

Two training-only follow-ups retained the learned CFG and completed-retreat
examples. A [whole-loop CEM fit](../../runs/contract-retreat-boundaries/20261110/cem-loop/comparison.json)
optimized all ten floats with Arc imitation distance in 298.6 seconds: three
rounds of 12 samples plus evaluated means/initialization, 40 variants total.
The epsilon pool retained exact numeric variants and ranked them by strict
whole-training PostScore. Its selected program passed five of five training
episodes including settling, reducing mean release XY error from 6.82 to
3.27 mm. This is an Arc metric ablation from the original MMD search.

Separately, a [beam-24 Arc shortlist](../../runs/contract-retreat-boundaries/20261110/arc-shortlist-beam24/result-summary.json)
evaluated all eight generated bodies on the 15 segments before selecting one.
It synthesized a complete loop in 123.4 seconds and passed both immediate and
settled training checks in five of five episodes. The search selected a
support-referenced descent without manual operand edits. A nominal audit found
why it could previously be discarded: zero-offset nominal grasps make the
self-payload and support X references nearly identical, so coarse beam diversity
pruning can retain only one. No new primitive capability was introduced.

Both follow-ups were frozen before evaluation on the same new seeds 300–314.
All 90 captures replayed exactly and received the independent physical audit:

| Candidate | 3 blocks: task + 5 mm audit | 4 blocks | 5 blocks |
| --- | --- | --- | --- |
| Original learned body, whole-loop CEM | 15/15 | 14/15 | 11/15 |
| Arc shortlist, CEM disabled | 15/15 | 15/15 | 3/15 |

The CEM candidate's task-only five-block rate is 13/15; two otherwise successful
runs move nonmovers by 5.80 and 5.49 mm. Its three task failures across four/five
blocks are premature loop exits. The Arc candidate has more taller-tower motion
and settling failures. The declared 90% combined-success target at every size
was **not met**. These are descriptive 15-case results, not a generalization
guarantee. The initial baseline used different seeds (200–214), so that comparison
is not a matched ablation. Both follow-ups passed all 12 symbolic checks,
including unbounded obligations, without repairs; neither has a motion or
termination proof.

A [read-only guard audit](../../runs/contract-retreat-boundaries/20261110/guard-reference-audit/summary.json)
finds that `Scattered(b,b_prime)` and `Scattered(b0,b_prime)` both fit all 15
positive and 20 exit examples, with identical continuing-head witness sets.
The current learner returns the first exact separator under lexical reference
ordering and prunes later equivalent truth signatures. Ordering unassigned
scope roles before reassigned roles makes the existing grammar enumerate the
fixed-base guard first. This is a possible future stability prior, not an adopted
guard change or correctness repair; it needs cross-task validation and cannot
fix nonmover displacement. Frozen candidates were not changed by this audit.

The [portable HTML report](../../runs/contract-retreat-boundaries/20261110/report/completed-retreat-loop.html)
includes videos and downloadable exact executables. The
[consolidated evidence](../../runs/contract-retreat-boundaries/20261110/result-summary.json)
records all three candidates, seeds, hashes, criteria and scope. This is an
experimental continuation of the learned CFG; the observation-based boundary
detector has not become a production CLI default. The maintained code change
normalizes only search-accepted empty blocks to `Skip(0)`; unresolved or failed
empty blocks still fail closed (31 focused tests passed).

### Contract-guided boundary refinement experiments

The [2026-11-10 follow-up](../../runs/contract-boundary-refinement/20261110/result-summary.json)
tests automatic boundary proposals, mandatory logical endpoints, and competing
guards. It continues the previously learned CFG from the same five independent
demonstrations per task. It supplies no body or teacher action labels and makes
no production default changes. All scripts, budgets, candidates and failures
are retained under `runs/contract-boundary-refinement/20261110/`.

The shared boundary engine searches bounded truth windows around the original
learned milestone. Existing primitive transitions propose admissible handoffs;
Stack ranks first-action hypotheses by the maintained imitation score. Complete
bodies are then searched, BMC-screened and executed. Stack acceptance requires
terminal `ON(b_prime,b)` with pre-update bindings, actual support contact,
contact-derived empty hand and converged controllers. The ten continuing
endpoints additionally receive a nominal check of the generated body's own
first action under the next demonstrated binding; this is not a physical or
universal successor-execution proof. Final iterations have no next-action
requirement. Prefixes, all demonstration transitions and tails are retained.

The [matched comparison](../../runs/contract-boundary-refinement/20261110/stack/comparison.json)
uses beam 12, maximum length 6, a 60-second proposal budget, pool 8, Arc distance,
CEM disabled and a 180-second case cap:

| Boundary policy | Pick admissible | Complete shared body | Immediate training | 50-step settled training |
| --- | --- | --- | --- | --- |
| Original geometric cut | 5/15 | No | — | — |
| First empty hand | 5/15 | No | — | — |
| Earlier manual retreat detector | 15/15 | Yes | 5/5 | 3/5 |
| Automatic contract-guided cuts | 15/15 | Yes | 5/5 | 5/5 |

Automatic cut proposals took 3.46 seconds and complete synthesis 62.2 seconds.
No predefined upward retreat, low-speed plateau or clearance threshold beyond
existing primitive contracts enters that policy. Its bounded search uses an
80-frame shift window, at most 32 first-action hypotheses and three cut vectors;
the first ranked vector succeeded. Demo 0's first continuing cut is frame 120,
versus 128 for the earlier manual detector. The two failed controls reached
depth six with zero nominal goals, without proposal timeout. Four released-only
partial bodies achieved terminal ON on all 15 examples but ended holding the
payload, illustrating why logical placement alone is insufficient.

The frozen [automatic executable](../../runs/contract-boundary-refinement/20261110/programs/stack-automatic-boundary/program.json)
uses five physical body primitives and only initial binding `b0`. Fresh seeds
500–514 give task/convergence/50-step-stability success of 15/15, 14/15 and 8/15
at three/four/five blocks. Including the independent 5 mm contact/frame audit
gives **15/15, 14/15 and 4/15**. All controllers converged, actual designated
support contacts persisted and there were no unintended contacts. Nine
five-block cases exceed the nonmover-frame limit during the final interval;
five completed loops lose the goal during settling. Release's `Z_m` operand is
a retreat reference; audit support is resolved separately from the saved learned
`ON(b_prime,b)` contract. The program passes all 12 symbolic checks, including
unbounded obligations with a learned invariant, in 27.2 seconds. No motion or
termination proof is established.

A [finite production-effect audit](../../runs/contract-boundary-refinement/20261110/stack/predicate-effect-audit/summary.json)
evaluates existing `put_effect`, enabling and ON*/Scattered domain formulas
without assuming their truth. All 132 actual transfers satisfy direct ON and
finite Put enabling. At Release return, ON* effects match 132/132 but Scattered
effects only 129/132. At interval end, including final settling, these counts
are 127/132 and 129/132. Eight effect mismatches arise from heads satisfying the
checked input axioms and enabling conditions. One immediate placement retains
ON while crossing Scattered's 100 mm boundary; five delayed cases lose numeric
ON* transitivity when top/base displacement exceeds 25 mm. Higher and full
motion refinement are outside this audit. These are refinement/domain gaps,
not counterexamples to conditional symbolic WP soundness.

The [endpoint acceptance ablation](../../runs/contract-boundary-refinement/20261110/endpoint-acceptance/summary.json)
reanalyzes all 40 earlier CEM variants on five starts: 196/200 executions pass a
per-transfer sometime-ON gate, 186 pass terminal ON, 180 pass terminal
phase/convergence checks, and 121 pass the settled whole-task gate. The latter
include only 120 sampled-frame passes; sampled observations are weaker than a
physics-substep audit. Three representative reexecutions exactly reproduce a
transient-ON failure, a settling failure and a success. This retrospective
corpus is not 200 independent scenes and does not show a changed selected winner.

For [Unstack](../../runs/contract-boundary-refinement/20261110/unstack/result-summary.json),
the same truth-window/contract mechanism preserves the actual learned
`Higher(b0,3)` milestone, all five traces and coverage. Moving the failed region's
cut 16 frames later makes outgoing Pick admissible in 5/5 cases instead of 0/5.
Original and two automatic variants still find zero nominal goal candidates
through length six. A separately recorded
[numeric-proposal addendum](../../runs/contract-boundary-refinement/20261110/unstack/offset-proposal-addendum/interpretation.json)
solves a shared Move-Z offset of -1.181713 mm for an existing generated
Move→Release pair under unchanged contracts. After freezing, fixed-offset BMC,
terminal Higher/released table support, convergence and nominal next-action
readiness pass 5/5; Higher persists through 50 extra steps. This additional
offset mechanism resolves a local fragment, not complete Unstack synthesis or
loop recovery. There is no exhaustive substep contact/frame audit or universal
motion proof for that fragment.

The [competing-guard experiment](../../runs/contract-boundary-refinement/20261110/competing-guards/summary.json)
uses the earlier CEM body, separately from the new automatic-boundary body.
Both consistent guard candidates remain in the existing grammar. A declared
preference for unchanged reference roles selects the fixed-base guard using
training data only; six variable-renaming checks preserve that choice. On
45 matched fresh starts (seeds 400–414 at three/four/five blocks), both guards
pass all physical criteria with exactly identical actions and observations.
There is no observed fresh success-rate gain. A separate, selected historical
three-case diagnostic repairs all known early exits with the same initial states
and unchanged physical body; it establishes the mechanism, not a rate estimate.
The stable-reference variant also passes all 12 symbolic checks.

The [portable HTML report](../../runs/contract-boundary-refinement/20261110/report/boundary-refinement-experiments.html)
includes exact-execution videos and downloadable programs/evidence. The data
supports automatic cuts plus mandatory logical endpoints, while exposing the
need for shared numeric proposals and complete symbolic-effect/domain checks.
It does not establish a fully integrated robust Stack/Unstack synthesizer or
arbitrary-action/teleoperation generalization.

### Integrated contract refinement and final Stack simulation

The opt-in [production integration](../cfg/CONTRACT-REFINEMENT.md) combines
bounded persistent-milestone cuts, phase-compatible loop entries, shared offset
proposals, mandatory endpoint contracts and complete-program validation. The
initial full CLI run recovered a five-primitive Stack loop from five complete
independent-controller traces in 100.1 pipeline seconds (107.9 process seconds).
No CFG, guard or body was supplied. It used MMD for outer refinement, Arc for the
shared body, and no CEM. All five training starts and all 15 actual transfer
contracts passed, including 50 final settling steps. Its original frozen program
still accumulated drift on five-block development scenes.

A generic bounded operand-neighborhood continuation then explored alternative
references around the learned bodies. It replayed 2130 operand combinations;
561 met the unchanged nominal goals across all 15 segments. An eight-candidate
pool preserved close-scoring shared XY references at each Move site. This
retained a body whose final Move takes XY from the fixed loop-entry reference
and Z from the current support. Neither role name nor task was hardcoded in
the selector. The exact five-action structure and float offsets were preserved;
the candidate passed the same BMC, physical imitation, endpoint and training
gates. Development seeds 900–909 at three/four/five blocks informed selection.

The [frozen executable](../../runs/contract-integrated/20261110/final-evaluation/stack-operand-neighborhood/program.json)
was then evaluated once on untouched seeds 1000–1014 at each size, with no tuning
or reselection afterward. The [comparison](../../runs/contract-integrated/20261110/stack-robustness-comparison.json)
retains every outcome, exact full-state capture, program hash and audit source:

| Frozen program / evaluation | Three blocks | Four blocks | Five blocks |
| --- | --- | --- | --- |
| Original full-CLI loop, development: task + settling + contact/support + 5 mm frame | 10/10 | 10/10 | 1/10 |
| Operand alternative, development: same physical criteria | 10/10 | 10/10 | 10/10 |
| Operand alternative, final untouched starts: same physical criteria | 15/15 | 15/15 | 15/15 |
| Operand alternative, final starts: all checked endpoint effects/domain obligations | 15/15 | 15/15 | 15/15 |

All 45 final episodes converged, retained task success for 50 settling steps and
replayed exactly. Independent audits inspect physics substeps at 2 ms, separately
from the production gate's controller-action sampling. Maximum nonmover drift
was 0.592/1.383/3.724 mm for three/four/five blocks. All 135 transfers retained
direct ON and matched the finite production Put ON*/Scattered effects and
checked domain axioms at Release and subsequent transfer/settling endpoints.

The development set still contains three Scattered failures, one at each size
for seed 908. A support-to-loose-block separation of 100.873 mm lies just outside
the 100 mm cutoff, while a roughly 1.70 mm payload placement bias crosses it.
Direct ON, support and physical stability remain true. These failures are retained
and reject acceptance on the full development-start contract gate; the final
successes do not erase them. Higher effects, unbounded motion refinement,
termination and universal scene/teleoperation generalization remain unproved.

This final candidate is a generated continuation of the learned CFG selected on
development data. Its 45/45 result must not be attributed to a fresh full-CLI run
of the new neighborhood stage or to automatic selection from training data alone.
The original full-CLI recovery and this continuation have distinct artifacts.
The [matched rerun of the integrated implementation](../../runs/contract-integrated/20261110/invocations/stack-reference-neighborhood-full/concise-summary.json)
also recovered a five-primitive loop automatically, in 207.7 pipeline seconds
(216.5 process seconds), after two refinements. Seven bodies passed the local
gates; complete training replay accepted the first, with 5/5 successes and 255/255
task/settling checkpoints. Its exact selected executable is preserved; that
frozen CLI recorded the other alternatives' count but did not save their
executables. Subsequent code now archives every retained body before selection.

This newly selected executable differs from the development-selected candidate.
On all 30 declared development starts, [independent evaluation](../../runs/contract-integrated/20261110/autonomous-versus-selected-stack.json)
found task/convergence/settling success of 10/10, 10/10 and 8/10 at three/four/five
blocks. Adding the 5 mm physical frame checks gives 10/10, 10/10 and 2/10;
including all audited relational effects/domain checks gives 9/10, 7/10 and 1/10.
All 90 direct ON endpoints held, but five-block nonmover drift reached 9.14 mm.
No untouched final seeds were reused. Training-only selection therefore still
fails to reliably select the physically stronger body available to the search.
The [all-development-start archive](../../runs/contract-integrated/20261110/data/stack-all-development-starts.npz)
contains all 30 unfiltered exact starts for subsequent production selection;
these are not additional imitation demonstrations or held-out evaluation.

Unstack phase-compatible cuts remove the previous mixed-hand obstruction:
all 15 learned entries become known empty, and 145 shared Move→Pick prefixes
exist where the earlier partition had none. This has not yet yielded a complete
accepted Unstack program. A diagnosed first-transfer tower collapse also led to
a generic nominal/BMC Pick condition: lateral approach must clear the source,
while an already XY-aligned grasp remains allowed. Physical controller code is
unchanged. Generated high-approach alternatives remove the observed initial
collapse, but the then-current contact observer still rejected the later
holding evidence. The follow-up below identifies false negatives in that
observer; those rejections did not establish payload slip.
A full post-Pick-correction/neighborhood attempt exhausted its search after
379.6 pipeline seconds (386.7 process seconds), with no complete program;
[its command and frozen source](../../runs/contract-integrated/20261110/invocations/unstack-source-clearance-neighborhood/command.json)
record the original five traces, Arc metric, CEM disabled and 600-second cap.

The next matched local probe identified a second pruning issue: all empty-hand
Move states shared a diversity bucket even when only some enabled a legal Pick.
Adding the existing generated-Pick enablement vector to that key changes zero
complete nominal bodies into 25 at the same beam/length/budget, in 5.27 seconds;
the optional neighborhood then retains eight alternatives. Readiness probes are
charged to the existing transition/time limits. All eight pass BMC and 10/15
physical endpoint audits; the five first-transfer entries have uncertain
holding immediately before Release under the historical observer. The active
contact correction below resolves those five observation failures.

The [final integrated Unstack attempt](../../runs/contract-integrated/20261110/invocations/unstack-action-readiness/diagnosis.json)
exhausts its search in 236.4 pipeline seconds (243.2 process seconds). Its saved
six-primitive candidate has Arc distance 0.03151 against the unchanged 0.03 limit
and matches the locally audited candidate's fingerprint. CFG acceptance fails,
so complete-program validation, fresh-scene evaluation and formal verification
do not run. This is structural search progress, not successful Unstack synthesis.

#### Active-contact correction and bounded float repair

Raw MuJoCo contact inspection found that the historical holding observer
discarded finger contacts with positive geometric separation above 1 micrometre.
The configured contact margin is 1 mm, and the inspected contacts were active
with positive normal forces despite separations of 0.037–0.334 mm. This is
consistent with [MuJoCo's contact model](https://mujoco.readthedocs.io/en/2.1.2/computation.html#contact).
The observer now requires an active constraint (`efc_address >= 0`) and finite
positive normal force from `mj_contactForce`. Two distinct fingers must contact
the same unique object. Missing or nonfinite force evidence remains unknown and
rejects acceptance; controllers and demonstrations are unchanged.

The [corrected local rechecks](../../runs/contract-integrated/20261110/unstack-active-contact/comparison.json)
pass all 15 endpoint audits for the original six-action Unstack body, the CEM
fits and a generated seven-action waypoint body. The original body's Arc
distance remains 0.03151, above the unchanged 0.03 threshold; the imitation-CEM
fit has distance 0.02789 and passes all 15 endpoint audits. Contact-derived
archive labels agree with the corrected observer on all 115 sampled snapshots.
These are local body results, not an accepted full Unstack program or a proof
that contact forces guarantee a stable grasp throughout arbitrary motion.

The [Stack float-refinement comparison](../../runs/contract-integrated/20261110/float-refinement/comparison.json)
uses the five original training starts and all 30 development starts. A bounded
coordinate search on the reference-diverse body changes its last Move X offset
by -1 mm and passes all 35 complete-start checks, all 15 original local endpoint
and handoff checks, fixed-offset BMC and the unchanged imitation threshold
(Arc 0.01956). The separately recovered CLI body needs two coordinate changes
in a diagnostic continuation to pass the same checks; its maximum sampled
nonmover displacement is 4.941 mm, close to the 5 mm limit. Neither result is
an independent final evaluation. The production implementation now supports
bounded coordinate repair of each retained structure, with a shared validation
time/action budget and renewed BMC, local imitation, endpoint and handoff gates.

The [fresh full CLI run with that repair stage](../../runs/contract-integrated/20261110/invocations/stack-full-multisize-float-repair/concise-summary.json)
now recovers and accepts a Stack loop automatically in 368.3 pipeline seconds
(376.8 process seconds), from the original five whole demonstrations. No CFG,
guard or body was supplied. All seven locally accepted structures first fail
complete validation. The first exhausts 40 numeric trials; the second passes on
its 14th, changing only its final Move X offset from zero to -1 mm. The resulting
[exact executable](../../runs/contract-integrated/20261110/cfg/20261111-023005-76cc044-stack-full-multisize-float-repair/artifacts/program.json)
and [readable program](../../runs/contract-integrated/20261110/invocations/stack-full-multisize-float-repair/program-readable.txt)
are preserved. Its SHA-256 is
`907f8717a43c9e17115d7e2b326b5b3e1077fa3eeba6f35328d0004f952bbe6b`.

All 35 requested starts pass complete execution, including 1785 task/settling
checkpoints, 105 ON transfers and 35 tail contracts. The repaired body's Arc
distance is 0.01955493 against the original pool minimum 0.01954694, retaining
the 0.03 threshold and 0.003 near-best radius; all 15 original local entries
pass BMC, endpoint and handoff checks. Maximum controller-action sampled frame
drift is 4.934 mm, only 0.066 mm below the 5 mm limit. This is a distinct body
from the earlier reference-diverse continuation. Independent fresh-seed
evaluation and the finer physics-substep audit are separate checks; neither
the 35 selection starts nor the prior candidate's 45/45 result establish this
new executable's held-out success rate.

The [fixed fresh evaluation on seeds 2000–2014](../../runs/contract-integrated/20261110/final-evaluation/stack-automatic-multisize-float-repair-seeds2000/summary.json)
then tests this exact executable once at each of three/four/five blocks, without
tuning on those cases:

| Automatically synthesized executable | Three blocks | Four blocks | Five blocks | Total |
| --- | --- | --- | --- | --- |
| Task, controller convergence and 50-step settling | 15/15 | 15/15 | 15/15 | 45/45 |
| Independent 2 ms physical/contact/support/frame audit | 15/15 | 15/15 | 13/15 | 43/45 |
| Finite Put effects and domain checks | 15/15 | 15/15 | 14/15 | 44/45 |
| Both stricter audit gates | 15/15 | 15/15 | 12/15 | 42/45 |

All 45 captures replay exactly. Five-block seeds 2001 and 2008 exceed the 5 mm
nonmover frame limit, reaching 7.886 and 5.599 mm. Seed 2012 instead violates a
Scattered effect on its third transfer and a domain axiom at the next head.
All 135 direct ON and ON* effect checks pass. These failures are retained;
task success alone does not establish the primitive refinement contracts.
The [seed-2012 geometry diagnosis](../../runs/contract-integrated/20261110/final-evaluation/stack-automatic-multisize-float-repair-seeds2000/diagnostics/n5-seed2012-scattered.txt)
shows why: the support is 101.953 mm from a loose block along X before the
transfer. After placement, the support remains beyond the 100 mm Scattered
cutoff, but the moved block is only 97.744 mm away. It still satisfies ON and
support contact, and nonmover drift is only 1.243 mm. The permitted ON placement
offset therefore does not imply exact inheritance of a sharp Scattered relation.
Final task success does not repair that intermediate abstraction mismatch.

The independent audit now recognizes positive-force contacts inside MuJoCo's
margin and explicitly requires actual support at Release and interval end.
It retains all prior near/penetrating contact collision checks and also checks
active margin contacts. Seven focused checks and an exact historical capture
replay precede the new evaluation. The earlier 45-case evaluation recorded true
support booleans for all 135 transfers, although those booleans were omitted
from its final acceptance conjunction; that omission did not alter its old
total under the original observations. Historical results are not silently
reclassified as passing the strengthened force-based audit.

Separately, [symbolic verification of the same executable](../../runs/contract-integrated/20261110/invocations/stack-auto-symbolic-120s/concise-summary.json)
passes all 12 checks in 24.8 pipeline seconds. Candidate replay provides 15
continuing heads and five normal exits from the original five starts. The
existing partition-based learner with PyEDA supplies the invariant; no invariant
was supplied and no executable or invariant refinement was needed. Establish,
preserve and exit VCs pass at sizes 2–4 and over the unbounded domain, each with
UNSAT counterexample queries and SAT premises. This is partial correctness of
the symbolic abstraction under the old WP rules, relational domain axioms,
task precondition and abstract action contracts. It establishes no motion
refinement or termination result; the physical audit failures remain relevant.

The [next full automatic run](../../runs/contract-integrated/20261110/invocations/stack-full-multisize-frame-margin/concise-summary.json)
uses `--candidate-selection frame-margin` on the same original 35 selection
starts. It finishes in 680.1 pipeline seconds (687.4 process seconds), comparing
seven structures and 170 coordinate vectors. Five repaired structures pass all
gates. The selected body's final Move takes XY from `b0` and Z from `b`, with its
X offset fitted from zero to -1 mm. All 35 complete starts, 15 original local
entries, 1785 goal/settling checkpoints and 140 block contracts pass. Its maximum
sampled nonmover drift is 1.713 mm, compared with 4.934 mm for the prior
first-passing selection. No threshold or controller changed, and no final seeds
entered selection.

The [exact new program](../../runs/contract-integrated/20261110/cfg/20261111-025848-nogit-stack-full-multisize-frame-margin/artifacts/program.json)
has SHA-256 `3b5ccd81c7b818c93870557c2e2c6d627cf3c8f158e7ebf7d5fd7bb81fd3d09c`.
The source also includes the fresh relational-node seeding correction, replacing
two unused terminal Get bindings with a zero-step Skip; this comparison does not
isolate ranking as its only code change. [Separate symbolic verification](../../runs/contract-integrated/20261110/invocations/stack-frame-margin-symbolic-120s/concise-summary.json)
of this exact candidate learns its invariant afresh and passes all 12 nonvacuous
checks, including the unbounded obligations, in 22.2 pipeline seconds. Its scope
remains abstract partial correctness under the old WP model. The earlier
candidate's fresh simulation results do not transfer to this executable.

Its [separate frozen evaluation on seeds 3000–3014](../../runs/contract-integrated/20261110/final-evaluation/stack-automatic-frame-margin-seeds3000/summary.json)
passes **45/45 cases**, 15 each at three/four/five blocks, for task completion,
controller convergence, 50-step settling, the strengthened independent physical
audit and every finite Put/domain obligation. All 135 transfers satisfy direct
ON, ON*/Scattered effects and the checked enabling/domain conditions. All 45
captures replay exactly. Maximum nonmover displacement at 2 ms resolution is
0.593/1.615/3.973 mm for three/four/five blocks, below the unchanged 5 mm limit.
No tuning or reselection used these starts. The prior seed-2000 failures remain
preserved; different seeds, executables and the seeding correction prevent
treating this as an isolated ranking ablation. These finite results still do not
establish motion refinement on arbitrary scenes or human teleoperation data.

[Validation records](../../runs/contract-integrated/20261110/validation/summary.json)
retain a broad 747-test run (seven optional skips and one outdated low-gripper
fixture failure), the fixture correction, and a subsequent passing 216-test
focused run covering the final search, BMC, CFG and artifact changes. The entire
broad suite was not rerun after that fixture-only correction. A later 151-test
run passes the active-contact observer, bounded parameter repair, candidate
selection, contract search, BMC and noise regressions together. A subsequent
190-test integration run also passes frame-margin selection, partial-CEM policy,
persistent deadlines, scoped seeding and shared collection regressions.

### Reusing the saved candidate for verification

The JSON loader preserves exact parameters and validates its fingerprint. The
verification CLI can replay a candidate from the accepted demonstration starts,
using the candidate's own events rather than requiring teacher identity:

```bash
uv run python -m synthesis.entry.synthesize_cfg \
  --mode verify --task stack --num-blocks 4 --candidate-program \
  --program runs/contract-end-to-end/20261110/programs/stack-four-block/program.json \
  --demos runs/contract-end-to-end/20261110/data/stack-independent-accepted-5-holding.npz \
  --verification-level symbolic
```

This is an entry point for later proof work, not a verified-result command.
The preserved program is fixed-size; its physical aliases do not justify an
unbounded proof for arbitrary block counts. The later proof must also respect
the actual controller assumptions and measured framing tolerance.

A [bounded smoke invocation](../../runs/contract-end-to-end/20261110/verification-smoke/smoke-result.json)
confirmed successful JSON reload and candidate execution on all 15 saved fresh
starts. It finished in 33.9 s externally (24.6 s inside the CLI). The old-WP
unbounded body VC was **invalid**, with a seven-value Box countermodel, and the
pipeline returned `needs_demonstrations`. No motion check ran. This is consistent
with the fixed-size limitation; no claim is made that the symbolic countermodel
was realized physically. A proof scoped to four blocks remains future work.

The final focused regression suite passed **171 tests**, covering search,
CFG refinement/quotienting, persistence, caching, BMC, the arc metric and MCMC
parity. The [validation record](../../runs/contract-end-to-end/20261110/validation.json)
preserves the earlier runs and the final log. The HTML report was also checked
offline for video playback, all 17 instruction seeks, exact executable download
and mobile layout.

## Maintained contract-search comparison

`search_comparison/` separates three questions: whether contract-guided beam
search finds useful straight-line bodies, whether it improves on local
Metropolis search given the same information, and whether the selected bodies
compose in complete supplied loops. The maintained backend is selected with
`--search-strategy contract`; MCMC remains the production default.

The [completed frozen comparison](../../runs/search-comparison/20261110-matched/report/search-comparison-report.pdf)
supports using contract-guided beam search for these straight-line proposals:

| Task | Beam: accepted fitting runs | Matched Metropolis: accepted fitting runs | Median beam time to checked candidate |
| --- | --- | --- | --- |
| Stack | 10/10 | 0/10 | 41.0 s |
| Unstack | 10/10 | 0/10 | 36.0 s |

Every accepted fit passed all 15 training scenes. All ten beam repeats selected
the same body per task, so this is **two distinct learned bodies**, not twenty
independent discoveries. The matched baseline explored lengths zero through
eight in every run and usually improved its nominal rank, but found no terminal
candidate during its 20-second structural allocation. This does not establish
failure under an unrestricted 48-second structural search or longer budgets.

The two frozen bodies then underwent **90 actual fresh complete-loop executions**:

| Task | 3 blocks | 4 blocks | 5 blocks |
| --- | --- | --- | --- |
| Stack | 15/15 | 15/15 | **1/15** |
| Unstack | 15/15 | 15/15 | **15/15** |

Five-block Stack has a material composition failure: all 15 runs satisfy the
simulator's weaker goal and all controls converge, but only three satisfy the
canonical global goal, one stays within the 5 mm frame bound, and twelve pass
contact checks. Maximum nonmover drift is 13.18 mm. The new search is a useful
replacement for structural MCMC here; local-transfer acceptance is insufficient
for accepting its learned programs across block counts. The supplied loop
scaffold and later formal verification remain separate from these finite tests.

The primary search consumed 300 simulator rollouts and retained the top nominal
parameter vector unchanged in every accepted fit. No CEM improvement was
demonstrated under this small budget. At the stricter 1 mm frame diagnostic,
the selected vectors pass 12/15 Stack and 10/15 Unstack training examples.
The production CLI continues to use its existing KL/MMD objective; these
experiment results use the shared physical-failure/arc objective described below.

The corrected legacy reference also has 0/10 accepted runs per task and no
production convergence: 888 attempted rollouts form 51 complete and nine
interrupted vector batches, with one mutation per run. An initial reference
adapter incorrectly rejected unspecified float parameters instead of using their
runtime zero defaults. All 13 first-version attempts are retained and excluded;
all twenty reference runs were repeated under a new frozen version with identical
settings. The forty primary runs do not call that adapter and are unaffected.
The [amendment](../../runs/search-comparison/20261110-matched/protocol-amendment-legacy-v2.json)
records the repair and the six focused regression checks.

See the [audited summary](../../runs/search-comparison/20261110-matched/audited-summary.json),
[primary protocol](../../runs/search-comparison/20261110-matched/protocol.json),
[corrected reference protocol](../../runs/search-comparison/20261110-matched/legacy-v2-protocol.json),
and [candidate lineage](../../runs/search-comparison/20261110-matched/frozen-candidates.json).
All 60 analyzed fitting invocations and all 90 fresh evaluations exited normally;
the longest fitting process took 49.90 seconds. The implementation passed 91
focused/regression tests before freezing, followed by six focused checks for the
reference-adapter repair. These repeats share fixed simulation demonstration pools
and do not establish human-teleoperation or new-primitive generalization.

The controlled comparison gives both arms the same 15 public training traces
per task, finite primitive/operand/offset corpus, nominal contracts, prefix rank,
terminal goal, BMC screen, physical acceptance checks, numeric objective and CEM.
Neither arm receives teacher commands, primitive labels, a supplied body or a
warm start. The Metropolis baseline chooses a primitive family uniformly before
choosing a member, so the many Move offsets do not dilute Pick proposals. Its
temperature was calibrated on excluded training-only pilots. The two arms differ
in structural exploration: beam expansion versus insert/delete/replace/swap
mutations with Metropolis acceptance. This is a matched algorithm comparison,
not the unchanged production MCMC pipeline.

Each fitting invocation has 48 seconds internally and an outer 55-second limit
with a two-second kill grace. Proposal construction and preprocessing count in
the first 20 seconds; structural work is additionally capped at 100,000 nominal
transitions. Both arms have at most 45 attempted simulator rollouts, with all
15 examples required to complete a shared-vector evaluation. CEM has one round
of two samples, but remaining time can prevent another complete batch. Only
fully evaluated vectors can win; errors and outer timeouts remain failed search
attempts even if a diagnostic vector was saved. BMC precedes fitting and uses
existential offsets separately per entry with `goal=True`; it is not a
shared-parameter, task-goal or physical-safety certificate.

Physical training selection prioritizes primitive convergence, observed grasp
and lift, successful release, local task success after 50 settling steps,
substep contact checks, and nonmover displacement at most 5 mm. Within equal
failure counts, it uses weighted arc-aligned spatial trajectory RMS. This is
the explicit experiment objective; the integrated CFG backend retains its
existing KL/MMD objective and acceptance thresholds.

`legacy.py` runs the actual existing `mutate_scoped` and
`straight_line_synthesize` path as a separate reference, with its KL objective,
default CEM and eight initially empty Skip slots. It retains repeated evaluations
and has no CFG BMC prefilter. Under the same wall/rollout cap, initial objective
and final-distance calls can consume 30 rollouts before the first mutation.
The reference sets `theta=0` to avoid loose imitation-only early convergence;
other search/CEM settings are recorded in its protocol. Its independent physical
audit does not change search scores, and the additional metric audit time is
recorded and included in its wall budget. The report
distinguishes production return/convergence from the best completely audited
vector; neither is inferred from budget exhaustion.

`freeze.freeze_protocol` and `legacy.freeze_protocol` write exclusive protocol
files containing all input and runtime source hashes. `run` and `legacy` require
either a matching frozen protocol or explicit `--pilot`. For example, after
freezing the protocol with the two training roots:

```bash
uv run python -m synthesis.experiment.search_comparison.run \
  --task stack --method beam --seed 0 --data PATH_TO_PUBLIC_TRAINING_DATA \
  --protocol PATH_TO_PROTOCOL_JSON --output NEW_TRIAL_DIRECTORY
```

The full-loop evaluator receives only frozen candidates and fresh seeds after
all fitting runs finish. It preserves the supplied initialization, guard and
pointer updates, with no resets or extra settling between iterations. Success
requires normal exit, the canonical global task goal, primitive convergence,
per-transfer contact/frame checks and final settling. Exact duplicate bodies
are evaluated once and retain search-run aliases. Alias-weighted outcomes must
be reported alongside actual unique-body physical executions. These experiments
test composition of learned bodies in supplied loops, not CFG/loop discovery,
human teleoperation, arbitrary primitive extensions or formal verification.

Inspect the bounded summaries without opening rollout logs:

```bash
uv run python -m synthesis.experiment.search_comparison.report \
  --run runs/search-comparison/20261110-matched
```

## Independent-controller Stack body experiment

The [2026-11-10 study](../../runs/contract-search/20261110-cd15-study/study-summary.json)
fits one shared straight-line transfer body to **15 independent-controller
demonstrations**, balanced across three loop positions. Fifteen different seeds
are held out. CFG boundaries and object-role bindings are supplied; fitting uses
observations and entry snapshots, without teacher commands, targets, action
labels, controller inversion, or a supplied body template.

Contract-guided beam search found five-action bodies. The main comparison uses
four CEM rounds, 24 samples per round, six elites, and all 15 fitting scenes per
simulated vector. BMC precedes CEM; it checks existential feasibility, not motion
safety. Numeric selection prioritizes convergence, observed grasp/lift, release,
stable ON after 50 settling steps, no unintended contacts, and at most 5 mm
nonmover displacement. The 1 mm result remains a separate diagnostic. Contact
monitoring checks every 2 ms physics step and distinguishes intentional grasp
and support contacts from collisions; existing static robot/environment contacts
are recorded exceptions.

| Structure proposals / numeric score | Optimizer seeds | Held-out spatial RMS |
| --- | --- | --- |
| Contract grid / arc-length RMS | 3 | 8.12 mm mean (6.41–10.62) |
| Contract + raw observations / arc-length RMS | 3 | **5.55 mm**, identical selected body |
| Contract grid / KL, 128 samples | 3 | 10.21 mm mean (7.82–13.67) |
| Contract grid / KL, 2,048 samples | 1 | 7.19 mm |
| Contract grid / DTW | 1 | 7.69 mm |
| Contract grid / MMD | 1 | 7.69 mm |

All selected programs passed all 15 held-out scenes. Exact duplicate bodies are
validated once: 12 run labels represent 10 parameter vectors and **150 actual
held-out rollouts**, not 180 independent trials. DTW and MMD parameters differ
only at rounding precision and produce identical recorded trajectories. The
training-selected observation-guided body has maximum held-out nonmover drift
1.419 mm (12/15 also meet 1 mm). Spatial RMS is a weighted, arc-aligned trajectory
score, not an endpoint error or formal motion bound.

Exact successor memoization (D) preserves the entire observation-guided candidate
pool, reducing action trials 24,090 → 15,111 and observed search time
41.10 → 37.89 seconds. Its identical numeric results are shared, not rerun.
Across the main comparison, CEM used 15,960 simulator rollouts. Returning the best
evaluated sample improved on the best evaluated mean in four of the twelve runs;
both standalone optimizers now retain samples as well as means. The observation
proposals already supplied the selected vector, so all four CEM rounds produced
no improvement for those three runs. This does not establish a general stopping
rule or change production budgets.

The [frozen protocol](../../runs/contract-search/20261110-cd15-study/main-protocol.json),
[metric diagnostics](../../runs/contract-search/20261110-cd15-study/metrics/summary.json),
[held-out results](../../runs/contract-search/20261110-cd15-study/heldout/summary.json),
and [source bundle](../../runs/contract-search/20261110-cd15-study/frozen-v2/manifest.json)
record the inputs, thresholds, exact parameter replay, and limitations. Those
scripts remain experimental artifacts; the later maintained backend is selected
with `--search-strategy contract` as described in the search guide.
These are empirical four-block, single-transfer results with supplied CFG
scaffolding, not new loop recovery, human-teleoperation evidence, exact symbolic
frame preservation, or an unbounded motion proof.

## Repeated-loop and Unstack follow-up

The [follow-up report](../../runs/contract-search/20261110-loop-unstack-study/report/loop-body-synthesis-report.pdf)
extends the local-transfer experiment with 135 complete Stack executions:
three frozen bodies, three block counts, and 15 matched fresh starts per count.
The original loop initialization, guard and pointer update are supplied; only
the physical body changes. There are no resets or extra settling steps between
iterations. These experiments test body composition, not loop discovery.

| Frozen Stack body | 3 blocks | 4 blocks | 5 blocks |
| --- | --- | --- | --- |
| Contract + observations | 15/15 | 15/15 | 4/15 |
| Contract grid + arc RMS | 15/15 | 0/15 | 0/15 |
| Contract grid + KL | 15/15 | 12/15 | 0/15 |

Success requires the canonical global goal, normal loop exit, controller
convergence, contact checks, a 5 mm per-transfer frame bound and final settling.
The simulator's goal alone is weaker: it accepts all 15 five-block observation
body runs, while only seven meet the canonical goal and four meet all criteria.
The grid/arc body accumulates alignment bias even when immediate release-on-target
checks pass. One five-block KL attempt reaches the iteration cap without final
settling. These findings limit the earlier isolated-transfer result; they do not
refute the separately checked supplied-program proof.

A 3/5/10/15-example training-size diagnostic finds five-action Stack bodies in
every setting, with almost identical local trajectory RMS. Evaluation on the
original 15-example pool passes throughout. This is a saturated training-pool
diagnostic, not independent evidence of sample complexity or full-loop success.

Unstack uses 15 training and 15 held-out independent-controller traces, balanced
across three positions in a four-block loop. All 30 replay exactly and satisfy
the settled local goal; two exceed the fixed contact classification margins and
remain explicitly flagged. Contract-progress diversity prevents equivalent
operand choices from crowding out useful beam states. The resulting six-action
observation-guided proposal passes all 15 held-out transfers and all 15 fresh
four-block full loops before numeric refinement. Its initial grid-only comparator
passes BMC but fails every physical test, exposing a missing support-clearance
condition in the nominal contract. Exact successor caching preserves the audited
candidate pool; diversity pruning is a separate heuristic.

The training-derived support-clearance rule also produces a grid-only six-action
body passing 15/15 held-out transfers and 15/15 full loops (35.73 mm trajectory
RMS). A tiny one-sample numeric refinement improves the observation body's
held-out RMS from 21.80 to 21.17 mm, but acceptance drops to 14/15 in each suite.
Each failure slightly exceeds the fixed 2 mm contact-geometry band during
intended table deposition; endpoints, controllers and frame checks still pass.
The training-selected final body is retained without test-driven reselection.

The report records all 12 search runs and 11 numeric trials, including two
unfinished numeric attempts, adaptive budgets and cumulative costs. Each Unstack
invocation has an outer 55-second limit; aggregate study time is longer. These
remain experimental runners with supplied CFG scaffolding, not a new production
search backend or formal motion result.

Comparison videos show the preselected scenes, including failures:

- [Stack, four blocks](../../runs/contract-search/20261110-loop-unstack-study/videos/stack-4blocks-seed30-comparison.mp4)
- [Stack, five blocks](../../runs/contract-search/20261110-loop-unstack-study/videos/stack-5blocks-seed30-comparison.mp4)
- [Unstack transfer: demonstration and initial proposals](../../runs/contract-search/20261110-loop-unstack-study/videos/unstack-transfer-seed15-comparison.mp4)
- [Unstack full loops with initial proposals](../../runs/contract-search/20261110-loop-unstack-study/videos/unstack-loop-seed30-comparison.mp4)
- [Unstack full loops with final training-selected bodies](../../runs/contract-search/20261110-loop-unstack-study/videos/unstack-final-loop-seed30-comparison.mp4)

The [Stack results](../../runs/contract-search/20261110-loop-unstack-study/stack-loop/summary.json),
[Unstack index](../../runs/contract-search/20261110-loop-unstack-study/unstack/experiment-summary.json)
and [completion audit](../../runs/contract-search/20261110-loop-unstack-study/completion-audit.json)
retain protocols, program/source hashes, individual outcomes and validation.
Videos restore saved full states without physics steps; restored observations
are checked exactly before rendering.

## Retired investigations

The old controller comparisons, tolerance sweeps, drift diagnosis, path-video
renderer, workspace interventions, manual ID-first continuation and local finite
Unstack proof adapter have been removed, together with their diagnostic-only
tests. The Unstack compatibility wrapper is also gone; use the shared collector
and production factory directly. No alternate demonstration runner remains here.

The [controller findings](CONTROLLER-PATHS.md) and
[workspace and placement findings](TOWER-WORKSPACE.md) retain the rationale for
current settings and the deferred payload-aware placement idea. Historical
reports and videos remain evidence, not current reproduction requirements.
The current controllers and motion contracts are unchanged.

PartialStack and the incorrect historical Reverse verifier remain removed.
Use Restack for that task and the shared Reverse collector for reversal demos;
neither has integrated formal verification support yet.

## Rechecking a collection

From `roboverify/`, with the simulator variables in [AGENTS.md](../../../AGENTS.md):

```bash
uv run python -m synthesis.entry.collect_demos --task stack --num-blocks 4 --num-trajectories 15
uv run python -m synthesis.entry.audit_demos \
  --demos demos/pipeline-audit-20261023/stack/demonstrations.npz \
  --output demos/pipeline-audit-20261023/stack/audit.json
```

Use the archive path actually printed by collection when collecting a new batch.
Audit defaults to the exact seed set 0–14; pass `--seeds` for another set. Run
every Unstack collection/audit invocation under `timeout --signal=TERM
--kill-after=2s 55s`, using smaller batches if rendering would exhaust the budget.
An incomplete or timed-out invocation is not a passing result.
