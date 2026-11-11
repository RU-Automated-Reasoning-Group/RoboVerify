# RoboVerify

RoboVerify synthesizes block-manipulation programs from demonstrations and checks
symbolic correctness and geometric motion obligations. Programs execute in a
Fetch/MuJoCo environment. The pipeline uses relational CFGs, flat-loop recovery,
learned invariants and counterexample-guided refinement.

## Project status

The seven current tasks are **Stack, Unstack, Reverse, Restack, Grid, Pyramid and Build3D**.
All have a provided groundtruth and use `collect_demos --task TASK`; `--program`
optionally replaces that default. Collection support is broader than proof support:

| Task | Shared execution and demo collection | Integrated synthesis / verification |
| --- | --- | --- |
| Stack | Supported | Supported in the documented model below |
| Unstack | Supported, including current lateral groundtruth | Inference and unbounded symbolic proof pass; extrema/gripper motion operands are encoded, but the full motion proof remains unsuccessful |
| Reverse | Supported | An earlier supplied program passed unbounded symbolic verification; the observed-height revision is not yet reverified, and table-coordinate motion and synthesis remain unsupported |
| Restack | Supported | Not integrated |
| Grid | Supported, nested goal loops | Not supported |
| Pyramid | Supported, nested goal loops | Not supported |
| Build3D | Supported, layer/row/column goal loops | Not supported |

[Symbolic pre/postconditions](roboverify/synthesis/cfg/TASK-CONDITIONS.md) are now
available for all seven tasks. Stack, Unstack, Reverse and Restack retain their
shared block conditions; Grid, Pyramid and Build3D share an identity-independent
occupancy contract with empty initial goals, scattered sources and one block per
goal. These specifications add no layout-shape requirement and do not enable the
unsupported verification pipelines listed above.

The **2026-10-23 collection audit passed all 105 demonstrations**: seeds 0–14
for four-block Stack/Unstack/Reverse, both five-block Restack source layouts,
3×3 Grid, and a 4+3+2+1 Pyramid. Every archive passed the then-current program/task
identity checks, task pre/postconditions, primitive convergence, full action
replay, physical task success and 50 extra settling steps at the end. All final
replay errors were zero. One original Pyramid seed failed at a far source corner;
the source region was moved 4 cm closer and all the same seeds passed on recheck.
The original failure and its replay video are retained. See the
[archive and audit guide](roboverify/synthesis/inference_lib/README.md#cross-environment-audit)
and [experiment inventory](roboverify/synthesis/experiment/README.md).
This audit did not run synthesis or establish new formal verification results.

On 2026-10-28, Reverse and Restack switched from block-count-based clearance to
instruction-entry `Z_m` (the highest physical block) plus 10 cm. Lateral moves
retain the current gripper height. Fifteen seeds each for four- and five-block
Reverse and both five-block Restack layouts passed collection, exact replay and
50-step final settling. The changed fingerprints require fresh demonstrations;
the earlier Reverse symbolic/motion results below have not been rerun on these
executables. See the [collection guide](roboverify/synthesis/inference_lib/README.md#collect-reverse-demonstrations).

The historical **2026-10-23 shared-pipeline tower investigation learned invariants
for both Unstack and Reverse and passed all 12 symbolic checks for each**, including
the unbounded proof. It reexecuted all 15 archived starts per task and used all
60/75 continuing and exit states. The 68-second Unstack symbolic run comprised
approximately 11 s setup/replay, 41 s learning/coverage and 16 s verification.
Later runs requesting both levels took 55.3 s (Unstack) and 164.9 s (Reverse),
retained symbolic success and reported incomplete motion support. Paper formulas
are separate verification-only controls, with explicitly documented repairs.
See the [commands, timings and scope](roboverify/synthesis/cfg/VERIFICATION.md#provided-unstack-and-reverse-verification).

The subsequent independent-extrema-witness implementation supports Unstack's
`Y_min`, `Z_m` and instruction-entry gripper references. The shared run takes
66.7 s and passes all 12 symbolic checks, but retains motion failures. With the
existing supported-tower premises, all collision checks pass; release/placement
and geometric loop checks still prevent a complete proof (103.6 s). Stack's
regression passes all 12 symbolic and 40 motion checks in 24.8 s. Extrema
expressions add no quantifiers; the full motion VCs retain their quantified
premises and effect checks. See the
[encoding and remaining obligations](roboverify/synthesis/cfg/VERIFICATION.md#independent-extrema-witness-recheck).

Unstack's adopted groundtruth now additionally requires **`b_prime != b0`**. The strengthened
program collected and audited 15 seeds each for 1–4 blocks (60 demonstrations);
single-block scenes exit without actions, and all four-block actions/observations
match the previous guard exactly. Fresh archives are under
`roboverify/demos/unstack-base-guard-20261023/`. Learning and all 12 symbolic checks
pass. With the existing supported-tower premises, release support and table
placement now pass along with collision, frame and predicate effects: 29 motion
checks are valid, three geometric preservation checks are SAT, and two geometric
entry checks time out. The full result remains `motion_unverified` (107.3 s).
See the [strengthened-guard results](roboverify/synthesis/cfg/VERIFICATION.md#strengthened-unstack-guard).

Increasing the motion-query budget to 120 seconds resolved both UNKNOWN entry
checks to SAT: that shared run has 12 valid symbolic checks and 29 valid /
5 refuted motion checks (122.3 s total). Small explicit scenes satisfy each failed
VC. Entry clearance and exact alignment exceed the initial premises; preservation
counterexamples exploit unnamed objects omitted by the finite extrema candidates.
See the [counterexamples and longer-timeout run](roboverify/synthesis/cfg/VERIFICATION.md#unstack-motion-counterexamples-and-longer-timeouts).

Unstack removes the auxiliary exact-column invariant, including both its checks
and its assumptions at loop body/exit boundaries. Its extrema references now use
physical witnesses bounded universally against **all** blocks at instruction entry.
This resolves both the arm-clearance and tower-height preservation counterexamples.
The descent collision is now proved from ground universal instances in **0.081 s**
through the shared pipeline, with no finite-domain restriction or new geometric
assumption. All 12 symbolic checks and all seven collision checks pass. The latest
shared run finishes in 128.5 s with **29 valid motion checks, one SAT** (initial
arm clearance), and **two UNKNOWN** at 30 seconds (Higher effects and tower-height
preservation, both proved in earlier runs). There is still no complete motion
proof. Stack retains all 12 symbolic and 40 motion checks.
See the [descent argument and solver comparison](roboverify/synthesis/cfg/VERIFICATION.md#unstack-descent-proof-from-universal-instances).

An optional **EqualHeight algebra** now replaces Higher throughout supplied-program
inference and verification (`--height-predicate EqualHeight`). The 2026-10-26
rechecks learned from the existing archived starts: Stack passed all 12 symbolic
and 40 motion checks (22 s); Unstack passed all 12 symbolic and 31 of 32 motion
checks (92 s), with initial arm clearance UNKNOWN. No supplied invariant or
refinement was needed. The table-placement rule is simpler, but block placement
still needs quantified support checks. Higher remains the default. See the
[semantics, reproduction commands and controls](roboverify/synthesis/cfg/VERIFICATION.md#equal-height-alternative).

Reverse's structural vocabulary also passes all 12 symbolic checks under the
EqualHeight domain. Adding EqualHeight and Scattered changes the learned
invariant: four-block training fails on a valid five-block transfer. Training
on 12 audited five-block executions (72 states) repairs the five-block proof,
but a six-block transfer still refutes unbounded preservation. The five-block
collection completed all 15 seeds; three lost task success during final settling
and were excluded from learning. Reverse motion support remains incomplete.
See the [vocabulary comparison and five-block results](roboverify/synthesis/cfg/VERIFICATION.md#reverse-vocabulary-investigation).

After the extrema encoding changes, revision `4f734bf` passed the full Python
regression suite: **475 tests across 68 modules**, with no failures, errors or
skips, including the optional real-solver baseline tests. On the current machine
the tests took 242.9 seconds, or 247.7 seconds including module loading. Results
are under `roboverify/runs/tests/20261023-motion-extrema/`.
The complete command is in [AGENTS.md](AGENTS.md#validation). The separate Lean
build and axiom audit previously passed all 709 declarations; the motion changes
do not modify that formalization or mechanically certify the Python translation.

The Stack workflow has a standalone multi-seed DSL demonstration collector,
full-state archives, optional 20 FPS videos, and shared `full` / `verify` pipeline
modes. Collection holds the initial gripper position for **50 settling steps**,
then saves that full simulator state as demonstration state zero. Settling is
excluded from recorded actions and video. Both modes and standalone MCMC restore
the archived start without resetting from the seed or settling again. Both modes
learn invariants from executions of the actual candidate and perform symbolic
and motion verification with feedback. See the
[collection guide](roboverify/synthesis/inference_lib/README.md).

Synthesis offers `--synthesis-approach relational` (the default) and
`--synthesis-approach id-first` (initial numeric MCMC/refinement, followed by
relational quotienting and named loop-body search).
Both return named programs to inference and verification. See the
[approach guide](roboverify/synthesis/cfg/VERIFICATION.md#synthesis-approaches).

Both the integrated and standalone search CLIs also accept
`--search-strategy mcmc|enumeration|contract` (default MCMC). Enumeration visits bounded
primitive structures in increasing length, checks each with BMC, then tunes its
float parameters using CEM. It retains the existing scoring and verification
stages; see the [search guide](roboverify/synthesis/mcmc/README.md) for scope and
budgets.
Contract search grows prefixes using nominal primitive contracts, observation
proposals, diversity and exact successor caching. The integrated CLI also exposes
opt-in [`--contract-refinement`](roboverify/synthesis/cfg/CONTRACT-REFINEMENT.md):
persistent milestone cuts, shared offset solving, reference-diverse proposals,
actual endpoint contracts and complete-program validation. A full CLI run
recovered a Stack loop from five whole demonstrations; a generated operand
alternative selected on development data passed 45/45 untouched three/four/five
block simulations and independent physical audits. This continuation result is
distinct from automatic training-only selection; its three development Scattered
counterexamples remain. A subsequent full CLI run with additional validation
starts and bounded float repair automatically recovers a different Stack loop
in 368.3 seconds and passes all 35 training/development starts, including the
checked endpoint effects. Its independent fresh evaluation is separate.
Complete Unstack synthesis and a physical refinement proof remain unresolved.
See the [evidence and scope](roboverify/synthesis/experiment/README.md#integrated-contract-refinement-and-final-stack-simulation).
Existing defaults are unchanged.
Contract search also screens candidates
with BMC before the existing CEM stage, while preserving CFG refinement and loop
inference. The [controlled comparison](roboverify/synthesis/experiment/README.md#maintained-contract-search-comparison)
separates matched beam/Metropolis search from the unchanged legacy CFG reference
and tests frozen bodies in complete supplied loops.
The separate [end-to-end experiment](roboverify/synthesis/experiment/README.md#end-to-end-synthesis-from-full-demonstrations)
uses whole independent-controller demonstrations without a supplied body or loop.
It synthesized a 17-primitive four-block Stack program that passed 15/15 fresh
simulations and the 5 mm contact/frame audit. This is a fixed-size program.
A [completed-retreat continuation](roboverify/synthesis/experiment/README.md#completed-retreat-recovers-a-loop-with-remaining-execution-failures)
resolved the geometric milestones' gripper-phase mismatch and synthesized a
five-primitive shared loop body. Whole-loop CEM on the same training data then
gave task/settling/5 mm contact-frame success of 15/15 with three blocks, 14/15
with four and 11/15 with five on fresh scenes. A separate Arc shortlist reached
15/15, 15/15 and 3/15 on the same scenes. Early guard exits and taller-tower
nonmover displacement remain; the 90% per-size physical-success target was not
met. Both candidates passed 12 symbolic checks, including unbounded proofs with
learned invariants; motion refinement and termination were not proved. Exact
executable JSON and candidate traces are saved. `--mode synthesize` skips final
proof; search acceptance and simulation alone are not verification.

The subsequent [automatic-boundary study](roboverify/synthesis/experiment/README.md#contract-guided-boundary-refinement-experiments)
replaces the manual retreat detector with existing primitive-contract checks
while preserving learned ON milestones. It recovers a complete Stack loop in
62.2 seconds after 3.46 seconds of cut proposals, passing 5/5 settled training
episodes. Fresh combined task/settling/5 mm success is 15/15, 14/15 and 4/15 at
three/four/five blocks. All 12 symbolic checks pass, but finite physical traces
show that local ON can hold while Put's Scattered effects or ON* domain assumptions
fail. A separate guard comparison repairs selected known early exits but finds
no improvement on 45 matched fresh starts. Unstack gains a locally successful
released Higher fragment only after an additional shared-offset solve; no full
Unstack program is established. These are experimental continuations, not new
production defaults or motion-refinement proofs.

Unstack's five-demo attempts synthesized three short segments but stopped at a
release boundary: geometric height success can occur while the payload remains
grasped, and the nominal Release model omits physical settling. No complete
Unstack program or fresh success rate is claimed by this study.

Stack reset scatters blocks within **0.70 m horizontally of the robot base**,
retaining block separation and initial gripper clearance. The shared Stack
precondition also requires one initial block-height level, expressed as
`forall x,y. Higher(x,y)`. The shared Higher predicate has a configurable
**1 mm tolerance** (`--higher-tolerance`, with 0 restoring exact comparison).
Primitive ID/ByName instructions share configurable controllers and retain their
50-step budgets.
See [controller settings](roboverify/synthesis/inference_lib/README.md#primitive-controller-settings).
Cartesian delta commands use uniform XYZ scaling in both the primitive
controller and active Fetch backend. Pick, Move and Release default to 2 mm
position tolerance. Release freezes XY before opening and corrects XYZ during
retreat, stopping on full 3D error. Head collision geometry is disabled in the
active CEE-US Fetch model; its appearance and mass are retained.
The four historical demonstration archives were deleted at the user's request;
reports and videos remain. The PyEDA verification recheck collected a fresh
five-trajectory, four-block archive with the current controllers; that archive
and its collection metadata are included in the repository. See the
[four-block Stack synthesis instructions](roboverify/synthesis/mcmc/README.md#four-block-stack-synthesis-without-verification)
to run standalone MCMC without verification, or the
[verification command](roboverify/synthesis/cfg/VERIFICATION.md#provided-stack-verification)
to verify the supplied program.
The [Stack controller measurements](roboverify/synthesis/experiment/CONTROLLER-PATHS.md)
diagnose Release drift and compare Pick stopping tolerances.

**Symbolic invariant inference uses the partition-based algorithm in
`inference.py`, through `InvInference`.** It is the intended algorithm for both
pipeline modes and standalone symbolic CEGIS; there is no learner-selection flag.
The truth-table minimizer is selectable with `--invariant-minimizer sympy|pyeda`
(SymPy by default). This preserves the partition algorithm and truth-table
semantics; see [minimizer settings](roboverify/synthesis/inference_lib/README.md#truth-table-minimization).

**Supplied Stack verification has passed with the intended learner and the vocabulary
`ON_star Higher Scattered equality`.** With the equal-height entry premise and
1 mm Higher tolerance, the bootstrap invariant passes without refinement.
The candidate passes unbounded symbolic verification and the documented
noiseless motion checks with explicit supported-tower geometry. Full synthesis
acceptance is resolved.
The **2026-10-01 PyEDA recheck passed on fresh demonstrations with the current
controllers**: all 12 symbolic checks (sizes 2–4 and unbounded) and all 63 noiseless
motion checks were valid, without invariant refinement or program repair.
The run returned `verified_model`; the earlier SymPy proof predates the combined
head-contact/Release/Pick update.
After [redundant obstacle VCs were consolidated](roboverify/synthesis/cfg/VERIFICATION.md#consolidating-redundant-obstacle-checks),
the same supported-model proof passes 12 symbolic and 36 motion checks.
[Unsat-core minimization](roboverify/synthesis/cfg/VERIFICATION.md#unsat-core-minimization)
now defaults off, preserving validity and premise-consistency checks. Ten parallel
Higher-retained rechecks passed all 120 symbolic VCs; unbounded exit VCs took
0.013–1.583 seconds instead of the earlier approximately 300-second totals.
See the [verification command](roboverify/synthesis/cfg/VERIFICATION.md#provided-stack-verification)
and the decisions on [initial heights](PAPER-DISCREPANCIES.md#31-stack-resets-equal-height-assumption-belongs-in-the-task-precondition)
and [Higher tolerance](PAPER-DISCREPANCIES.md#32-higher-tolerance-for-contact-induced-height-differences).
Archives must contain full simulator states and match the current task specification;
collections from before the equal-height precondition require recollection for
pipeline use. Collect demonstrations when an example's archive is absent.

A [Higher-removal ablation](roboverify/synthesis/cfg/VERIFICATION.md#higher-removal-ablation)
found that simply dropping Higher breaks the learned Stack proof. An explicitly
supplied invariant using only ON*, Scattered and equality passes both levels
under the existing supported-height model. Higher is not indispensable for this
task; recovering the replacement automatically is separate work.
Removing [all extra motion premises](roboverify/synthesis/cfg/VERIFICATION.md#removing-all-extra-motion-premises)
also breaks the Higher baseline's motion proof: the learned invariant does not
establish empty-gripper clearance or tight input alignment. Its symbolic proof
and carried-block collision checks still pass. The supported-tower result above
uses separate geometric assumptions and is not a quantifier-free motion proof.

For paper Section 6.2, `synthesis.entry.learn_invariant` starts with an empty
dataset and invariant False, searches increasing block counts directly for valid
initial environments reaching a failed VC, and runs the fixed program in MuJoCo.
Symbolic guard witnesses have no ID-order restriction; physical replay checks
those choices and confirms the selected failure at the predicted iteration.
It supports symbolic-only verification (default) or both symbolic and motion
verification. Stack is
implemented with an adapter interface for future environments. See the
[counterexample-learning experiment](roboverify/synthesis/experiment/invariant_learning/README.md).

Unstack and Reverse both sample their initial tower 54–70 cm ahead of the robot
base and 15–25 cm on its higher-Y side, independent of block count. All Unstack
factories use this single right-to-left layout. The
[provided Unstack program](roboverify/synthesis/examples/unstack.py) retains
15 cm leftward placements and executes through the shared controller. The old
lateral experiment and compatibility wrapper have been removed. Reverse rebuilds
the reversed tower on the lower-Y side using trainable table-relative Move
offsets. The supplied
[`synthesis.examples.reverse`](roboverify/synthesis/examples/reverse.py) program
starts with `b <- tbl`; one While loop handles every block using ordinary
Pick/Move/Release primitives and `b <- b_prime` after each placement. All 15 tested
executions passed (five seeds each at 2–4 blocks); `collect_demos --task reverse`
checks the reversal against frozen entry geometry. See the
[Reverse collection guide](roboverify/synthesis/inference_lib/README.md#collect-reverse-demonstrations).
This is empirical execution validation. The historical standalone Reverse
verifier encoded a different task and has been removed (discrepancy 39).
A [workspace and Reverse placement investigation](roboverify/synthesis/experiment/TOWER-WORKSPACE.md)
validated 200 executions in the now-adopted larger source region; fresh
production-factory rechecks passed all 30 executions across both tasks and
2–4 blocks. The user deferred the investigated payload-feedback placement skill
because it would strengthen the primitive and require reviewing its verification
assumptions. The investigation runners have been retired; their findings remain
documented. Production controllers and motion contracts remain unchanged.
Historical Unstack motion checks cover their original narrower source region.

A [Lean mechanization of WP soundness](roboverify/proofs/wp_soundness/README.md)
checks the complete, separated-chain model, including guarded top-to-top
relocation, quantified assertions and conditional program partial correctness.
The Python dispatcher now implements guarded source detachment and attachment,
including table-valued supports. Mechanical Python/Lean correspondence and
physical refinement remain separate obligations; see
[discrepancy 30](PAPER-DISCREPANCIES.md#30-stack-invariant-vocabulary-and-attachment-semantics).

The Lean development also checks an exact shared-height-witness Put WP and a
[restricted AER/EPR VC theorem](roboverify/proofs/wp_soundness/README.md#restricted-aerepr-vc-theorem),
with execution/no-fault soundness for the restricted symbolic generator.
Concrete Put and assignment discharge the action obligations. For AF
specifications and QF selection matrices, an exact small-model theorem and
closed EPR reduction decide VC validity over all well-formed current/frozen
block scenes. Preservation of the CAV paper's broader AF^R fragment is refuted
for our Put by a checked finite-counterexample theorem; AER preservation suffices for the
decidability result. This does not establish EPR membership of the production
pipeline or arbitrary extra background axioms; see
[discrepancy 1](PAPER-DISCREPANCIES.md#1-theorem-52-contradicts-the-papers-own-table-7-r_higher).

Python now offers [three selectable WP backends](roboverify/synthesis/verification_lib/WP-BACKENDS.md)
through `--wp-system old|new|reduction`. The default retains the existing rules
and axioms; `new` uses shared height representatives with succinct relational
axioms, and `reduction` compiles supported queries directly into Boolean choices
of complete current/frozen tower scenes using the derived small-model bound.
Unsupported queries and exhausted compilation budgets return `UNKNOWN`.
Independent finite-scene tests cover both implementations; no mechanical
Python/Lean correspondence or new full benchmark verification is claimed.

[Higher and Scattered action summaries](roboverify/synthesis/verification_lib/RELATIONAL-SUMMARIES.md)
now have an explicit framed, nondeterministic WP contract. CFG summary inference
recognizes these endpoint atoms, and motion checking must establish the endpoint,
preservation of other objects and the successor domain. Lean checks the abstract
rewrite and weakestness theorems for arbitrary object types; it does not certify
the Python translator or physical controllers. The finite solver encoding is
exact; unbounded quantified-array queries can remain inconclusive. This extends
the verification API without establishing a new environment proof.

The separate [baseline framework](roboverify/baselines/README.md) currently contains
pinned mypyvy UPDR models for Stack, Unstack and Reverse. With cvc5, all three
inferred invariants without supplied supporting invariants and passed a fresh
certificate check. Unstack took 21.18 seconds and Reverse 163.18 seconds in the
recorded runs; Z3 returned `unknown` under the tested query limits. These are
unbounded abstract partial-safety results under the documented model assumptions,
separate from physical motion verification and termination. The
[baseline guide](roboverify/baselines/mypyvy/README.md) provides dependencies,
reproduction commands, model boundaries and a bounded progress reporter.

A historical right-to-left Unstack result is recorded in the
[retired verification diagnostic](roboverify/synthesis/cfg/VERIFICATION.md#verification-of-the-current-unstack-revision):
its learned ON*/equality invariant passed all 12 symbolic checks (2–4 blocks and
unbounded), under the recorded domain axioms. Actual-reset motion checking failed
exact Release support because the base center starts slightly below 0.425 m.
A separately labeled exact-resting-height model passed 182 noiseless motion
checks across 2–4 blocks; this conditional result does not certify the unchanged
simulator/controller or establish unbounded motion safety.

Grid and Pyramid execute the shared reference program using nested loops over
goal-only links. Pyramid accepts `bottom_row_blocks=k` for k+(k−1)+...+1 blocks;
Grid accepts row/column dimensions. `collect_demos --task grid|pyramid` uses the
provided groundtruth by default and saves full-state archives with goal topology,
cursor bindings and nested-loop events; see the
[collection guide](roboverify/synthesis/inference_lib/README.md#collect-grid-and-pyramid-demonstrations).
Goal-aware synthesis and verification remain separate work. Execution and
collection evidence is recorded in
[discrepancy 40](PAPER-DISCREPANCIES.md#40-grid-and-pyramid-execution-and-verification-have-separate-support).

Build3D extends this execution model to **m rows × n columns × k layers**.
Only goals carry traversal links: `r` advances across a row, `d` between rows
within a layer, and `u` between layer heads. Three nested loops fill one layer
before starting the next, using the existing Pick/Move/Release controllers and
15 cm goal-relative clearance. The shared collector accepts `--task build3d`
with `--build-rows`, `--build-cols` and `--build-layers`. All 25 collected trials
passed task validation, exact action replay and 50 extra settling steps:
15 seeds for 2×2×2 and five each for 2×3×2 and 2×2×3. See the
[Build3D collection guide](roboverify/synthesis/inference_lib/README.md#collect-build3d-demonstrations).
This adds execution and collection; goal-aware synthesis and verification remain
unsupported, and larger dimensions remain subject to physical workspace limits.

Restack now supports a lateral source tower plus loose blocks and collection
with an adapted version of the paper's two-phase program. `--num-blocks` sets
the total and `--initial-tower-blocks` sets the source height; b0 starts inside
the tower. Source blocks stay on the higher-Y side, leaving the lower-Y side for
temporary placements and the rebuilt tower. `collect_demos --task restack`
defaults to the supplied program. The printed guard's nontermination and the
lateral waypoint adaptations are documented in
[discrepancy 41](PAPER-DISCREPANCIES.md#41-restack-needs-a-terminating-first-loop-and-an-executable-lateral-layout).
See the [collection guide](roboverify/synthesis/inference_lib/README.md#collect-restack-demonstrations).
Execution/collection is supported; Restack synthesis and formal verification
have not been added.
The legacy PartialStack environment, factory, standalone verifier and inference
option have been removed; Restack is the supported task.

The supported scope is structured chains and flat loops for the tower tasks;
the integrated synthesis CLI exposes Stack and Unstack. A successful
`verified_model` result establishes partial correctness in the documented
geometric model, not total termination or physical-controller refinement.

## Documentation

| Document | Purpose |
| --- | --- |
| [AGENTS.md](AGENTS.md) | Development rules, environment, commands, architecture and experiment reporting. |
| [PAPER-DISCREPANCIES.md](PAPER-DISCREPANCIES.md) | The single paper-review record: stable entries, decisions/proofs, status and remaining actions. |
| [Benchmark conditions](roboverify/synthesis/cfg/TASK-CONDITIONS.md) | Symbolic pre/postconditions for all seven tasks and their model scope. |
| [CFG verification](roboverify/synthesis/cfg/VERIFICATION.md) | Integrated synthesis/verification workflow and model assumptions. |
| [Motion API](roboverify/synthesis/verification_lib/README.md) | Motion contracts, collision/support checks, bounded noise and BMC distinctions. |
| [Draft WP soundness proof](roboverify/synthesis/verification_lib/WP-SOUNDNESS-DRAFT.md) | Relational model, atomic rewrite proofs, and explicit assumptions for review. |
| [Mechanized WP proof](roboverify/proofs/wp_soundness/README.md) | Lean build, checked theorem statements, paper correspondence and review boundaries. |
| [Higher/Scattered summaries](roboverify/synthesis/verification_lib/RELATIONAL-SUMMARIES.md) | Endpoint-summary WP contract, CFG/motion integration, Lean theorems and solver limits. |
| [Baselines](roboverify/baselines/README.md) | Independent baseline families, current mypyvy models and reproduction instructions. |
| [Standalone CEGIS](roboverify/synthesis/verification_lib/CEGIS.md) | APIs and commands for refining existing programs. |
| [Counterexample learning](roboverify/synthesis/experiment/invariant_learning/README.md) | Section 6.2 experiment from a fixed program and no demonstrations; symbolic or both proof stages. |
| [Trace inference](roboverify/synthesis/inference_lib/README.md) | Collecting loop-head states and learning invariants with DemoStore. |

Implementation lives in `roboverify/`. Configure MuJoCo through AGENTS.md and run
entry points as modules from that directory. Tests use `unittest`. The reviewed
paper is [POPL2027.pdf](POPL2027.pdf). Completed implementation plans and audit
history are retained in Git rather than maintained as active task documents.
