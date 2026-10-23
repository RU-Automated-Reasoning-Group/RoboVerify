# RoboVerify

RoboVerify synthesizes block-manipulation programs from demonstrations and checks
symbolic correctness and geometric motion obligations. Programs execute in a
Fetch/MuJoCo environment. The pipeline uses relational CFGs, flat-loop recovery,
learned invariants and counterexample-guided refinement.

## Project status

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
five-trajectory, four-block archive with the current controllers; its path is in
the [verification command](roboverify/synthesis/cfg/VERIFICATION.md#provided-stack-verification).
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
factories use this single right-to-left layout; the
[lateral experiment](roboverify/synthesis/cfg/VERIFICATION.md#right-to-left-unstack-revision)
uses the same factory and retains 15 cm leftward placements. Reverse rebuilds
the reversed tower on the lower-Y side using trainable table-relative Move
offsets. The supplied
[`synthesis.examples.reverse`](roboverify/synthesis/examples/reverse.py) program
starts with `b <- tbl`; one While loop handles every block using ordinary
Pick/Move/Release primitives and `b <- b_prime` after each placement. All 15 tested
executions passed (five seeds each at 2–4 blocks); `collect_demos --task reverse`
checks the reversal against frozen entry geometry. See the
[Reverse collection guide](roboverify/synthesis/inference_lib/README.md#collect-reverse-demonstrations).
This is empirical execution validation; the historical standalone Reverse
verifier does not verify this program (discrepancy 39).
A [workspace and Reverse placement investigation](roboverify/synthesis/experiment/TOWER-WORKSPACE.md)
validated 200 executions in the now-adopted larger source region; fresh
production-factory rechecks passed all 30 executions across both tasks and
2–4 blocks. The user deferred the investigated payload-feedback placement skill
because it would strengthen the primitive and require reviewing its verification
assumptions. Production controllers and motion contracts remain unchanged.
Historical Unstack motion checks cover their original narrower source region.

A [Lean mechanization of WP soundness](roboverify/proofs/wp_soundness/README.md)
checks the complete, separated-chain model, including guarded top-to-top
relocation, quantified assertions and conditional program partial correctness.
The Python attachment-only dispatcher and physical refinement remain separate
correspondence obligations; this does not change the verification status above.

Grid and Pyramid execute the shared reference program using nested loops over
goal-only links. Pyramid accepts `bottom_row_blocks=k` for k+(k−1)+...+1 blocks;
Grid accepts row/column dimensions. `collect_demos --task grid|pyramid` uses the
provided groundtruth by default and saves full-state archives with goal topology,
cursor bindings and nested-loop events; see the
[collection guide](roboverify/synthesis/inference_lib/README.md#collect-grid-and-pyramid-demonstrations).
Goal-aware synthesis and verification remain separate work. Execution and
collection evidence is recorded in
[discrepancy 40](PAPER-DISCREPANCIES.md#40-grid-and-pyramid-execution-and-verification-have-separate-support).

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

The supported scope is structured chains and flat loops for the tower tasks;
the integrated synthesis CLI exposes Stack and Unstack. A successful
`verified_model` result establishes partial correctness in the documented
geometric model, not total termination or physical-controller refinement.

## Documentation

| Document | Purpose |
| --- | --- |
| [AGENTS.md](AGENTS.md) | Development rules, environment, commands, architecture and experiment reporting. |
| [PAPER-DISCREPANCIES.md](PAPER-DISCREPANCIES.md) | The single paper-review record: stable entries, decisions/proofs, status and remaining actions. |
| [CFG verification](roboverify/synthesis/cfg/VERIFICATION.md) | Integrated synthesis/verification workflow and model assumptions. |
| [Motion API](roboverify/synthesis/verification_lib/README.md) | Motion contracts, collision/support checks, bounded noise and BMC distinctions. |
| [Draft WP soundness proof](roboverify/synthesis/verification_lib/WP-SOUNDNESS-DRAFT.md) | Relational model, atomic rewrite proofs, and explicit assumptions for review. |
| [Mechanized WP proof](roboverify/proofs/wp_soundness/README.md) | Lean build, checked theorem statements, paper correspondence and review boundaries. |
| [Standalone CEGIS](roboverify/synthesis/verification_lib/CEGIS.md) | APIs and commands for refining existing programs. |
| [Counterexample learning](roboverify/synthesis/experiment/invariant_learning/README.md) | Section 6.2 experiment from a fixed program and no demonstrations; symbolic or both proof stages. |
| [Trace inference](roboverify/synthesis/inference_lib/README.md) | Collecting loop-head states and learning invariants with DemoStore. |

Implementation lives in `roboverify/`. Configure MuJoCo through AGENTS.md and run
entry points as modules from that directory. Tests use `unittest`. The reviewed
paper is [POPL2027.pdf](POPL2027.pdf). Completed implementation plans and audit
history are retained in Git rather than maintained as active task documents.
