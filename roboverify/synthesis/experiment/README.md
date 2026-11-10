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
