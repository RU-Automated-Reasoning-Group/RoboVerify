# Experiment inventory and current workflow

Routine demonstrations use `synthesis.entry.collect_demos --task TASK`, with a
provided groundtruth for Stack, Unstack, Reverse, Restack, Grid and Pyramid.
Use `synthesis.entry.audit_demos` to check saved archives, replay their actions
and test final physical success after settling. See the
[105-demonstration audit](../inference_lib/README.md#cross-environment-audit)
and [pipeline support matrix](../../../README.md#project-status). Execution and
collection support do not imply synthesis or formal verification support.

## Removed or consolidated

| Former implementation | Current state |
| --- | --- |
| PartialStack environment/factory/reset helpers | Removed at the user's request. Use the lateral Restack environment. The old `partial` and `partial_stack` task names are rejected. |
| `entry.verify_partial_with_learned_invariant` and its inference option/unused fixture | Removed with the legacy task. There is no replacement Restack proof claim. |
| `entry.verify_reverse_with_learned_invariant` | Removed because it encoded Unstack, not reversal. The physical Reverse program and its frozen-entry reversal validation remain supported. |
| `unstack_lateral.py` execution wrapper and separate collection loop | Replaced by `examples/unstack.py`, shared observation-relative coordinate resolution, and `collect_demos`. The short module retains compatibility imports and forwards CLI calls to collection. |

Unstack's `Y_min`, `Z_m` and `gripper` references are evaluated once at instruction
entry. Shared execution produced exactly the same actions and states as saved
pre-migration two- and four-block executions. Their semantics enter current
fingerprints, so historical archives do not automatically match the new source
identity. Keep the original program artifacts with historical proof results;
the current integrated loader explicitly rejects these unencoded references.

## Retained tools

| Tools | Why retained / scope |
| --- | --- |
| `mcmc/`, `run_logger.py`, `report.py`, `config.py` | Instrumented search and bounded reporting infrastructure, not one-off demonstrations. |
| `invariant_learning/` | The fixed-program counterexample-learning experiment; see its [guide](invariant_learning/README.md). |
| `ablate_higher.py`, `motion_premise_ablation.py` | Reproduce proof-premise and vocabulary findings; neither changes production verification defaults. |
| `compare_stack_heights.py` | Diagnose predicate interpretation on saved traces without rerunning the simulator. |
| `compare_stack_paths.py`, `render_stack_paths.py` | Inspect and visualize primitive paths; used by controller investigations and their tests. |
| `compare_stack_control.py`, `controller_baseline.py`, `tune_pick_tolerance.py`, `diagnose_release_drift.py` | Reproduce historical controller comparisons and explain the adopted settings. They are diagnostic tools, not the current demonstration collectors; see [controller measurements](CONTROLLER-PATHS.md). |
| `tower_workspace.py` | Reproduce the workspace survey and deferred payload-feedback investigation. It now uses shared Unstack execution; feedback variants remain diagnostic only. See the [workspace findings](TOWER-WORKSPACE.md). |
| `id_first_continuation.py` | Diagnose continuation/folding with supplied straight-line candidates. It does not establish end-to-end synthesis success. |
| Historical Unstack and 2D symbolic verification examples | Keep their abstraction/proof-regression role separate from current physical programs. The 2D mark-all example is not Grid/Pyramid physical verification. |

The local `verify_unstack_lateral.py` diagnostic and its test predate this audit
and remain separate local work. Their reported proof results retain their
original executable/fingerprint, reset and geometric-premise scope. The audit
did not rerun those proofs or merge them into the main pipeline.

Generated demos and run artifacts remain outside source commits. Historical
reports and videos are retained as evidence. The pre-existing plotting helper's
`partial_stack` log key denotes historical experiment data, not an active task;
the environment figure helper now imports Restack.

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
