# Revision 1: shared observations and condition modes for both CaP methods

Status: implemented and validated on 2026-10-16. Both `cap_google_baseline` and
`cap_method_1` now expose current gripper/block positions and accept the same two
condition modes. The configuration contract is in
[Method1_Configuration.md](Method1_Configuration.md); data flows are in
[BASELINE-FLOW.md](BASELINE-FLOW.md) and [METHOD-1-FLOW.md](METHOD-1-FLOW.md).

## Implemented interface

```text
O_t = (p_gripper(t), p_1(t), ..., p_n(t))
get_gripper_position() -> Position3
get_block_position(handle) -> Position3
--condition-mode dsl_only|coordinate_predicates
```

Position reads return immutable copies of finite world-frame XYZ values in
metres, without stepping physics. Saved triples are snapshots; query again after
movement. Method 1 supports literal component indices 0/1/2. Existing handles,
frozen trial-entry predicates, and Pick/Move/Release meanings are unchanged.
The default mode is `dsl_only` for both methods.

| Method | dsl_only | coordinate_predicates |
| --- | --- | --- |
| cap_google_baseline | Prompts require DSL-only conditions; positions may parameterize actions | Prompts also permit generated coordinate predicates and general numeric conditions, including counters |
| cap_method_1 | Same contract, enforced by validation and the checked interpreter | Same contract, enforced by validation and the checked interpreter |

There is no requirement for extended-mode numeric conditions to depend on
coordinates. Constant comparisons, counter guards, and numeric helper arguments
are supported alongside coordinate predicates. Conditions remain Boolean and
free of physical-action side effects in method 1. Its restricted Python subset
still applies; this is not unrestricted Python or new symbolic-proof support.

## Baseline preservation

The baseline deliberately remains close to the original Google CaP
implementation. The pinned `LMP -> LMPFGen -> exec_safe` source is unchanged.
Mode-specific prompt construction and shared position bindings live in the
integration/compatibility layer. Recursive helper discovery/binding and Python
execution remain upstream behavior.

No complete DSL checker, checked interpreter, or semantic repair loop was added
to the baseline. Both modes retain `validation=not_run`,
`validation_enabled=false`, and `refinement_enabled=false`. A numeric condition
that violates a baseline `dsl_only` prompt can still execute. Adapter argument,
handle, query-action, primitive, and budget checks remain active; they do not
establish complete loop/branch compliance.

Method 1 retains local source checking, recursive helper generation, bounded
interface-only repair, complete root checking, and checked AST execution. A
bundle records its mode, and runtime revalidation rejects a mismatched mode.
Position results are runtime data, not automatic LLM inputs or repair feedback.

## Completed implementation work

| Area | Implementation |
| --- | --- |
| Configuration and CLI | [config.py](../config.py), [run_cap.py](../../entry/run_cap.py), and [runner.py](../runner.py) validate/propagate the mode for all four combinations. |
| Shared observation API | [contracts.py](../contracts.py) defines Position3 and mode-specific API text; [adapter.py](../adapter.py) reads current positions and records coordinate-query events. Generation-only mode advertises the APIs but cannot read physical state. |
| Method-1 checking | [validation.py](../validation.py) checks Position3 indexing, mode-dependent comparison types, indirect helper/assignment flows, and action effects. Numeric comparisons are rejected in strict mode and accepted in extended mode. |
| Method-1 execution | [runtime.py](../runtime.py) interprets position reads and numeric comparisons, revalidates under the resolved mode, and retains finite-value and condition-effect checks. |
| Generation | [generation.py](../cap_method_1/generation.py) passes mode rules to policy/helper/repair requests. [Baseline compatibility](../cap_google_baseline/compatibility.py) supplies the same mode/API text to upstream policy/helper prompts. |
| Provenance and reporting | [checkpoints.py](../checkpoints.py) records mode, observation API version, checker version/hash, and validation/repair status. [report.py](../../experiment/report.py) displays the mode and versions within its existing output limit. |

Observation API version: `positions-v1`. Method-1 checker version:
`condition-modes-v1`; baseline checker version/hash remain null. Initial policy
prompts match across methods for identical task/API/mode inputs. No live model
availability or task-performance claim follows from these changes.

## Validation

All **57 CaP unittest cases passed** using the existing project environment:

```bash
.venv/bin/python -m unittest discover -s synthesis/cap -t . -p 'test_*.py' -v
```

The tests cover immutable/fresh position reads, invalid handles and indices,
finite values, exact integer comparisons, coordinate-derived action parameters in both modes, indirect
numeric conditions, nested helpers and quantifiers, numeric counters, and
condition purity. Scripted generation tests cover mode propagation to helper and
repair requests, no runtime coordinates in those requests, and mode-bearing
checkpoints. A baseline regression executes numeric conditions even in strict
prompt mode while asserting that method-1 validation/repair is never called.
Pinned-notebook/core equivalence and original helper-generation tests pass.
An offline CLI check also exercised all four combinations with a counter-loop
policy: both baseline modes and extended method 1 returned `generated`; strict
method 1 returned `constraint_failed` before execution, as required.

A real MuJoCo integration diagnostic ran the same scripted policy from one saved
three-block Stack start across all four method/mode combinations. The script
reads positions, executes Pick, reads again, computes a relative Move offset
from saved positions, and reads the final gripper position. All four starts,
coordinate-read results, and action sequences matched exactly.

| Measurement | All four combinations |
| --- | --- |
| Pick | Converged in 14 controller steps |
| Move | Converged in 5 controller steps |
| Total physical actions | 19 |
| Position reads | 4, with current values refreshed after action |
| Model transport | Scripted response, one generation call; no paid/live model request |
| Trial verdict | `goal_failed`, expected for a partial Pick/Move diagnostic |

Artifacts are under `runs/cap-revision-1/20261016-213331/`: the diagnostic script,
scripted policy, compact `summary.json`, bounded per-run reports, initial states,
trajectories, source checkpoints, and timelines. The code revision was `fd36754`.
Video was disabled for this parity diagnostic. Normal completion and controller
convergence here establish integration parity, not successful Stack completion.

The completed revision does not add per-step LLM replanning, rollout-feedback
repair, a new controller, Goal adapters, or proof rules for generated numeric
predicates. New multi-trial/live-model benchmarking remains separate work.
