# Revision 5: extend task and demonstration-collection support

- **Status:** planned; not implemented.
- **Goal:** support Unstack, Reverse, ReStack, Grid, and Pyramid with consistent task definitions, executable demonstrations, and matched CaP evaluation.
- **References:** [shared task conditions](../../cfg/tasks.py), [collector](../../entry/collect_demos.py), [CaP task adapter](../tasks.py), and [environment definitions](../../environment/cee_us_env/fpp_construction_env.py).

## 1. Reuse the Stack adaptation structure

| Layer | Existing Stack approach | Required for each additional task |
| --- | --- | --- |
| Task specification | Shared pre/postconditions used by collection, synthesis, and verification | Reconcile existing definitions and expose a shared task identity and validation interface |
| Concrete interpretation | Evaluate relational predicates on recorded block scenes | Define the domain, bindings, frozen entry geometry, and tolerances consistently |
| Demonstration program | Explicit Pick/Move/Release program with supported control flow | Supply a physically executable program and validate its behavior; a symbolic program alone is insufficient |
| Initial scene | Task reset or exact archived state; fresh Stack has 50 settling steps | Define a task-appropriate start and settling policy rather than copying Stack blindly |
| Recording | Full simulator snapshots, observations, actions, bindings, events, optional video | Reuse the archive/recording infrastructure and preserve faithful reset/replay |
| Validation | Normal completion, primitive convergence, shared pre/postconditions | Also check agreement with the task's physical success evaluator; preserve failures |
| CaP input and evaluation | Task wording derived from the shared specification; independent terminal evaluation | Align all methods on instruction, APIs, scene, and success criteria |

## 2. Current task coverage and missing integration

| Task | Existing definitions | Planned work |
| --- | --- | --- |
| Unstack | Environment, shared relational pre/postconditions, and historical program fixtures | Add collector support; provide and validate an explicit primitive demonstration factory; handle logical `tbl` and check every block rests on the table |
| Reverse | Canonical initial tower and physical order-reversal evaluator; standalone verification definitions | Reconcile formal and physical specifications; preserve initial order/frozen geometry; provide a primitive program and collection validation |
| ReStack | Partial-stack environment with multiple towers and a designated b0; final single-tower evaluator | Add a matching shared specification and primitive program. Do not assume the standalone Partial verifier defines the same ReStack task |
| Grid | Goal positions, reset, and physical layout evaluator | Add shared goal/reference semantics, specification, executable demonstration, and collector/CaP integration |
| Pyramid | Six-block 3+2+1 layered goal positions, reset, and physical evaluator | Add shared goal/reference semantics, specification, executable demonstration, and collector/CaP integration |

- These tasks do not lack all specifications: several definitions already exist in different subsystems. The missing work includes reconciliation, executable programs, and integration.
- The collector CLI currently accepts only Stack. `task_spec` supports Stack and Unstack only.
- Grid/Pyramid currently fail explicitly at the CaP adapter boundary; adding a CLI choice alone does not resolve the missing goal interface.

### 2.1 Open question: Reverse and ReStack specifications (deferred)

- **Status:** deferred for later discussion on 2026-10-22. The CaP adapter is unchanged until the definition is settled.
- **Finding:** Reverse and ReStack have environment cases, success evaluators and standalone verifiers outside `synthesis/cap/`. However, they have no entry in the shared [`task_spec`](../../cfg/tasks.py). The standalone verifiers' formulas also do not specify the complete task.

| Task | Shared pre/post ([`cfg/tasks.py`](../../cfg/tasks.py)) | Physical success ([environment](../../environment/cee_us_env/fpp_construction_env.py)) | Standalone verifier pre/post |
| --- | --- | --- | --- |
| Unstack | pre `∀x. x = tbl ∨ ON_star(x, b0)`; post `∀x,y. ON_star(x, y) → x = y` | `_roboverify_unstack_all_on_ground`: every block at table height and no block on another | [verify_unstack](../../entry/verify_unstack_with_learned_invariant.py) |
| Reverse | None | `_roboverify_reverse_tower_complete`: one tower whose bottom-to-top order is the reverse of the canonical start `[b0, …]` | `verify_reverse` (removed upstream) checks only the first loop, which puts every block on the table. Its postcondition matches Unstack's (singleton `ON_star`, equal height, Scattered) and says nothing about reversed order. Its precondition pins `ON_star_zero = ON_star` at entry. |
| ReStack (`partial_stack`) | None; the name `restack` exists only in CaP | `_roboverify_stack_tower_complete`, the same evaluator as Stack: one tower rooted at b0, with b0 on the table. Start: `_reset_sim_roboverify_partial_stack`, a random partition into at least two towers. | `verify_partial` (removed upstream) uses the Unstack postcondition. Its precondition, every block `ON_star(·, b0)`, describes a single tower and does not match the multi-tower start. |

**Update after merging upstream `main` (`67d48dc`, 2026-10-23).** This settles the source of the specifications. Upstream now defines both tasks in the shared [`task_spec`](../../cfg/tasks.py), and the entries above describe the earlier state.

- **ReStack (`restack`):**
  - pre: `b0 ≠ tbl` and every two physical blocks are tower-related or Scattered;
  - post: every physical block is `ON_star(·, b0)`.
  - Environment: `RoboVerifyRestack`, one lateral source tower containing `b0` plus loose blocks. It uses the same success check as Stack.
- **Reverse:**
  - pre: every physical block is `ON_star(·, b0)` and `ON_star_zero = ON_star`;
  - post: for physical blocks, `ON_star_zero(x, y) ↔ ON_star(y, x)`, i.e. the order is reversed.
- The standalone `verify_reverse`/`verify_partial` entry points above were removed upstream (`3537361`).
- CaP `restack` now creates the upstream Restack environment (see [pre-merge review](pre_merge_upstream_main_20261023.md)). Adopting the shared pre/postconditions in CaP's `validate_start`/`evaluate_terminal` remains a Revision 5 item.

Current CaP handling in [tasks.py](../tasks.py):

- **Unstack:** `validate_start` checks the shared precondition. `evaluate_terminal` reports the physical result and the DSL postcondition.
- **Reverse:** `validate_start` requires the canonical initial tower. Only the physical result is evaluated; `dsl_postcondition` is `None`.
- **ReStack:** `restack` maps to the `partial_stack` environment. The start is not validated, and only the physical result is evaluated.

Options to decide:

- **(A) Use the shared `task_spec` as the source of definitions.** Keep Stack and Unstack, and remove Reverse and ReStack from the CaP adapter until shared specifications exist. These two tasks currently cannot be evaluated with a DSL postcondition, unlike Stack and Unstack.
- **(B) Use the environment cases as the source of definitions.** Keep both tasks, and add shared pre/postconditions that agree with the physical evaluators. Reverse needs frozen entry geometry for the original order. Treat the standalone verifiers' first-loop formulas as partial specifications, not task definitions.

## 3. Planned changes

- Introduce task-aware collection configuration, including symbolic context, logical-table handling, aliases, initial-state validation, and terminal validation.
- Expose task specifications without silently weakening existing pre/postconditions or replacing physical evaluation with a less restrictive symbolic condition.
  - For Unstack, absence of block-on-block relations alone does not establish that every block rests on the table.
  - Logical `tbl` is not an object-coordinate reference for Move.
- Integrate primitive demonstration factories compatible with the current loader; identify any required control-flow extensions explicitly.
- Preserve full-state archives and task identity/version metadata. Do not reuse incompatible historical data simply because it has the current NPZ format.
- Maintain Unstack's 60-second end-to-end limit and explicit failure handling.
- Derive task prompt text from the agreed definitions and record exact instructions. Initial observation input remains a separately configured research choice.

## 4. Validation and acceptance

- Start with Unstack, then integrate the remaining tasks after specification/program readiness is established.
- Test valid and invalid starts, goal and non-goal terminal states, object bindings, and evaluator agreement.
- Collect demonstrations across distinct seeds; retain failed seeds instead of replacing them. Publish a valid archive only when the requested collection passes its contract.
- Check snapshot restoration, action/observation alignment, primitive convergence, and optional video frame counts.
- Run CaP comparisons on paired held-out scenes once task adapters are ready. Keep demonstration and evaluation scenes separate.
- This document does not authorize implementing these changes or collecting new demonstrations yet.
