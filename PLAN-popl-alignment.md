# Align RoboVerify with the POPL submission

> **Status — read this first.**
>
> Branch: `phase-f-synthesis`. Phases A–E are integrated into `main`.
> Phase F's supported implementation and audit fixes are complete; **end-to-end
> learning acceptance on validated task demonstrations remains open**.
>
> Latest code validation: **215 unittest tests passed in 50.735 seconds**, including
> simulator tests, for commit `fdf5277` (entry 17). This documentation cleanup does
> not change code or claim a new test run.
>
> Entries 16–18 have no immediate implementation task under the agreed model.
> Their paper corrections remain recorded in [PAPER-DISCREPANCIES.md](PAPER-DISCREPANCIES.md).
> General supported-height premises for motion verification are deferred until
> needed; simulator/controller refinement and total termination are outside scope.
>
> **Next:** obtain or validate task-correct demonstrations, then run the integrated
> synthesis/verification pipeline and assess full loop recovery. Existing demos and
> paper experiment results are not correctness targets. Keep every Unstack
> end-to-end run under the user's **60-second wall-clock limit**.

## Scope and standing decisions

The paper is an artifact under test, not a specification. Neither paper nor code
wins automatically when they disagree. Log discrepancies and justify changes by
correctness arguments and tests. The algorithm review covers POPL2027.pdf §§2–5,
Algorithms 1–6, Appendix A/Table 7 and the corresponding appendix algorithms and
proofs; experimental numbers are not regression targets.

In scope: Stack, Unstack, Reverse and ReStack/Partial; structured chains and flat
loops. The current integrated synthesis CLI exposes Stack and Unstack. Branch
synthesis, nested/starred quotient, Grid/Pyramid, total termination and physical
controller refinement remain outside this plan. Existing out-of-scope code stays
available; this is not a claim that those tasks have verified physical programs.

Use [AGENTS.md](AGENTS.md) for environment, commands and working conventions;
[CLAUDE.md](CLAUDE.md) for architecture; and
[CFG verification](roboverify/synthesis/cfg/VERIFICATION.md) for the integrated
pipeline's proof scope. This status header is the session handoff; update it when
work changes. Detailed historical investigations remain in Git history rather
than in this active task list.

## Remaining work

- [ ] **Demonstration acceptance:** reconcile the historical Unstack source with
  the intended task or collect replacement demonstrations. Validate initial and
  final conditions before using them. Reaching the goal transiently is insufficient.
  See discrepancy 10. The current collector is not a task-correctness oracle.
- [ ] **Full learning acceptance:** with validated recordings, recover the complete
  loop and verify the same synthesized CFG against the same task specification.
  Record budget exhaustion, unsupported candidates and inconclusive verification
  explicitly. Structural tests already exercise synthesis and feedback without
  saved demonstrations; learning success is a separate acceptance gate.
- [ ] **Paper corrections:** work through the active discrepancy ledger, including
  Higher rules/model assumptions (16), split validation notation (17), and primitive
  contact semantics (18). Code completion does not mean the PDF has been edited.
- **Deferred by user:** encode general supported-height premises for all blocks,
  including unnamed objects, only if inadmissible intermediate-height models
  obstruct needed motion proofs. Until then, failed/unknown obligations remain
  unsuccessful verification; do not silently discard counterexamples.
- **Outside current scope:** controller/settling refinement (entry 4), termination
  proofs (15), branch/nested-loop synthesis and additional task families.

## Completed phases

| Phase | Implemented result | Current reference |
| --- | --- | --- |
| A — Fail closed | Uncovered motion, unknown instructions, solver unknown and inconsistent premises cannot certify success; interactive debugger traps removed; RNG state isolated. | Architecture and verification guides |
| B — Predicate/parameter alignment | Physical blocks separated from `tbl`; numeric/symbolic predicate agreement; fresh existential witnesses; `Top` removed; frozen-entry relations; xyz Move parameters; physical table height exposed. | Conflicts 5–7 below; discrepancy ledger |
| C — Real trace inference | `DemoStore`, per-invocation frozen geometry, explicit trace inputs to tower verifiers and regression-only golden fixtures. | [Trace workflow](roboverify/synthesis/inference_lib/README.md) |
| D — Motion obligations | Optional bounded noise; placement, frame, support, collision, complete relation-effect and root-alignment VCs; explicit unsupported results. | [Motion API](roboverify/synthesis/verification_lib/README.md) |
| E — Symbolic and motion refinement | Labeled VCs, vacuity checks, finite countermodel search/unbounded proof, geometric realization, invariant progress checks and accumulated motion penalties. | [Standalone CEGIS](roboverify/synthesis/verification_lib/CEGIS.md) |
| F — Synthesis | Scoped relational CFGs, exact predicate learning, recursive refinement, straight-line search, flat-loop recovery and integrated verification/resynthesis. Real learning acceptance remains open. | [Integrated workflow](roboverify/synthesis/cfg/VERIFICATION.md) |

### Phase F implementation record

| Item | Completed behavior |
| --- | --- |
| F0 — Segment starts | Full simulator snapshots include control/mocap/warm-start arrays and bindings; deterministic replay is the default, direct reset is optional. Tests compare restored state and the next action at recorded starts. Observation-only demos require a faithful replay source. |
| F1 — IR/lowering | The CFG is the search IR; `Program` is the execution/verification IR. Absolute inclusive demo segments share cut states. Structured regions lower to sequences, Get and While. Get requires witness existence and correctness for every permitted choice. |
| F2 — Validate | Atomic whole-CFG boundary, binding and adjacency checks. A split compares the new condition with the original outgoing target, as detailed below. |
| F3 — Predicates | Canonical terms with numeric/Z3 semantics; bounded classifier/guard search; direct ON and zero-binder formulas supported. Results distinguish exact separation, no separator and exhausted budget. |
| F4 — Refinement/scope | All transition witnesses, execution negatives, fresh existential bindings, must-scope analysis and synchronized demo partitions; rejected refinement leaves the graph unchanged. |
| F5 — Straight-line search | Shared acceptance/objective logic, cached KL/MMD distances, annealing, bounded epsilon pool and PostScore ranking on convergence. PostScore 1 on demo seeds is required for block acceptance, but is not final-state verification. |
| F6 — Driver | Instrumented recursive synthesis CLI; `main.py` forwards to it. The actual synthesized CFG and task enter verification, demonstration requests/resynthesis and structural motion repair. |
| F7 — Flat quotient | Fresh templates, consistent partial carry composition, multi-block units, suffix iteration extraction, fixed-point folding and refinement inside unresolved loop bodies. Generated loops have no demo-derived cap. |

For `P -> v0 -> Q`, refinement produces `P -> v1 --C--> v2 -> Q`.
Compare `first(C) <= last(Q)` on each **original unsplit** segment. P holds at its
start, first(C) is strictly interior, and Q holds at its end. These boundary checks
already imply the inequality. Entry and later blocks use the same rule; the entry
precondition, not C, holds at local time zero. P and C need not persist until the
next condition. See [entry 17](PAPER-DISCREPANCIES.md#17-temporal-validation-acceptance-and-the-incorrect-rejection-sentence).

Multiple guard witnesses are allowed. Demonstrated bindings are positives;
unselected continuing-state choices are unlabeled; all bindings at demonstrated
exits are negatives. Runtime may choose the first match because preservation
verification covers every matching choice. Explicit execution budgets raise
`LoopBudgetExceeded` when exhausted; they do not simulate a normal guard-false exit.

### Audit remediation — completed

The original 2026-09-20 audit examined `132cfee`. Its pre-fix descriptions and
probe outputs are retained in Git history (`AUDIT-popl-alignment.md` as of
`fdf5277`); the current checklist replaces that superseded document. IDs A1–A11
below are audit IDs, distinct from phase A's original substeps.

- [x] **A1 — Abstract effects and alignment:** motion proves ON*/Higher/Scattered
  effects against Put WP, block-only Scattered isolation, quantified root discovery
  and tight alignment for constructed towers. Existing input towers satisfy the
  declared alignment assumption; see [resolved entry 12](PAPER-RESOLUTIONS.md#12-root-discovery-and-tight-alignment-premises--implemented-with-an-explicit-input-assumption).
- [x] **A2 — Loop budgets:** no generated demo-count cap; explicit budget exhaustion
  reports incomplete execution instead of a verified loop exit.
- [x] **A3 — Template freshness:** generated template variables avoid existing
  names; substitution round trips are checked by regressions.
- [x] **A4 — Carried updates:** every adjacent substitution uses the same partial
  composition; inconsistent/noninjective extensions stop matching.
- [x] **A5 — Flat recovery:** discover all recorded iterations, fold to a fixed
  point and refine unresolved bodies while preserving the outer loop.
- [x] **A6 — Verification handoff:** verify the actual synthesized CFG, request and
  incorporate validated demos, relearn invariants and repair physical instruction
  structure while preserving the entire symbolic program.
- [x] **A7 — Primitive coverage:** Pick/Move/Release geometry, intentional contact,
  support/fall outcomes and state propagation across blocks/loop contexts.
- [x] **A8 — Bindings:** close numeric operands through in-scope aliases or typed
  Get; search/mutation/scoring respect runtime scope and assignments.
- [x] **A9 — Vocabulary:** zero-binder classifiers and direct ON are available.
- [x] **A10 — Whole-CFG validation:** boundaries, witnesses and adjacent partitions
  checked atomically after splits/folds; entry 17 supplies the settled semantics.
- [x] **A11 — Inference inputs:** CFG learning includes terminal heads and frozen
  entry geometry; CLI exposes vocabulary, variable count and legacy/monotone
  learners, with legacy as the integrated default.
- [x] **D1 — Demo diagnostics:** reject empty/invalid recordings and distinguish
  transient postcondition success from success at the recorded end.
- [x] **D2 — Demo-independent integration:** synthetic tests exercise loop recovery,
  actual verification, counterexamples, demo requests/resynthesis and motion repair.
- [x] **D3 — Saved-demo independence:** optimizer parity generates its Pick/Move
  trace in memory. Repository tests do not require saved demonstration files.

Shared symbol freshness is also complete. Use `synthesis/util/symbols.py` for
auxiliary solver symbols, generated program names and capture-safe quantifier
opening/rebuilding. Persistent program identities and intentionally shared state
symbols retain named lookup. Entry 16 records the corrected Higher rules and
quantifier-capture regressions.

## Conflicts

All seven original decisions are settled. This table records their current form;
completed actions are not new tasks.

| # | Settled decision |
| --- | --- |
| 1 | **Higher/WP theorem:** the paper's no-new-quantifiers argument conflicts with its own Table 7. Do not alter valid code to force that claim. Later Higher rule corrections are separately justified in entry 16; entry 1 no longer claims exact agreement with the old table. |
| 2 | **Distance:** use KL where convergence and epsilon thresholds depend on its scale; keep MMD selectable. Cache demo-side density and bound sampling costs. Do not reuse MMD thresholds as KL thresholds. |
| 3 | **Objective:** straight-line search optimizes imitation distance, then ranks the near-best pool by PostScore. Keep legacy weighted-goal scoring selectable for comparison. Pool default is 10; rank at convergence. |
| 4 | **Vacuity:** retain premise consistency checks and distinguish valid, invalid, vacuous and unknown. An unsat core is only a shortcut when it actually establishes inconsistent premises. |
| 5 | **Top:** removed; it is not part of the predicate vocabulary. |
| 6 | **Frozen ON\*:** retain `ON_star_zero` as the invocation's entry relation. Pin it to current ON* in the task precondition when used, never with a global link axiom; Put cannot rewrite the past. |
| 7 | **Move parameters:** optimize x, y and z. Reassess CEM budgets when dimensionality changes; a single smoke run does not justify changing defaults. |

## Model and algorithm boundaries

- Supported towers use uniform upright blocks of height L, a common flat table,
  exact support and complete layers. General height premises in the solver are
  deferred as noted above; the two-height restriction is only a regression fixture.
- Existing towers satisfy strict root-relative XY bounds `< L/4`. Constructed
  placements prove that bound against a proved root; frame/support VCs preserve it.
  Triangle inequality yields pairwise `< L/2`. Root-only checking is sufficient
  under this invariant; ON* alone does not imply the tighter premise.
- Motion checks retain the shared-`t` straight-segment collision query. An enclosing
  endpoint box would be an overapproximation, not an equivalent replacement.
- Noise is opt-in and off by default. BMC goal checking is distinct from robust
  motion/collision verification; solve mode is existential even when noise is used.
- Symbolic countermodels must preserve all relation tables when realized as
  geometry. Unrealisable/unknown cases remain explicit. Preservation refinement
  adds an uncovered **successor** through the abstract body; entry/exit failures
  cannot be treated as guaranteed repair by weakening an invariant.
- The standalone CEGIS learner defaults to monotone Boolean rows; the integrated
  CFG CLI defaults to the legacy learner. Accepted updates must pass explicit
  progress checks. Log False then bootstrap from demos; do not claim universal
  convergence from False or tune to the paper's reported iteration counts.
- Integrated motion repair may change physical instruction structure while keeping
  the full symbolic program unchanged. The older standalone offset-repair API has
  a narrower contract; see its guide.
- A successful integrated result is `verified_model`: partial correctness in the
  documented model. It is not proof of total termination, simulator settling,
  grasp reliability, full-arm/table collision freedom or hardware execution.

## Validation and acceptance

Configure the environment from AGENTS.md, then run from `roboverify/`:

```bash
uv run python - <<'PYTEST'
from pathlib import Path
import unittest
modules = sorted('.'.join(p.with_suffix('').parts)
                 for p in Path('synthesis').rglob('test_*.py'))
result = unittest.TextTestRunner(verbosity=2).run(
    unittest.defaultTestLoader.loadTestsFromNames(modules))
raise SystemExit(not result.wasSuccessful())
PYTEST
```

Run focused tests for changed behavior and the appropriate broader checks. Scope
formatting to changed files (`uvx isort --profile black`, then `uvx black`) and run
`git diff --check`. Documentation-only edits need link/reference checks, not a new
simulator run. Read experiment results through `synthesis.experiment.report`.

The completed timing gates are measurements, not regression targets: the Phase D
exact noisy collision spike took 0.0032 s; the existing Stack motion fixture took
0.122 s noiseless / 0.107 s noisy and refuted its endpoint contract. The Phase F
straight-line spike took 0.731 s versus 0.244 s for one objective evaluation, with
0.240 s in ranking. These motivated retaining the exact collision query and a
bounded candidate pool, not claims of task success.

Historical Stack/Unstack runs did not establish full verified learning. In
particular, the old Unstack demonstration failed its final condition and some
verification runs hit the one-minute cap. Use new validated recordings for the
open acceptance gate; do not preserve faulty demo verdicts as correctness oracles.
