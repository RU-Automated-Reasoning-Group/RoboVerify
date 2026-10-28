# Revision 8: verifying CaP-generated programs with RoboVerify

- **Status:** plan, 2026-10-28. Nothing in this document is implemented yet.
- **Input:** the 300 final programs of the `final_v2` Opus 5.5 experiment
  ([final_v1 report](final_v1_report.md)): `runs/cap-final-v2/20261027-final-50/claude-opus-5.5/<task>/`,
  plus `runs/cap-final-v2/20261027-final-50-supplement/` (seed 50 for Unstack and
  Reverse; seed 39 of those tasks was excluded). Every trial succeeded, so each
  trial contributes exactly one program: the executed checkpoint of its
  successful round (`attempts/round_<k>/cap/<run>/artifacts/trial_001/checkpoints/<active_checkpoint_id>/policy.py`,
  with `k = task_spec_success_round`).
- **Goal:** run RoboVerify's verification (`synthesize_cfg --mode verify`) on
  each of these programs, or report precisely why it cannot run, without changing
  any program and without weakening the verifier.

## 1. Summary

1. **Success on one scene is not verification.** Each program was executed on
   exactly one scene: its own trial seed. Some programs needed up to four
   recovery rounds on that same scene. `task_spec_success` shows that the
   program worked once. `verified_model` would show partial correctness for
   every precondition state and any block count, in RoboVerify's geometric
   primitive model ([scope](../../cfg/VERIFICATION.md#scope-and-model)).
2. **The primitives already match.** The CaP runtime executes `Pick`, `Move`
   and `Release` as RoboVerify's `PickByName`, `MoveByName` and
   `ReleaseByName`, with the same controllers
   ([adapter.py](../adapter.py), around line 280). `get`, `exists` and
   `forall` enumerate physical blocks in ascending ID order and return the first
   witness, which matches RoboVerify's guard semantics
   ([adapter.py:214](../adapter.py)). Predicates use the shared
   `scene_from_obs` evaluator. The gap is in program structure, not in the
   primitives.
3. **Only 14 of the 300 programs, all Stack, avoid every construct the current
   pipeline rejects** (Section 3.3).
   - The main blockers are:
     - Python `if` (224 programs);
     - offsets computed from measured positions (183);
     - `Release` given a block other than the one held (186);
     - table-relative Move operands, which the motion verifier does not encode (90);
     - nested goal loops in Grid and Pyramid (100).
   - Each blocker maps to a specific verifier extension. Section 6 lists them,
     and Section 7 orders them into milestones with projected coverage.
4. **Recommended order:**
   - **M0:** extraction and cross-scene replay, with no proofs and no model calls.
   - **M1:** a translator with action-identical translation validation.
   - **M2:** the 14 Stack programs on the unchanged pipeline.
   - **M3–M6:** one verifier extension at a time. Each extension must match the
     runtime semantics and comes with its own tests; none may make a CaP
     program pass by relaxing a check.

## 2. What "verified" means here

The integrated verify mode reports `verified_model`. It means that symbolic
verification passed at finite sizes 2–4 and in the unbounded domain, and that
every motion obligation passed. Both are with respect to:
- the task's pre- and postconditions (`cfg/tasks.py`);
- learned loop invariants;
- the explicit premises: `--supported-towers`, `--table-surface-height`,
  `--initial-arm` and the Higher tolerance.

It does not cover termination, controller refinement, settling or contacts.
CaP's success criterion (`task-spec-v2+final-release-v1`) is a physical
check of a single execution after 50 settling steps. The two results are
separate, and both are reported.

Every program gets exactly one terminal outcome:

| Outcome | Meaning |
| --- | --- |
| `verified_model` | Both stages passed in the declared scope |
| `translation_unsupported` | The source uses a construct with no faithful RoboVerify encoding; the construct and source line are listed |
| `translation_mismatch` | The translated program does not reproduce the CaP rollout on the trial's own scene (a translator defect; must be 0) |
| `verifier_unsupported_task` | The task has no integrated verification (today Reverse, ReStack, Grid, Pyramid) |
| `verifier_unsupported_shape` | The adapter, summary proposal or motion encoding returns `unsupported` (for example a table-relative operand) |
| `refuted_by_execution` | The program fails a precondition-satisfying scene in MuJoCo (M0 cross-scene replay or a realized counterexample) |
| `symbolic_not_proved` | A VC stays invalid after invariant refinement, or `needs_demonstrations`; this is not a refutation unless a counterexample is realized |
| `motion_refuted` | A motion obligation is invalid, with its geometric counterexample |
| `unknown` | Solver unknown or timeout |

The verifier fails closed: an unsupported, unknown or vacuous result is never
counted as verified ([AGENTS.md](../../../../AGENTS.md)). Structural repair and
resynthesis are disabled (`--motion-iterations 0`, no `--additional-demos`).
The program under test is never changed. Invariant refinement, which changes
only the invariant, stays on.

## 3. The 300 programs

Measured with an AST scan of all 300 successful checkpoints. Helpers are
inlined through the call graph, and no helper is recursive.

### 3.1 Size and duplicates

| Task | Programs | Distinct (exact AST) | Distinct (alpha-renamed) | Non-blank lines min/median/max | Successful round 0/1/2/3/4 |
| --- | ---: | ---: | ---: | --- | --- |
| Stack | 50 | 47 | 46 | 12 / 13 / 27 | 50/0/0/0/0 |
| Unstack | 50 | 50 | 50 | 14 / 22 / 33 | 50/0/0/0/0 |
| Reverse | 50 | 50 | 50 | 21 / 34 / 54 | 23/17/9/0/1 |
| ReStack | 50 | 50 | 50 | 28 / 57 / 83 | 34/12/3/1/0 |
| Grid | 50 | 46 | 36 | 19 / 23 / 28 | 50/0/0/0/0 |
| Pyramid | 50 | 29 | 14 | 15 / 19 / 25 | 50/0/0/0/0 |
| Total | 300 | 272 | 246 | | |

Deduplicate on the alpha-renamed AST hash, which ignores comments, docstrings
and local names. That is 246 verification jobs. Results are reported per
trial, linked to the job of its program.

### 3.2 Constructs (number of programs using each)

| Construct | Stack | Unstack | Reverse | ReStack | Grid | Pyramid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `while` loop nesting depth | 1 | 1 | 1 | 1 | 2 | 2 |
| `if` | 5 | 27 | 45 | 47 | 50 | 50 |
| `break` / `continue` | 3 | 19 | 7 | 4 | 0 | 1 |
| `get_block_position` / `get_gripper_position` read | 22 | 42 | 39 | 49 | 26 | 7 |
| Move/Release offset depends on a measurement | 22 | 42 | 37 | 49 | 26 | 7 |
| All offsets are literals | 28 | 6 | 8 | 0 | 18 | 43 |
| `get` or measurement while a block is held | 22 | 37 | 32 | 48 | 26 | 7 |
| `get` while a block is held | 6 | 12 | 16 | 29 | 0 | 0 |
| `Release(r, h)` where `r` is not the picked name (syntactic) | 15 | 29 | 21 | 21 | 50 | 50 |
| `tbl` as a Move/Release operand | 0 | 39 | 35 | 16 | 0 | 0 |
| `ON_star_zero` | 0 | 0 | 3 | 0 | 0 | 0 |
| Multi-binder `exists`/`forall` | 0 | 5 | 2 | 0 | 0 | 0 |
| Goal API (`HasBlock`, `AtGoal`, `MoveRight`, `MoveDown`) | 0 | 0 | 0 | 0 | 50 | 50 |

Every program calls `Pick`, `Move` (four arguments) and `Release` (two
arguments), plus `get` and `exists`. No program uses `for`, recursion or
imports. The Python subset is the one the shared checker enforces
([SHARED-DSL-CHECKER.md](SHARED-DSL-CHECKER.md)).

Typical shapes:
- **Measurement-free Stack** (seed 0): `while exists(x. not ON*(x,b0))`.
  The body binds `blk = get(...)` and `top = get(tower top)`, then calls
  `Pick(blk)`, three `Move`s referencing `top`, and `Release(blk, 0.15)`.
- **Measured variants:** these read `gp - bp` after the lift and add it to the
  transport offsets, which compensates for the carried block's offset in the
  gripper.
- **Unstack:** builds a row on the table relative to `b0` and `last`, often
  with `tbl` as the Z reference.
- **ReStack:**
  - phases written as helpers, such as `lift_block`, `stack_on` and `put_on_table`;
  - numeric layout constants;
  - `if` to choose the buffer tower.
- **Grid and Pyramid:** RoboVerify's reference shape (Figs. 17–18): nested goal
  loops, an `if not HasBlock(g)`, and `Release(goal, h)`.

### 3.3 Blockers in the current pipeline

These checks are syntactic and conservative. A program without a listed
blocker can still fail summary proposal or proof.

| Blocker | Where the current code refuses it |
| --- | --- |
| `if`/`else`, `break`, `continue`, early `return` | No `If` instruction exists. `Program.wp` and CFG lowering have no branch ([DSL.md §6](DSL.md#6-conditions-object-selection-and-control-flow)) |
| Offsets computed from measured positions | `MoveByName.target_offset` holds constants; motion uses `concrete_float` ([primitive_motion.py](../../verification_lib/primitive_motion.py)) |
| `get` or `Assign` between `Pick` and `Release` | [program_adapter.py:57](../../cfg/program_adapter.py), `propose_summaries` ([cfg/verification.py:42](../../cfg/verification.py)) |
| `Release(r, h)` with `r` ≠ held block | Motion obligation `release_object` ([primitive_motion.py:253](../../verification_lib/primitive_motion.py)); the runtime uses `r` only as the retreat-height reference ([control.py `release`](../../api/control.py)) |
| `tbl` operand in Move/Release | Motion check `table_reference_*` is `unsupported` ([primitive_motion.py:183](../../verification_lib/primitive_motion.py)) |
| Nested loops, goal handles | [program_source.py:157, 185](../../cfg/program_source.py); goal-aware verification is deferred ([discrepancy 40](../../../../PAPER-DISCREPANCIES.md#40-grid-and-pyramid-execution-and-verification-have-separate-support)) |
| `ON_star_zero` scope | CaP freezes entry geometry at program entry ([adapter.py:256](../adapter.py)); RoboVerify `While` freezes it at each loop invocation ([instructions.py:752](../../api/instructions.py)) |
| Task not exposed | `synthesize_cfg --task` accepts `stack` and `unstack` only ([synthesize_cfg.py:392](../../entry/synthesize_cfg.py)) |

Projected coverage. A program is counted at a milestone when all of its
blockers are covered by that milestone. These are upper bounds, not predicted
proof results.

| Milestone (cumulative) | Stack | Unstack | Reverse | ReStack | Grid | Pyramid | Total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M2: existing pipeline | 14 | 0 | 0 | 0 | 0 | 0 | 14 |
| M3: + Release retreat reference, + table operands | 22 | 2 | 1 | 0 | 0 | 0 | 25 |
| M4a: + measured offsets | 39 | 14 | 5 | 1 | 0 | 0 | 59 |
| M4b: + observations while holding | 45 | 23 | 5 | 3 | 0 | 0 | 76 |
| M4c: + `if`/`break` | 50 | 50 | 47 | 50 | 0 | 0 | 197 |
| M5: + task-scope `ON_star_zero`, Reverse/ReStack tasks | 50 | 50 | 50 | 50 | 0 | 0 | 200 |
| M6: + goal tasks, nested loops | 50 | 50 | 50 | 50 | 50 | 50 | 300 |

The Reverse and ReStack rows also require M5's task integration before any
proof can run.

## 4. Translating CaP source to a RoboVerify `Program`

New module: `synthesis/cap/verification/translate.py`.
- **Input:** the checked policy source, the task, and the input handles
  (`b0`, `tbl`, or `h0`, `null`).
- **Output:** either a `Program` factory or `translation_unsupported` with the
  construct, source line and reason.
- **Approximation:** none. The translator never drops, reorders or folds
  anything whose value depends on the state at runtime.

1. **Parse and normalize.** Parse with `ast` after re-running the shared DSL
   checker; the source passed it at generation time. Inline the helpers:
   - *Boolean helpers* (predicate combinations) become formula macros.
   - *Object-valued pure helpers* become a `Get` with a defining formula. For
     example, `return a if Higher(a, b) else b` becomes
     `get(x. (x = a ∧ Higher(a,b)) ∨ (x = b ∧ ¬Higher(a,b)))`, whose witness is
     unique.
   - *Statement helpers*, which contain primitives, are inlined with
     capture-avoiding names (`util/symbols.fresh_name`).
2. **Object variables.**
   - `x = get(lambda y: P)` becomes `Get(x, P)`.
   - `x = y` becomes `Assign(x, y)`.
   - `x = get(lambda y: eq(y, b0))` becomes `Assign(x, b0)`.
3. **Loops.** `while exists(lambda x: P(x)):` whose body starts with
   `v = get(lambda x: P(x))` (the same formula) becomes `While(P(v), [v], body)`.
   Any other `while exists(...)` becomes a binder-free `While` with an explicit
   `Get` in the body, which keeps the Get witness-existence obligation.
   `max_iters` is set from the CaP execution budget so that replay never stops
   early.
4. **Conditions and quantifiers.** CaP's conditions translate to `predicates.term`
   formulas: `ON`, `ON_star`, `ON_star_zero`, `Higher`, `Scattered`, `eq`,
   `not`/`and`/`or`, `exists`/`forall` with one or more binders.
   - CaP quantifiers range over physical blocks only (`self.handles`). In
     table tasks, RoboVerify scenes include `tbl`. The translator therefore
     guards every binder with `x != tbl`, so that the translated formula has
     CaP's meaning.
5. **Primitives.** These translate one to one to the `...ByName` instructions,
   whose offsets must be constants:
   - **Literals** are copied.
   - **Numeric locals** assigned only from literals and literal arithmetic are
     constant-folded.
   - **Counters** (`i += 1` used in offsets) and **measured values** are
     `translation_unsupported` until M4a.
6. **Unsupported until their milestone:**
   - `if`, `break`, `continue` and early `return` (until M4c);
   - goal loops (until M6);
   - task-scope `ON_star_zero` (until M5).

   Each is reported with its line number.

**Translation validation** (`translation_mismatch`), for every translated
program:
- Restore the trial's `initial_state.npz` (format v2 snapshot) and execute the
  `Program` with `cfg.collection.record_execution(initial_snapshot=...)`. This
  needs no settling, because the saved state is already settled.
- Compare with the CaP rollout of the successful round,
  `execution/rollout_001/trajectory.npz`:
  - the primitive sequence;
  - the resolved physical operands (from the CaP `primitive_start` events);
  - every action (`d0_actions`), with equal length and within `1e-9`;
  - the final full state.

Both runs use the same controllers from the same snapshot, so they should be
bit-identical. Any difference is a translator bug, not a property of the
program.

## 5. Traces for invariant inference

Verify mode learns invariants from executions of the exact candidate on the
archive's initial states, and the archive fingerprint must match the program
([VERIFICATION.md, workflow](../../cfg/VERIFICATION.md#verification-workflow)).
Per job:

1. **Scenes.** The scenes are:
   - the trial's own scene;
   - K − 1 other evaluation scenes of the same task from `final_v2` (all of them
     satisfy the precondition);
   - optionally, fresh reset seeds at other block counts (for example 2, 3 and
     5 blocks for Stack), since the instructions say counts are unspecified.

   The supplied Stack proof used five four-block demonstrations (K = 5).
2. **Recording.** Record the translated program from each scene's saved
   snapshot into a current full-state archive, with `metadata.validation` and
   the settled check (`post_settling_steps: 50`). This uses a new collector
   entry that accepts snapshots; `collect_demos` takes seeds only.
3. **Failed scenes.** A scene where the program fails is not added to the
   archive. It is recorded as `refuted_by_execution` evidence for that program,
   because the program then has a concrete precondition-satisfying failure.
4. **Verification.** Run `synthesize_cfg --mode verify --program <job>/program.py:build_program --demos <job>/archive/demonstrations.npz`
   with the flags of the supplied Stack proof:
   - `--invariant-relations ON_star Higher Scattered equality`
   - `--invariant-minimizer pyeda`
   - `--supported-towers`, `--table-surface-height 0.4`, `--initial-arm` (the task's settled reset pose)
   - `--higher-tolerance 0.001`
   - `--motion-iterations 0`, `--verification-timeout-ms 60000`

   Unstack invocations keep the 60-second wall-clock alarm.

**M0, before any translation:** replay every distinct source unchanged
through the CaP runtime on all 50 scenes of its task, and optionally on other
block counts. This uses `ScriptedModel` and the scripted-response batch mode
(`create_batch(responses=[source])`), with recovery and demonstrations off, so
there are no model calls and no cost. It gives each program's empirical success
rate across scenes. It is the cheapest strong signal and produces the first
`refuted_by_execution` results.

## 6. Verifier extensions

Each extension follows the same rules:
- it matches the runtime semantics (control.py and the CaP adapter), never
  only the CaP programs;
- it gets synthetic unit tests and a regression on the supplied Stack proof;
- it is recorded in [PAPER-DISCREPANCIES.md](../../../../PAPER-DISCREPANCIES.md)
  where it changes the paper's model.

| ID | Extension | Size | Notes |
| --- | --- | --- | --- |
| E1 | `Release(r, h)` with `r` ≠ held: the retreat goes to `(arm.x, arm.y, Z(r) + h)`, which is what `PrimitiveController.release` does | Small | The current model requires `r` = held and takes the retreat XY from `r`. Decision D2 |
| E2 | Table-relative operands: `tbl` resolves to the table frame's XY and to surface − L/2 for Z, as in `control.reference_position` | Small–medium | Needs a `table_position` premise next to `--table-surface-height` |
| E3 | Summary proposal with alternatives: try the Z reference of the last Move, then `tbl`, then the outgoing ON edge; accept only the one the motion stage proves | Small | Needed when a placement's Z reference is not its support (for example `Move(b0, last, b0, [0,-0.15,0.001])` places next to `b0`). Every alternative is still proved; none is assumed |
| E4 | Measured offsets: Move/Release offsets become linear terms over `arm` and named blocks' current positions, evaluated at runtime and encoded in the motion model | Medium | In the noiseless model a held block's centre equals the arm, so `gp − bp` is 0. With `--motion-noise` it is the grasp error that the program compensates for. A new DSL operand type, rejected by loaders unless enabled, like `ObservationReference` |
| E5 | Observations while holding: `Get`/`Assign` between `Pick` and `Release` | Medium | Needs an abstract mid-placement state: the held block is out of every ON* relation and Higher is undefined for it. Each Get needs its witness-existence obligation in that state |
| E6 | `If`/`else` (plus `break`, `continue` and early `return`, normalized to guards when they match a pattern, otherwise unsupported) | Large | Needs: an `If` instruction; a WP rule `(c ⇒ wp(S1)) ∧ (¬c ⇒ wp(S2))`; a CFG `BranchRegion`; candidate traces per branch; motion verification over both paths; invariant inference unchanged |
| E7 | Reverse and ReStack in the integrated CLI: task specs exist in `cfg/tasks.py`; several sequential loops; task-scope `ON_star_zero` | Medium–large | For CaP, `ON_star_zero` must read program-entry geometry ([DSL.md §5.3](DSL.md#53-higher-and-frozen-relations)) |
| E8 | Goal-aware verification: goal sort, HasBlock/AtGoal, MoveRight/MoveDown, nested loops, goal-relative Move/Release | Large | Open since discrepancy 40; RoboVerify's own Grid/Pyramid reference programs have no proof either |

## 7. Milestones

| Milestone | Work | Result |
| --- | --- | --- |
| M0 | Extract the 300 programs into a manifest (task, seed, round, source path, hash); dedupe to 246 jobs; scripted cross-scene replay on 50 scenes per task | Empirical success matrix; first `refuted_by_execution` |
| M1 | Translator and translation validation for all 300 | Every program is either translated with an action-identical replay, or `translation_unsupported` with its constructs |
| M2 | Archive recording and verify runs for translatable Stack programs (about 14) | First `verified_model` / `symbolic_not_proved` / `motion_refuted` results |
| M3 | E1, E2, E3; Unstack through the integrated CLI | About 25 programs in scope |
| M4 | E4 (a), E5 (b), E6 (c), each as its own commit and decision | Up to about 197 programs |
| M5 | E7 (Reverse/ReStack) | Up to 200 |
| M6 | E8 (Grid/Pyramid), only if the user wants it | Up to 300 |

**Code layout:**
- `synthesis/cap/verification/`:
  - `translate.py`
  - `equivalence.py`
  - `traces.py` (snapshot collector)
  - `outcomes.py` (outcome classification from verify-run artifacts)
- `synthesis/cap/experiments/revision8_verify.py`: driver, parallel per job, no model access.
- `synthesis/cap/reports/verification_report.py`
- Results: `docs/revision_8_report.md` with a per-task outcome table and figures.

**Artifacts:** `runs/cap-verification/<run>/<task>/<job_hash>/`:
- `source.py`, `trials.json` (the trials that share the source),
  `translation.json`, `program.py`;
- `equivalence.json`, `archive/`, `cross_scene.json`;
- `verify/`, a link to the `runs/cfg/...` verify run;
- `result.json`, with the outcome and the failed obligation or construct.

**Tests** (`synthesis/cap/tests/test_verification_translate.py`, plus the owning
packages for E1–E8):
- translation of every construct listed in Section 3.2, from synthetic sources;
- unsupported constructs are reported with their line numbers;
- the binder `tbl` guard;
- an action-identical replay of a small Stack program in MuJoCo;
- outcome classification from synthetic verify results;
- no extension makes a check that was previously refuted pass, unless the
  extension changes the modelled semantics to match the runtime.

## 8. Risks

- **Translation fidelity.** Semantic differences that action-identical replay
  on one scene would not catch:
  - quantifier domains;
  - first-witness order;
  - frozen-geometry scope;
  - the loop budget.

  These are covered by unit tests and by replay on all K scenes.
- **Inference data.** One scene per program is too few to learn useful
  invariants. Cross-scene archives fix this, but a program that fails some
  scenes yields fewer traces.
- **"Not proved" versus "wrong".** An invalid preservation VC after
  refinement can come from the learner's vocabulary, not the program.
  `symbolic_not_proved` stays separate from `refuted_*`, and a refutation needs
  a realized counterexample (the `counterexamples.py` / invariant-learning
  realization path) or a failing scene.
- **Summary proposal.** Placements that use an unusual Z reference can fail
  structurally even when they are correct. E3 addresses this without assuming
  any summary.
- **Model premises.** Supported towers, tight alignment and an exact `L`
  support height mean that CaP programs which place with a 1 mm margin (for
  example a `0.051` Z offset) will fail `release_support` in the exact model.
  This is a real model-level finding, not something to tune away; see D4.
- **Time.** About 90 s per Stack proof, so 246 jobs take hours on CPU without
  GPUs.

## 9. Open decisions

1. **D1. Scope of the first report.** Recommendation: M0–M3, reporting every
   program's outcome and leaving the large extensions (E4–E8) to later
   decisions. The alternative is to commit up front to all six tasks (M6).
2. **D2. Release retreat reference (E1).** Recommendation: model the
   controller's actual semantics, retreat XY at the arm and Z from the named
   reference, and log it as a discrepancy against the paper's `Release(b, h)`
   precondition that `b` is held. The alternative is to keep the current model
   and classify these programs as `motion_refuted`.
3. **D3. Measured offsets (E4).** Recommendation: implement them as linear
   observation operands and prove them in both the noiseless and noisy models.
   The alternative is to classify them as `translation_unsupported`. Rejected:
   substituting `gp − bp = 0` in the translator, because that changes the program.
4. **D4. Placement height premise.** Release support requires the exact height
   `L` above the support. Offsets such as `0.051` violate it. Recommendation:
   keep it exact and report these as `motion_refuted` (premise).
   The alternative is a separately named, explicit support-tolerance premise,
   which would need its own soundness argument.
5. **D5. Traces per job.** Recommendation: K = 5 four-block scenes (the trial's
   own scene plus four other `final_v2` scenes), plus one 3-block scene. The
   alternative is all 50 scenes, which costs more.
6. **D6. Deduplication key.** Recommendation: the alpha-renamed AST hash
   (246 jobs). The alternative is the exact AST hash (272 jobs).
7. **D7. Grid/Pyramid (E8).** Recommendation: defer, because it is the open
   goal-verification item from discrepancy 40 and also applies to RoboVerify's
   own programs. Report these 100 programs as `verifier_unsupported_task` after
   M0's replay results.
