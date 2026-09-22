# Agent guide and architecture

`AGENTS.md` is the single, tool-neutral guide for coding agents in this repository.
Read the environment, project-status and convention sections before making changes;
use the architecture and workflow sections for the affected subsystem.
[README.md](README.md) indexes the remaining project documentation.

- [Environment](#environment)
- [Project status and decisions](#project-status-and-decisions--read-before-starting)
- [Conventions](#conventions)
- [Project overview](#what-this-project-is)
- [Commands and workflows](#commands-and-workflows)
- [Architecture](#architecture)
- [Experiment monitoring](#monitoring-runs)

## Environment

Both lines are required before anything touches the simulator:

```bash
cd roboverify
unset LD_PRELOAD
export LD_LIBRARY_PATH="$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia"
```

Without `LD_LIBRARY_PATH`, `import mujoco_py` raises `Missing path to your environment
variable` and every simulator-backed test fails at import.

Run everything as a module from `roboverify/` — the `synthesis` package uses relative
imports and files under `synthesis/entry/` are not runnable as bare scripts:

```bash
uv run python -m synthesis.entry.collect_stack_loop_traces --output /tmp/stack-loop-traces.json
uv run python -m synthesis.entry.verify_stack_with_learned_invariant --demo-store /tmp/stack-loop-traces.json
uv run python -m unittest synthesis.verification_lib.test_bmc_lib -v
uv run python -m unittest synthesis.experiment.test_run_logger -v     # fast, no simulator
uv run python -m unittest synthesis.experiment.test_mcmc_parity -v    # drives MuJoCo
bash format.sh                                                        # isort then black
```

Tests are `unittest`, not pytest. No linter is configured.

## Project status and decisions — read before starting

Read [README.md](README.md#project-status) for current project status and
[PAPER-DISCREPANCIES.md](PAPER-DISCREPANCIES.md) for numbered findings, settled
reasoning and remaining actions. Implementation is complete within the supported
scope; validated demonstrations and end-to-end learning acceptance remain open.
Update the relevant status or entry when it changes, rather than maintaining a
separate implementation-plan history.

- **The paper is an artifact under test, not a specification.** Neither paper nor
  code automatically wins a disagreement. Do not change code solely because the
  paper says so, and do not use its experimental numbers as regression targets.
- **Discrepancies get logged, not silently fixed.** Preserve entry IDs and the
  distinction between implementation defects, paper corrections and model limits.
- Keep Unstack end-to-end invocations under the user's **60-second wall-clock
  limit**. A timeout or exhausted budget is not successful verification.

### Settled implementation decisions

These seven original decisions are settled, not a new task list. Related proofs
and paper corrections are recorded in the numbered review entries.

| # | Decision |
| --- | --- |
| 1 | Higher WP rules introduce quantifiers; do not force the paper's original AFR-closure claim by changing valid code. The corrected rules and theorem issue are entries 16 and 1. |
| 2 | Use KL where convergence/epsilon thresholds depend on its scale; retain MMD as an option, cache demo density and bound sampling costs. MMD and KL thresholds are not interchangeable. |
| 3 | Straight-line search optimizes imitation distance, then ranks the near-best pool by PostScore. Retain legacy weighted-goal scoring for comparison. The default pool limit is 10; rank at convergence. |
| 4 | Keep premise consistency checks and distinguish valid, invalid, vacuous and unknown. An unsat core is a shortcut only when it establishes inconsistent premises. |
| 5 | Top is removed from the predicate vocabulary. |
| 6 | ON_star_zero is frozen entry geometry. Tasks using it equate it with current ON* in the precondition, never through a global link axiom (entry 5). |
| 7 | Optimize all three Move coordinates. Reassess CEM budgets when dimensionality changes; a single smoke run does not justify new defaults. |

## Conventions

- All real work lives under `roboverify/`; the repository root holds project
  guidance, the paper and review decisions.
- Commit in meaningful increments, one coherent change per commit, rather than one large
  commit at the end.
- Work on a topic branch; do not commit directly to `main`.
- When staging, use explicit paths. The tree carries unrelated untracked files
  (`roboverify/demos/`, `plot.py`, `create_env_figure.py`), and `git add -A` sweeps them in.
- If a tool needs its own rules configuration, point it to `AGENTS.md` rather than
  duplicating these instructions in a separate agent-specific Markdown file.

## What this project is

RoboVerify synthesizes and *formally verifies* robot manipulation programs (block
stacking/unstacking/reversing on a Fetch pick-and-place rig). A program is a small DSL
(`Pick`/`Move`/`Release`/`PickPlace`, `While`, `Put`/`Assign` for the logical/verification
view) that can be: executed in a MuJoCo simulator, mutated/optimized via MCMC + CEM against
expert demonstrations, and verified two ways — high-level (quantified Z3 reasoning over an
abstract `ON`/`ON_star`/`Higher`/`Scattered` block algebra, with loop invariants *learned*
from example traces) and low-level (bounded model checking / geometric reasoning over actual
box coordinates).

Implementation lives under `roboverify/`; the repository root holds onboarding,
project status, the paper and its review/decision record.

## Commands and workflows

Use the environment and module commands above; the full-suite command is below.

- [Trace workflow](roboverify/synthesis/inference_lib/README.md): collection and inference.
- [Motion API](roboverify/synthesis/verification_lib/README.md): geometric checks and noise.
- [Standalone CEGIS](roboverify/synthesis/verification_lib/CEGIS.md): existing-program refinement.
- [CFG workflow](roboverify/synthesis/cfg/VERIFICATION.md): integrated synthesis and verification.

Instrumented MCMC entry point: `uv run python -m synthesis.experiment.mcmc.run`.
Use `--smoke --demo-dir demos` for a bounded search, or task/iteration options for
longer runs. Read the resulting directory through the report tool described below.

### Validation

From `roboverify/`, with the simulator environment above configured:

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
`git diff --check`. Documentation-only edits need reference checks, not simulator
runs. Tests use synthetic scenes or generated traces, not saved demonstration files.

## Architecture

Read the relevant package notes before non-trivial changes. This section describes
the DSL, verification backends, inference, search and integrated CFG pipeline.

- **`synthesis/api/`** — the program representation.
  - `instructions.py`: `Instruction` subclasses. Physical instructions (`Pick`, `Move`,
    `Release`, `PickPlace`, and their `...ByName` variants that resolve symbolic box names via
    `env.symbolic_name_to_box_id`) implement `eval()` to drive a MuJoCo env, and
    `register_trainable_parameter`/`update_trainable_parameter` to expose float offsets
    (`Parameter`) as a flat vector for MCMC/CEM optimization. Verification-only instructions
    (`Put`, `GoalAssign`, `MarkGoal`, `MoveRight`, `MoveDown`) raise on `eval()` and
    must be lowered to physical instructions before execution. `Assign` updates
    runtime aliases, and `Get` finds an object satisfying its binding condition.
    `While` evaluates a restricted subset of Z3 formulas against concrete block
    positions to find/bind existential guard variables each loop iteration.
  - `program.py`: `Program` (holds a list of instructions plus trainable parameters), the
    weakest-precondition machinery (`wp`, `VC_aux`) and the `rewrite_for_put_for_*`/
    `rewrite_for_put_on_tbl_for_*` family that specializes VC generation for each predicate
    (`ON_star`, `Higher`, `Scattered`) across a `Put`. `Program.highlevel_verification(...)`
    and `Program.lowlevel_verification(...)` are the two verification entry points a program
    exposes.

- **`synthesis/verification_lib/`** — the two verification backends.
  - `highlevel_verification_lib.py`: `HighLevelContext` sets up the Z3 sort for boxes in one
    of two modes — `"declare"` (an uninterpreted `DeclareSort`, used for generic
    inference and unbounded verification) or `"enum"` (a finite `EnumSort` with a
    concrete `num_blocks`,
    used to *check* a learned invariant is sound for a specific finite instance, optionally
    rendering a scene). Defines the `ON_star`/`ON_star_zero`/`Higher`/`Scattered` predicates as Z3
    functions.
  - `bmc_lib.py`: bounded model checking. Encodes a fixed-length sequence of `Pick`/`Move`/
    `Release` instructions as Z3 constraints over per-timestep box positions
    (`BMCTraceSymbols`, `encode_step`), then `bmc_feasible`/`bmc_solve`/`bmc_verify` check
    reachability, solve for unknown offsets, or verify a fully-instantiated low-level program
    against a goal.
  - `symbolic_verify.py`: labeled VC results, exact vacuity detection, increasing-size
    finite counterexample search and optional unbounded proof. `counterexamples.py`
    realizes relation tables as geometry or explicitly refuses the model.
  - `cegis.py`: bounded symbolic/motion refinement, monotone finite-vocabulary
    learning, `NeedsResynthesis` for entry failures, and counterexample penalties.
    See [Phase E workflow](roboverify/synthesis/verification_lib/CEGIS.md).
  - `motion_verification.py`: explicit placement contracts, frame preservation, and
    swept-cube checks for lowered loop bodies. Results retain proof mode, failed
    obligations, counterexamples, and timings. `NoiseSpec` is opt-in, off by default;
    all tower verification CLIs accept `--motion-noise GRASP MOVE RELEASE`.
    Missing physical programs (Reverse/Partial) and uncovered instructions fail closed.
    See [motion verification semantics](roboverify/synthesis/verification_lib/README.md).
  - `lowlevel_verification_lib.py`: geometric low-level context, box-corner/cube drawing
    helpers used to visualize/verify concrete 3D placements.

- **`synthesis/inference_lib/demo_store.py`** — loop-head demonstration storage and
  adaptation to invariant inference. `Program.eval(..., on_loop_head=store.add)`
  records one `LoopHeadState` after each successful guard binding, before the body.
  All physical blocks are copied, including ones with no symbolic alias; every row
  keeps its invocation's entry geometry for `ON_star_zero`. Loop IDs are instruction
  paths (`"1"` for a loop following an initial assignment). `DemoStore.save/load`
  round-trips JSON without Z3 objects and preserves the relational table marker.
  `InvInference(store, loop_id, vocab, context)` reuses the existing learner.
  See [the trace workflow](roboverify/synthesis/inference_lib/README.md).

- **`synthesis/inference_lib/inference.py`** — invariant learning. Given positive traces
  (expert demos) and negative traces (random/failed programs), builds a boolean-formula
  vocabulary over the block predicates, partitions per-timestep states into truth-table rows,
  and extracts a minimal quantified invariant (`loop_inference`, `forall_exists_loop_inference`,
  `learn_from_partition`) that is later instantiated into an `EnumSort` context for finite
  checking (`instantiate_invariant`/`serialize_invariant` round-trip an invariant between the
  inference context and a verification context).

- **`synthesis/mcmc/`** — program synthesis by search, not primarily verification.
  - `synthesis.py`: the orchestration hub — collects/replays expert trajectories against a
    MuJoCo env, mutates programs (`mutate_program`), scores candidates by trajectory
    divergence from demos (via `cost_func`) combined with BMC feasibility checks
    (`check_bmc_candidate`/`score_candidate_program`), and drives the outer `MCMC(...)` search
    loop. Also has the `make_roboverify_*_env` factories (stack/unstack/reverse/partial/grid/
    pyramid) and video/frame saving utilities.
  - `decision_tree.py`: a compatibility alias only. `ON_feature` now resolves to
    `predicates.atoms.GroundON`; the shallow-tree feature learner it used to hold was
    superseded by the bounded predicate enumerator in `synthesis/predicates/`, which can
    express quantified separators rather than a single ground `ON(b1, b2)`.
  - `search_core.py`: the acceptance rule, annealing schedule, imitation objective and
    epsilon candidate pool shared by the original and instrumented searches — keep the
    Metropolis ratio here rather than writing it out a second time.
  - `distance.py`: cached KL/MMD trajectory distances for the candidate pool.
  - `cem.py` / `cost_func.py`: cross-entropy-method parameter optimizer, and KL/MMD-based
    trajectory-distribution distance metrics used as the optimization objective.

- **`synthesis/util/symbols.py`** — shared symbol allocation and quantifier hygiene.
  Use `fresh_const(sort, prefix, avoid=(...))` for auxiliary solver variables;
  include the surrounding formulas and operands in `avoid` when adding a binder.
  Use `rewrite_quantifier` to transform an existing quantified body, or
  `open_quantifier` when moving binders: these alpha-rename and substitute de
  Bruijn indices without capturing free names or merging nested shadowed binders.
  Use `fresh_name(preferred, occupied)` for generated program-level names; it
  reserves each result in the supplied set. Keep `get_consts`/named constructors
  for program identities, fixed vocabulary, and intentionally shared BMC state
  symbols. Never rebuild an auxiliary by guessing or reusing its printed name.

- **`synthesis/util/on.py`** — the ground-truth geometric implementations of the block algebra
  (`on_star_implementation`, `higher_implementation`, `scattered_implementation`)
  operating on raw `obs` arrays; both the `While` runtime interpreter and
  the MCMC reward/feature code call into these rather than duplicating geometry logic.

- **`synthesis/entry/`** — runnable pipelines.
  `verify_{stack,unstack,reverse,partial}_with_learned_invariant.py` require explicit
  `--demo-store` input, learn an invariant and check a high-level Put/Assign/While
  program, optionally in a finite `"enum"` context. Stack/Unstack also check a
  lowered physical body; Reverse/Partial report unsupported motion because they
  lack one. The separate 2D entry point is outside the supported tower-task scope.
  `synthesize_cfg.py` is the instrumented relational CFG synthesis CLI and writes a standard
  run directory; `main.py` is now only a shim that forwards to it, the scratch experiment
  script it used to hold having been replaced by that driver.

- **`synthesis/entry/verified_synthesis.py`** — instrumented Phase E Stack driver.
  Runs symbolic refinement before motion repair, retains counterexamples and
  invariant progression, and records failure/finite proof scope explicitly.

- **`synthesis/environment/`** — MuJoCo/Gymnasium environments (Fetch pick-and-place block
  construction, ant maze, etc.), largely vendored/adapted from CEE-US and
  `fetch_block_construction`. Observations pack agent state first, then per-block state in
  fixed-width slots; the `10 + 12*i : 10 + 12*i + 3` box-position slice convention recurs
  across `instructions.py`, `on.py`, and the `While` guard evaluator — keep it in sync if the
  observation layout ever changes. The four tower tasks expose
  `table_surface_height` separately (the world z of the simulator's table plane);
  `height_offset` is the resting block-center height, and relational `tbl` has no
  coordinates.

- **`synthesis/experiment/`** — run logging and reporting, plus an instrumented copy of
  the MCMC search. `run_logger.py` owns the run-directory contract; `report.py` is the
  bounded reader; `config.py` captures every knob into `config.json`. Under `mcmc/`,
  `search.py`, `cem.py` and `run.py` reimplement only the four functions that need to
  emit records (`MCMC`, `score_candidate_program`, `optimize_program`, `cem_optimize`)
  and import the remaining implementation from `synthesis.mcmc`. Both copies share
  acceptance and objective
  helpers, and `test_mcmc_parity.py` pins the same accept/reject sequence.

- **`synthesis/predicates/`** — the predicate language and its bounded search.
  `term.py` holds canonical interned first-order terms, `scene.py` the concrete
  observation semantics, `enumerate.py` the bottom-up prenex enumerator with explicit
  depth/binder/candidate/timeout bounds, and `classifier.py`/`guard.py` the two call
  shapes (`LearnClassifier`, loop-guard synthesis). Search reports `found`, `no_separator`
  or `budget_exhausted` — an approximate separator is never returned as an exact one.

- **`synthesis/cfg/`** — the relational CFG and the synthesis algorithms over it:
  `graph.py`/`region.py` (IR), `lower.py` (CFG to `Program`, emitting the scoped `Get`
  a refinement's existential prefix requires), `refine.py` (Algorithm 3),
  `straightline.py` (Algorithm 5), `synthesize.py` (the Algorithm 2 recursive driver),
  `quotient.py`/`kleene.py` (Algorithm 4, flat case only), plus demo recording,
  segment reset/replay and split validation. Numeric physical operands can be
  generalized to named operands during folding, but this does not establish a
  relational summary: symbolic lowering rejects such candidates. Learned guards
  accept demonstrated witnesses and reject all bindings at demonstrated exits;
  unselected continuing-state bindings are unlabeled. Runtime may choose the first
  matching witness; symbolic preservation verification covers every matching choice.
  Multiple witnesses are permitted, without a separate uniqueness requirement.
  Extracted iterations share their invocation's frozen entry geometry.

  Run `uv run python -m synthesis.entry.synthesize_cfg --task unstack --num-blocks 3
  --smoke --quotient` (on one line) for a bounded integration smoke. `--demos` accepts
  a lossless `.npz` recording; without it, the driver collects the historical oracle.
  `--reset-mode replay` is the default, and Unstack has a 60-second process alarm.
  The driver validates demonstrations and verifies the actual synthesized CFG
  through `cfg/verified_synthesis.py`, including demo requests/resynthesis and
  structural motion repair. Success is `verified_model` in the documented scope.
  The historical Unstack oracle fails its final task condition in the seed-0
  smoke; invalid demonstrations are rejected before synthesis.

- **`synthesis/topdown/`** — retired. `topdown.py` is a thin compatibility wrapper over
  the `synthesis/predicates/` enumerator; the hand-rolled BFS and its DSL are gone.

## Monitoring runs

Instrumented runs write `runs/<name>/<utc>-<sha>-<slug>/`, with `runs/<name>/latest`
symlinked to the newest one. The contract:

| file | pattern | notes |
|---|---|---|
| `config.json` | written once | resolved config, git sha + dirty flag, argv, library versions |
| `status.json` | **overwritten** each update | one small object forever; `alive` plus a stale `heartbeat` means the run is wedged, not finished |
| `metrics.jsonl` | one flat record per iteration | aggregate it; never read it line by line |
| `events.jsonl` | rare, rate-limited per kind | new bests, first feasible candidate, exceptions |
| `stdout.log` | fd-level capture | the firehose, including MuJoCo/OpenGL output; grep only |
| `result.json` | written once at exit | final verdict |
| `artifacts/` | as needed | programs, pickles, videos, tracebacks (referenced by path, never inlined) |

**Read a run with the report tool, not by opening the files:**

```bash
uv run python -m synthesis.experiment.report --run runs/mcmc/latest
uv run python -m synthesis.experiment.report --glob 'runs/mcmc/*' --table
```

Output is capped at `--max-lines` (default 60) so inspecting a run costs the same
whether it is at iteration 10 or 10,000, and the shape is stable so two reports diff
cleanly. **Never `cat` `metrics.jsonl` or `stdout.log`**; if you must grep the log,
bound it (`grep -m 20`). Avoiding those two reads is the entire point of the run
directory.

### Diagnosing a bad run

The metrics are chosen so each symptom points at a specific lever:

| symptom in the report | likely cause | lever |
|---|---|---|
| `bmc_feasible` near 0, `bmc_reason` dominated by one label | goal spec, operand pool, or program length | `--program-slots`, `--goal-feature`, `--num-blocks` |
| `accept` near 1.0 | acceptance rule is nearly unselective (a cost delta of 0.0065 gives ratio 0.993) | `--beta` |
| `at_floor` near 1.0, `first_feasible` unset | chain pinned at `bmc_failed_cost`, so acceptance is unconditional | seed program, BMC penalty shape |
| `cem` delta mean ~0 or many zero-delta iters | inner parameter optimization not improving anything | `--cem-N/-K/--cem-iterations`, `--cem-init-std` |
| `success` mean ~0 while `best_cost` improves | objective is not tracking task success | reward weights, objective design |

`--smoke` (20 iterations, 2 CEM iterations, no videos) is for testing one of these
hypotheses in minutes rather than hours.
