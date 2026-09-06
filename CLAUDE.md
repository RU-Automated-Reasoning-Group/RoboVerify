# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

RoboVerify synthesizes and *formally verifies* robot manipulation programs (block
stacking/unstacking/reversing on a Fetch pick-and-place rig). A program is a small DSL
(`Pick`/`Move`/`Release`/`PickPlace`, `While`, `Put`/`Assign` for the logical/verification
view) that can be: executed in a MuJoCo simulator, mutated/optimized via MCMC + CEM against
expert demonstrations, and verified two ways — high-level (quantified Z3 reasoning over an
abstract `ON`/`ON_star`/`Higher`/`Scattered` block algebra, with loop invariants *learned*
from example traces) and low-level (bounded model checking / geometric reasoning over actual
box coordinates).

All real work lives under `roboverify/`; the repo root only holds experiment logs/notes.

## Commands

All commands run from the `roboverify/` directory using `uv` (see `roboverify/pyproject.toml`
for the pinned dependency set — `torch`, `z3-solver`, `mujoco-py`, `gymnasium`, etc.).

```bash
cd roboverify
unset LD_PRELOAD   # avoids "Failed to initialize OpenGL Runtime" before running any sim code
```

Run a script as a module (required — the `synthesis` package uses relative imports and files
under `synthesis/entry/` are not meant to be run as bare scripts):

```bash
uv run python -m synthesis.entry.verify_stack_with_learned_invariant
uv run python -m synthesis.entry.verify_stack_with_learned_invariant --verification-mode finite --num-blocks 4
uv run python -m synthesis.entry.verify_unstack_with_learned_invariant
uv run python -m synthesis.entry.verify_reverse_with_learned_invariant
uv run python -m synthesis.entry.verify_partial_with_learned_invariant
uv run python -m synthesis.entry.verify_2d_with_learned_invariant
uv run python -m synthesis.entry.main   # big ad hoc experiment/demo-collection script
```

Run tests (unittest, not pytest):

```bash
uv run python -m unittest synthesis.verification_lib.test_bmc_lib -v
```

Format code (isort then black, over the package dirs — no linter is configured):

```bash
bash format.sh
```

## Architecture

- **`synthesis/api/`** — the program representation.
  - `instructions.py`: `Instruction` subclasses. Physical instructions (`Pick`, `Move`,
    `Release`, `PickPlace`, and their `...ByName` variants that resolve symbolic box names via
    `env.symbolic_name_to_box_id`) implement `eval()` to drive a MuJoCo env, and
    `register_trainable_parameter`/`update_trainable_parameter` to expose float offsets
    (`Parameter`) as a flat vector for MCMC/CEM optimization. Verification-only instructions
    (`Put`, `Assign`, `GoalAssign`, `MarkGoal`, `MoveRight`, `MoveDown`) raise on `eval()` —
    they exist purely for the high-level Z3 semantics (weakest-precondition rewriting) and
    must be lowered to physical instructions before execution. `While` also doubles as a
    runtime interpreter: it evaluates a restricted subset of Z3 formulas directly against
    concrete block positions to find/bind the existential guard variable each loop iteration.
  - `program.py`: `Program` (holds a list of instructions plus trainable parameters), the
    weakest-precondition machinery (`wp`, `VC_aux`) and the `rewrite_for_put_for_*`/
    `rewrite_for_put_on_tbl_for_*` family that specializes VC generation for each predicate
    (`ON_star`, `Higher`, `Scattered`) across a `Put`. `Program.highlevel_verification(...)`
    and `Program.lowlevel_verification(...)` are the two verification entry points a program
    exposes.

- **`synthesis/verification_lib/`** — the two verification backends.
  - `highlevel_verification_lib.py`: `HighLevelContext` sets up the Z3 sort for boxes in one
    of two modes — `"declare"` (an infinite `DeclareSort`, used for *inference*, i.e. learning
    invariants generically) or `"enum"` (a finite `EnumSort` with a concrete `num_blocks`,
    used to *check* a learned invariant is sound for a specific finite instance, optionally
    rendering a scene). Defines the `ON_star`/`Higher`/`Scattered`/`Top` predicates as Z3
    functions.
  - `bmc_lib.py`: bounded model checking. Encodes a fixed-length sequence of `Pick`/`Move`/
    `Release` instructions as Z3 constraints over per-timestep box positions
    (`BMCTraceSymbols`, `encode_step`), then `bmc_feasible`/`bmc_solve`/`bmc_verify` check
    reachability, solve for unknown offsets, or verify a fully-instantiated low-level program
    against a goal.
  - `lowlevel_verification_lib.py`: geometric low-level context, box-corner/cube drawing
    helpers used to visualize/verify concrete 3D placements.

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
  - `decision_tree.py`: learns a single discriminating `ON(b1, b2)`-style feature (via a
    shallow decision tree) that separates two subsets of demo trajectories — used to split a
    long demo into stages before running MCMC per-stage.
  - `cem.py` / `cost_func.py`: cross-entropy-method parameter optimizer, and KL/MMD-based
    trajectory-distribution distance metrics used as the optimization objective.

- **`synthesis/util/on.py`** — the ground-truth geometric implementations of the block algebra
  (`on_star_implementation`, `higher_implementation`, `scattered_implementation`,
  `top_implementation`) operating on raw `obs` arrays; both the `While` runtime interpreter and
  the MCMC reward/feature code call into these rather than duplicating geometry logic.

- **`synthesis/entry/`** — runnable pipelines. `verify_{stack,unstack,reverse,partial,2d}_with_learned_invariant.py`
  all follow the same shape: run inference in a `"declare"` context to learn an invariant,
  optionally instantiate it into a finite `"enum"` context, build both a high-level (`Put`/
  `Assign`/`While`) and a lowered physical (`PickPlaceByName`) version of the same program, then
  call `highlevel_verification` and `lowlevel_verification` and report both results. `main.py`
  is a large, mostly-scratch experiment script (trajectory collection, feature learning,
  MCMC) rather than a clean library entry point — read it for examples, don't extend it as if
  it were an API.

- **`synthesis/environment/`** — MuJoCo/Gymnasium environments (Fetch pick-and-place block
  construction, ant maze, etc.), largely vendored/adapted from CEE-US and
  `fetch_block_construction`. Observations pack agent state first, then per-block state in
  fixed-width slots; the `10 + 12*i : 10 + 12*i + 3` box-position slice convention recurs
  across `instructions.py`, `on.py`, and the `While` guard evaluator — keep it in sync if the
  observation layout ever changes.

- **`synthesis/topdown/`** — an alternate top-down program synthesis DSL (`dsl.py`,
  `topdown.py`), separate from the MCMC search path.
