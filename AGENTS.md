# Agent onboarding

Entry point for any coding agent working in this repository — Codex, Cursor, Claude Code,
or otherwise. Read this first. It is deliberately short; it points at the documents that
carry the detail.

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

## Work in progress — read before starting

`PLAN-popl-alignment.md` is the active plan: bringing this code into line with the POPL
submission (`POPL2027.pdf`), verification soundness first. **Its status header records
which stages are done, which is in progress, and which design questions are already
settled.** Start there, and update that header as you go — it is the only handoff channel
between sessions.

Two standing decisions from that plan, so they are not re-litigated:

- **The paper is an artifact under test, not a specification.** It was written by the same
  people as the code and may describe intended rather than implemented behaviour. Where
  the two disagree, neither automatically wins. Do not change code whose only
  justification is "the paper says so," and do not treat the paper's reported numbers as
  regression targets.
- **Discrepancies get logged, not silently fixed.** `PAPER-DISCREPANCIES.md` records places
  where the paper is wrong or underspecified, to be worked once the code is sound. Add to
  it when you find another.

The plan's conflicts table (items 1–7) is fully adjudicated. Treat those as decided.

## Reading experiment runs

Instrumented runs write `runs/<name>/<utc>-<sha>-<slug>/`. **Use the report tool; do not
open the files directly.**

```bash
uv run python -m synthesis.experiment.report --run runs/mcmc/latest
uv run python -m synthesis.experiment.report --glob 'runs/mcmc/*' --table
```

Output is capped so inspecting a run costs the same at iteration 10 or 10,000. Never `cat`
`metrics.jsonl` or `stdout.log`; if you must grep the log, bound it (`grep -m 20`).
Avoiding those two reads is the entire point of the run directory.

## Architecture

`CLAUDE.md` holds the full architecture notes — the DSL and program representation, the
two verification backends, invariant inference, the MCMC search, and the run-directory
contract. **Despite the filename its content is tool-neutral**, and it is the most
detailed description of the codebase. Read it before making non-trivial changes.

If your tool uses its own rules file (Cursor's `.cursor/rules/`, for instance), point it
at this file rather than duplicating the content here.

## Conventions

- All real work lives under `roboverify/`; the repository root holds notes, plans and
  experiment logs.
- Commit in meaningful increments, one coherent change per commit, rather than one large
  commit at the end.
- Work on a topic branch; do not commit directly to `main`.
- When staging, use explicit paths. The tree carries unrelated untracked files
  (`roboverify/demos/`, `plot.py`, `create_env_figure.py`), and `git add -A` sweeps them in.
