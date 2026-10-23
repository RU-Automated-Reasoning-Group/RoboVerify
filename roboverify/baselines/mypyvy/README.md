# mypyvy UPDR baseline

This family asks upstream mypyvy to **infer** invariants from transition systems,
starting with safety assertions and no supplied supporting invariants. Each
successful inference is exported and checked by `mypyvy verify` in a fresh
process. Input consistency is checked first. The models are unbounded in the
number of blocks; the finite regression traces are tests, not a proof bound.

## Installation

The tested configuration is Python 3.12.3, Z3 4.16.0.0 and cvc5 1.3.1
(`git ea1b484`). Source is pinned to
[`wilcoxjay/mypyvy` cd80557ae11f5958c7c81dbecf3b24f7a3b16c1f](https://github.com/wilcoxjay/mypyvy/tree/cd80557ae11f5958c7c81dbecf3b24f7a3b16c1f).
The upstream source and binary are external dependencies, not vendored copies.
From the repository root, the following installs them outside the checkout:

```bash
git clone https://github.com/wilcoxjay/mypyvy /tmp/roboverify-mypyvy
git -C /tmp/roboverify-mypyvy checkout cd80557ae11f5958c7c81dbecf3b24f7a3b16c1f
uv venv --python 3.12 /tmp/roboverify-mypyvy-venv
uv pip install --python /tmp/roboverify-mypyvy-venv/bin/python \
  -r roboverify/baselines/mypyvy/requirements.txt
curl -L --fail -o /tmp/roboverify-cvc5-1.3.1.zip \
  https://github.com/cvc5/cvc5/releases/download/cvc5-1.3.1/cvc5-Linux-x86_64-static.zip
echo '1a1cda20d2df4938fa4944a69f33ddc9172e319ece0eed0aa09c4d7abede3ed1  /tmp/roboverify-cvc5-1.3.1.zip' | sha256sum -c -
unzip /tmp/roboverify-cvc5-1.3.1.zip -d /tmp/roboverify-cvc5-1.3.1
chmod +x /tmp/roboverify-cvc5-1.3.1/cvc5-Linux-x86_64-static/bin/cvc5
```

The archive is the official [cvc5 1.3.1 release](https://github.com/cvc5/cvc5/releases/tag/cvc5-1.3.1)
for Linux x86-64. Other platforms require their corresponding binary and a new
recorded result. mypyvy vendors its parser. No simulator dependencies are needed.

## Reproduce and monitor

From `roboverify/`:

```bash
export RV_BASELINE_PYTHON=/tmp/roboverify-mypyvy-venv/bin/python
export ROBOVERIFY_MYPYVY_ROOT=/tmp/roboverify-mypyvy
export ROBOVERIFY_CVC5=/tmp/roboverify-cvc5-1.3.1/cvc5-Linux-x86_64-static/bin/cvc5

timeout --signal=TERM --kill-after=2s 55s "$RV_BASELINE_PYTHON" \
  -m baselines.mypyvy.run --task unstack \
  --mypyvy-root "$ROBOVERIFY_MYPYVY_ROOT" --cvc5 "$ROBOVERIFY_CVC5"

"$RV_BASELINE_PYTHON" -m baselines.mypyvy.run --task reverse \
  --mypyvy-root "$ROBOVERIFY_MYPYVY_ROOT" --cvc5 "$ROBOVERIFY_CVC5" \
  --timeout-seconds 240

"$RV_BASELINE_PYTHON" -m baselines.mypyvy.run --task stack \
  --mypyvy-root "$ROBOVERIFY_MYPYVY_ROOT" --cvc5 "$ROBOVERIFY_CVC5" \
  --timeout-seconds 90
```

Use `--solver z3` to compare Z3 with the same model, `--mode typecheck` for parsing
and typing only, and `--output-dir runs/baselines/mypyvy/NAME` for a named new
directory. The default total budget is 45 seconds and each solver query has a
5,000 ms limit. Unstack rejects budgets over 50 seconds and nonfinite budgets;
keep its external 55-second cap. A timeout or `unknown` is never verification.

The runner prints a bounded terminal JSON summary. During a run:

```bash
"$RV_BASELINE_PYTHON" -m baselines.report --run runs/baselines/mypyvy/NAME
```

`status.json` is atomically refreshed at least every two seconds during a worker
phase, with phase, elapsed time, PID and UTC heartbeat. `result.json` retains each
phase's outcome, exit code, timeout flag and duration. Phases are `typecheck`,
`trace` (initial SAT plus two negative carrier probes), `updr`, and `verify`.
`verified` requires every phase. `typechecked` does not mean proved. Other
terminal outcomes include `unknown`, `timeout`, `abstract_counterexample`,
`error` and `inconclusive`, all unsuccessful for inference mode.

Each directory also contains exact input and inferred `.pyv` files, source and
adapter hashes, solver version/binary hash, commands, per-phase logs, and the
first actual SAT/model and UNSAT solver exchanges. The parent kills a worker's
whole process group on deadline. It does not infer success from exit code alone.

## Model semantics and scope

`models/stack.pyv` is the supplied Stack reference/control. Unstack and Reverse
translate the current provided groundtruths with ON*, Higher and Scattered.
Physical blocks inhabit an uninterpreted sort together with the distinct `tbl`
marker. Table isolation is explicit: table/block ON* and Higher are false,
table/table is reflexive, and Scattered with table is false. Table placement is a
fresh separated column. Higher updates assume supported, equal-height blocks;
the next occupied level above the destination represents one block height.
These abstractions contain no coordinates, workspace bounds or collisions.

Both tasks initially require every physical block to be above `b0`, which gives
one total source tower under the domain axioms. There is **no minimum of two
physical blocks**, finite cardinality bound, supplied loop invariant, or assumed
reversal conclusion. The initial source tower and reliable abstract Put effect
are premises of this model, not results about arbitrary physical scenes.

- Unstack copies the executable quantified guard exactly. It cannot be replaced
  by `p != b0` without additional assumptions. With one physical block it selects
  that block repeatedly, so this is partial safety, not a termination proof.
  For larger finite towers the last base stays in place. Its exit property is
  that ON* contains only identity pairs.
- Reverse begins with `b = tbl`; the same loop handles the first block and every
  later block, including the original base last. Immutable `on_star_zero`
  freezes initial ON* once in `init`. At exit, current ON* must equal its
  transpose over physical blocks. It is never globally linked to current ON*.
  A step detaches the source before attaching it to `b`. **This differs from
  the existing attachment-only production block-Put rewrite:** it models the
  groundtruth's intended relocation, and does not certify that dispatcher.
- `put_enabled` is a separate safety assertion. Guarded transitions additionally
  require clear/suitable supports, but a bad guard choice cannot disappear
  merely because the transition precondition excludes it. The model treats Put
  atomically and does not prove primitive grasping or motion contracts.

The 14 supplied domain axioms (including top existence) are carried by
`derived relation domain_axioms(): domain_axioms & ...`; Unstack/Reverse add
table isolation. In the pinned source, the derived expression is asserted at
each state, **not** interpreted as `D <-> (D & axioms)`. Positive initial-SAT and
negative `!on_star(b0,b0)` / `!domain_axioms` probes check this behavior.
These remain endpoint assumptions. UPDR safety under these assumptions does not
by itself establish that arbitrary abstract updates preserve the domain or that
all abstract pre-states admit a successor. Complete concrete 1–4 block traces
exercise every update and endpoint against an independent height/column oracle;
they rule out accidental deadlocks in those cases, not an unbounded closure
theorem. The use of finite-model finding for SAT countermodels does not place a
cardinality limit on the UNSAT proof obligations.

## Compatibility adapter

The pinned upstream implementation exposes its SMT-LIB backend as `--cvc4`.
`worker.py` launches the actual cvc5 binary, using `--finite-model-find`, and
adapts cvc5's outer model delimiter, sort-cardinality comments and typed model
values to the existing parser. Model values receive a reserved collision-checked
prefix because UPDR later uses their names in quantified formulas. Unsupported
quoted tokens, malformed models, crashes and unknown answers fail closed.
Raw cvc5 responses are retained alongside the normalized presentation.

For assumption queries the adapter temporarily asserts the assumption
conjunction and restores the solver stack afterward; cores are disabled for this
backend. It expands safety definitions with upstream's own capture-avoiding
macro expander because upstream's concrete state evaluator cannot evaluate
definition calls. This applies equally to Z3 and cvc5; the UPDR algorithm and SMT
formulas are otherwise unchanged.

Some automatically inferred facts mention only immutable entry geometry.
Upstream's file checker rejects invariant declarations without mutable symbols,
so certificate serialization conjoins `(done | !done)` to each inferred clause.
This is logically equivalent and keeps those facts as initialization and
preservation obligations. They are never promoted to axioms. A negative
regression deliberately exports a false frozen fact and requires verification
to reject it.

## Tests

```bash
ROBOVERIFY_BASELINE_PYTHON="$RV_BASELINE_PYTHON" "$RV_BASELINE_PYTHON" \
  -m unittest baselines.mypyvy.test_adapter baselines.mypyvy.test_models -v
```

Integration tests require the two dependency environment variables above and
otherwise skip explicitly. They check complete executions on 1–4 physical
blocks, all ON*/Higher/Scattered values after each transfer, frozen reversal,
illegal choices, premature finish, stale old supports, wrong final order,
singleton behavior, carrier consistency, certificate obligations and reserved
model names. Adapter tests cover nested multiline models, multiple sorts,
assumption cleanup on exceptions and unsuccessful/mixed output classification.
These tests live outside `synthesis/`. The repository's full-suite discovery
includes `baselines/`; set the dependency variables to enable its integration
tests. The command above runs this family alone.

## Current measured results

2026-10-23, seed 0, the pinned dependencies above, no supplied supporting
invariants. Timings are one wall-clock observation, not a performance study.
The small result manifest records exact timings and model hashes; full artifacts
remain under ignored `runs/baselines/mypyvy/`.

| Model / backend | Result | Total | Evidence directory |
| --- | --- | --- | --- |
| Stack / cvc5 | inferred and independently verified | 5.79 s | `stack-cvc5-checked` |
| Unstack / cvc5 | inferred and independently verified | 21.18 s | `unstack-cvc5-checked` |
| Reverse / cvc5 | inferred and independently verified | 163.18 s | `reverse-cvc5-checked` |
| Unstack / Z3 | unknown during inference; no proof | 22.83 s | `unstack-z3-checked` |
| Stack / Z3 | unknown during inference; no proof | 16.41 s | `stack-z3-checked` |
| Reverse / Z3 | unknown during inference; no proof | 18.74 s | `reverse-z3-checked` |

The successful Reverse run learned 15 clauses. An earlier export failed the
immutable-invariant syntax check and was recorded as `error`; the result above
comes from a fresh inference and separate successful check after the equivalent
serialization fix. Neither that failed run nor a finite test is counted as a
proof. These results establish abstract partial safety under the stated model
assumptions, separate from RoboVerify's symbolic and motion verification.
All 14 focused tests passed, including the real-solver and process-group deadline
checks.
