# Counterexample-guided verification (Phase E)

This guide covers the standalone Phase E APIs for refining an existing
single-loop program. The [integrated CFG workflow](../cfg/VERIFICATION.md) now
connects synthesis, verification, demonstration requests and structural repair;
its capabilities are broader than the standalone offset-repair API below.

`symbolic_verify.py` labels each VC as establishment,
preservation, exit, or straight-line body, and returns structured
valid/invalid/vacuous/unknown checks. A result is truthy only when all checks pass.
`Program.highlevel_verification` now returns this object while remaining compatible
with Boolean callers. Vacuous means the premise and domain axioms are inconsistent.
Unsat cores provide a one-sided shortcut; a core containing the negated conclusion
requires a separate premise satisfiability check. Solver timeouts remain failures.

`SymbolicVerify(build_problem, ...)` searches finite sizes in increasing order,
then requests an unbounded proof unless disabled. `build_problem(n)` returns
`(program, precondition, postcondition, context)`; `n=None` requests DeclareSort.
Finite success is labeled with its checked range. Unknown/vacuous smaller instances
stop the search rather than making a later counterexample appear smallest.

`run_symbolic_cegis` records `False`, learns from the supplied `DemoStore`, checks
strict enlargement, verifies, and adds preservation-failure successors. Its default
`MonotoneInvariantLearner` builds a universal Boolean formula over the observed
vocabulary rows. The finite set of allowed rows only grows. `InvInference` can be
passed instead, but nonmonotone updates are rejected. An exit failure requests a
stronger invariant; a vacuous/unknown query or unrealizable geometric model stops
with its own status. An establishment/body failure saves the model and raises
`NeedsResynthesis`, whose `.s0` is the concrete scene when realization succeeds.

Counterexample scene construction preserves ON*, frozen ON*, Higher, Scattered,
aliases, and table identity. Abstract tables not realizable by geometry are refused.
Symbolic replay supports straight-line `Put`, `Assign`, and `Skip`; it executes the
placement abstraction, not the physical controller. The successor may be the final
loop head where the guard is false, which is still required for exit verification.
This standalone driver supports one non-nested loop. The integrated workflow
propagates state across supported structured CFGs.

`run_motion_cegis` takes explicit `MotionBlockSpec`s covering every loop, accumulates
`PenStore` environments, and calls the supplied resynthesis callback. Entry
conditions must equal the program's invariant and guard. Full motion verification
runs after every repair. Constants, conditions, placement contracts, relational
summary, symbolic assignments, guards, and motion operand/release structure must
remain unchanged. This built-in repair searches offsets only. The integrated
CFG workflow supports instruction-structure repair while preserving the entire
symbolic program. Unknown, inconsistent, and unsupported checks
cannot become successes or training counterexamples.

`MotionPenalty` rechecks candidates in each saved environment, fixing current and
frozen positions and aliases while retaining the configured universal noise bounds.
Multiple failing obligations in one environment count once. The maximizing score
is `-MMD - weight * failed_environments` when no optional legacy goal reward is used.
The original and instrumented Runner/optimizer/MCMC APIs accept `motion_penalty` and
`motion_penalty_weight`; defaults preserve their old scores. Penalty-bearing CEM
runs serially because live Z3 objects cannot be sent through the multiprocessing
pickle queue. `optimize_motion_parameters` also supports a supplied demonstration
score; without one it is explicitly penalty-only repair. A zero training penalty
is never substituted for the full verification query.

Run the initial Stack integration from `roboverify/`, with the simulator environment
variables from `AGENTS.md` set:

```bash
uv run python -m synthesis.entry.verified_synthesis \
  --demo-store /tmp/phase-c-stack-traces.json \
  --run-root runs/phase-e --max-blocks 4 --symbolic-iterations 20
uv run python -m synthesis.experiment.report --run runs/phase-e/cegis/latest
```

For motion repair, provide `--expert-states observations.npy` (a two-dimensional
NumPy array in the normal environment observation layout) and matching
`--num-blocks`/`--seeds`, or explicitly request diagnostic `--penalty-only` repair.
`--motion-noise GRASP MOVE RELEASE` opts into bounded errors. `--bounded-only`
limits the symbolic claim to sizes 2 through `--max-blocks`; it is not an unbounded
proof. `--learner legacy` selects Phase C inference with progress checks. The
current runnable task is Stack; the generic library interfaces support other
single-loop programs. No Unstack run may exceed the standing 60-second cap.

RunLogger stores each invariant's full S-expression under `artifacts/invariants/`,
models and scenes under `artifacts/counterexamples/`, and accumulated motion
examples in `artifacts/penalties.json`. Metrics contain bounded fields and artifact
paths, rather than repeating potentially large formulas. The report shows the
latest eight refinement steps. Exit code 0 means both stages passed in the reported
scope; 2 means a recorded verification/refinement failure. Budget exhaustion never
means verification succeeded.

The real seed-0 Phase C Stack data reaches `NeedsResynthesis` after its first
learned invariant. No verified Stack program is claimed. A separate regression
fixture starts with False, strictly enlarges twice, and discharges all VCs including
an unbounded check. A motion fixture generates genuine solver counterexamples,
repairs its placement height, and passes the full motion recheck.
