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

Run the shared supplied-program workflow from `roboverify/`, with the simulator
environment variables from `AGENTS.md` set. Generated demo archives are not
bundled; use the actual output path printed by the collection command:

```bash
uv run python -m synthesis.entry.collect_demos \
  --program synthesis.examples.stack:build_program --save-video
uv run python -m synthesis.entry.synthesize_cfg --mode verify \
  --program synthesis.examples.stack:build_program \
  --demos demos/stack/3-blocks-5-trajectories/demonstrations.npz \
  --run-name stack-verification
uv run python -m synthesis.experiment.report --run runs/cfg/latest
```

`--mode full` searches first; verify mode starts from the supplied program and
enters the same inference and verification stages. Both rerun the actual physical
candidate from recorded initial states to obtain loop heads and normal exits.
New Stack collections save their state after 50 settling steps; candidate
execution restores it directly without repeating that preparation. Motion repairs
trigger fresh execution, inference and both verification checks.
The supplied program's executable fingerprint must match the archive. Expert
recordings remain separate for imitation scoring and later resynthesis.
For this integrated CLI, `--demos` replaces the old `--demo-store` and
`--expert-states` inputs, while `--output-dir` and `--run-name` replace its old
`--run-root` and `--slug` options. The separate
`synthesis.experiment.mcmc.run` CLI still uses `--run-root` and `--slug`. The old
`synthesis.entry.verified_synthesis` command now forwards to this verify mode.

`--motion-noise GRASP MOVE RELEASE` opts into bounded errors. `--learner legacy`
is the default; `monotone` selects the Boolean-row learner. The generic standalone
library APIs above remain available. No Unstack run may exceed the standing
60-second cap.

RunLogger stores candidate programs, actual execution traces and bootstrap
invariants under `artifacts/candidates/<revision>/`, counterexamples and later
invariant progression as referenced artifacts, and unresolved demonstration
requests in `artifacts/resynthesis_request.json`. Use the report tool to inspect
bounded summaries. Exit code 0 means both verification stages passed with result
`verified_model`; 2 means an explicit unsuccessful result. Budget exhaustion never
means verification succeeded.

End-to-end Stack learning acceptance remains open. Generated regression fixtures
exercise both verification backends and the repair/resynthesis paths; collection
and replay checks alone do not establish a verified program. See
[the integrated workflow](../cfg/VERIFICATION.md) for scope and
[the collection guide](../inference_lib/README.md) for seeds and video.
