# Learning invariants from counterexample executions

This standalone experiment implements the fixed-program workflow of paper
Section 6.2. It starts with **an empty dataset and invariant False**, generates
initial environments with an SMT query, executes the supplied DSL program in
MuJoCo, and learns from the resulting loop heads and normal exits. It uses the
intended partition learner (`InvInference`), without synthesis or program repair.

Stack is the first supported task. The program must contain one non-nested loop,
use `b0=0` as its only fixed entry alias, and preserve block geometry before the
loop. The existing Stack example meets these requirements. The executable
fingerprint and entry bindings must remain identical at every block count.

## Running the experiment

From `roboverify/`, configure the simulator environment as in
[AGENTS.md](../../../../AGENTS.md#environment). No demonstration collection is
needed before these commands.

Symbolic verification only (the default):

```bash
uv run python -m synthesis.entry.learn_invariant \
  --task stack --program synthesis.examples.stack:build_program \
  --verification-level symbolic --save-video \
  --run-name stack-counterexamples-symbolic
```

Both symbolic and motion verification, with the supported-tower premises used
by the supplied Stack proof:

```bash
uv run python -m synthesis.entry.learn_invariant \
  --task stack --program synthesis.examples.stack:build_program \
  --verification-level both \
  --supported-towers --table-surface-height 0.4 \
  --initial-arm 1.3446426 0.74911606 0.5314612 \
  --save-video --run-name stack-counterexamples-both
```

Motion verification runs only after symbolic verification succeeds. It uses the
learned invariant and unchanged physical program. These arm/table values are
explicit formal premises; full snapshot restoration is used for physical
executions. See the [motion model](../../verification_lib/README.md).
`--motion-noise GRASP MOVE RELEASE` opts into bounded errors; the default is
noiseless. A failed motion check produces diagnostics, without repairs or extra
training data.

| Option | Default / meaning |
| --- | --- |
| `--max-counterexample-blocks` | 4; search sizes 1 through this bound, starting again at 1 after each update. |
| `--max-rounds` | 10 accepted learner updates at most; up to 11 verification attempts including the final check. |
| `--max-loop-iterations` | Stack uses `num_blocks - 1`; override for a different bounded execution horizon. |
| `--verification-timeout-ms` | 10000 per symbolic/initial-domain/coverage solver query; does not bound Python formula construction or learning. |
| `--trajectory-timeout-seconds` | 60 for generated-scene preparation and execution together. |
| `--invariant-relations` | `ON_star Higher Scattered equality`. |
| `--invariant-variables` | 2 quantified learner variables. |
| `--higher-tolerance` | 0.001 metres, shared by simulation predicates, inference, and geometric checking. |
| `--seed` | 0; execution seeds increase by update number. Seeds do not replace solver-generated layouts. |
| `--motion-timeout-ms` | 10000 per motion query. |
| `--save-video` | Optional MP4 for each generated execution, at 20 FPS. Preparation is excluded. |
| `--output-dir` | `runs`; experiment results root. |
| `--run-name` | Optional readable run label. |

## What counts as a counterexample

The unbounded verifier checks establishment, preservation, and exit using the
current invariant. A preservation countermodel can be an unreachable intermediate
state; it is **not** automatically a valid simulator reset. The experiment saves
verification countermodels separately from generated initial-state witnesses.

After an establishment or preservation failure, the initial-state query asks:

```text
valid_initial_environment(s0)
and execution of the supplied symbolic program from s0
    reaches a loop head outside the current invariant within the configured bound
```

The query is a bounded preimage built with the existing placement WP rules.
Finite guards select the first matching physical ID, including deterministic
`Get` bindings. The final guard-false head counts as an invariant state. Thus a
one-block Stack environment contributes its normal exit despite executing zero
loop iterations. The unbounded preservation proof still covers every matching
witness, as in the existing verifier.

Block counts are tried in increasing order. Each size first checks initial-domain
consistency, then existence of a missing head. UNSAT at smaller sizes establishes
minimality **within the adapter's initial domain and execution bounds**. UNKNOWN
stops the search. The selected environment is a witness to missing invariant
coverage; it need not be the same relation table as the induction countermodel.
The query uses the current abstract placement semantics, so physical replay must
independently confirm that a missing state is actually encountered.

Stack's solver domain uses the existing reset workspace, gripper clearance,
Scattered separation, and equal tabletop heights. The adapter installs the
requested block poses into a full simulator state, settles for 50 steps, and
rechecks the task precondition and coverage query on the settled geometry. That
full snapshot is state zero and is restored without further settling. Failed
settling checks are retained as diagnostic archives.

An accepted execution must complete with converged primitives and satisfy the
actual task precondition and postcondition. At least one observed head/exit must
violate the previous invariant. All observed heads/exits then enter the cumulative
dataset, with each trajectory's own block universe and frozen entry geometry.
The next invariant must cover all accumulated states and satisfy the existing
strict-enlargement checks. No abstract successor or handwritten clause enters
this dataset.

An exit failure needs a stronger invariant; enlargement cannot remove its bad
state, so this experiment reports `needs_stronger_invariant`. A failed induction
proof with no reachable missing state in the searched bounds reports
`no_reachable_counterexample`. Neither case is silently repaired. The abstract
WP limitations and deferred Scattered mismatch remain documented in
[PAPER-DISCREPANCIES.md](../../../../PAPER-DISCREPANCIES.md).

## Results and artifacts

```bash
uv run python -m synthesis.experiment.report --run runs/invariant-learning/latest
```

Use a distinct `--run-name` for simultaneous experiments. Each run records:

- `artifacts/summary.md` and `progression.json`: checked invariants, failed
  obligations, generated sizes, state counts, phase timings, and outcomes.
- `artifacts/invariants/`: readable formulas, SMT expressions, and implication
  queries for accepted enlargement checks. Revision 0 is False.
- `artifacts/queries/`: executable SMT-LIB queries and outcomes at each size.
- `artifacts/verification/`: unbounded symbolic checks and countermodels;
  requested motion checks include their geometric counterexamples.
- `artifacts/counterexamples/`: generated coordinates and execution metadata.
- `artifacts/trajectories/`: current full-state NPZ archives, including rejected
  executions when states were recorded. These remain experiment artifacts.
- `artifacts/learning-states.json`: the accumulated learning dataset;
  `artifacts/videos/`: optional videos of those physical executions.

Exit code 0 means the requested stages passed: `verified_symbolic` for symbolic
only, or `verified_model` for both. A motion failure retains the symbolic result
separately. Exit code 2 means failure or an inconclusive/budget-limited experiment.
A finite search bound never substitutes for the unbounded proof. The motion
result concerns the configured geometric model, not universal MuJoCo dynamics.

Count verification attempts, accepted counterexample executions, and learner
updates separately. The initial False check and final successful check are
verification attempts. Do not assert the paper's table counts or exact printed
formulas as regression targets.

## Adding an environment

Implement `ExperimentTask` in `tasks.py` and register its factory in `TASKS`.
The CLI obtains its task choices from this registry. The adapter supplies:

1. Context, vocabulary, loop ID, fixed-program/task description, and invariant
   installation into that program's verification representation.
2. Unbounded symbolic verification and `search_query(size, timeout_ms)`, returning
   a `WitnessQuery` with initial-domain constraints, missing-head query, execution
   bound, and model decoder. The shared flat-loop preimage helper is optional.
3. Physical execution from the decoded scene, initialization/settling, complete
   trace validation, and learning states with correct frozen geometry.
4. Motion verification for `both` mode, or an explicit unsupported result.

`run_experiment` owns dataset accumulation, the fixed `InvInference` call,
progress checks, result statuses, and artifacts. A test adapter with a binding-only
program checks that this orchestration does not depend on Stack names or geometry.
New predicate languages, nested/multiple loops, and missing physical or symbolic
backends need their own implementation; registering an environment does not
implicitly provide those capabilities. Preserve the 60-second total limit when
adding Unstack.
