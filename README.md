# RoboVerify

RoboVerify synthesizes block-manipulation programs from demonstrations and checks
symbolic correctness and geometric motion obligations. Programs execute in a
Fetch/MuJoCo environment. The pipeline uses relational CFGs, flat-loop recovery,
learned invariants and counterexample-guided refinement.

## Project status

The Stack workflow now has a standalone multi-seed DSL demonstration collector,
full-state archives, optional 20 FPS videos, and shared `full` / `verify` pipeline
modes. Collection holds the initial gripper position for **50 settling steps**,
then saves that full simulator state as demonstration state zero. Settling is
excluded from recorded actions and video. Both modes and standalone MCMC restore
the archived start without resetting from the seed or settling again. Both modes
learn invariants from executions of the actual candidate and perform symbolic
and motion verification with feedback. See the
[collection guide](roboverify/synthesis/inference_lib/README.md).

Synthesis offers `--synthesis-approach relational` (the existing default) and
`--synthesis-approach id-first` (numeric MCMC/refinement, followed by quotienting).
Both return named programs to inference and verification. See the
[approach guide](roboverify/synthesis/cfg/VERIFICATION.md#synthesis-approaches).

Stack reset scatters blocks within **0.70 m horizontally of the robot base**,
retaining block separation and initial gripper clearance. Primitive ID/ByName
instructions share configurable controllers and retain their 50-step budgets.
See [controller settings](roboverify/synthesis/inference_lib/README.md#primitive-controller-settings).

**Symbolic invariant inference uses the partition-based algorithm in
`inference.py`, through `InvInference`.** It is the intended algorithm for both
pipeline modes and standalone symbolic CEGIS; there is no learner-selection flag.

**Supplied Stack symbolic verification passes with the intended learner and the
explicit vocabulary `--invariant-relations ON_star equality`.** It learns from
candidate loop heads and normal exits, then proves establishment, preservation
and exit, including unbounded verification. This configuration does not carry
separation facts needed by motion verification: collision checks still fail.
End-to-end supplied-program verification and full synthesis acceptance remain open.
See the [verification command](roboverify/synthesis/cfg/VERIFICATION.md#provided-stack-verification)
and [invariant diagnosis](PAPER-DISCREPANCIES.md#30-stack-invariant-vocabulary-and-attachment-semantics).
Current archives contain full simulator states; old demo formats are unsupported.
Collect demonstrations before running the pipeline examples.

The supported scope is structured chains and flat loops for the tower tasks;
the integrated synthesis CLI exposes Stack and Unstack. A successful
`verified_model` result establishes partial correctness in the documented
geometric model, not total termination or physical-controller refinement.

## Documentation

| Document | Purpose |
| --- | --- |
| [AGENTS.md](AGENTS.md) | Development rules, environment, commands, architecture and experiment reporting. |
| [PAPER-DISCREPANCIES.md](PAPER-DISCREPANCIES.md) | The single paper-review record: stable entries, decisions/proofs, status and remaining actions. |
| [CFG verification](roboverify/synthesis/cfg/VERIFICATION.md) | Integrated synthesis/verification workflow and model assumptions. |
| [Motion API](roboverify/synthesis/verification_lib/README.md) | Motion contracts, collision/support checks, bounded noise and BMC distinctions. |
| [Standalone CEGIS](roboverify/synthesis/verification_lib/CEGIS.md) | APIs and commands for refining existing programs. |
| [Trace inference](roboverify/synthesis/inference_lib/README.md) | Collecting loop-head states and learning invariants with DemoStore. |

Implementation lives in `roboverify/`. Configure MuJoCo through AGENTS.md and run
entry points as modules from that directory. Tests use `unittest`. The reviewed
paper is [POPL2027.pdf](POPL2027.pdf). Completed implementation plans and audit
history are retained in Git rather than maintained as active task documents.
