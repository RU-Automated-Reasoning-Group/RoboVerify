# RoboVerify

RoboVerify synthesizes block-manipulation programs from demonstrations and checks
symbolic correctness and geometric motion obligations. Programs execute in a
Fetch/MuJoCo environment. The pipeline uses relational CFGs, flat-loop recovery,
learned invariants and counterexample-guided refinement.

## Project status

The Stack workflow now has a standalone multi-seed DSL demonstration collector,
full-state archives, optional 20 FPS videos, and shared `full` / `verify` pipeline
modes. Both modes learn invariants from executions of the actual candidate and
perform symbolic and motion verification with feedback. See the
[collection guide](roboverify/synthesis/inference_lib/README.md).

Synthesis offers `--synthesis-approach relational` (the existing default) and
`--synthesis-approach id-first` (numeric MCMC/refinement, followed by quotienting).
Both return named programs to inference and verification. See the
[approach guide](roboverify/synthesis/cfg/VERIFICATION.md#synthesis-approaches).

Stack reset now scatters blocks within **0.70 m horizontally of the robot base**,
retaining block separation and initial gripper clearance. With this region,
**500/500 four-block demos (seeds 0–499)** finish in exactly three iterations,
pass initial/final task validation, and converge in every primitive within at
most 22 steps. The earlier region passed 96/100; see review entry 23 for the
reset bounds and physical diagnosis. This validates the tested four-block
collection, not arbitrary tower sizes or robot configurations. Ten selected
seeds also have validated 20 FPS videos (review entry 23).
Primitive ID/ByName instructions share configurable controllers and retain their
50-step budgets. See [controller settings](roboverify/synthesis/inference_lib/README.md#primitive-controller-settings).
**End-to-end learning acceptance remains open:**
the verification-mode smoke requests additional demonstrations after symbolic
checking; the full-search smoke exhausts its configured budget. Neither result
is verified. See [the review record](PAPER-DISCREPANCIES.md) for model boundaries,
remaining acceptance work, and recorded implementation findings. Old demonstration
formats are removed; recollect instead of using the historical datasets.

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
