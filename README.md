# RoboVerify

RoboVerify synthesizes block-manipulation programs from demonstrations and checks
symbolic correctness and geometric motion obligations. Programs execute in a
Fetch/MuJoCo environment. The pipeline uses relational CFGs, flat-loop recovery,
learned invariants and counterexample-guided refinement.

## Project status

The supported implementation and audit fixes are complete and integrated into
local `main`. **End-to-end learning acceptance remains open:** validate or replace
task demonstrations, recover the complete loop and verify the same synthesized
CFG against the intended specification. See
[entry 10](PAPER-DISCREPANCIES.md#10-the-historical-unstack-oracle-does-not-establish-its-final-task-condition)
for the next steps. Existing demos and the paper's experimental results are not correctness
oracles. Paper corrections and explicitly deferred work remain in the review record.

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
