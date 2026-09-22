# RoboVerify

RoboVerify synthesizes block-manipulation programs from demonstrations and checks
symbolic correctness and geometric motion obligations. Programs execute in a
Fetch/MuJoCo environment. The supported synthesis pipeline uses relational CFGs,
flat-loop recovery, learned invariants and counterexample-guided refinement.

## Start here

- [Agent and contributor setup](AGENTS.md): environment, commands and conventions.
- [Current implementation plan](PLAN-popl-alignment.md): completed work, settled
  decisions and remaining demonstration/learning acceptance.
- [Architecture](CLAUDE.md): packages, APIs and experiment reporting.
- [Integrated synthesis and verification](roboverify/synthesis/cfg/VERIFICATION.md):
  pipeline, model assumptions and the meaning of `verified_model`.
- [Motion verification](roboverify/synthesis/verification_lib/README.md),
  [standalone CEGIS](roboverify/synthesis/verification_lib/CEGIS.md), and
  [trace-based inference](roboverify/synthesis/inference_lib/README.md): API guides.
- [Paper discrepancies](PAPER-DISCREPANCIES.md) and
  [settled findings and proofs](PAPER-RESOLUTIONS.md): decisions from reviewing
  [POPL2027.pdf](POPL2027.pdf), independent of its experiment claims.

Implementation lives in `roboverify/`. Configure MuJoCo as described in AGENTS.md
and run Python entry points as modules from that directory. Tests use `unittest`.
The implementation and synthetic regression coverage are complete for the agreed
scope; full learning acceptance still requires validated task demonstrations.
