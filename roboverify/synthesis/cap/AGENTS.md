# CaP experiment execution

Follow the repository-wide [AGENTS.md](../../../AGENTS.md) for environment setup,
project decisions, implementation conventions, and validation. This file adds
experiment guidance specific to `synthesis/cap/`.

- Before starting any task under `synthesis/cap/`, read the [README](README.md)
  overview and the sections relevant to the task. Read historical experiment
  reports only when needed.
- Run independent experiment trials in parallel by default. Maximize useful
  utilization of authorized CPU, GPU, memory, and model-request capacity to reduce
  experiment completion time.
- Use the shared CaP batch runner and optional GPU scheduler. Select only GPUs in
  the explicit allowlist, coordinate reservations across concurrent batches, and
  respect global and per-GPU worker limits and provider request limits.
- Choose concurrency from measured worker memory and observed throughput. Increase
  it when resources permit; reduce it when contention, memory pressure, encoding,
  I/O, or request throttling lowers throughput. High GPU utilization alone is not
  evidence of faster experiments.
- Preserve independent trial processes, fixed initial-state replay, per-trial
  budgets, artifact ownership, and the declared experiment denominator. Keep all
  recovery attempts in the same trial and rendering assignment.
- Record the scheduling configuration, actual assignments, queue time, active
  time, resource measurements, and failures. Resume pending trials without
  duplicating live workers, completed trials, or paid model requests.
- Use serial execution when a dependency, resource limit, debugging procedure,
  or controlled comparison requires it, and record the reason.
