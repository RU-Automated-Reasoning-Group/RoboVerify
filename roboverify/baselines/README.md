# Baselines

Each baseline family owns its models, adapter, dependencies, tests and reproduction
instructions in a separate package. Add future families alongside `mypyvy/`;
do not route unrelated baselines through a solver-specific runner.

- [mypyvy](mypyvy/README.md): UPDR invariant inference for the supplied Stack
  control and abstractions of the current Unstack and Reverse groundtruths.

These experiments are separate from RoboVerify's demonstration learner and
symbolic/motion pipeline. A baseline's abstract safety result does not establish
physical collision freedom, primitive convergence, synthesis or termination.

Run artifacts go under ignored `runs/baselines/FAMILY/`. The small shared schema
uses `config.json`, live `status.json`, and terminal `result.json`; each phase
records status and elapsed seconds. Inspect a run without opening large solver
logs:

```bash
cd roboverify
python -m baselines.report --run runs/baselines/mypyvy/RUN
```

The report reads only JSON summaries, prints at most 20 phases and truncates
failure reasons to 500 characters. A heartbeat is a snapshot, not a liveness
guarantee. `synthesis.experiment.report` handles the different synthesis/MCMC
schema and should not be used for these directories.
