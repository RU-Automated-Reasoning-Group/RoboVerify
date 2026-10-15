# Pinned Google CaP source

Source: [Google Research Code as Policies](https://github.com/google-research/google-research/tree/8c116f61a61b4339cbbae92be89ff83d9897d4b5/code_as_policies),
commit `8c116f61a61b4339cbbae92be89ff83d9897d4b5`.

`Interactive_Demo.ipynb` and `LICENSE` are upstream copies. `core.py` contains
the definitions of `LMP`, `LMPFGen`, `FunctionParser`, `var_exists`,
`merge_dicts`, and `exec_safe`, extracted from the notebook without algorithm
edits, with trailing whitespace normalized. `PROVENANCE.json` records the
notebook hash and source cell indices; tests compare the extracted ASTs.
Do not format or modernize the extracted definitions in place.

The baseline compatibility loader supplies dependencies in a fresh namespace:

- The historical completion API delegates to the shared text model client.
- Standard-library `ast.unparse` replaces the external `astunparse` formatter,
  including its statement-list behavior. Formatting differences may change
  helper-signature prompt text and are a declared compatibility difference.
- Syntax highlighting is omitted; printed text is mirrored to the common log.
- The shared client removes one surrounding Markdown code fence when present;
  raw response text is preserved separately for both methods.
- Execution instrumentation compiles source with checkpoint filenames, traces
  statements and enforces the common budget. Original `exec_safe` filtering and
  Python execution/binding semantics are retained. This is not a Python security
  sandbox and is intended for controlled simulator experiments.
- Wrapper subclasses log helper resolution and apply the shared depth budget.
  DSL validation and compliance refinement are never invoked by the baseline.

The environment bindings and prompts use the shared RoboVerify DSL interface.
The default experiment configures no auxiliary query LMPs. This is an adapted
baseline, not a reproduction of Google's original tabletop simulator/results.
