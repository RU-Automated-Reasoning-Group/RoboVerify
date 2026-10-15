# CaP methods on RoboVerify

Two implementations share text model transport, DSL environment bindings,
immutable code checkpoints, recording, and independent trial evaluation:

- `cap_google_baseline`: the pinned Google `LMP`/`LMPFGen` definitions, with
  environment/model compatibility and observational logging hooks. No new DSL
  validator or compliance-refinement loop runs on this path.
- `cap_method_1`: deterministic subset validation, recursive helper generation,
  bounded local repair, and an explicit checked interpreter.

See [CAP-INTEGRATION.md](CAP-INTEGRATION.md) for the design and
[vendor provenance](vendor/google_cap/README.md) for the baseline's exact source
and compatibility differences. Plan 2 is documented but not implemented.

## Implemented scope

The adapter connects Stack, Unstack, Reverse, and ReStack to the existing
block-reference `PickByName`/`MoveByName`/`ReleaseByName` controllers and shared
predicate evaluator. It exposes no raw observations, positions, images, reward,
or simulator objects. `ON_star_zero` is available for Reverse and retains the
trial-entry geometry. Other tasks expose the current-state tower vocabulary.
There is no coordinate-bearing `tbl` handle or added placement macro.

Grid/Pyramid explicitly return `adapter_gap` before model calls: their shared
Goal traversal, occupancy, and physical-reference contract remains unfinished.
ReStack uses the environment's PartialStack reset and terminal criterion;
Reverse checks the canonical initial order and evaluates complete reversal.
Shared DSL pre/post checks are additionally reported for Stack/Unstack. Initial
states must satisfy the shared precondition. The primary success metric uses
normal completion, primitive convergence, and the environment terminal check;
the DSL postcondition is a separate diagnostic, not an extra scoring gate.

Method 1 supports plain top-level function definitions, assignment, arithmetic
for action parameters, object equality, Boolean conditions, nested `if`/`while`,
`return`, `break`, `continue`, and `exists`/`forall`/`get` with query lambdas.
It rejects imports, attributes, subscripts, function aliases, dynamic calls,
decorators, default arguments, comprehensions, `for` loops, and numeric branch
tests. This is a declared Python subset, not a complete analyzer of Python.
Complete validation rechecks dependencies and value provenance before any action.
Local helper validation uses unknown parameter types until caller-level checking
can supply them; a local pass is not labeled an executable complete pass.

The current runner generates a fresh policy per trial. Both methods configure
**no auxiliary query LMPs**. Ordinary generated helpers work in both methods;
optional runtime query-LMP registration and cross-invocation method-1 session
persistence remain extension points. The logger can record new runtime-generated
sources, but this release does not claim those optional module configurations.

## Model providers and observation modality

Both methods use a **text-only LLM**. Prompts contain instructions, the API
contract, symbolic input names, source code, and (method 1 only) compliance
diagnostics. The model is not automatically sent a full symbolic state table.
Predicate truth is evaluated by the host on current simulator state during
execution. Videos and raw states are host-side artifacts, not model inputs.

`llm.py` defines a normalized text request/response boundary. The initial
OpenRouter configuration uses the user-selected model
`deepseek/deepseek-v4.1-flash`; its availability is not asserted by offline tests.
Provider, model, base URL, credential environment-variable name, output limit,
sampling options, request timeout, and bounded transport retries are configurable.
Credentials are read at request time and are not written into request artifacts.

Chat Completions is the shared compatibility path for
[OpenRouter](https://openrouter.ai/docs/api/api-reference/chat/create-a-chat-completion),
[OpenAI](https://developers.openai.com/api/reference/chat-completions/overview),
and an [OpenAI-compatible vLLM server](https://docs.vllm.ai/en/latest/serving/online_serving/openai_compatible_server/).
A separate `--api-style responses` adapter supports text Responses requests;
provider availability and parameter support still depend on the chosen model.
`--token-field max_completion_tokens` handles chat models needing that field.
`--extra-body options.json` supplies explicit provider-specific settings such as
routing or reasoning options. Unsupported requests fail explicitly; the client
does not silently change models or drop sampling parameters.

No tool calling, image input, or provider-side code execution is required.
Record and match actual model/provider settings between methods. Missing token
usage is tracked separately from known token counts. Transport retries consume
the same model-call budget and do not become code repair for the baseline.

## Run

Run from `roboverify/`. For physical execution, configure the simulator environment
as documented in the root `AGENTS.md`:

```bash
unset LD_PRELOAD
if [ -d "$PWD/.runtime/mujoco210" ]; then
    export MUJOCO_PY_MUJOCO_PATH="$PWD/.runtime/mujoco210"
else
    export MUJOCO_PY_MUJOCO_PATH="$HOME/.mujoco/mujoco210"
fi
export LD_LIBRARY_PATH="$MUJOCO_PY_MUJOCO_PATH/bin:/usr/lib/nvidia"
uv run python -m synthesis.entry.run_cap --method cap_method_1 --task stack --num-blocks 3 --provider openrouter --model deepseek/deepseek-v4.1-flash
```

Set `OPENROUTER_API_KEY` in the launching environment. Do not put credentials in
command-line arguments or JSON configuration files. Video recording is enabled
by default and requires ffmpeg; `--no-save-video` disables it explicitly.

To compare on an identical state, reuse the saved `initial_state.npz` from the
first run, or use a current demonstration archive with `--trace-index`:

```bash
uv run python -m synthesis.entry.run_cap --method cap_google_baseline --task stack --num-blocks 3 --initial-state runs/cap/<first-run>/artifacts/trial_001/initial_state.npz
```

Replace `<first-run>` with the actual directory before running. Saved snapshots
restore directly; fresh Stack preparation performs the same 50 settling steps
before recording. Trials record success only after normal completion and the
terminal task check. An earlier success flag or a successful state reached by
a failed program is not counted as a success. Unstack's timeout is capped at
60 seconds. `--max-model-calls`, `--max-attempts`, `--max-depth`, and `--max-steps`
additionally bound generation/repair/execution. `--max-tokens` limits reported
token usage; when usage is unavailable, call/time limits still apply and the
unknown-usage call count is reported explicitly.

Alternative transports keep the method and environment unchanged:

```bash
uv run python -m synthesis.entry.run_cap --method cap_method_1 --provider openai --model YOUR_MODEL --api-style responses
uv run python -m synthesis.entry.run_cap --method cap_method_1 --provider vllm --model YOUR_SERVED_MODEL --base-url http://localhost:8000/v1
```

OpenAI reads `OPENAI_API_KEY`; local vLLM defaults to no authentication header.
Use `--api-key-env VARIABLE_NAME` when a local server requires authentication.
Selecting a model without the required API parameter support is a configuration
failure, not grounds to silently alter the experiment.

## Offline generation diagnostic

This command makes no network requests and requires no simulator. It tests code
generation/helper resolution using fixed responses; it is not a task benchmark:

```bash
python -m synthesis.entry.run_cap --method cap_method_1 --generate-only --responses-file synthesis/cap/examples/generation_only.json --output-dir /tmp/cap-diagnostics
python -m synthesis.entry.run_cap --method cap_google_baseline --generate-only --responses-file synthesis/cap/examples/generation_only.json --output-dir /tmp/cap-diagnostics
```

Generation-only runs have `success=null`; their status records generation success
or failure without entering physical benchmark scores. Baseline generation-only mode uses upstream's
debug option: top-level execution is skipped, while original helper binding still
occurs. Method 1 validates code without executing generated definitions.

## Logs and checkpoints

Outputs use `runs/cap/<timestamp>-<revision>-<method>-<task>-<unique-id>/` and the
existing `latest` symlink. `config.json`, `status.json`, metrics, events, stdout,
and final results retain the existing RunLogger contract. Detailed unthrottled
events and artifacts live in `artifacts/trial_001/`:

- `lmp/call_*/`: actual request, prompt, response text, provider response/usage.
- `checkpoints/cp_*/`: policy, all known/pending helper sources, parent-linked
  manifest, source hashes, validation diagnostics, and source diff.
- `execution/rollout_001/`: full-state trajectory archive, per-step JSONL,
  MP4, and frame-to-state/source index; partial failed rollouts are retained.
- `manifest.json`: artifact paths and completion states. Code checkpoints are
  not resumable interpreter snapshots.

Use the bounded report interface:

```bash
python -m synthesis.experiment.report --run runs/cap/latest
```

Do not read full stdout or metrics files for ordinary monitoring. Baseline
checkpoints explicitly use `validation=not_run`; method-1 failures retain exact
diagnostics and pre-repair source. Console output is also saved to `stdout.log`.

## Tests and validation limits

```bash
python -m unittest synthesis.cap.test_validation synthesis.cap.test_methods synthesis.cap.test_llm synthesis.cap.test_logger -v
uv run python -m unittest discover -s synthesis/cap -t . -p 'test_*.py' -v
```

Core tests use only the standard library. Adapter/runner tests additionally use
the project's NumPy/Z3 dependencies and exercise real shared predicates and
controllers against a deterministic fixture. The ffmpeg test records synthetic
frames and checks partial-rollout video/state correlation. These tests do not
establish MuJoCo physical success. No six-task performance results are claimed.

A user-authorized live OpenRouter Stack generation smoke was also attempted.
The initial 4096-output-token request ended with `finish_reason=length`. An
explicit non-reasoning smoke option is available in
[openrouter_smoke_options.json](examples/openrouter_smoke_options.json), passed
with `--extra-body`; it is not a changed default or a benchmark recommendation.
See [OpenRouter reasoning options](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens).
Subsequent diagnostics exercised model output, checkpointing, and compliance
repair. An early unused-function validation omission was corrected and covered
by a regression test; that earlier accepted diagnostic is not validation evidence.
The post-fix bounded method-1 diagnostic exhausted its two local attempts on an
invalid condition, and the original baseline stopped on a generated helper-name
mismatch. These outcomes are generation diagnostics, not physical task results.
The earlier `environment_unavailable` diagnostic used an interpreter without
the project's dependencies. The existing RoboVerify runtime was subsequently
located in a preserved workspace backup and reconnected through local `.venv`
and `.runtime/mujoco210` links. No dependency reinstall was needed; see AGENTS.md
for selecting this runtime instead of the system Python.

The original Stack factory passed reset, step, and offscreen rendering with a
58-dimensional observation, a four-dimensional action, and a 500x500 RGB frame.
Both CaP methods then ran the same scripted Pick in real MuJoCo, starting from
the same saved snapshot: each controller converged in 14 steps, the action
sequences matched exactly, and each video contained 15 frames. These were
integration diagnostics using fixed responses, not model-generated Stack
solutions. Their `goal_failed` results are expected because a single Pick does
not complete the Stack task.

A subsequent real-model three-block Stack pilot used identical full initial
states, a specification-only task prompt, and matched DeepSeek/OpenRouter
settings for both methods. The baseline completed one Pick (15 controller steps)
before failing on a six-argument Move call. Method 1 rejected the same invalid
API call in all three candidates (two repairs), so it executed no robot actions.
Neither trial completed Stack. The shared API requires
`Move(x_ref, y_ref, z_ref, [dx, dy, dz])`. Programs, diagnostics, trajectories,
videos, and the paired report are retained under
`runs/cap-stack-comparison/20261015-135348/`. This is one paired pilot, not an
aggregate success-rate comparison; broader task-policy evaluation remains open.

Five independent method-1 experiments then used `max_attempts=5` on the same
three-block initial state, with all other settings retained (including
temperature 0). They produced five distinct initial program texts but no task
success: observed pass@5 on this instance was false (0/5 successes). Each used
five model calls and four repairs; four trials ended on Move arity and one on
unsupported `is` syntax. No robot actions ran. The batch used 25 model calls and
20248 reported tokens. Per-trial programs, diagnostics, and the report are under
`runs/cap-stack-pass5/20261015-150331/`. This is a single-instance pass@5 outcome,
not a benchmark-wide probability estimate.
