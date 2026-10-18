# CaP methods on RoboVerify

Two implementations share text model transport, DSL environment bindings,
immutable code checkpoints, recording, and independent trial evaluation:

- `cap_google_baseline`: the pinned Google `LMP`/`LMPFGen` definitions, with
  environment/model compatibility and observational logging hooks. No new DSL
  validator or compliance-refinement loop runs on this path.
- `cap_method_1`: deterministic subset validation, recursive helper generation,
  bounded local repair, and an explicit checked interpreter.

See [CAP-INTEGRATION.md](docs/CAP-INTEGRATION.md) for the design and
[vendor provenance](vendor/google_cap/README.md) for the baseline's exact source
and compatibility differences. Plan 2 is documented but not implemented.

Observation boundaries and flowcharts with labeled data edges:

- [Google baseline: flow and two prompt modes](docs/BASELINE-FLOW.md).
- [Method 1: two-mode execution flow](docs/METHOD-1-FLOW.md).

Shared position observations and two condition modes are implemented for both
methods. Baseline modes are prompt-only; method-1 modes are checked:

- [Method 1 configuration](docs/Method1_Configuration.md).
- [Revision 1 implementation and validation](docs/revision_1.md).
- [Revision 2 plan: prompts, versions, and parallel batches](docs/revision_2.md).
- [Shared DSL checker architecture](docs/SHARED-DSL-CHECKER.md) (planned integration).

## Implemented scope

The adapter connects Stack, Unstack, Reverse, and ReStack to the existing
block-reference `PickByName`/`MoveByName`/`ReleaseByName` controllers and shared
predicate evaluator. Both methods expose current gripper/block XYZ through
`get_gripper_position()` and `get_block_position(handle)`. They return immutable
`Position3` snapshots in world-frame metres without stepping physics. Full raw
observations, images, rewards, and simulator objects remain host-only. `ON_star_zero` is available for Reverse and retains the
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
It rejects imports, attributes, general subscripts, function aliases, dynamic
calls, decorators, default arguments, comprehensions, and `for` loops. Literal
Position3 component reads at indices 0/1/2 are supported. With the default
`--condition-mode dsl_only`, numeric conditions remain forbidden;
`--condition-mode coordinate_predicates` also permits pure coordinate predicates
and general numeric conditions, including counters and constant comparisons. This is a declared Python subset, not a complete analyzer of Python.
Complete validation rechecks dependencies and value provenance before any action.
Local helper validation uses unknown parameter types until caller-level checking
can supply them; a local pass is not labeled an executable complete pass.

The current runner generates a fresh policy per trial. Both methods configure
**no auxiliary query LMPs**. Ordinary generated helpers work in both methods;
optional runtime query-LMP registration and cross-invocation method-1 session
persistence remain extension points. The logger can record new runtime-generated
sources, but this release does not claim those optional module configurations.

## Condition modes

Both `--method cap_google_baseline` and `--method cap_method_1` accept:

| Flag | Allowed loop/branch/selection conditions |
| --- | --- |
| `--condition-mode dsl_only` (default) | DSL predicates, identity, Boolean logic, quantifiers, and their pure helper compositions. Coordinates may still parameterize actions. |
| `--condition-mode coordinate_predicates` | The same DSL plus generated coordinate predicates and general numeric conditions, including counters. No coordinate-origin restriction. |

Baseline modes change policy/helper prompts only: upstream `LMP`, `LMPFGen`, and
`exec_safe` stay unchanged, with no added DSL gate or repair. A baseline prompt
violation can therefore execute. Method 1 enforces the selected mode through
complete checking and its interpreter; repairs receive static diagnostics only.
The initial policy prompts are identical for the same task/API/mode inputs.

For example, from `roboverify/` with the documented simulator environment:

```bash
.venv/bin/python -m synthesis.entry.run_cap --method cap_method_1 --condition-mode coordinate_predicates --task stack --num-blocks 3
.venv/bin/python -m synthesis.entry.run_cap --method cap_google_baseline --condition-mode dsl_only --task stack --num-blocks 3
```

These commands use the configured model provider and require its credentials.
Use `--generate-only --responses-file ...` for an offline generation diagnostic.
Mode, observation API version, validation configuration, and method-1 checker
version are retained in config/checkpoints and the bounded run report.

## Prompt versions

Both methods accept `--prompt-version legacy_v1|typed_v2`, independently of
`--condition-mode`. The default is **legacy_v1**, preserving the previous
policy/helper/repair text and method-specific whitespace/context assembly.
**typed_v2** adds typed parameter/return descriptions, precise primitive and
predicate semantics, configured Higher tolerance, and interface examples.
Policy requests require an execution entry; helper requests require only the
requested definition; Method-1 repairs follow the source unit's role.

For example, an offline generation diagnostic using the existing scripted inputs:

```bash
.venv/bin/python -m synthesis.entry.run_cap --method cap_method_1 --prompt-version typed_v2 --condition-mode coordinate_predicates --generate-only --responses-file synthesis/cap/examples/generation_only.json
```

The versioned resources are [legacy_v1.json](prompts/legacy_v1.json) and
[typed_v2.json](prompts/typed_v2.json). Add a new version for future prompt changes;
keep existing versions fixed for reproducible comparisons. Package bytes are
frozen on first load in each process. The existing contracts.py constants and
legacy template files are preserved and checked against legacy_v1.

Config, results, checkpoints, and the bounded report record `prompt_version` and
`prompt_package_sha256`; trial artifacts retain a copy of `prompt_package.json`.
Every request retains `prompt.txt`, `request.json`, and `prompt_metadata.json`
with exact prompt and canonical request hashes, role, call ID, and retry index.
Checkpoint source `generation_matches` link matching returned code to those
requests (outer whitespace ignored; all matches retained if ambiguous).
Prompt provenance is separate from observation/checker versions.

This changes model-facing text and provenance only. The Google core, baseline's
prompt-only constraints, Method-1 checker/interpreter, physical primitives, and
both condition modes retain their behavior. Offline tests check old prompt bytes,
role-specific repairs, retries, and fixture action parity across versions.
No live performance improvement is established. See
[Revision 2 section 1](docs/revision_2.md#1-make-api-types-and-semantics-explicit-in-the-prompt).

## Model providers and observation modality

Both methods use a **text-only LLM**. Prompts contain instructions, the API
contract, symbolic input names, source code, and (method 1 only) compliance
diagnostics. All requests include the configured condition-mode rules. The
model is not automatically sent coordinates or a full symbolic state table.
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

## Resource budgets and timing

CaP now defaults to a **600-second wall-clock safety cap** (`--timeout`).
Unstack retains its 60-second end-to-end cap. Trial effort is also bounded by
model calls (`--max-model-calls`), provider-reported cumulative tokens
(`--max-tokens`), helper/call depth (`--max-depth`), execution/query ticks
(`--max-steps`), and policy simulation-step calls (`--max-sim-steps`, default
10000). Existing per-primitive controller limits remain unchanged.
These resource limits can stop a trial well before its wall deadline. Reports
record the first exhausted budget in `budget_exhausted_by`, separately from
primitive nonconvergence and ordinary runtime/model failures.

The shared runner accepts `queue_wait_seconds` and `batch_context` from a batch
scheduler. Queue delay is recorded separately and does not consume the active
trial deadline. Independent processes own their simulator and signal handlers;
parallel output directories must remain trial-specific. No generic parallel
batch CLI is added here; the Batch C rerun retains its reproducible scheduler
under the run artifacts.

`timings.json`, `result.json`, and the bounded report expose exclusive phase
seconds: `model_policy`, `model_helper`, `model_repair`, `transport_backoff`,
`validation`, `generation_control`, `execution`, `simulation`, `recording`,
`rendering`, `video_finalize`, `logging_io`, `setup`, `evaluation`, `cleanup`,
and residual `overhead`, where applicable. Nested phases are not double counted;
for example, simulation and logging are removed from the execution subtotal.
Model-request time includes network/provider waiting and response decoding; it
is not pure model inference time. Failed attempts and retry backoff are timed.
Validation and repair-request time remain distinct. Simulation means policy
`env.step` calls; fresh-scene settling belongs to setup.

`active_wall_seconds` covers measured work through environment cleanup, before
final result/timing serialization. Cleanup can extend beyond an expired active
trial deadline and is reported rather than treated as successful execution.
Saved-snapshot restoration and observation semantics are unchanged. The pinned
Google core and all model-facing prompt text remain unchanged by instrumentation.
The 65-test CaP suite covers independent budget exhaustion, failed requests,
nested exclusive timing, queue accounting, simulation-step limits, and Unstack's
retained cap.

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

A matched Gemini batch used `google/gemini-4-argon`, five independent trials per
method, and `max_attempts=5` on the same saved three-block scene. Method 1
succeeded in 2/5 trials (trials 3 and 4); the baseline succeeded in 0/5. Observed
pass@5 was therefore true for method 1 and false for the baseline on this scene.
All five method-1 policies passed validation on their first candidate, with no
repairs; the other three trials failed Move convergence. Baseline failures were
three helper-name visibility errors, one Move convergence failure, and one
normal return with a failed physical goal despite a true DSL postcondition.
Method 1 used five model calls and 4261 reported tokens; the baseline used 15
calls and 11481 tokens. This does not establish a benefit from the larger repair
budget or a benchmark-wide advantage. The endpoint required reasoning to be
enabled; rejected requests with reasoning disabled are retained separately as
setup diagnostics. All evaluation trials used reasoning enabled, temperature 0,
and exactly matching full initial states. The report and per-trial artifacts are
under `runs/cap-stack-gemini-pass5/20261015-160329-reasoning/`.

The same Gemini configuration was also evaluated on one five-block Stack scene,
with five fresh trials per method. Both methods returned 0/5 successes, so both
observed pass@5 outcomes were false. All five method-1 policies passed initial
validation without repair; three failed Pick convergence and two failed Move
convergence. The baseline had three helper-name errors, one generated syntax
error, and one Move convergence failure. Method 1 used five model calls and
4233 reported tokens; the baseline used 15 calls and 11433 tokens. Every trial
restored the same precondition-validated, seed-0 snapshot after 50 settling steps;
all ten videos were checked against their recorded frame counts. Reports and
artifacts are under `runs/cap-stack-gemini-5blocks-pass5/20261015-164132/`.
This is one fixed five-block scene, not a benchmark-wide result; comparison with
the three-block batch changes the layout as well as the block count.

Revision 1 validation (2026-10-16): all 57 CaP tests passed, including strict/extended
condition behavior, recursive mode propagation, query freshness, and unchanged
pinned Google core. A real MuJoCo scripted Pick/Move diagnostic compared all four
method/mode combinations from the same saved three-block Stack start. Initial
observations, all position-read values, and all 19 controller actions matched
exactly; Pick converged in 14 steps and Move in 5. Each run returned `goal_failed`
because the short diagnostic does not complete Stack. No live model requests
were made. Artifacts and the compact parity summary are under
`runs/cap-revision-1/20261016-213331/`; see [Revision 1](docs/revision_1.md).

A post-Revision-1 DeepSeek batch on 2026-10-18 used
`deepseek/deepseek-v4.1-flash`, `dsl_only`, five independent policy trials per
method, and the same saved three-block Stack start. Both methods had **0/5
physical successes**. Baseline outcomes were three Move-arity failures, one
helper response truncated at the output-token limit, and one no-witness query
failure. Four baseline trials completed a Pick before failing. All five method-1
trials exhausted five local attempts (four repairs each) on Move arity or invalid
object-reference types, with no physical actions. Baseline used 14 model calls
and 14,937 reported tokens; method 1 used 25 calls and 21,882 tokens.
Settings were temperature 0, reasoning disabled, 4096 output tokens, a 180-second
trial deadline, and 12 model calls maximum per trial. Initial states and
nonmethod configuration matched; the initial prompt texts differed only by one
trailing newline retained by the upstream baseline. All ten videos were verified
against recorded frame/state counts. This is one fixed scene and does not
evaluate the coordinate-predicate mode or establish benchmark-wide performance.
The report, exact configuration, prompts/source checkpoints, recordings, videos,
and audit are under `runs/cap-stack-deepseek-revision1/20261018-172554/`.

A matched reasoning-enabled comparison on 2026-10-18 used **16,384 output tokens
per request**, `dsl_only`, and five fresh trials per method/model from the same
three-block Stack snapshot:

| Model | Method | Physical successes | Model calls | Reported tokens |
| --- | --- | --- | --- | --- |
| `openai/gpt-6.1-sol` | Google baseline | 1/5 | 30 | 40,567 |
| `openai/gpt-6.1-sol` | Method 1 | 5/5 | 6 | 8,072 |
| `deepseek/deepseek-v4.1-flash` | Google baseline | 1/5 | 5 | 63,755 |
| `deepseek/deepseek-v4.1-flash` | Method 1 | 2/5 | 7 | 82,434 |

GPT baseline failures were three 180-second trial deadlines and one Pick
nonconvergence; method 1 completed all five with one static repair in total.
DeepSeek baseline failures were two invalid object-reference errors, one response
truncated at 16k, and one Move nonconvergence. DeepSeek method 1 had two Move
nonconvergences and one response truncated at 16k, with two repairs across its
five trials. Token totals include returned usage only; three GPT baseline
requests had no returned response at the trial deadline.

Both models used reasoning enabled, temperature 0, a 50-second request timeout,
one transport retry, a 180-second trial deadline, at most 12 model calls and
50,000 reported tokens per trial, and up to five local method-1 attempts.
The two model batches ran concurrently; each batch alternated methods
sequentially. Initial observation states and nonmodel configuration matched.
Prompt texts matched apart from the existing trailing-newline difference between
baseline and method 1. Seventeen saved videos passed frame/state-count checks;
three deadline-interrupted GPT baseline videos failed to finalize and are marked
unavailable. All failures remain in the denominator.

The GPT endpoint required reasoning, so its initial reasoning-disabled request
is retained as a separate setup diagnostic. Earlier 4k-output batches were
interrupted at the user's request and marked superseded; their partial outcomes
are excluded here. Revision 2 documentation was drafted during the runs, but
executable and model-facing prompt sources remained unchanged from `d7bcbb0`.
This is one fixed scene, not a benchmark-wide estimate or evidence isolating a
single cause of the method difference. The combined report, detailed trial
links, audits, source hashes, and recordings are under
`runs/cap-stack-reasoning-comparison/20261018-180533-16k/`; individual batches are
under the matching `20261018-180533-reasoning-16k` directories in
`runs/cap-stack-gpt61sol-revision1/` and `runs/cap-stack-deepseek-revision1/`.

A subsequent matched comparison on 2026-10-18 switched only the condition-mode
configuration and its prompt rule text to **`coordinate_predicates`**, retaining
reasoning enabled, 16,384 output tokens, the same three-block initial snapshot,
and the previous B/C budgets. Five fresh trials were run for each method/model:

| Model | Method | Physical successes | Model calls | Reported tokens |
| --- | --- | --- | --- | --- |
| `openai/gpt-6.1-sol` | Google baseline | 4/5 | 21 | 33,394 |
| `openai/gpt-6.1-sol` | Method 1 | 5/5 | 7 | 9,422 |
| `deepseek/deepseek-v4.1-flash` | Google baseline | 0/5 | 8 | 75,190 |
| `deepseek/deepseek-v4.1-flash` | Method 1 | 2/5 | 5 | 75,019 |

GPT baseline had one Pick nonconvergence; all method-1 trials succeeded, with
two static repairs in total. DeepSeek baseline had three invalid-object-reference
failures, one total-deadline failure, and one Move nonconvergence. DeepSeek method
1 had two output-limit failures and one Move nonconvergence, without repair calls.
Nineteen saved videos passed frame/state-count checks; the deadline-interrupted
DeepSeek baseline video did not finalize and is marked unavailable.

Every trial row in the new combined and model-specific reports includes
`# of tokens used`, summing provider-reported input and output usage across all
initial/helper/repair requests. All twenty totals matched the saved usage data;
reasoning tokens were not added twice. The previous DSL-only B/C reports also
received the per-trial token column, retaining incomplete-usage markers for their
unreturned requests.

The combined report, prior-mode comparison, configuration/prompt audits, source
hashes, and video links are under
`runs/cap-stack-coordinate-comparison/20261018-184618-reasoning-16k/`.
Individual batches are under the corresponding
`20261018-184618-coordinate-reasoning-16k` directories in
`runs/cap-stack-gpt61sol-revision1/` and `runs/cap-stack-deepseek-revision1/`.
The current samples are separate from the prior DSL-only trials; five trials on
one scene do not establish a general advantage for either condition mode.
Position reads alone also do not show that a policy used coordinates in its
conditions: they may only parameterize actions. Revision 2 remained documentation
only during these runs, and all checked executable/prompt sources matched the
prior B/C code. Its proposed trial-level parallel scheduler was not applied to
this batch.

A Batch C DSL-only Google-baseline rerun on 2026-10-18 used the **600-second
wall-clock safety cap**, unchanged GPT-6.1-sol/reasoning/16k settings and the
original fixed three-block start. Five fresh trials ran in two isolated processes
with dispatch order `[4, 5, 1, 3, 2]` from a recorded seed. The original 12-call,
50,000-reported-token, depth-12, and 10,000-execution-tick limits were retained;
the explicit 10,000-policy-simulation-step cap was nonbinding.

The result was **3/5 successful**, with two Move-convergence failures and no
resource or wall-clock exhaustion. The five trials used 28 model calls and
42,272 reported tokens. Successful active durations were 138.94, 187.46, and
134.00 seconds. The 187.46-second trial spent 185.15 seconds in model requests
and 0.18 seconds in simulation; it also waited 217.83 seconds in the batch queue,
which was recorded separately and excluded from its trial deadline. On these
observed traces, successful completion counts by 180/300/600 seconds were
2/5, 3/5, and 3/5; these thresholds are not independent rerun experiments.

All five saved videos matched trajectory state counts, phase totals matched
active wall time, token totals matched returned usage, and initial observations
and first-policy prompts matched the earlier Batch C baseline. The pinned Google
core and prompt text were unchanged. The code passed 65 CaP tests and a five-job
scripted parallel-fixture check before live requests. Reports, queue/timing
records, the reproducible process scheduler, snapshots, and audit are under
`runs/cap-stack-gpt61sol-budget/20261018-192834-dsl-600s/`.
Only the Batch C DSL-only baseline result was rerun: the updated table retains
all other A/B/C and coordinate-mode results at their original settings. Its new
3/5 cell must be marked 600 seconds (previously 1/5 at 180 seconds); differences
between fresh samples cannot be attributed entirely to the longer timeout.
