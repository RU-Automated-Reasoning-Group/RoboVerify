# CaP methods on RoboVerify

Three implementations share text model transport, DSL environment bindings,
immutable code checkpoints, recording, and independent trial evaluation:

- `cap_google_baseline`: the pinned Google `LMP`/`LMPFGen` definitions, with
  environment/model compatibility and observational logging hooks. No pre-execution DSL
  gate or compliance-refinement loop runs on this path; final output is checked
  independently after the attempt.
- `cap_method_1`: deterministic subset validation, recursive helper generation,
  bounded local repair, and an explicit checked interpreter.
- `cap_method_2`: complete policy/helper generation and whole-candidate static
  repair, using the same checker/interpreter as Method 1.

See [CAP-INTEGRATION.md](docs/CAP-INTEGRATION.md) for the design and
[vendor provenance](vendor/google_cap/README.md) for the baseline's exact source
and compatibility differences. Method 2 and the final-output baseline audit are implemented.

Observation boundaries and flowcharts with labeled data edges:

- [Google baseline: flow and two prompt modes](docs/BASELINE-FLOW.md).
- [Method 1: two-mode execution flow](docs/METHOD-1-FLOW.md).

Shared position observations and two condition modes are implemented for both
methods. Baseline modes are prompt-only; method-1 modes are checked:

- [Method 1 configuration](docs/Method1_Configuration.md).
- [Revision 1 implementation and validation](docs/revision_1.md).
- [Revision 2 plan: prompts, versions, and parallel batches](docs/revision_2.md).
- [Shared DSL checker architecture](docs/SHARED-DSL-CHECKER.md).
- [Revision 3: checker correctness](docs/revision_3.md).
- [Revision 4: shared failure recovery](docs/revision_4_failure_recover.md).
- [Revision 4.1: GPU scheduling — implemented](docs/revision_4_1_gpu_scheduling.md).
- [Revision 4.2: configurable primitive prompts and feedback — implemented](docs/revision_4_2_fix_primitive_noncovergence.md).
- [Revision 4.2.1: deferred video rendering — implemented](docs/revision_4_2_1_deferred_video_rendering.md).
- [Revision 4: deferred improvements](docs/revision_4_unimplemented.md).
- [Revision 5: task expansion plan](docs/revision_5_add_tasks.md).
- [Revision 6: demonstration input plan](docs/revision_6_add_demonstration.md).

Revision 3 fixes loop-carried effect propagation and generic helper type merging
in the shared checker (`shared-contract-v3`). All 106 CaP tests pass, including
18 new regressions. Baseline execution is unchanged; its audit uses the revised
contract for new runs. See [implementation and validation](docs/revision_3.md).
Configurable shared failure recovery is implemented as `cap-recovery-v1`; task expansion and demonstration input remain deferred plans.

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
ReStack uses the upstream Restack reset (one source tower containing b0 plus
loose blocks) and the Stack terminal criterion; its start is not yet validated.
Reverse checks the canonical initial order and evaluates complete reversal.
Shared DSL pre/post checks are additionally reported for Stack/Unstack; only
those starts must satisfy the shared precondition. The per-trial `success`
uses normal completion, primitive convergence, and the environment terminal
check. Batch `first_attempt_success` and recovery `task_success` additionally
use the DSL postcondition for Stack/Unstack. Upstream now defines shared
specifications for every task. Because CaP is RoboVerify's baseline, Revision 5
adopts RoboVerify's demonstration-acceptance criterion for every task: normal
completion, primitive convergence, the shared precondition at entry, and the
shared postcondition (goal occupancy for Grid/Pyramid) at the final state. It
is reported as a new primary field, `task_spec_success`; the existing fields keep
their definitions, and the physical evaluator becomes a final and settled diagnostic. Revision 5 also adds
a `tbl` reference for Move/Release. See the
[task expansion plan](docs/revision_5_add_tasks.md).

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

The runner accepts `--prompt-version legacy_v1|typed_v2|typed_v3`, independently of
`--condition-mode`. The default is **legacy_v1**, preserving the previous
policy/helper/repair text and method-specific whitespace/context assembly.
Method 2 requires `typed_v3`, which retains typed_v2 shared API and existing
method templates and adds whole-candidate generation/repair templates.
**typed_v2** adds typed parameter/return descriptions, precise primitive and
predicate semantics, configured Higher tolerance, and interface examples.
Policy requests require an execution entry; helper requests require only the
requested definition; Method-1 repairs follow the source unit's role.

For example, an offline generation diagnostic using the existing scripted inputs:

```bash
.venv/bin/python -m synthesis.entry.run_cap --method cap_method_1 --prompt-version typed_v2 --condition-mode coordinate_predicates --generate-only --responses-file synthesis/cap/examples/generation_only.json
```

The versioned resources are [legacy_v1.json](prompts/legacy_v1.json) and
[typed_v2.json](prompts/typed_v2.json), and [typed_v3.json](prompts/typed_v3.json). Add a new version for future prompt changes;
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

## Shared checker, final-output audit, and Method 2

`checker.check_bundle` returns scoped, source-located compliance reports under an
explicit `DSLContract`. Method 1 retains local recursive repair; Method 2 checks
and revises the complete source module, without per-helper generation requests.
Both require a complete pass before execution. Controller and observation APIs
are unchanged, and both condition modes remain available.

Baseline executes through Google's unchanged core. The compatibility layer
records `final_output.json` and immutable output snapshots. After the attempt,
`audit.audit_run` writes `dsl_compliance.json` independently of physical success;
the bounded report reads it. Final source/binding normalization is analysis-only.
Incomplete generation is `incomplete`, absent source is `not_available`, and
unrepresentable final bindings are `unsupported`; none counts as a pass.
`--no-baseline-audit` disables postprocessing for diagnostics, without changing
model calls or actions. `--audit-timeout` defaults to 10 seconds outside the
active trial wall cap. The original result/checkpoint files are not rewritten.

## Parallel batches

Provide a JSON file with `run_config` and `model_config` fields accepted by
[config.py](config.py); credentials remain in the environment. Set
`run_config.prompt_version` to `typed_v3` to compare all three methods. For example:

```bash
.venv/bin/python -m synthesis.entry.run_cap_batch --config experiment.json --initial-state saved/initial_state.npz --trials 5 --max-workers 3 --condition-modes dsl_only coordinate_predicates --schedule-seed 20261018 --output-dir runs/cap-comparison/my-batch
.venv/bin/python -m synthesis.entry.run_cap_batch --output-dir runs/cap-comparison/my-batch --resume
```

The first command schedules 30 trials (three methods, two modes, five trials).
The parent creates an immutable manifest/hash and copies the initial snapshot,
then dispatches balanced randomized trial groups to isolated subprocesses.
Each job owns its environment, recorder, logs, and latest link. The parent owns
atomic `schedule.json`, `summary.json`, `report.md`, and worker-count metadata.
Queue time is separate from active trial time. Reports include per-trial tokens,
physical outcome, DSL status, calls, timings, and failure reasons.

Resume runs only pending jobs; completed, failed, and interrupted jobs are never
silently replaced. An existing live worker or another active scheduler prevents
duplicate dispatch. Worker failures remain in the report denominator. The parent
can audit saved baseline output after a worker crash without rerunning generation.
Scripted responses can be supplied with `--responses-file` for offline diagnostics.

Revision 2 validation: 87 CaP tests passed. A six-combination real-MuJoCo scripted
preflight with three workers preserved exact initial states and all 19 controller
actions, and all six videos matched the recorded 20 states. Every DSL check
passed; `goal_failed` is expected for the partial script. The preflight artifacts
are under `runs/cap-revision2-preflight/20261018-214704/`; live experiment results
are reported separately.

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
by default in **deferred** mode: record full snapshots during rollout, then render
videos in separate workers. `--video-mode inline|deferred|off` selects the behavior.
Rendering requires ffmpeg/ffprobe. Legacy explicit `--save-video` selects inline
when `--video-mode` is absent; `--no-save-video` selects off. Contradictory explicit
settings are rejected.

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
Unstack currently retains a 60-second end-to-end cap; Revision 5 removes this
special case so that every task uses the same defaults. Trial effort is also bounded by
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
parallel output directories remain trial-specific. The reusable batch CLI is
`synthesis.entry.run_cap_batch`; the earlier Batch C scheduler remains preserved
under its run artifacts.

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

## Revision 2 DeepSeek comparison

A Revision 2 comparison on 2026-10-18 used `deepseek/deepseek-v4.1-flash`,
reasoning enabled, **16,384 output tokens per request**, and `typed_v3` across
three methods and both condition modes. Five fresh policy requests per group
produced 30 trials on the same saved three-block Stack scene, with three
isolated workers and a recorded balanced randomized dispatch order.

| Method | Condition mode | Physical successes | Complete DSL passes | Reported tokens |
| --- | --- | --- | --- | --- |
| `cap_google_baseline` | `coordinate_predicates` | 1/5 | 2/5 | 83,672 |
| `cap_google_baseline` | `dsl_only` | 0/5 | 2/5 | 80,402 |
| `cap_method_1` | `coordinate_predicates` | 2/5 | 2/5 | 90,779 |
| `cap_method_1` | `dsl_only` | 1/5 | 3/5 | 79,975 |
| `cap_method_2` | `coordinate_predicates` | 0/5 | 0/5 | 91,330 |
| `cap_method_2` | `dsl_only` | 1/5 | 1/5 | 89,576 |

All trials used temperature 0, a 600-second active wall cap, 12-call and
50,000-reported-token trial budgets, 10,000 execution/query ticks, 10,000 policy
simulation steps, depth 12, and up to five Method-1 local or Method-2 whole-candidate
attempts. Request timeout was 50 seconds with one transport retry available.
No trial exhausted a shared resource/wall budget; per-response truncation and
per-primitive exhaustion are separate outcomes.

There were **five physical successes, twenty response-length failures with no
returned code content, four Move nonconvergences, and one Release
nonconvergence**. The failed primitives each used their unchanged 50-step budget;
Move residuals were approximately 0.033 m and the Release residual was 0.00398 m.
These residuals alone do not uniquely establish the physical cause.

The ten complete programs passed the DSL contract. Other DSL outcomes were
14 incomplete constrained generations and six unavailable baseline outputs.
All 30 trials made exactly one policy request: no helper-generation, repair,
or transport-retry calls occurred. Thus this live sample does not isolate a
benefit of either repair strategy; those mechanics were exercised in scripted
regression tests. This is a single-scene pilot, not a benchmark-wide estimate.

Total usage was **515,734 reported input/output tokens**, including 457,078
reported reasoning tokens counted once. Full initial simulator snapshots,
model settings, request/prompt hashes, token totals, exclusive timing totals,
and all 30 videos were audited. Every video matched its trajectory state count.

See the [full report](../../runs/cap-revision2-deepseek/20261018-215345/report.md),
[failure analysis](../../runs/cap-revision2-deepseek/20261018-215345/analysis.md),
and [policy/check/video index](../../runs/cap-revision2-deepseek/20261018-215345/artifacts.md).
Every trial row includes `# of tokens used`; no usage is missing in this batch.

## Haiku 5.5 follow-up comparison

A follow-up comparison on **2026-10-19 UTC** used
`anthropic/claude-haiku-5.5`, reasoning enabled with explicit **medium effort**,
and the same **16,384-token output cap**. All 30 trials reused the DeepSeek
batch's exact full initial snapshot, instruction, typed_v3 prompt package,
three-worker dispatch order, and per-trial resource limits. Temperature was
omitted because the Haiku route did not advertise support; DeepSeek used 0.
This is a model/configuration comparison, not a model-only controlled ablation.

| Method | Condition mode | Haiku physical success | Prior DeepSeek success | Haiku tokens |
| --- | --- | --- | --- | --- |
| `cap_google_baseline` | `coordinate_predicates` | 3/5 | 1/5 | 39,115 |
| `cap_google_baseline` | `dsl_only` | 2/5 | 0/5 | 48,577 |
| `cap_method_1` | `coordinate_predicates` | 5/5 | 2/5 | 30,239 |
| `cap_method_1` | `dsl_only` | 2/5 | 1/5 | 28,259 |
| `cap_method_2` | `coordinate_predicates` | 4/5 | 0/5 | 46,873 |
| `cap_method_2` | `dsl_only` | 3/5 | 1/5 | 62,668 |

Haiku produced **19 physical successes**, with **six Move failures, two Pick
failures, two Release failures, and one no-witness get failure**. There were no
output-limit failures. The 39 model calls comprised 30 policy generations,
seven baseline helper generations, and two whole-candidate repairs. Method 2
DSL-only trials 2 and 4 each repaired `Expected position, got vector` and then
completed Stack. No failed trial was replaced or rerun.

Total usage was **255,731 input/output tokens**, including 128,723 reported
reasoning tokens counted once. All 39 responses provided usage and cost; the
provider-reported total cost was **$0.080496**. Initial policy prompts matched
the corresponding DeepSeek requests byte-for-byte. All 30 full initial
simulator snapshots, request hashes, token totals, timing sums, and videos were
validated; video frames matched recorded states.

Final DSL results are **29 pass and one unsupported**. Baseline coordinate trial
3 exposed CAP-AUDIT-1: the old recorder omitted a global helper dependency inside
a query lambda, causing a false unresolved-helper diagnostic. The physical
no-witness failure is unchanged. The original audit is retained, and the
corrected audit reports insufficient captured binding information, not a definite
source violation. Capture now includes nested code-object globals; regression
tests cover both complete new captures and incomplete legacy captures. No paid
request or physical rollout was repeated for the correction. All 88 CaP tests
pass after this reporting fix.

These are five fresh policy samples per method/mode on one fixed scene.
They do not establish general performance on other scenes or tasks.

See the [full report](../../runs/cap-revision2-haiku55/20261019-021933/report.md),
[analysis](../../runs/cap-revision2-haiku55/20261019-021933/analysis.md), and
[source/check/video index](../../runs/cap-revision2-haiku55/20261019-021933/artifacts.md).

## Configurable failure recovery

All three methods support `--recovery` (default off) and optional `--recovery-config path.json`.
Every revised candidate starts from the same full initial snapshot, with a fresh execution context.
Up to four recovery rounds follow the initial rollout. The shared deterministic feedback module
adds current failure evidence and short summaries of earlier failures to a versioned prompt
supplement. Baseline remains free of a pre-execution DSL gate.

Batch configuration can include `"recovery_variants": [false, true]` alongside `run_config`
and `model_config`. With three methods, one condition mode, and five trials this schedules
30 trials. Recovery attempts stay inside their original trial and all usage is accumulated.
See [configuration, budgets, evidence, and logging](docs/revision_4_failure_recover.md#implementation-and-validation).

The [Revision 4 recovery comparison](docs/revision_4_recovery_report.md) records 120 Haiku trials (20 per setting) and 300 MiniMax trials (50 per setting), with separate model results, per-trial token usage, recovery outcomes, and artifact audits. Those experiments used the implementation validated by 124 CaP tests; GPU scheduling validation is recorded in Revision 4.1.

## Rate-limited experiment batches

`create_batch(..., transport_control={"state_path": "/absolute/path/requests.json",
"interval_seconds": 6.0, "request_limit": 800})` optionally shares request-start
pacing and a wire-request ledger across workers. No credentials or prompts enter
this ledger. Waiting remains inside the existing request/trial deadlines.

`execute_batch(..., dispatch_gate=callback)` optionally checks admission before
starting each trial. A refusal drains active workers and leaves unstarted jobs
pending; `resume=True` skips completed trials. A quota-aware caller reserves the
remaining model-call allowances of active trials and the candidate trial through
`request_control.dispatch_reservation`, using the current provider quota. These
controls are disabled unless explicitly configured and do not change generation,
DSL checking, recovery prompts, controller behavior, or existing model budgets.

## GPU-scheduled parallel experiments

[Revision 4.1 Section 1](docs/revision_4_1_gpu_scheduling.md#1-gpu-scheduling) is implemented.
GPU scheduling is optional and works with all three methods and recovery on/off.
[Revision 4.2](docs/revision_4_2_fix_primitive_noncovergence.md) implements configurable primitive prompts and failure diagnostics; its 220-trial comparison is complete. Historical-program injection and other proposals are retained in the [deferred backlog](docs/revision_4_unimplemented.md).

From `roboverify/`, configure the existing simulator environment as described in
[root AGENTS.md](../../../AGENTS.md), then inspect physical identities and verified
EGL mappings:

```bash
.venv/bin/python -m synthesis.cap.gpu_scheduler --inventory
```

Choose GPUs you are permitted to use. Replace the UUID placeholders below with
those devices, and save this as `batch.json`. `initial_state` should point to a
saved full-state archive. Measure worker memory before setting headroom and
worker allowances; 2,048/1,024 MiB below are integration-test examples.

```json
{
  "run_config": {
    "task": "stack",
    "num_blocks": 3,
    "prompt_version": "typed_v3",
    "save_video": true
  },
  "model_config": {
    "provider": "openrouter",
    "model": "anthropic/claude-haiku-5.5",
    "max_output_tokens": 16384,
    "extra_body": {"reasoning": {"effort": "medium"}}
  },
  "initial_state": "/absolute/path/to/initial_state.npz",
  "methods": ["cap_google_baseline", "cap_method_1", "cap_method_2"],
  "modes": ["dsl_only"],
  "recovery_variants": [false, true],
  "trials": 5,
  "max_workers": 2,
  "schedule_seed": 0,
  "gpu_config": {
    "gpu_policy": "auto",
    "gpu_allowlist": ["GPU-REPLACE-WITH-ALLOWED-UUID-1", "GPU-REPLACE-WITH-ALLOWED-UUID-2"],
    "max_workers_per_gpu": 1,
    "min_free_memory_mb": 2048,
    "worker_memory_mb": 1024,
    "gpu_poll_interval_seconds": 2,
    "gpu_sample_window_seconds": 10,
    "admission_timeout_seconds": 60,
    "reservation_directory": "/tmp/roboverify-cap-gpu-reservations"
  }
}
```

Use the **same reservation directory** for cooperating batches on this host.
Different directories do not coordinate worker slots. Load is an admission
signal, not permission to use someone else's GPU. Auto mode requires working
NVML and a verified mapping; `fixed` accepts exactly one explicit UUID and logs
any monitoring limitation. Both modes enforce shared worker slots. The installed
EGL mapper supports the shim's eight-device enumeration prefix and requires the
persistent device UUID extension; it fails explicitly on unsupported mappings.

```bash
.venv/bin/python -m synthesis.cap.batch --config batch.json --directory runs/cap-my-batch
.venv/bin/python -m synthesis.cap.batch --directory runs/cap-my-batch --resume
```

This example schedules 30 trials. Every trial owns a fresh worker process, and
its GPU assignment remains fixed through recovery. Existing per-trial budgets
and shared request controls still apply. To retain rate limiting, include the
existing `transport_control` object in the JSON. API callers can also retain
`execute_batch(..., dispatch_gate=...)` for quota-aware admission. Keep credentials
in the provider's environment variable, outside batch configuration.

When no GPU qualifies, trials wait outside their active budgets. After the
admission timeout, active workers finish and unstarted trials remain pending.
Resume uses the immutable manifest, skips finished trials, and refuses to launch
a duplicate while a prior worker remains live. Worker/mapping/launch failures
remain in the results; they are not automatically replaced.

The batch saves `gpu_samples.jsonl`, `execution_metadata.json`, per-worker
`gpu_probe.json`, and assignments in schedules, trial summaries, and runner
configuration. Existing reports retain queue/active time, phase timings, outcome,
and token accounting. Compare throughput and failures when increasing workers;
GPU utilization alone is insufficient. See [CaP experiment guidance](AGENTS.md).

For a no-model diagnostic, add a `responses` array of scripted code strings to
the JSON. The opt-in hardware test additionally checks serial/parallel trajectory
and video parity; its environment variables and results are documented in
[Revision 4.1 validation](docs/revision_4_1_gpu_scheduling.md#16-validation).


## Revision 4.2 primitive prompt and feedback ablations

All methods accept optional `primitive_prompt` and `primitive_feedback` objects
inside batch JSON `run_config`; see [the configuration and verified contracts](docs/revision_4_2_fix_primitive_noncovergence.md).
All new Booleans default to false. General API supplements apply to initial,
recovery, helper, and existing static-repair prompts; they do not change the
controller or DSL checker. Actual primitive diagnostics are appended only to
recovery-policy requests following a primitive failure.

The four new prompt blocks and their heading are stored in
[prompts/primitive_v1.json](prompts/primitive_v1.json).
[primitive_prompts.py](primitive_prompts.py) loads that resource, selects enabled
blocks, substitutes controller values, and records provenance. The resource move
preserves the exact rendered text and existing supplement hashes; the base
`legacy_v1`, `typed_v2`, and `typed_v3` packages remain unchanged.

The reproducible 220-trial design is provided by:

```bash
.venv/bin/python -m synthesis.cap.revision42_experiment --directory runs/cap-revision42/NEW_BATCH
.venv/bin/python -m synthesis.cap.revision42_experiment --directory runs/cap-revision42/EXISTING_BATCH --resume
.venv/bin/python -m synthesis.cap.revision42_report --directory runs/cap-revision42/EXISTING_BATCH --videos
```

`--model` reruns the same design with another model; only the model identifier
changes in the job definitions. A model without the `:free` suffix uses a fixed
recorded request cap (`--paid-request-limit`) instead of the free daily-quota
lookup. Name its credential variables with repeated `--credential` options and set
`--request-interval` and `--video-mode` explicitly. Repeated
`--variant PROFILE=METHOD[,METHOD]` options select an extension batch, such as
Baseline ablations or the `only_prompt` profile, which keeps the four prompt
additions and disables all three feedback enrichments. `--accumulates-with`
records the batch it complements. The report accepts repeated `--directory`
options: batches with the same model accumulate, different models are shown side
by side, and `--output` holds the combined analysis. `--interpretation` inserts
the retained hand-written sections.

Use the existing simulator environment setup. The experiment entry point uses
the declared Stack snapshot and GPU UUID allowlist from its frozen configuration;
inspect/adapt these explicitly for another host or scene. The four authorized
credential slots are named `OPENROUTER_API_KEY`, `OPENROUTER_API_KEY_2`,
`OPENROUTER_API_KEY_3`, and `OPENROUTER_API_KEY_4`. Store their values outside the
repository. The shared request ledger records environment names only, rotates
requests under individual pacing/caps, and reserves remaining active-trial call
allowances before admitting additional trials. Quota-limited admission leaves
pending trials resumable, without replacing failed trials.

`create_batch(..., experiment_variants=[...])` supports named prompt/feedback
variants with optional method subsets. Variant overrides are restricted to those
two configuration groups. Its immutable manifest records the balanced dispatch
order and unique identities; the parent report separates profiles.

The main/ablation [report](docs/revision_4_2_report.md) includes initial/final
success, primitive nonconvergence, conditional recovery transitions, per-trial
tokens and artifact links, plus [plots](docs/plots/). Scripted preflight runs are
explicitly excluded from the live-trial denominator.


The completed MiniMax Revision 4.2 batch has 20 trials in each of 11 settings. Full-treatment
success is Baseline 18/20, Method 1 18/20, and Method 2 20/20; ablation results are
in the linked report. Extension batches add Baseline ablations and `only_prompt`
for MiniMax (140 trials), the Claude Haiku 5.5 rerun (360 trials in 18 settings),
and a Qwen3.8 27B `only_prompt` stress test (60 trials); all share that report. All 446 state restores and videos and 226 delivered recovery
payloads passed the artifact audit. One full Baseline trial stopped because its
required feedback plus metadata exceeded the unchanged 8,192-byte cap; this
limitation and all failures remain reported. The final interpretation is also
saved in `final_interpretation.md` (the combined MiniMax/Haiku version is in
`runs/cap-revision42/20261021-combined-minimax-haiku55/`) for use after
regenerating numerical tables and plots.


## Deferred video rendering

New experiments default to `video_mode: "deferred"` across all three methods.
Set it explicitly when adapting an older batch JSON that has `save_video: true`.
See [Revision 4.2.1](docs/revision_4_2_1_deferred_video_rendering.md).

```json
{
  "run_config": {"video_mode": "deferred"},
  "render_workers": 2,
  "render_timeout": 600
}
```

These are additions to the normal batch configuration. `max_workers` caps the
combined trial and render worker count. `render_workers` additionally caps
rendering processes; both pools share GPU reservations and the explicit allowlist.
Batch execution drains queued videos after trials finish, while marking experiment
completion separately. Recovery may continue while earlier attempts render.

Single-trial CLI runs drain their queue after policy execution. Unstack only
enqueues videos, preserving its 60-second execution limit; drain its queue
separately using the command below. Python `run()`
only records/enqueues jobs; call `drain_queue(result["run_dir"])` when convenient.
A saved queue can also be processed without new model requests or policy rollouts:

```bash
.venv/bin/python -m synthesis.cap.deferred_video --directory runs/PATH_TO_BATCH_OR_RUN --max-workers 2
.venv/bin/python -m synthesis.cap.deferred_video --directory runs/PATH_TO_BATCH_OR_RUN --retry-failed
```

Use the normal simulator environment. Batch queues reuse the manifest GPU policy;
standalone queues use the caller's existing GPU environment. `--timeout` bounds
each render worker (default 600 seconds). Interrupted jobs are resumable;
failed jobs require `--retry-failed`. Existing completed videos are not overwritten.
Only queued trajectories with captured render metadata are accepted; older archives
are not silently treated as exact visual replays.

Each rollout retains `render_job.json` (trajectory/render identity),
`render_status.json` (mutable video outcome, attempts, timing, assignment), and
`frame_index.jsonl`. The queue root has `video_summary.json`. Policy `result.json`
remains immutable; the bounded report and batch summaries resolve current video
status through its `status_path`. A video failure never changes task success or
triggers recovery. Deferred render time is separate from policy execution budgets.
