# Revision 4.1: recovery history and CaP GPU scheduling

- **Status:** Section 1: **not implemented yet**. Section 2: **implemented** (optional CaP GPU scheduling).
- **Goal:** let the synthesizer compare earlier failed programs with the latest program, so it can recognize recurring mistakes and avoid undoing useful corrections.
- **Scope:** the shared recovery module used by Baseline, Method 1, and Method 2.
- **Implemented scheduling scope:** GPU selection and parallel-trial resource scheduling entirely within `synthesis/cap/`; see Section 2.
- **References:** [Revision 4](revision_4_failure_recover.md), [MiniMax Baseline trial 6 case study](case_study/revision_4_failure_recovery_cases/minimax_baseline_dsl_only_trial_006.md).

## 1. Recovery history

**Status: not implemented yet.** Existing recovery-history prompts remain unchanged.

### 1.1 Motivation and evidence

Current recovery prompts contain the latest failed program and selected execution evidence, plus short summaries of older failures. Older summaries include a candidate hash, status, reason, and goal/compliance outcomes, but do not include the corresponding source code. A hash identifies an artifact; it does not let the model reconstruct that program.

The case study shows a sequence of unsuccessful revisions: an impossible selection condition, an uninitialized variable, unsuccessful placement, and two different errors in calculating the Z target for `Move`. The last two rounds both report Move nonconvergence, but their formulas and targets differ. This supports investigating repeated or related mistakes; it does not establish that all repeated failure labels have the same root cause or that missing historical code caused the failures.

Including historical programs is a reasonable next change because it gives the model the missing connection between an earlier failure and the code that produced it. Whether this reduces recurrence or improves recovery success remains an experimental question. Extra source can also increase input cost and distract from the latest failure.

### 1.2 Proposed information contract

| Information in a recovery request | Current Revision 4 | Proposed historical-source profile |
| --- | --- | --- |
| Original task, API, and condition-mode rules | Included | Included |
| Latest failed executed program and attributable helper bindings | Included | Included |
| Latest failure's selected events, keyframes, and goal evidence | Included | Included |
| Earlier failed programs that actually entered rollout | Not included | Include complete executed-source bundles |
| Earlier failure status, reason, and goal/compliance outcomes | Included as summaries | Retain and associate with the corresponding source bundle |
| Earlier failure locations | Not guaranteed in older summaries | Include source unit/checkpoint and line when recorded |
| Full earlier trajectories, videos, previous prompts, and model reasoning | Not included | Keep in artifacts; do not replay them into the prompt |

- Include all earlier failed rollout attempts from the **same trial**, in chronological order. At recovery round 4, the prompt contains the failed programs from rounds 0–3, with round 3 receiving the latest detailed evidence.
- An attempt that began top-level execution but failed before its first primitive still counts, such as the `NameError` in trial 6. A generated candidate rejected before rollout is not a historical executed program.
- Associate each attempt with its round, candidate identity, source/checkpoint identities, observed failure, failure location, and terminal task/compliance results. Keep observed failures separate from inferred diagnoses.
- Retain complete source units containing executed code, including branches that were not taken. Do not reconstruct a program only from executed statement fragments.
- Preserve helper definitions and the bindings relevant to execution. For Baseline, include attributable execution-time source versions when a later helper definition would otherwise hide an earlier executed version. Do not merge different helper versions into a fictitious single executable. This debugging record does not change the final-output-only DSL compliance metric.
- Identical source units may be serialized once in an inline source registry and referenced by multiple attempts. Every referenced body must be present in that same prompt; a hash or filesystem path alone is insufficient.
- Keep repeated candidates in the attempt history even when their source bodies are deduplicated. Different outcomes must remain attributable to their respective rollouts.

### 1.3 Configuration and prompt structure

Add a configurable recovery-history profile so the original behavior remains available for ablation. The following names are proposed, not existing configuration fields:

| Proposed setting | Meaning |
| --- | --- |
| `recovery_history_mode = summaries_only` | Preserve Revision 4: latest source and evidence, earlier failure summaries |
| `recovery_history_mode = executed_programs` | Latest source and evidence, plus every earlier failed rollout's source bundle and associated failure summary |

Keep `summaries_only` as the compatibility default until the comparison is evaluated. The history setting applies only when recovery is enabled and must work through the same shared API for all three methods. Generation workflows and their existing static checks remain unchanged.

Suggested request structure:

```text
ORIGINAL TASK, API, AND CONDITION RULES
{existing_initial_prompt}

RECOVERY INSTRUCTIONS
Revise the policy to complete the original task.
Every replacement starts from the same restored initial state.
Historical programs and trajectories are debugging evidence, not executable state.
Compare the latest program with earlier failed programs before revising it.
Use recorded evidence to assess whether an edit repeats an earlier mistake
or removes a useful correction. Do not infer a shared cause from a failure label alone.
Return a complete replacement policy under the existing output rules.

EXECUTED SOURCE REGISTRY
{complete_source_units_and_attributable_binding_versions}

EARLIER FAILED ATTEMPTS — CHRONOLOGICAL
{round, source_references, bindings, failure_status, failure_reason,
 failure_source_location, task_outcomes, dsl_compliance, duplicate_identity}

LATEST FAILED ATTEMPT
{latest_round_and_source_references}
{latest_selected_execution_evidence_and_goal_results}
```

The latest program appears once in the source registry and is explicitly identified. Earlier history is placed before the latest detailed evidence so the current failure remains close to the output instruction. Do not request hidden reasoning or add a new diagnosis-generation model call. Any optional source diff must be deterministic and supplemental to complete source, not a substitute for it.

### 1.4 Context limits and accounting

- Retain the **16,384 output-token cap per request** for the initial comparison. This change primarily increases input size; it does not by itself justify a larger output cap.
- Keep the current **8,192 feedback evidence bound** for selected evidence and failure summaries. Historical source bodies need a separately configurable allowance, alongside the existing current-source context. Measure saved source sizes before choosing that allowance; no new numeric source limit is settled by this document.
- Count the entire serialized request—including source registry, histories, task/API text, and formatting—against the model context window and per-round/cumulative token reservations. Do not treat historical code as free context.
- Record the counting method. The current implementation uses a conservative UTF-8 byte upper bound for reservations; this is not a measured provider token count.
- Fit required source and failure associations first, then add optional execution evidence within its existing budget. Deduplicate identical source bodies without dropping attempt records.
- If all required historical source cannot fit, record an explicit history/context-budget stop. Do not silently truncate code or fall back to summaries while labeling the request `executed_programs`.
- Initially retain the existing recovery count and overall effort limits. If source inclusion makes those limits bind more often, report that effect and evaluate any budget increase as a separately configured comparison.

### 1.5 Planned implementation and validation

| Area | Planned change | Acceptance check |
| --- | --- | --- |
| Failure collection and history storage | Preserve immutable per-attempt executed-source bundles, bindings, and failure locations | Later edits or helper redefinitions cannot mutate older records |
| Shared prompt builder | Serialize the chosen history profile, inline source registry, and attempt-to-source associations | The fourth recovery request includes source for rounds 0–3; every reference resolves inside the prompt |
| Baseline source attribution | Preserve relevant source/binding versions observed during execution | A final helper version cannot overwrite evidence of an earlier executed version |
| Budget accounting | Include historical source in request/context and round/trial reservations | Oversized history stops explicitly; code is never silently truncated |
| Logging | Save history mode, included rounds, source hashes, bindings, serialized-size accounting, and exact delivered payload | Prepared history matches the actual next model request |
| Compatibility | Retain the existing summary profile and recovery-off behavior | No historical-source injection when the feature is not selected |

Use synthetic regression tests for history immutability, source deduplication, repeated candidates, helper-version attribution, pre-primitive exceptions, missing required artifacts, and context-budget exhaustion. Keep missing evidence explicit; do not invent code, bindings, or failure locations.

### 1.6 Proposed experiment

Compare `summaries_only` and `executed_programs` for all three methods with recovery enabled. Hold the model, reasoning effort, output cap, task, condition mode, initial state, and maximum recovery count fixed. Run the batch settings in parallel under the shared request limits.

For a focused recovery comparison, branch both profiles from the same saved initial failed program and its exact initial state. Record this as a recovery comparison from a shared candidate, separate from a fresh end-to-end generation experiment. It controls the starting policy without claiming later stochastic generations are identical.

| Measure | Purpose |
| --- | --- |
| Final task success and recovery rounds | Determine whether historical source improves recovery |
| Recurrence of an already observed code defect | Test the motivation directly; require source/evidence review, not just matching failure labels |
| Reintroduction of a previously corrected defect | Detect regressions between revisions |
| Repeated source candidates | Distinguish identical-program retries from different programs with similar failures |
| Input/output tokens, calls, and elapsed time | Quantify the additional cost |
| History/context/round-budget stops | Detect whether the larger input reduces the opportunity to recover |
| DSL compliance and task outcomes separately | Preserve the distinction between valid source and successful execution |

The trial 6 case study is a diagnostic example, not sufficient evidence of an improvement. No implementation, rerun, or experiment is authorized by this planning document alone.

## 2. GPU scheduling

**Status: implemented.** GPU admission is optional and applies to all three
methods, with recovery enabled or disabled. It is implemented entirely within
`synthesis/cap/`. Section 1 remains **not implemented yet**.

### 2.1 Scope and architecture

MuJoCo physics and primitive control remain on CPU; GPUs provide offscreen
rendering. The scheduler targets trial throughput across model requests,
CPU simulation, rendering, encoding, and artifact I/O. It does not change policy
prompts, generation/checking behavior, recovery feedback, task semantics, or
primitive controllers.

| File | Implemented responsibility |
| --- | --- |
| [gpu_scheduler.py](../gpu_scheduler.py) | NVML monitoring, UUID-to-EGL mapping, shared atomic reservations, process liveness, worker verification, and rendering probe |
| [batch.py](../batch.py) | Existing isolated trial queue, total worker limit, GPU admission, resume, telemetry, and `python -m synthesis.cap.batch` entry point |
| [batch_worker.py](../batch_worker.py) | Claim assignment and verify rendering before importing trial machinery; retain the assignment across recovery attempts |
| [config.py](../config.py) | Validated optional `GPUConfig`, separate from policy and recovery configuration |
| [test_gpu_scheduler.py](../test_gpu_scheduler.py) | Mocked admission/lifecycle tests and opt-in real simulator/rendering parity test |
| [AGENTS.md](../AGENTS.md) | Default parallel experiment execution within authorized resource limits |

```mermaid
flowchart LR
    Q[Immutable trial manifest / pending queue] --> A[Parent admission]
    N[NVML samples + explicit UUID allowlist] --> A
    R[Shared locked reservations] <--> A
    A -->|Reserve UUID / EGL index / slot| W[Fresh worker process]
    W --> P[Verify mapping and render probe]
    P -->|GPUS fixed before simulator imports| T[Original method + optional recovery]
    T -->|Outcome, timings, tokens, artifacts| B[Batch report]
    T -->|Confirmed process exit| R
    A -->|No eligible GPU| Q
```

No simulator package, primitive API, CFG module, shared experiment reporter,
existing entry script, vendored Google core, or dependency is changed.
Omitting `gpu_config` preserves the existing batch execution path.

### 2.2 Configuration and admission

Pass `gpu_config` to `create_batch(...)`, or include it in the JSON configuration
for the new batch CLI. See the [README usage example](../README.md#gpu-scheduled-parallel-experiments).

| Setting | Behavior |
| --- | --- |
| `gpu_policy` | `auto` (default): choose within the allowlist using reservations and recent monitored load; `fixed`: exactly one explicit UUID |
| `gpu_allowlist` | Required nonempty list of unique physical `GPU-...` UUIDs; never expanded automatically |
| `max_workers` | Existing batch-level field: maximum active trial processes in this batch |
| `max_workers_per_gpu` | Maximum cooperating workers per GPU, including starting workers; default 1 |
| `min_free_memory_mb` | Memory headroom after all worker reservations; required positive in auto mode |
| `worker_memory_mb` | Conservative memory allowance reserved for each worker before launch; required positive in auto mode |
| `gpu_poll_interval_seconds` | NVML sampling interval; default 2 seconds |
| `gpu_sample_window_seconds` | Recent window for mean utilization and minimum free memory; default 10 seconds; use available samples during warm-up |
| `reservation_directory` | Required shared host-local directory used by all cooperating batch schedulers |
| `admission_timeout_seconds` | Default 60 seconds without admission; drain active workers and return with unstarted trials pending, then resume explicitly |

Eligible devices need a verified EGL mapping, a successful latest NVML health
sample, a free slot, and sufficient memory. Admission subtracts the **full**
allowance of every reserved worker, even when NVML already sees some of its
allocation; this deliberately over-reserves instead of spending the same
headroom twice. Within eligible devices, prefer fewer reservations, then lower
mean utilization, then more free memory. Different cooperating policies honor
the stricter active per-GPU slot limit.

NVML is accessed through the installed driver library, without installing a
Python package. Samples include UUID, monitoring index, PCI bus ID, free/used
memory, utilization, and available compute/graphics process memory. The health
field means the memory/utilization/PCI queries succeeded; it is not a full
hardware fault diagnosis. Unavailable process counters are labeled explicitly.

In auto mode, unavailable monitoring prevents new admission. Existing workers
continue; they are not killed because monitoring or external load changed.
Fixed mode can proceed when monitoring itself is unavailable, but records this
limitation and still requires an independently verified EGL mapping and a free
reserved slot. A device reported unhealthy is not admitted. Fixed fallback
cannot enforce a memory threshold without monitoring.

Tune memory allowances and worker counts from measurements. The tested values
are diagnostic settings, not universal production defaults. Slots coordinate
participating schedulers; they do not reserve hardware against other lab users.

### 2.3 Atomic reservations and failure handling

`reservation_directory/scheduler.lock` protects a host-specific
`reservations.json`. Each reservation records the trial's absolute job path,
UUID, EGL index, slot, memory allowance, parent identity, worker identity, state,
and admission sample. Linux process identities combine PID, process start ticks,
and boot ID; stale heartbeats alone never reclaim a live worker's slot.

- Reserve before spawn, including the startup interval before visible allocation.
- The worker claims its token before importing or using the rendering stack.
- Launch failure releases the reservation and retains a failed trial record.
- Confirmed worker exit or crash releases its slot; failed trials are not retried
  under new identities.
- If a parent dies, a live recorded worker keeps its slot. A child launched before
  its PID was recorded is detected by its exact job path, or claims the token
  itself. A child whose reservation was reclaimed cannot initialize the GPU.
- Resume refuses duplicate dispatch while a prior worker is live. Once that
  worker exits, resume collects its artifacts or records interruption and runs
  only previously pending trials.
- File permissions, ambiguous mapping, or foreign-host reservation directories
  fail explicitly rather than silently assigning GPU 0.

The existing batch lock prevents two parents from dispatching the same manifest.
The shared GPU lock coordinates different batches. Request pacing and wire-call
limits remain separate controls and continue to apply inside each worker.

### 2.4 Device mapping and initialization

The installed `mujoco_py` shim enumerates the first eight EGL devices. The mapper
uses the same enumeration prefix and `EGL_EXT_device_persistent_id` to match each
binary device UUID to its physical NVML UUID. Missing, unsupported, or ambiguous
mappings fail before a policy trial. This implementation intentionally does not
infer EGL identity from CUDA visibility or the monitoring index.

The distinction matters on the tested server: monitoring GPU 6 maps to EGL 4,
and monitoring GPU 7 maps to EGL 5. These are recorded observations, not portable
hard-coded mappings.

The parent passes a private `env` dictionary to a fresh subprocess with `GPUS`
set to the verified EGL index. It does not mutate the parent's environment or
inherit a rendering context through fork. The worker verifies the mapping again
and performs a small MuJoCo offscreen rendering probe **before any model call**.
A failure produces `gpu_initialization_failed`, with zero model calls, in the
same trial's artifacts. Non-rendering test fixtures skip only the rendering probe.

The assignment remains fixed through the entire trial, including helper calls
and every recovery attempt. Each attempt retains the existing fresh execution
context and exact initial-state restore. Environment cleanup uses the existing
runner, and process exit releases the graphics resources.

### 2.5 Budgets, artifacts, and measurements

Resource admission occurs before the existing active trial budget starts.
Queue time remains separate. The batch hard deadline, model-call/token limits,
simulation/execution limits, and Unstack's 60-second limit remain unchanged.
The worker holds its slot during model requests because Baseline generation can
interleave with execution, and recovery alternates both stages.

| Artifact | Added scheduling information |
| --- | --- |
| `manifest.json` | Immutable GPU policy, allowlist, memory allowances, limits, and reservation path |
| `schedule.json`, each `job.json`, and trial summaries | UUID/EGL assignment, slot, admission sample, environment settings, and process identity |
| Each runner `config.json` → `config.batch_context.gpu_assignment` | Same assignment for all recovery attempts |
| `gpu_probe.json` | Worker identity, verified UUID/EGL mapping, effective environment, and rendered image dimensions/checksum |
| `gpu_samples.jsonl` | Timestamped device load, memory, identity, and available process measurements |
| `execution_metadata.json` | Peak active workers, pause reasons, sampled device memory/load peaks, and per-worker CPU/RSS/I/O/GPU-memory observations |
| Existing trial artifacts/reports | Queue/active time, phase timings, retries, policy outcome, source attribution, videos, and token totals |

Device memory peaks include other processes. Per-worker memory observations use
NVML process attribution when available; all peaks are sampled, not exact
instantaneous maxima. CPU/RSS/I/O observations cover the worker process, not
all encoder descendants. Existing rendering and video-finalization timings help
interpret these measurements. Scheduling itself adds no model tokens.

### 2.6 Validation

The validation uses mocked GPU state, isolated subprocesses, scripted model
responses, and real MuJoCo rendering. No paid model requests are required.

The focused tests cover simultaneous reservations, startup memory accounting,
allowlist enforcement, recent-load windows, incompatible/missing mapping,
monitoring failure and fixed fallback, launch failure, worker crash, surviving
children, PID reuse, duplicate-trial refusal, parent-environment isolation,
resource-wait pause/resume, and zero-call worker initialization failure.

The opt-in integration test runs all three methods with recovery off/on in
serial and parallel batches from one saved three-block Stack state. It checks
primitive trajectories, exact restore, fixed assignment through recovery,
source checkpoints, independently decoded video frame counts, final outcomes,
and completed-trial resume without repeated requests.

Validation completed on 2026-10-20:

- Default CaP suite: **151 passed**, with the opt-in hardware test skipped.
- Focused GPU tests: **17 passed**, including resume of completed batches when
  GPU discovery is unavailable.
- Hardware integration: **passed** separately, with 12 trials and 18 rollouts
  across three methods, recovery off/on, and one/two workers. All 18 videos
  passed independent frame-count checks. State/action arrays matched exactly
  between serial and parallel execution; delivered source matched active
  checkpoint source. All six recovery-on trials restored the initial state and
  succeeded after one repair. The six recovery-off trials retained the scripted
  no-witness failure. These are expected test outcomes, not model-quality results.
- Completed-batch resume left worker results unchanged. Shared reservations were
  empty after both batches. No remote model requests were made.

| Diagnostic | One worker | Two workers |
| --- | ---: | ---: |
| Trials / rollouts | 6 / 9 | 6 / 9 |
| Observed peak active workers | 1 | 2 |
| First dispatch to last worker exit | 49.69 s | 30.14 s |
| Sampled maximum per-worker GPU memory | 69.89 MiB | 69.89 MiB |
| Sampled maximum worker RSS | 923.54 MiB | 919.68 MiB |

These measurements are one small scripted comparison on a shared server, with
other validation work running concurrently. They validate scheduling and
accounting; they do not establish optimal concurrency or production throughput.
The test used two explicit allowed GPUs, one slot per GPU, 1,024 MiB per-worker
reservations, and 2,048 MiB headroom. Larger worker counts require separate tuning.

Artifacts: [validation directory](../../../runs/cap-gpu-scheduling-validation/20261020-section2-final/),
[serial report](../../../runs/cap-gpu-scheduling-validation/20261020-section2-final/workers-1/report.md),
[parallel report](../../../runs/cap-gpu-scheduling-validation/20261020-section2-final/workers-2/report.md),
and [audit](../../../runs/cap-gpu-scheduling-validation/20261020-section2-final/audit.json).
Earlier fixture-development diagnostics are retained separately; the final
comparison uses the self-contained top-level scripted policy for all methods.

Reproduce focused tests from `roboverify/`:

```bash
.venv/bin/python -m unittest synthesis.cap.test_gpu_scheduler.GPUTests -v
```

For hardware validation, configure the existing MuJoCo runtime first, set the
following environment variables, then run the opt-in test. Use a fresh output
directory for new rollouts; an existing completed output is checked without
repeating trials.

```bash
export CAP_GPU_TEST_UUIDS=GPU-ALLOWED-UUID-1,GPU-ALLOWED-UUID-2
export CAP_GPU_TEST_SNAPSHOT=/absolute/path/to/three-block-stack-initial_state.npz
export CAP_GPU_TEST_OUTPUT=/absolute/path/to/new-validation-directory
.venv/bin/python -m unittest synthesis.cap.test_gpu_scheduler.GPUIntegrationTests -v
```
