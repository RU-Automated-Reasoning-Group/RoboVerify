# Revision 4.3: configurable recovery feedback and full executed-policy history

- **Status:** implemented and tested on 2026-10-22; no live experiment has been run with it. Section 8 records the implementation, validation, and deviations from the plan.
- **Goal:**
  1. Make every information item in the [Revision 4 recovery feedback](revision_4_failure_recover.md) independently configurable.
  2. Implement the deferred executed-program history (formerly [backlog Section 1.4](revision_4_unimplemented.md#14-historical-programs-and-the-editoutcome-ledger)) as a configurable history mode. This item moves here from the backlog.
- **Compatibility:**
  - Recovery off is unchanged.
  - The default configuration reproduces `cap-recovery-v1` prompts byte for byte.
  - Revision 4.2 primitive prompt/feedback switches keep their names and meanings.
- **Scope:** all three methods share the change. It touches only the shared feedback module and the recovery request.
  - Each method's generation, checking, and helper workflow stays the same.
  - Baseline gets no pre-execution DSL gate.
  - Every replacement still rolls out from the saved initial state.

## 1. Motivation

- **The current feedback is a fixed bundle.** Revision 4.2 could ablate only its three new primitive enrichments. Older items cannot be turned off, so their individual contribution is unknown: keyframes, goal clauses, supporting events and earlier-failure summaries.
  - At 8,192 bytes, the median feedback size was 8,057, so items also compete for space.
- **Earlier programs are absent.**
  - Each earlier failed attempt appears only as one line: round, source hash, status, a reason truncated to 140 characters, outcome flags and duplicate/changed flags.
  - In the Revision 4 review, 169 feedbacks contained earlier summaries without any earlier source.
  - Reviewed cases show defects being repaired and then reintroduced. Example: [MiniMax Method 1 trial 18](case_study/revision_4_failure_recovery_cases/minimax_method_1_dsl_only_trial_018.md), rounds 0 → 1 → 4.
  - The model cannot compare against a program it cannot see.
- **History affects only rounds 2–4.** The round-1 request has no earlier attempt.
  - Across the saved Revision 4 and 4.2 recovery requests, 379 of 810 had at least one earlier executed attempt.
  - Measured size: see Section 4.4.

## 2. Inventory of the current feedback

The table lists every item that the current shared module ([recovery.py](../recovery.py) `select_relevant_evidence` and `recovery_prompt`) places in a recovery request.

| # | Item | Current key / prompt section | Content |
| --- | --- | --- | --- |
| 1 | Latest outcome | `evidence.latest` | Round, candidate hash, status, reason (≤140 chars), physical goal, logical postcondition, DSL compliance, `duplicate_of`, `changed_from_parent` |
| 2 | Current executed source and bindings | `CURRENT EXECUTED SOURCE AND BINDINGS` | Source files that actually executed and, for Baseline, the execution-time binding record (policy + generated helper bodies) |
| 3 | Failure location | `evidence.last_source_location` | Last statement or Python line event: checkpoint, unit, line, statement, state index |
| 4 | Failure events | `evidence.failure_context` | Events from the last `primitive_start` onward: primitive arguments, controller phases and targets, `primitive_end` result, exception or query failure |
| 5 | Failed-query candidates | Inside `query_failure` events in item 4 | Up to 16 candidate bindings evaluated by a failed `get`, plus an omitted count |
| 6 | Keyframes | `evidence.keyframes` | Gripper and block XYZ at up to three states: initial, start of the last primitive, terminal |
| 7 | Goal clauses | `evidence.derived_goal_evidence` | Up to 16 task clauses evaluated on the terminal state (e.g. `ON_star(block_i, block_0)`) |
| 8 | DSL diagnostics | `evidence.dsl_diagnostics` | Checker diagnostics when the attempt's DSL audit failed |
| 9 | Supporting events | `evidence.supporting_context` | Recent predicate, query, primitive, controller, branch, loop and line events, newest first, within the remaining bytes (window: last 80 candidates) |
| 10 | Event counts | `evidence.event_counts` | Count of every logged event kind in the attempt |
| 11 | Scene metadata | `object_handle_to_physical_id`, `world_frame_units`, `higher_tolerance` | Handle-to-block mapping, units, Higher tolerance |
| 12 | Earlier-failure summaries | `evidence.earlier_failures` | Item 1's fields for every earlier failed attempt |
| 13 | Selection metadata | `selected_event_ids`, `omitted_event_count`, `token_counting` | Which events were selected, how many were omitted, and the counting method |
| 14 | Missing-field markers | `evidence.missing_fields` | Explicit unavailability, e.g. no exception location for a normal goal failure |
| 15–17 | Primitive enrichments (Revision 4.2) | `evidence.primitive_diagnostics` | `target_derivation`, `controller_phases`, `action_context`, already configurable through `RunConfig.primitive_feedback` |

The fixed recovery instructions are not configurable in this revision:
- complete the task with the same API and condition rules;
- the trajectory is debugging evidence only;
- restart from the saved initial state;
- labels are not bindings;
- do not assume unobserved causes;
- output contract.

## 3. Part 1: configurable feedback items

### 3.1 Configuration

Add `RecoveryConfig.feedback` (`RecoveryFeedbackConfig`).

- **Software default.** Every Boolean below defaults to `true`, history defaults to `summaries`, and the evidence layout defaults to `legacy_fill`. Together these reproduce `cap-recovery-v1`.
  - "All on" here means the Revision 4 items only.
  - The Revision 4.2 switches keep their existing `false` defaults: the four `primitive_prompt` blocks and the three `primitive_feedback` enrichments.
- **Experiment reference.** The experiment's reference profile is a separate choice, made in the [ablation plan](revision_4_3_ablation_experiment_plan.md).

| Setting | Item(s) | Default | Notes |
| --- | --- | --- | --- |
| `current_source` | 2 | `true` | Off: the request carries no source. It becomes a fresh-generation request with failure evidence. In history mode, the latest attempt's source is a registry reference (Section 4). |
| `failure_location` | 3 | `true` | |
| `failure_events` | 4 | `true` | Does not remove item 5 (see `query_candidates`) |
| `query_candidates` | 5 | `true` | Emitted as its own record, independent of `failure_events`, so either can be disabled alone |
| `keyframes` | 6 | `true` | Selection rule unchanged |
| `goal_clauses` | 7 | `true` | |
| `dsl_diagnostics` | 8 | `true` | Only applies when the audit failed |
| `supporting_events` | 9 | `true` | `supporting_event_window` (int, default 80, range 0–512) sets the candidate window; under `fixed_items` it is also the number delivered |
| `event_counts` | 10 | `true` | |
| `selection_metadata` | 13 | `true` | |

- **Always included (not configurable):**
  - the schema/version identifier;
  - the latest outcome (item 1), because a recovery request must state that and how the attempt failed;
  - missing-field markers (item 14), because missing data must remain explicit;
  - scene metadata (item 11), because the handle-to-block mapping is needed to read every other item.
- **Earlier-failure summaries** (item 12) are controlled by the history mode (Section 4.1), not by a separate Boolean. History is one factor with three levels, which avoids a dependent pair of switches.
- **Revision 4.2 switches:** `target_derivation`, `controller_phases` and `action_context` stay in `RunConfig.primitive_feedback`. Reports list them with the items above as one feedback profile.
- **Configuration loading and recording:** `--recovery-config path.json` and batch `run_config` accept the block. Unknown keys and non-Boolean values are rejected. The resolved block is recorded in `recovery_config.json` and in every `feedback.json`.

```json
{
  "recovery": {
    "enabled": true,
    "feedback": {
      "current_source": true,
      "failure_location": true,
      "failure_events": true,
      "query_candidates": true,
      "keyframes": true,
      "goal_clauses": true,
      "dsl_diagnostics": true,
      "supporting_events": true,
      "supporting_event_window": 80,
      "event_counts": true,
      "selection_metadata": true,
      "evidence_layout": "legacy_fill",
      "evidence_bytes": null
    },
    "history": {
      "mode": "summaries",
      "edit_ledger": true,
      "source_diff": false,
      "source_bytes": 16384
    }
  }
}
```

### 3.2 Semantics

- **Collection is unchanged.**
  - Configuration controls what the model sees, not what is logged.
  - Every attempt artifact keeps the full evidence, so offline analysis is unaffected.
- **Disabled items are declared, not hidden.**
  - A non-default profile emits schema `cap-recovery-v2` and an `omitted_by_configuration` list of disabled item names.
  - The model can then distinguish "not provided by this profile" from "unavailable" (`missing_fields`) and from "observed false".
  - The default profile keeps emitting `cap-recovery-v1` without that list, preserving byte identity.
- **Evidence layout.** `evidence_layout` selects how items are sized.

  | Layout | Sizing | Use |
  | --- | --- | --- |
  | `legacy_fill` (default) | Current behavior: optional items fill whatever remains under the 8,192-byte cap | Byte-identical v1 reproduction |
  | `fixed_items` | Each item is limited only by its own count limit: supporting events `supporting_event_window`, action context `context_actions`, goal clauses 16, keyframes 3, query candidates 16. `evidence_bytes` is raised above the measured maximum and acts only as a guard. | Ablations |

  - Under `fixed_items`, an enabled item's content does not depend on which other items are enabled.
  - Each request records, per enabled item, whether it was delivered complete.
  - Exceeding the guard stops with `required_feedback_exceeds_token_limit`, the existing stop, and the audit reports it as a violation.
  - Measured sizes and proposed guard values are in the [ablation plan](revision_4_3_ablation_experiment_plan.md#5-evidence-limits).
- **Stop rules are unchanged:**
  - A preparation failure still stops with `preparation_failure_no_rollout`.
  - Missing essential attribution still stops.
  - Oversized required content still stops with `required_feedback_exceeds_token_limit`.
  - Disabling an item never converts a stop into a request containing less than the stated core.
- **Recovery off:** disabled recovery bypasses this configuration entirely.

## 4. Part 2: full executed-policy history

### 4.1 History modes

`RecoveryConfig.history.mode`:

| Mode | Earlier attempts in the request |
| --- | --- |
| `none` | No earlier attempts in the request |
| `summaries` (default) | Current behavior: item 12's one-line summaries; no earlier source |
| `executed_programs` | A chronological ledger of every earlier **executed** attempt in the same trial. Each entry has its complete attributable source and bindings, plus a source registry that deduplicates identical bodies. |

At the round-*k* request (k = 2…4), the ledger contains attempts 0 to k−2. Attempt k−1 is the latest attempt and keeps its detailed failure evidence.

### 4.2 What enters the history

- **Executed attempts only.**
  - An attempt that entered top-level execution counts, even if it failed before its first primitive.
  - A preparation failure ends the trial today, so no later request could show it. If preparation handling is added later, prepared-but-unexecuted candidates must be recorded separately and not as executed history.
- **Complete source units, including branches not taken.** Use the same attribution as the current-source item:
  - Method 1/2: checkpoint sources;
  - Baseline: `executed.py`, plus the execution-time binding record with the helper versions actually invoked.
  - Never merge different helper versions into one fictitious program.
  - Never reconstruct a program from executed lines.
- **Ledger entry fields:**

  | Field | Content |
  | --- | --- |
  | Identity | Round, candidate hash, parent round, `duplicate_of`, `changed_from_parent` |
  | Source | Unit role (policy / helper / executed bundle) → registry ID; binding map → registry IDs |
  | Outcome | Status, reason (≤140 chars), physical goal, logical postcondition, DSL compliance |
  | Failure point | Unit:line and statement; for a primitive failure, primitive name and arguments |
  | Edit summary (`edit_ledger`) | Against the previous executed attempt: units added/removed/changed, and lines added/removed per unit. Chronological order associates each edit with the next observed outcome. No causal label is attached. |
  | Diff (`source_diff`, default off) | Deterministic unified diff against the previous executed attempt. This is an aid; complete bodies remain in the registry. |

- **Excluded:**
  - old trajectories, keyframes and events;
  - videos;
  - earlier prompts;
  - model reasoning.

  These remain artifacts. Detailed physical evidence stays limited to the latest attempt.

### 4.3 Request layout

`summaries` mode keeps the current layout exactly. `executed_programs` mode renders the following, keeping the `cap-recovery-v2` schema:

```text
RECOVERY REQUEST (cap-recovery-v2)
{unchanged recovery instructions}
Earlier executed programs are listed for comparison. Each replacement still runs
from the saved initial state; earlier failures are debugging evidence only.

SOURCE REGISTRY
[src_<sha8>] <body>                    # each distinct body once, oldest first

EXECUTED ATTEMPT HISTORY (oldest first)
round 0: {identity, source refs, binding refs, outcome, failure point}
round 1: {..., edit summary vs round 0[, diff]}

CURRENT EXECUTED SOURCE AND BINDINGS (latest attempt, round k-1)
{source refs into the registry, binding refs}

OBSERVED FAILURE (latest attempt)
{evidence under the selected feedback profile; earlier_failures omitted}
```

- The latest attempt's bodies also live in the registry, so an unchanged helper is never printed twice.
- Every reference must resolve to a body in the same request. A path or hash alone is insufficient.
- The one-line summaries are not repeated, because each ledger entry already contains those fields.
- `executed_programs` requires `current_source`; configuration validation rejects the combination with `current_source=false`.

### 4.4 Size and budget

- **Separate allowance.** `history.source_bytes` (default **16,384** UTF-8 bytes) bounds the registry plus ledger for earlier attempts.
  - It is separate from the existing 8,192-byte evidence cap, so history does not displace latest-failure evidence.
  - The complete request still counts against the round reservation (100k tokens) and the 200k context check.
- **Measured basis.**
  - Replaying all 379 saved requests (Revision 4 and 4.2) that had at least one earlier executed attempt.
  - With bodies deduplicated by content hash, including source text repeated inside Baseline binding records, and about 300 bytes of ledger per attempt, the history would be:

    | Statistic | Bytes |
    | --- | ---: |
    | Median | 1,560 |
    | 90th percentile | 3,848 |
    | 99th percentile | 7,718 |
    | Maximum | 10,972 |

  - All 379 fit in 16,384 bytes. Without deduplication, which matters mainly for Baseline binding records that repeat source text, histories reach 37,007 bytes, so the registry is required. (These counts follow real paths only; an earlier draft counted each request twice through the per-trial `cap/latest` symlink.)
- **No silent truncation or fallback.**
  - If the required history exceeds the allowance, stop with `history_exceeds_source_limit`.
  - Do not drop attempts, cut bodies, or silently revert to summaries.
  - If `source_diff` would exceed the remaining allowance, omit diffs and record the omission. Diffs are optional, bodies are required.
- **Output cap unchanged.** The 16,384-token output cap, four recoveries / five rollouts, and all round/trial limits are unchanged. Report extra input tokens and calls.

### 4.5 Integrity

- **Freeze copies (lesson from CAP-RECOVERY-1).**
  - Store an immutable copy of each attempt's sources, bindings and outcome as soon as it finishes.
  - Later helper generation or a later round must not mutate earlier entries.
  - Build the ledger from these frozen copies.
- **Save provenance per request:**
  - prepared history;
  - registry IDs and hashes;
  - included rounds;
  - sizes.
- **Verify the actual HTTP payload against the prepared history,** as the current audit does for feedback.
- **Helper requests are unchanged.**
  - History enters the root-policy recovery request only.
  - Propagating runtime feedback to helper requests (Baseline execution-time helpers) remains a [separate backlog item](revision_4_unimplemented.md#15-independent-configuration-and-prompt-routing).

## 5. Implementation plan

| File | Change |
| --- | --- |
| [config.py](../config.py) | Add `RecoveryFeedbackConfig` and `RecoveryHistoryConfig`, nested in `RecoveryConfig`. Validate types, ranges, modes and dependencies; reject unknown keys. Load from JSON. |
| [llm.py](../llm.py) | Local OpenAI-compatible provider mode: thinking through the chat template, no request pacing or free-quota admission, server/model provenance |
| [recovery.py](../recovery.py) | Gate each item in `select_relevant_evidence`. Add the `fixed_items` layout and per-item delivery records, the `omitted_by_configuration` list and schema switch. Store frozen attempt records. Add `build_history` (registry, ledger, edit summary, optional diff). Extend `recovery_prompt` with the history layout. Add the `history_exceeds_source_limit` stop. Add a seed loader that starts recovery from an archived round-0 attempt. |
| [batch.py](../batch.py) / launcher | Accept the new blocks in batch `run_config`. Variant selection works like Revision 4.2 profiles. Record the resolved profile in `comparison_plan.json`. |
| Report generator | Show feedback profile and history mode per group. Add history size, included rounds and the new stop reason. |
| `test_revision43.py` | Tests in Section 6 |

## 6. Validation

- **Byte identity.** With default configuration, regenerate feedback and recovery prompts offline from archived Revision 4.2 attempts. They must match the delivered `feedback.json` and request prompts byte for byte.
- **Per-item gating.**
  - For each switch, turning it off removes exactly its item and lists it in `omitted_by_configuration`.
  - All other required content is unchanged; only optional fill may differ.
  - Invalid configurations are rejected.
- **History content.**
  - A scripted 5-round trial must satisfy all of the following:
    - the round-4 request contains complete bodies for attempts 0–2 and the latest attempt 3;
    - every reference resolves;
    - identical bodies appear once;
    - a later helper change does not alter earlier entries;
    - a Baseline binding record shows the helper version actually invoked.
  - The round-1 request has no history section.
- **Limits.**
  - An oversized history stops with `history_exceeds_source_limit`.
  - An oversized diff is omitted and recorded.
  - The full request is checked against round and context limits.
- **Unchanged behavior:**
  - recovery off;
  - initial prompts;
  - Revision 4.2 switches;
  - initial-state restoration;
  - method boundaries;
  - absence of a Baseline DSL gate.
- **Preflight.** A scripted physical preflight runs all three methods with `executed_programs` before any live batch.

## 7. Evaluation

The experiment design moved to [revision_4_3_ablation_experiment_plan.md](revision_4_3_ablation_experiment_plan.md).

## 8. Implementation and validation

### 8.1 What was implemented

- **Configuration** ([config.py](../config.py)):
  - `RecoveryFeedbackConfig` and `RecoveryHistoryConfig`, nested in `RecoveryConfig`.
  - Unknown keys, non-Boolean switches, out-of-range windows and invalid modes are rejected.
  - Dependency checks: `executed_programs` requires `current_source`, and `source_diff` requires `executed_programs`.
  - `RecoveryConfig.default_profile` identifies the byte-identical `cap-recovery-v1` profile. `evidence_cap` resolves the cap.
- **Batch variants** ([batch.py](../batch.py)):
  - May override `recovery.feedback` and `recovery.history`, merged field by field into the batch recovery settings.
  - Budgets remain batch-wide.
- **Evidence selection** ([recovery.py](../recovery.py) `select_relevant_evidence`):
  - The default profile runs the unchanged v1 path.
  - Other profiles:
    - emit `cap-recovery-v2` and `omitted_by_configuration`;
    - gate each item;
    - emit failed-query candidates as their own `query_candidates` record;
    - support `fixed_items`.
  - Non-default feedback artifacts also record:
    - `item_delivery`: per item, `complete`, `not_applicable`, `disabled`, or a partial count;
    - `history_mode`, `evidence_layout` and `evidence_cap`.

    These stay outside the model-visible evidence.
- **History:**
  - `freeze_attempt` stores immutable JSON copies of each attempt as soon as it is collected.
  - `build_history` builds the content-addressed registry and ledger, including Baseline binding records whose policy and helper bodies become registry references.
  - It also builds edit summaries keyed by unit path within the checkpoint, plus optional unified diffs.
  - Over the allowance it returns `history_exceeds_source_limit`; diffs alone are dropped first and recorded as `diffs_omitted`.
- **Seed loader:**
  - Call `run(..., seed_attempt=<archived rounds/000/result.json>)`.
  - The seed is validated before any run directory is created. It must be a failed attempt with saved artifacts and an initial state matching any requested initial state.
  - The seed's recorded calls, tokens, ticks, simulation steps and active wall time are charged to the trial ledger.
  - `seed.json` records the result path, hashes and charges. Round 0 is not regenerated.
- **Local model mode** ([llm.py](../llm.py)):
  - `local_model_config` returns a `vllm` `ModelConfig` for a loopback endpoint, with thinking set through `chat_template_kwargs`.
  - The key is read from an environment variable (default `VLLM_API_KEY`) at request time.
  - `server_provenance` records the served model ids and roots, `max_model_len` and the vLLM version, without credentials.
  - A `vllm` configuration that carries OpenRouter `reasoning` options is rejected.
  - The generic batch path applies no request pacing or quota admission unless a launcher supplies them.

### 8.2 Validation

- **v1 golden file.** [testdata/revision43_v1_golden.json](../tests/testdata/revision43_v1_golden.json) was recorded from the unmodified code: seven scripted scenarios covering all three methods, no-witness histories, primitive failures with all three Revision 4.2 enrichments, goal failure, NameError and tick budget. With default configuration, the implementation reproduces every prepared feedback object and recovery prompt exactly.
- **Offline replay of archived Revision 4.2 attempts.** The default configuration regenerated the feedback and recovery prompt of every ready archived recovery round, using each run's own recorded configuration.
  - **485/485 feedback objects** and **485/485 delivered recovery prompts** were byte-identical.
  - Coverage: MiniMax, Haiku and Qwen batches.
  - Each trial's `cap/latest` symlink duplicates its run directory; counting those links would give 970.
- **Focused tests** ([test_revision43.py](../tests/test_revision43.py)), 13 tests. They cover:
  - configuration validation and round trips;
  - batch variant merging;
  - per-item gating, checked against the evidence actually serialized into the prompt;
  - independence of query candidates and failure events;
  - `fixed_items` count limits, delivery records and guard stop;
  - history `none`;
  - a five-round `executed_programs` trial;
  - frozen-record immutability;
  - Baseline binding references, edit summaries, diff omission and the history-size stop;
  - a seeded branch from an archived failed attempt, plus rejection of a successful seed;
  - local model configuration and credential-free provenance.

  The five-round trial checks that rounds 0–2 are in the ledger and round 3 is current, that all bodies are present, deduplicated and resolvable, and that prepared equals delivered feedback.
- **Full suite.** `synthesis/cap/`: 194 tests passed, 2 opt-in hardware tests skipped.

### 8.3 Deviations from the plan

- **`evidence_bytes` defaults to `null`,** which uses `RecoveryConfig.feedback_tokens`, rather than a literal 8192. This keeps one cap setting and preserves the v1 default.
- **Failed-query candidates are always split out under `cap-recovery-v2`.** Even with both items enabled, they move into `query_candidates` rather than staying inside the `query_failure` event. The split is what makes the two items independently configurable. v1 keeps the original layout.
- **No-history request layout.** A round-1 `executed_programs` request has no earlier attempt, so it uses the plain current-source section without a registry, as specified in Section 6. Later rounds use the registry layout of Section 4.3.
- **Report generator.** Reports do not yet show feedback-profile or history columns; these will be added with the Revision 4.3 experiment launcher. Per-request profile, delivery and history records are already saved in each `feedback.json`.
- **Experiment launcher.** The two-stage launcher from the ablation plan is not implemented here, because its open points are still being decided. The seed loader and local model mode provide the building blocks: Stage A pool, Stage B branches.
