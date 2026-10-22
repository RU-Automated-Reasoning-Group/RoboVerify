# Revision 4.3: configurable recovery feedback and full executed-policy history

- **Status:** planned; not implemented. This document defines the scope for implementation review.
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
  - Across the saved Revision 4 and 4.2 recovery requests, 758 of 1,620 had at least one earlier executed attempt.
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

Add `RecoveryConfig.feedback` (`RecoveryFeedbackConfig`). Every Boolean defaults to `true`, which reproduces `cap-recovery-v1`.

| Setting | Item(s) | Default | Notes |
| --- | --- | --- | --- |
| `current_source` | 2 | `true` | Off: the request carries no source. It becomes a fresh-generation request with failure evidence. In history mode, the latest attempt's source is a registry reference (Section 4). |
| `failure_location` | 3 | `true` | |
| `failure_events` | 4 | `true` | Off also removes item 5 and any failure-event data used by item 17 |
| `query_candidates` | 5 | `true` | Off strips only the candidate list from `query_failure` events |
| `keyframes` | 6 | `true` | Selection rule unchanged |
| `goal_clauses` | 7 | `true` | |
| `dsl_diagnostics` | 8 | `true` | Only applies when the audit failed |
| `supporting_events` | 9 | `true` | `supporting_event_window` (int, default 80, range 0–512) sets the candidate window |
| `event_counts` | 10 | `true` | |
| `scene_metadata` | 11 | `true` | Off makes block labels in other items harder to interpret; intended for ablation only |
| `earlier_summaries` | 12 | `true` | Ignored in `executed_programs` history mode, where the ledger carries these fields (Section 4.3) |
| `selection_metadata` | 13 | `true` | |

- **Always included (not configurable):**
  - the schema/version identifier;
  - the latest outcome (item 1), because a recovery request must state that and how the attempt failed;
  - missing-field markers (item 14), because missing data must remain explicit.
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
      "scene_metadata": true,
      "earlier_summaries": true,
      "selection_metadata": true
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
- **The byte cap is unchanged.** The cap of 8,192 bytes applies after item selection.
  - Turning items off frees space that supporting events (item 9) and action context (item 17) may then fill.
  - Ablations therefore compare bounded end-to-end payloads, as in Revision 4.2.
  - Record actual delivered sizes and the composition of supporting events per request.
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
- `earlier_summaries` is ignored in this mode because each ledger entry already contains those fields. The resolved configuration records this.

### 4.4 Size and budget

- **Separate allowance.** `history.source_bytes` (default **16,384** UTF-8 bytes) bounds the registry plus ledger for earlier attempts.
  - It is separate from the existing 8,192-byte evidence cap, so history does not displace latest-failure evidence.
  - The complete request still counts against the round reservation (100k tokens) and the 200k context check.
- **Measured basis.**
  - Replaying all 758 saved requests (Revision 4 and 4.2) that had at least one earlier executed attempt.
  - With bodies deduplicated by content hash, including source text repeated inside Baseline binding records, and about 300 bytes of ledger per attempt, the history would be:

    | Statistic | Bytes |
    | --- | ---: |
    | Median | 1,560 |
    | 90th percentile | 3,848 |
    | 99th percentile | 7,718 |
    | Maximum | 10,972 |

  - All 758 fit in 16,384 bytes. Without deduplication, Baseline binding records alone reach 46,324 bytes, so the registry is required.
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
| [config.py](../config.py) | Add `RecoveryFeedbackConfig` and `RecoveryHistoryConfig`, nested in `RecoveryConfig`. Validate types, ranges and modes; reject unknown keys. Load from JSON. |
| [recovery.py](../recovery.py) | Gate each item in `select_relevant_evidence`. Add the `omitted_by_configuration` and schema switch. Store frozen attempt records. Add `build_history` (registry, ledger, edit summary, optional diff). Extend `recovery_prompt` with the history layout. Add the `history_exceeds_source_limit` stop. |
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

## 7. Evaluation (to be decided before running)

No experiment is authorized by this document. Two designs are proposed; choosing models, counts and profiles is a separate decision.

- **History ablation: `summaries` vs `executed_programs`, other settings at default.**
  - History matters only from round 2.
  - Independent trials like Revision 4.2 would dilute the effect with trials that succeed in round 0 or 1.
  - Preferred design: **branch from the same archived failed attempts.**
    - Select trials whose rounds 0 and 1 both failed in saved Revision 4.2 batches.
    - Resume each one at round 2 under both modes.
    - Hold fixed the frozen attempts 0–1, the initial state, model and budgets.
    - This needs a small seed loader that replays archived attempts into the frozen history without regenerating them, plus a matching audit.
  - Fallback: independent full trials, analysed only over trials that reach round 2.
- **Feedback-item ablations.**
  - Do not run all combinations.
  - Candidates:
    - `minimal`: core + current source;
    - `legacy`: defaults;
    - a few leave-one-out profiles for the items most likely to matter: keyframes, goal clauses, supporting events, earlier summaries.
  - As in Revision 4.2, initial attempts are unaffected by feedback, so compare recovery among initially failed trials. Matched branching from the same failed round-0 attempts is preferable.
- **Metrics, reported separately:**
  - final success;
  - rounds to success;
  - reintroduced edits;
  - repeated candidates;
  - new stops;
  - input/output tokens;
  - calls;
  - physical, logical and DSL outcomes.
