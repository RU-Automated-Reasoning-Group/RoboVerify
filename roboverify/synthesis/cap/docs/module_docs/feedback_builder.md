# Feedback builder

Turns a failed attempt into the text that is appended to the next round's synthesis request. It is deterministic and makes no LLM calls.

## Pseudocode

The flow under the frozen configuration (all feedback items on, history mode `executed_programs`). Each function is described below.

```python
def build_feedback(attempt, earlier_attempts, config):
    record = collect_failure(attempt)        # what ran and what happened
    if not record.rolled_out or record.trajectory is None:
        return Stop(reason)                  # no rollout to give feedback on

    # 1. Required evidence about the latest failure
    failure  = last_failure_events(record)   # last primitive's events, or the final exception
    evidence = {
        "latest":           outcome_summary(attempt),
        "failure_events":   failure,
        "failure_location": last_source_location(record),
        "keyframes":        keyframes(record.trajectory, failure),
        "goal_clauses":     goal_clauses(record.trajectory),
        "query_candidates": failed_get_candidates(failure),
        "bookkeeping":      handle_map_units_and_tolerance(record),
    }
    if attempt.status == "primitive_failed":
        evidence["primitive_diagnostics"] = primitive_diagnostics(record)

    # 2. Supporting events and selection metadata
    evidence["supporting_events"] = recent_events(record, n=config.supporting_event_window)
    evidence["selection_metadata"] = event_counts_and_selected_ids(record)
    if size(evidence) > config.evidence_cap:
        return Stop("required_feedback_exceeds_token_limit")

    # 3. Earlier executed programs
    history = build_history(earlier_attempts, record)
    if size(history) > config.history.source_bytes:
        return Stop("history_exceeds_source_limit")

    return Feedback(current_source=record.sources, evidence=evidence, history=history)
```

The [recovery loop](recovery_loop.md) calls these steps in order ([`recovery.py`](../../recovery.py#L1003-L1094)), and the [LLM client](llm_client.md) renders the result into the next policy request with [`recovery_prompt`](../../recovery.py#L587-L638).

## Functions

| Function | What it does | Code |
| --- | --- | --- |
| `collect_failure` | Loads the attempt's event timeline, the sources and helper bindings that actually ran, and the recorded trajectory. Hashes the sources so that a repeated candidate can be flagged. | [`collect_failure`](../../recovery.py#L160-L209) |
| Stop checks | No rollout (e.g. the checker never passed), or the trajectory or sources are missing. | [`recovery.py`](../../recovery.py#L320-L328) |
| `last_failure_events` | The start, controller phases and end of the last primitive call, plus a failed query or exception if any. Empty for a normal finish that misses the goal. | [`recovery.py`](../../recovery.py#L329-L347) |
| `outcome_summary` | Round, candidate hash, status, failure reason (140 characters), success flags, and whether the candidate repeats an earlier one | [`summarize`](../../recovery.py#L217) |
| `last_source_location` | Source unit, line and statement of the last executed statement | [`recovery.py`](../../recovery.py#L393-L400) |
| `keyframes` | Gripper and block XYZ at three states: the start, the start of the last primitive, and the end | [`recovery.py`](../../recovery.py#L348-L369) |
| `goal_clauses` | Truth value of each goal clause at the final state, up to 16 (e.g. `ON_star(block_i, b0)` or `HasBlock(goal_j)`) | [`goal_evidence`](../../recovery.py#L232-L285) |
| `failed_get_candidates` | For a `get` with no match, up to 16 candidates with their predicate values | [`split_query_candidates`](../../recovery.py#L291) |
| `handle_map_units_and_tolerance` | Object handle to physical ID map, world-frame metres, Higher tolerance | [`recovery.py`](../../recovery.py#L378-L392) |
| `primitive_diagnostics` | Only for a failed primitive. **Target derivation:** for each axis, the reference block's coordinate, the offset and the resulting target. **Controller phases:** which phase completed or failed, measured value against its threshold, steps used. **Action context:** the preceding converged actions with geometry before and after. | [`build_diagnostics`](../../primitive_feedback.py#L327-L356) |
| `recent_events` | The last 40 (frozen) other events: primitive calls, queries, predicates, branches, loop iterations | [`recovery.py`](../../recovery.py#L459-L495) |
| `event_counts_and_selected_ids` | Count of each event kind, IDs of selected events, number omitted | [`recovery.py`](../../recovery.py#L496-L504) |
| Size check | UTF-8 byte bound of the evidence against the cap (frozen: 32 KB). Evidence is never silently truncated. | [`recovery.py`](../../recovery.py#L505-L516) |
| `build_history` | Every earlier attempt that ran: its sources (stored once per distinct body), outcome, failure point, and an edit summary against the previous round (units added, removed, changed, with line counts). Limit `source_bytes` (default 16 KB). | [`build_history`](../../recovery.py#L738-L835), [`freeze_attempt`](../../recovery.py#L671) |

## Configuration

Each evidence item can be switched off in [`RecoveryFeedbackConfig`](../../config.py#L76) (`current_source`, `failure_location`, `failure_events`, `query_candidates`, `keyframes`, `goal_clauses`, `dsl_diagnostics`, `supporting_events`, `event_counts`, `selection_metadata`), and the primitive diagnostics in [`PrimitiveFeedbackConfig`](../../config.py#L207). A disabled item is listed as omitted in the feedback.

- **History mode** ([`RecoveryHistoryConfig`](../../config.py#L113)): `executed_programs` (frozen) sends earlier programs as above; `summaries` sends one line per earlier failure instead; `none` sends nothing.
- **`dsl_diagnostics`:** post-hoc DSL audit results. Only Baseline produces them; Method 2 runs only checked programs.
- **Evidence layout:** `fixed_items` (frozen) takes each item at its own count limit. The older `legacy_fill` instead adds supporting events until the byte cap is reached.
