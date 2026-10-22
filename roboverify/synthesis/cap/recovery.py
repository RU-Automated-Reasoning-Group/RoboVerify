"""Shared deterministic failure feedback and bounded initial-state recovery."""

import ast
import difflib
import hashlib
import json
import re
import time
import uuid
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

from .budget import Budget
from .contracts import BudgetExceeded

VERSION = "cap-recovery-v1"
VERSION_V2 = "cap-recovery-v2"


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def token_bound(text):
    """Conservative UTF-8 byte bound, reconciled with provider usage after dispatch."""
    return len(text.encode("utf-8"))


class RoundBudget(Budget):
    """Round resources also debit a single trial ledger; no allowance resets."""

    parent = None
    charged_tokens = 0
    reservation = 0
    context_tokens = 200000
    output_tokens = 16384

    def remaining(self):
        local = super().remaining()
        return min(local, self.parent.remaining()) if self.parent else local

    def tick(self):
        if self.parent:
            self.parent.tick()
        super().tick()

    def simulation_step(self):
        if self.parent:
            self.parent.simulation_step()
        super().simulation_step()

    def model_call(self):
        if self.parent:
            self.parent.model_call()
        super().model_call()

    def reserve_request(self, prompt):
        self.remaining()
        if self.calls >= self.max_model_calls:
            self.exhaust("model_calls", "Round model-call budget exhausted")
        if self.parent.calls >= self.parent.max_model_calls:
            self.parent.exhaust("model_calls", "Trial model-call budget exhausted")
        amount = token_bound(prompt) + self.output_tokens
        if amount > self.context_tokens:
            self.exhaust(
                "context_tokens", "Input plus output exceeds context allowance"
            )
        if self.charged_tokens + amount > self.max_tokens:
            self.exhaust("tokens", "Request reservation exceeds round token allowance")
        if self.parent.charged_tokens + amount > self.parent.max_tokens:
            self.parent.exhaust(
                "tokens", "Request reservation exceeds trial token allowance"
            )
        self.charged_tokens += amount
        self.parent.charged_tokens += amount
        self.reservation = amount
        return {
            "input_bound": token_bound(prompt),
            "output_cap": self.output_tokens,
            "reserved_tokens": amount,
            "counting_method": "utf8-byte-upper-bound",
        }

    def abandon_request(self):
        self.unknown_usage_calls += 1
        self.parent.unknown_usage_calls += 1
        self.reservation = 0

    def account_tokens(self, count):
        if count is not None:
            difference = int(count) - self.reservation
            self.charged_tokens += difference
            self.parent.charged_tokens += difference
        # Unknown/failed usage retains the reservation rather than becoming free.
        if count is None:
            self.unknown_usage_calls += 1
            self.parent.unknown_usage_calls += 1
        else:
            self.tokens += int(count)
            self.parent.tokens += int(count)
        if self.charged_tokens > self.max_tokens:
            self.exhaust("tokens", "Actual usage exceeds round token allowance")
        if self.parent.charged_tokens > self.parent.max_tokens:
            self.parent.exhaust("tokens", "Actual usage exceeds trial token allowance")
        self.reservation = 0


def snapshot_hash(snapshot):
    digest = hashlib.sha256()
    for name, value in [
        ("gt_state", snapshot.gt_state),
        *sorted(snapshot.arrays.items()),
    ]:
        digest.update(name.encode())
        digest.update(str(value.shape).encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.tobytes())
    digest.update(encoded(snapshot.bindings).encode())
    return digest.hexdigest()


def verify_restoration(env, reference, logger):
    import numpy as np

    from synthesis.cfg.reset import capture

    actual = capture(env)
    equal = (
        reference.bindings == actual.bindings
        and reference.arrays.keys() == actual.arrays.keys()
        and np.array_equal(reference.gt_state, actual.gt_state)
        and all(
            np.array_equal(v, actual.arrays[k]) for k, v in reference.arrays.items()
        )
    )
    logger.emit(
        "initial_state_restoration",
        reference_sha256=snapshot_hash(reference),
        actual_sha256=snapshot_hash(actual),
        exact_match=equal,
        bindings=actual.bindings,
    )
    if not equal:
        raise ValueError("Initial-state restoration did not reproduce full snapshot")


def read_json(path):
    return json.loads(Path(path).read_text())


def collect_failure(result):
    """Load actual execution sources, not just the most recently generated helper."""
    from synthesis.cfg.recordings import load_traces

    root = Path(result["run_dir"]) / "artifacts/trial_001"
    events = []
    with (root / "timeline.jsonl").open() as handle:
        for line in handle:
            events.append(json.loads(line))
    sources = {}
    for event in events:
        path = event.get("source_path")
        if path and (Path(result["run_dir"]) / path).is_file():
            sources[path] = (Path(result["run_dir"]) / path).read_text()
    checkpoint = result.get("active_checkpoint_id")
    if checkpoint:
        directory = root / "checkpoints" / checkpoint
        for path in sorted(directory.rglob("*.py")):
            sources[str(path.relative_to(Path(result["run_dir"])))] = path.read_text()
    final = None
    if (root / "final_output.json").exists():
        final = read_json(
            Path(result["run_dir"]) / read_json(root / "final_output.json")["artifact"]
        )
    trajectory = root / "execution/rollout_001/trajectory.npz"
    trace = load_traces(trajectory)[0] if trajectory.exists() else None
    rolled = any(e["kind"] == "policy_rollout_start" for e in events)
    source_texts = sorted(set(sources.values()))
    exact = hashlib.sha256(encoded(source_texts).encode()).hexdigest()
    try:
        normalized = sorted(
            ast.dump(ast.parse(s), include_attributes=False) for s in source_texts
        )
    except SyntaxError:
        normalized = source_texts
    identity = hashlib.sha256(encoded(normalized).encode()).hexdigest()
    return {
        "result": result,
        "root": root,
        "events": events,
        "sources": sources,
        "bindings": final,
        "trace": trace,
        "rolled_out": rolled,
        "source_sha256": exact,
        "candidate_sha256": identity,
        "binding_sha256": (
            hashlib.sha256(encoded(final).encode()).hexdigest() if final else None
        ),
    }


def compact_event(event):
    omit = {"ts", "elapsed_seconds", "trial_id", "run_id", "recovery_round"}
    return {k: v for k, v in event.items() if k not in omit}


def summarize(record, round_index):
    result = record["result"]
    return {
        "round": round_index,
        "candidate": record["candidate_sha256"],
        "status": result["status"],
        "reason": result.get("reason", "")[:140],
        "environment_success": result.get("environment_success"),
        "dsl_postcondition": result.get("dsl_postcondition"),
        "dsl_compliance": result.get("dsl_compliance"),
        "duplicate_of": record.get("duplicate_of"),
        "changed_from_parent": record.get("changed_from_parent"),
    }


def goal_evidence(trace):
    """Derived terminal clause values use the same concrete predicate semantics."""
    from synthesis.predicates.scene import evaluate, scene_from_obs
    from synthesis.predicates.term import atom, block_id

    scene = scene_from_obs(
        trace.states[-1],
        trace.num_blocks,
        trace.snapshots[0].bindings,
        entry_obs=trace.states[0],
    )
    base = trace.snapshots[0].bindings.get("b0")
    clauses = []
    if trace.task == "stack" and base is not None:
        for i in range(min(trace.num_blocks, 16)):
            clauses.append(
                {
                    "clause": f"ON_star(block_{i},block_{base})",
                    "value": bool(
                        evaluate(atom("ON_star", block_id(i), block_id(base)), scene)
                    ),
                }
            )
    elif trace.task == "unstack":
        for i in range(trace.num_blocks):
            for j in range(trace.num_blocks):
                if i != j and len(clauses) < 16:
                    clauses.append(
                        {
                            "clause": f"not ON_star(block_{i},block_{j})",
                            "value": not bool(
                                evaluate(
                                    atom("ON_star", block_id(i), block_id(j)), scene
                                )
                            ),
                        }
                    )
    return {
        "source": "host predicate evaluation of recorded terminal state",
        "clauses": clauses,
        "scope": "up to 16 clauses; physical goal remains separate",
    }


QUERY_DETAIL = ("candidates", "evaluated_count", "omitted_candidates")


def split_query_candidates(events):
    """Separate failed-query candidate evaluations from their failure events."""
    stripped, candidates = [], []
    for event in events:
        if event["kind"] == "query_failure" and "candidates" in event:
            candidates.append(
                {"event_id": event["event_id"], "query": event.get("query")}
                | {k: event[k] for k in QUERY_DETAIL if k in event}
            )
            event = {k: v for k, v in event.items() if k not in QUERY_DETAIL}
        stripped.append(event)
    return stripped, candidates


def select_relevant_evidence(
    record, earlier, token_limit, primitive_feedback=None, config=None
):
    """Required failure context first, then recent supporting evidence by event ID.

    The default recovery configuration reproduces cap-recovery-v1 exactly. Other
    profiles gate individual items, declare what they omit, and may size items by
    their own count limits (``fixed_items``) instead of filling leftover bytes.
    """
    from .config import FEEDBACK_ITEMS, RecoveryConfig

    cfg = config or RecoveryConfig()
    items, legacy = cfg.feedback, cfg.default_profile
    fixed = items.evidence_layout == "fixed_items"
    trace, events = record["trace"], record["events"]
    if not record["rolled_out"]:
        return {"ready": False, "reason": "preparation_failure_no_rollout"}
    if trace is None or not record["sources"]:
        return {"ready": False, "reason": "missing_essential_source_or_snapshot"}
    if record["bindings"] and record["bindings"].get("status") != "complete":
        return {
            "ready": False,
            "reason": "missing_essential_baseline_binding_attribution",
        }
    important = [
        e
        for e in events
        if e["kind"]
        in {
            "primitive_start",
            "primitive_end",
            "controller_phase",
            "query_failure",
            "trial_failure",
            "python_exception",
            "evaluation_failure",
        }
    ]
    latest = important[-1:] if important else []
    starts = [e for e in important if e["kind"] == "primitive_start"]
    if starts:
        start = starts[-1]
        latest = [e for e in important if e["event_id"] >= start["event_id"]]
    indices = sorted(
        {
            0,
            len(trace.states) - 1,
            *[int(e["state_index"]) for e in latest if "state_index" in e],
        }
    )
    # Preserve pre-action/failure frames without adding a full trajectory.
    indices = sorted({0, len(trace.states) - 1, indices[1] if len(indices) > 2 else 0})
    frames = []
    for index in indices:
        obs = trace.states[index]
        frames.append(
            {
                "state_index": index,
                "gripper_xyz": obs[:3].tolist(),
                "blocks_xyz": {
                    f"block_{i}": obs[10 + 12 * i : 13 + 12 * i].tolist()
                    for i in range(trace.num_blocks)
                },
            }
        )
    adapter = read_json(record["root"] / "adapter_config.json")
    try:
        goal_details = goal_evidence(trace)
    except Exception as exc:
        goal_details = {
            "clauses": "unknown",
            "evaluator_error": f"{type(exc).__name__}: {exc}",
        }
    required = {
        "derived_goal_evidence": goal_details,
        "schema": VERSION,
        "latest": summarize(record, len(earlier)),
        "world_frame_units": "world XYZ in metres",
        "object_handle_to_physical_id": adapter["object_handles"],
        "higher_tolerance": adapter["higher_tolerance"],
        "keyframes": frames,
        "failure_context": [compact_event(e) for e in latest],
        "earlier_failures": [dict(summary) for summary in earlier],
        "missing_fields": [],
        "omitted_event_count": len(events),
        "selected_event_ids": [],
        "token_counting": "utf8-byte-upper-bound",
    }
    locations = [
        e
        for e in events
        if e["kind"] in {"statement_start", "python_exception", "python_line"}
    ]
    required["last_source_location"] = (
        compact_event(locations[-1]) if locations else None
    )
    audit_path = record["root"] / "dsl_compliance.json"
    if record["result"].get("dsl_compliance") == "fail" and audit_path.exists():
        audit = read_json(audit_path)
        required["dsl_diagnostics"] = audit.get("diagnostics", [])
    if not latest:
        required["missing_fields"].append(
            "exception location: not applicable for normal goal failure"
        )
    query_candidates = []
    if not legacy:
        required["schema"] = VERSION_V2
        required["omitted_by_configuration"] = items.disabled() + (
            ["earlier_attempts"] if cfg.history.mode == "none" else []
        )
        if cfg.history.mode != "summaries":
            # executed_programs carries these fields in its ledger instead.
            del required["earlier_failures"]
        context, query_candidates = split_query_candidates(required["failure_context"])
        required["failure_context"] = context
        if items.query_candidates and query_candidates:
            required["query_candidates"] = query_candidates
        for item, key in (
            ("failure_events", "failure_context"),
            ("failure_location", "last_source_location"),
            ("keyframes", "keyframes"),
            ("goal_clauses", "derived_goal_evidence"),
            ("dsl_diagnostics", "dsl_diagnostics"),
        ):
            if not getattr(items, item):
                required.pop(key, None)
    # Preserve the legacy path exactly unless an explicit enrichment is selected.
    from .config import PrimitiveFeedbackConfig
    from .primitive_feedback import build_diagnostics

    options = primitive_feedback or PrimitiveFeedbackConfig()
    diagnostics, action_candidates = build_diagnostics(
        record, adapter["object_handles"], options
    )
    if diagnostics is not None:
        required["primitive_diagnostics"] = diagnostics
    if token_bound(encoded(required)) > token_limit:
        return {"ready": False, "reason": "required_feedback_exceeds_token_limit"}
    if diagnostics is not None and "action_context" in diagnostics:
        context = diagnostics["action_context"]
        for candidate in action_candidates:
            context["actions"].append(candidate)
            context["omitted_candidate_ids"].remove(candidate["action_id"])
            if not fixed and token_bound(encoded(required)) + 768 > token_limit:
                context["actions"].remove(candidate)
                context["omitted_candidate_ids"].append(candidate["action_id"])
        context["actions"].sort(key=lambda item: item["action_id"])
        context["omitted_candidate_ids"].sort()
    optional = [
        e
        for e in events
        if e["kind"]
        in {
            "predicate",
            "query_result",
            "query_failure",
            "primitive_start",
            "primitive_end",
            "controller_phase",
            "python_line",
            "statement_start",
            "branch",
            "loop_iteration",
            "coordinate_query",
        }
        and e not in latest
    ]
    window = (
        optional[-items.supporting_event_window :]
        if items.supporting_event_window
        else []
    )
    selected = list(latest) if legacy or items.failure_events else []
    supporting = []
    if items.supporting_events:
        for event in reversed(window):
            candidate = compact_event(event)
            supporting.insert(0, candidate)
            if not fixed:
                required["supporting_context"] = supporting
                if token_bound(encoded(required)) + 512 > token_limit:
                    supporting.pop(0)
                    continue
            selected.append(event)
        required["supporting_context"] = supporting
    required["selected_event_ids"] = sorted(e["event_id"] for e in selected)
    required["omitted_event_count"] = len(events) - len(selected)
    required["event_counts"] = dict(Counter(e["kind"] for e in events))
    if not legacy:
        if not items.event_counts:
            del required["event_counts"]
        if not items.selection_metadata:
            for key in ("selected_event_ids", "omitted_event_count", "token_counting"):
                del required[key]
    # Counts/IDs are metadata too: never silently exceed the declared cap.
    while (
        not fixed
        and token_bound(encoded(required)) > token_limit
        and required.get("supporting_context")
    ):
        removed = required["supporting_context"].pop(0)
        if "selected_event_ids" in required:
            required["selected_event_ids"].remove(removed["event_id"])
            required["omitted_event_count"] += 1
    if token_bound(encoded(required)) > token_limit:
        return {"ready": False, "reason": "required_feedback_exceeds_token_limit"}
    feedback = {
        "ready": True,
        "evidence": required,
        "feedback_token_bound": token_bound(encoded(required)),
        "current_sources": record["sources"],
        "execution_bindings": record["bindings"],
    }
    if legacy:
        return feedback
    if not items.current_source:
        feedback["current_sources"] = feedback["execution_bindings"] = None
    available = {
        "current_source": True,
        "failure_location": bool(locations),
        "failure_events": bool(latest),
        "query_candidates": bool(query_candidates),
        "keyframes": True,
        "goal_clauses": True,
        "dsl_diagnostics": record["result"].get("dsl_compliance") == "fail"
        and audit_path.exists(),
        "supporting_events": True,
        "event_counts": True,
        "selection_metadata": True,
    }
    delivery = {}
    for item in FEEDBACK_ITEMS:
        if not getattr(items, item):
            delivery[item] = "disabled"
        elif not available[item]:
            delivery[item] = "not_applicable"
        else:
            delivery[item] = "complete"
    if items.supporting_events and len(supporting) < len(window):
        delivery["supporting_events"] = {
            "partial": len(supporting),
            "available": len(window),
        }
    for item in ("target_derivation", "controller_phases", "action_context"):
        if not getattr(options, item):
            delivery[item] = "disabled"
        elif diagnostics is None or item not in diagnostics:
            delivery[item] = "not_applicable"
        elif item == "action_context" and diagnostics[item]["omitted_candidate_ids"]:
            delivery[item] = {
                "partial": len(diagnostics[item]["actions"]),
                "omitted": len(diagnostics[item]["omitted_candidate_ids"]),
            }
        else:
            delivery[item] = "complete"
    feedback.update(
        item_delivery=delivery,
        history_mode=cfg.history.mode,
        evidence_layout=items.evidence_layout,
        evidence_cap=token_limit,
    )
    return feedback


RECOVERY_INSTRUCTIONS = (
    "Revise the current policy to complete the original task using the same API and condition rules.\n"
    "The failed trajectory is debugging evidence only. The replacement runs from its entry point "
    "after exact restoration of the saved initial state, never from the failure state. "
    "Diagnostic block labels/handles are not executable bindings. Do not assume an unobserved cause.\n"
)
RECOVERY_OUTPUT = (
    "\nOutput complete replacement Python policy code only, with its top-level invocation. "
    "Preserve this method's helper-generation and whole-program requirements above.\n"
)


def recovery_prompt(original, feedback):
    schema = feedback["evidence"]["schema"]
    prompt = (
        original + "\n\nRECOVERY REQUEST (" + schema + ")\n" + RECOVERY_INSTRUCTIONS
    )
    if schema == VERSION:
        return (
            prompt
            + "CURRENT EXECUTED SOURCE AND BINDINGS\n"
            + encoded(
                {
                    "sources": feedback["current_sources"],
                    "bindings": feedback["execution_bindings"],
                }
            )
            + ("\nOBSERVED FAILURE AND ACCUMULATED HISTORY\n")
            + encoded(feedback["evidence"])
            + RECOVERY_OUTPUT
        )
    history = feedback.get("history")
    if history:
        prompt += (
            "Earlier executed programs are listed for comparison. Each replacement still runs "
            "from the saved initial state; earlier failures are debugging evidence only.\n"
            "SOURCE REGISTRY\n"
            + encoded(history["registry"])
            + "\nEXECUTED ATTEMPT HISTORY (oldest first)\n"
            + encoded(history["attempts"])
            + "\nCURRENT EXECUTED SOURCE AND BINDINGS (latest attempt)\n"
            + encoded(history["current"])
            + "\n"
        )
    elif feedback["current_sources"] is not None:
        prompt += (
            "CURRENT EXECUTED SOURCE AND BINDINGS\n"
            + encoded(
                {
                    "sources": feedback["current_sources"],
                    "bindings": feedback["execution_bindings"],
                }
            )
            + "\n"
        )
    heading = (
        "OBSERVED FAILURE AND ACCUMULATED HISTORY"
        if feedback.get("history_mode") == "summaries"
        else "OBSERVED FAILURE (latest attempt)"
    )
    return prompt + heading + "\n" + encoded(feedback["evidence"]) + RECOVERY_OUTPUT


UNIT = re.compile(r"(?:^|/)checkpoints/[^/]+/(.+)$")


def unit_name(path):
    """Checkpoint directories change between rounds; the unit path within does not."""
    match = UNIT.search(path)
    return match.group(1) if match else path


def failure_point(record):
    events = record["events"]
    point = {}
    locations = [
        e
        for e in events
        if e["kind"] in {"statement_start", "python_exception", "python_line"}
    ]
    if locations:
        point.update(
            {
                k: locations[-1][k]
                for k in ("source_unit", "source_line", "statement")
                if k in locations[-1]
            }
        )
    if record["result"].get("status") == "primitive_failed":
        starts = [e for e in events if e["kind"] == "primitive_start"]
        if starts:
            point["primitive"] = starts[-1].get("primitive")
            point["arguments"] = starts[-1].get("arguments")
    return point


def freeze_attempt(record, round_index):
    """Immutable copy of an executed attempt for later history requests."""
    result = record["result"]
    return json.loads(
        encoded(
            {
                "round": round_index,
                "candidate": record["candidate_sha256"],
                "duplicate_of": record.get("duplicate_of"),
                "changed_from_parent": record.get("changed_from_parent"),
                "rolled_out": record["rolled_out"],
                "sources": record["sources"],
                "bindings": record["bindings"],
                "outcome": {
                    "status": result["status"],
                    "reason": result.get("reason", "")[:140],
                    "environment_success": result.get("environment_success"),
                    "dsl_postcondition": result.get("dsl_postcondition"),
                    "dsl_compliance": result.get("dsl_compliance"),
                },
                "failure_point": failure_point(record),
            }
        )
    )


def _line_changes(old, new):
    added = removed = 0
    matcher = difflib.SequenceMatcher(a=old.splitlines(), b=new.splitlines())
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "equal":
            removed += i2 - i1
            added += j2 - j1
    return added, removed


def edit_summary(previous, current, include_diff):
    before = {unit_name(k): v for k, v in previous["sources"].items()}
    after = {unit_name(k): v for k, v in current["sources"].items()}
    summary = {
        "compared_with_round": previous["round"],
        "units_added": sorted(set(after) - set(before)),
        "units_removed": sorted(set(before) - set(after)),
        "units_changed": [],
    }
    diffs = {}
    for unit in sorted(set(before) & set(after)):
        if before[unit] == after[unit]:
            continue
        added, removed = _line_changes(before[unit], after[unit])
        summary["units_changed"].append(
            {"unit": unit, "lines_added": added, "lines_removed": removed}
        )
        if include_diff:
            diffs[unit] = "\n".join(
                difflib.unified_diff(
                    before[unit].splitlines(),
                    after[unit].splitlines(),
                    f"round_{previous['round']}/{unit}",
                    f"round_{current['round']}/{unit}",
                    n=1,
                    lineterm="",
                )
            )
    return summary, diffs


def build_history(earlier, latest, settings):
    """Content-addressed source registry plus a ledger of earlier executed attempts.

    Returns None when there is no earlier executed attempt, a stop dictionary when the
    required history exceeds its allowance, or the history payload otherwise.
    """
    executed = [a for a in earlier if a["rolled_out"]]
    if not executed:
        return None
    registry = {}

    def register(body):
        ident = "src_" + hashlib.sha256(body.encode()).hexdigest()[:12]
        registry.setdefault(ident, body)
        return ident

    def references(attempt):
        bindings = attempt["bindings"]
        if bindings is not None:
            bindings = json.loads(encoded(bindings))
            if isinstance(bindings.get("policy"), str):
                bindings["policy"] = register(bindings["policy"])
            for function in (bindings.get("functions") or {}).values():
                if isinstance(function, dict) and isinstance(
                    function.get("source"), str
                ):
                    function["source"] = register(function["source"])
        return {k: register(v) for k, v in attempt["sources"].items()}, bindings

    chain = executed + [latest]
    entries, diffs = [], {}
    for index, attempt in enumerate(chain):
        sources, bindings = references(attempt)
        entry = {
            "round": attempt["round"],
            "candidate": attempt["candidate"],
            "parent_round": chain[index - 1]["round"] if index else None,
            "duplicate_of": attempt["duplicate_of"],
            "changed_from_parent": attempt["changed_from_parent"],
            "sources": sources,
            "bindings": bindings,
        }
        if attempt is not latest:
            entry["outcome"] = attempt["outcome"]
            entry["failure_point"] = attempt["failure_point"]
        if index and settings.edit_ledger:
            entry["edit_summary"], unit_diffs = edit_summary(
                chain[index - 1], attempt, settings.source_diff
            )
            if unit_diffs:
                diffs[attempt["round"]] = unit_diffs
        entries.append(entry)
    current = entries.pop()
    latest_ids = set(current["sources"].values())
    if current["bindings"]:
        latest_ids |= set(re.findall(r"src_[0-9a-f]{12}", encoded(current["bindings"])))

    def size(ledger):
        bodies = sum(
            token_bound(body)
            for ident, body in registry.items()
            if ident not in latest_ids
        )
        return bodies + token_bound(encoded(ledger))

    omitted_diffs = None
    if diffs:
        with_diffs = [
            e | ({"diff": diffs[e["round"]]} if e["round"] in diffs else {})
            for e in entries
        ]
        current_with = current | (
            {"diff": diffs[current["round"]]} if current["round"] in diffs else {}
        )
        if size(with_diffs + [current_with]) <= settings.source_bytes:
            entries, current = with_diffs, current_with
        else:
            omitted_diffs = "exceeds_source_bytes"
    used = size(entries + [current])
    if used > settings.source_bytes:
        return {
            "ready": False,
            "reason": "history_exceeds_source_limit",
            "history_bytes": used,
            "history_limit": settings.source_bytes,
        }
    history = {
        "mode": "executed_programs",
        "registry": registry,
        "attempts": entries,
        "current": current,
        "history_bytes": used,
        "history_limit": settings.source_bytes,
        "included_rounds": [e["round"] for e in entries],
    }
    if omitted_diffs:
        history["diffs_omitted"] = omitted_diffs
    return history


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_seed(path, initial_state=None):
    """Archived failed round-0 attempt used as the frozen start of a recovery branch."""
    path = Path(path).resolve()
    result = read_json(path)
    root = Path(result["run_dir"]) / "artifacts/trial_001"
    if not (root / "timeline.jsonl").is_file():
        raise ValueError(f"Seed attempt artifacts are missing: {root}")
    if result.get("task_success") is True or (
        "task_success" not in result
        and result.get("success") is True
        and result.get("dsl_postcondition") is not False
    ):
        raise ValueError("Seed attempt succeeded; there is no failure to recover")
    state = root / "initial_state.npz"
    if not state.is_file():
        raise ValueError("Seed attempt has no saved initial state")
    if initial_state is not None and file_sha256(initial_state) != file_sha256(state):
        raise ValueError("Seed initial state differs from the requested initial state")
    identity = {
        "seed_result": str(path),
        "seed_result_sha256": file_sha256(path),
        "seed_run_dir": result["run_dir"],
        "initial_state_sha256": file_sha256(state),
        "method": result.get("method"),
        "status": result.get("status"),
        "charged": {
            "model_calls": result.get("model_calls", 0),
            "known_tokens": result.get("known_tokens", 0),
            "unknown_usage_calls": result.get("unknown_usage_calls", 0),
            "execution_ticks": result.get("execution_ticks", 0),
            "simulation_steps": result.get("simulation_steps", 0),
            "active_wall_seconds": result.get("active_wall_seconds", 0),
        },
    }
    return result, identity


def run_recovery(config, model_config, **kwargs):
    from synthesis.experiment.run_logger import RunLogger

    from .logger import CapLogger
    from .runner import run_once

    cfg = config.recovery
    seed_path = kwargs.pop("seed_attempt", None)
    # Validate a seed before creating any run directory or ledger.
    seed = load_seed(seed_path, kwargs.get("initial_state")) if seed_path else None
    timeout = (
        min(cfg.total_timeout, 60) if config.task == "unstack" else cfg.total_timeout
    )
    ledger = Budget(
        timeout=timeout,
        max_model_calls=cfg.total_calls,
        max_steps=cfg.total_steps,
        max_depth=config.max_depth,
        max_tokens=cfg.total_tokens,
        max_sim_steps=cfg.total_sim_steps,
    )
    ledger.charged_tokens = 0
    parent = RunLogger(
        config.output_root,
        "cap",
        asdict(config)
        | {
            "model": asdict(model_config),
            "recovery_schema": VERSION if cfg.default_profile else VERSION_V2,
            "seed_attempt": str(seed_path) if seed_path else None,
            "instruction": kwargs.get("instruction"),
            "initial_state": str(kwargs.get("initial_state")),
        },
        slug=f"{config.method}-recovery-{uuid.uuid4().hex[:8]}",
        capture_stdout=False,
    )
    logger = CapLogger(parent.run_dir, run=parent, console=False)
    logger.artifact("recovery_config.json", asdict(cfg))
    history, attempts, records, frozen, feedback = [], [], [], [], None
    if seed:
        charged = seed[1]["charged"]
        # Every branch of a seed starts with the same recorded round-0 usage.
        ledger.calls += charged["model_calls"]
        ledger.tokens += charged["known_tokens"]
        ledger.charged_tokens += charged["known_tokens"]
        ledger.unknown_usage_calls += charged["unknown_usage_calls"]
        ledger.steps += charged["execution_ticks"]
        ledger.simulation_steps += charged["simulation_steps"]
        ledger.started -= charged["active_wall_seconds"]
        logger.artifact("seed.json", seed[1])
        logger.emit("recovery_seed", **seed[1])
    snapshot = kwargs.get("initial_state")
    trace_index = kwargs.get("trace_index", 0)
    stop = "max_recovery_rounds"
    try:
        for index in range(cfg.max_rounds + 1):
            try:
                remaining = ledger.remaining()
            except BudgetExceeded:
                stop = "trial_budget_exhausted"
                break
            if (
                ledger.calls >= cfg.total_calls
                or ledger.charged_tokens >= cfg.total_tokens
            ):
                stop = "trial_budget_exhausted"
                break
            round_config = replace(
                config,
                output_root=str(parent.run_dir / "attempts" / f"round_{index:03d}"),
                timeout=min(
                    config.timeout if index == 0 else cfg.round_timeout, remaining
                ),
                max_tokens=config.max_tokens if index == 0 else cfg.round_tokens,
                max_model_calls=(
                    config.max_model_calls if index == 0 else cfg.round_calls
                ),
            )
            logger.emit(
                "recovery_round_start",
                round=index,
                parent_round=index - 1 if index else None,
            )
            attempt_kwargs = dict(kwargs)
            attempt_kwargs.update(
                initial_state=snapshot,
                trace_index=trace_index,
                queue_wait_seconds=(
                    kwargs.get("queue_wait_seconds", 0) if index == 0 else 0
                ),
                batch_context=dict(kwargs.get("batch_context") or {})
                | {"recovery_round": index, "render_queue_root": str(parent.run_dir)},
            )
            if index == 0 and seed:
                result = dict(seed[0]) | {"seeded_from": seed[1]["seed_result"]}
            else:
                result = run_once(
                    round_config,
                    model_config,
                    shared_budget=ledger,
                    recovery_feedback=feedback,
                    **attempt_kwargs,
                )
            # Retain the old attempt outcome and expose the stricter task criterion separately.
            result["task_success"] = result.get("success") is True and (
                result.get("dsl_postcondition") is True
                if config.task in {"stack", "unstack"}
                else True
            )
            attempts.append(result)
            logger.artifact(f"rounds/{index:03d}/result.json", result)
            try:
                record = collect_failure(result)
            except Exception as exc:
                stop = "missing_essential_evidence"
                logger.emit(
                    "recovery_evidence_error",
                    round=index,
                    error=f"{type(exc).__name__}: {exc}",
                )
                break
            record["duplicate_of"] = next(
                (
                    i
                    for i, r in enumerate(records)
                    if r["candidate_sha256"] == record["candidate_sha256"]
                ),
                None,
            )
            record["changed_from_parent"] = bool(
                records
                and records[-1]["candidate_sha256"] != record["candidate_sha256"]
            )
            records.append(record)
            frozen.append(freeze_attempt(record, index))
            logger.artifact(
                f"rounds/{index:03d}/identity.json",
                {
                    k: record[k]
                    for k in (
                        "source_sha256",
                        "candidate_sha256",
                        "binding_sha256",
                        "duplicate_of",
                        "changed_from_parent",
                        "rolled_out",
                    )
                },
            )
            logger.emit(
                "recovery_round_end",
                round=index,
                status=result["status"],
                task_success=result["task_success"],
                duplicate_of=record["duplicate_of"],
                calls=ledger.calls,
                tokens=ledger.tokens,
                charged_tokens=ledger.charged_tokens,
            )
            trigger = not result["task_success"] or (
                cfg.on_dsl_noncompliance and result.get("dsl_compliance") == "fail"
            )
            logger.emit(
                "recovery_decision",
                round=index,
                triggered=trigger,
                remaining_rounds=cfg.max_rounds - index,
                environment_success=result.get("environment_success"),
                dsl_postcondition=result.get("dsl_postcondition"),
                status=result["status"],
            )
            if not trigger:
                stop = "success"
                break
            if index == cfg.max_rounds:
                break
            if snapshot is None:
                snapshot = record["root"] / "initial_state.npz"
                trace_index = 0
            if not Path(snapshot).is_file():
                stop = "missing_initial_snapshot"
                break
            from synthesis.util.on import using_higher_tolerance

            with using_higher_tolerance(config.higher_tolerance):
                feedback = select_relevant_evidence(
                    record,
                    history,
                    cfg.evidence_cap,
                    config.primitive_feedback,
                    config=cfg,
                )
            if feedback["ready"] and cfg.history.mode == "executed_programs":
                built = build_history(frozen[:-1], frozen[-1], cfg.history)
                if built is not None and built.get("ready") is False:
                    feedback = built
                else:
                    feedback["history"] = built
            logger.artifact(f"rounds/{index:03d}/feedback.json", feedback)
            if not feedback["ready"]:
                stop = feedback["reason"]
                break
            history.append(summarize(record, index))
    finally:
        final = (
            dict(attempts[-1])
            if attempts
            else {"success": False, "status": "budget_exhausted"}
        )
        final.update(
            run_dir=str(parent.run_dir),
            recovery_enabled=True,
            recovery_schema=VERSION if cfg.default_profile else VERSION_V2,
            success=bool(
                attempts and attempts[-1]["task_success"] and stop == "success"
            ),
            first_attempt_success=bool(attempts and attempts[0]["task_success"]),
            recovery_rounds=max(0, len(attempts) - 1),
            rollout_count=sum(r["rolled_out"] for r in records),
            stop_reason=stop,
            attempts=attempts,
            model_calls=ledger.calls,
            known_tokens=ledger.tokens,
            unknown_usage_calls=ledger.unknown_usage_calls,
            charged_tokens=ledger.charged_tokens,
            execution_ticks=ledger.steps,
            simulation_steps=ledger.simulation_steps,
            active_wall_seconds=time.monotonic() - ledger.started,
            queue_wait_seconds=kwargs.get("queue_wait_seconds", 0),
        )
        phases = Counter()
        for attempt in attempts:
            phases.update(attempt.get("phase_seconds", {}))
        final["phase_seconds"] = dict(phases)
        final["max_observed_depth"] = max(
            (a.get("max_observed_depth", 0) for a in attempts), default=0
        )
        final["recovery_overhead_seconds"] = max(
            0,
            final["active_wall_seconds"]
            - sum(a.get("active_wall_seconds", 0) for a in attempts),
        )
        final["status"] = (
            "completed" if final["success"] else final.get("status", "failed")
        )
        if not final["success"] and final["status"] == "completed":
            final["status"] = "goal_failed"
        logger.artifact("recovery_result.json", final)
        logger.emit(
            "recovery_end",
            stop_reason=stop,
            success=final["success"],
            rounds=len(attempts),
        )
        logger.close()
        parent.finish(
            final["status"],
            **{k: v for k, v in final.items() if k not in {"status", "run_dir"}},
        )
    return final
