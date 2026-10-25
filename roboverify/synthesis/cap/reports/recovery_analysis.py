"""Artifact-only analysis of feedback delivery and recovery source changes.

Run from roboverify/: python -m synthesis.cap.reports.recovery_analysis --help.
No simulator imports, model requests, policy execution, or artifact mutation.
"""

import argparse
import ast
import copy
import difflib
import hashlib
import json
import os
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

VERSION = "cap-repair-analysis-v1"
METHODS = {
    "cap_google_baseline": "Baseline",
    "cap_method_1": "Method 1",
    "cap_method_2": "Method 2",
}


def read(path):
    return json.loads(Path(path).read_text())


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def source_units(sources):
    """Deduplicate identical text, preserving paths and distinct source versions."""
    units = {}
    for path, text in sorted(sources.items()):
        key = hashlib.sha256(text.strip().encode()).hexdigest()
        if key not in units:
            units[key] = {"paths": [], "code": text.strip()}
        units[key]["paths"].append(path)
    return sorted(units.values(), key=lambda u: (Path(u["paths"][0]).name, u["code"]))


class MaskNumbers(ast.NodeTransformer):
    def visit_Constant(self, node):
        if type(node.value) in (int, float):
            return ast.copy_location(ast.Constant(value="<number>"), node)
        return node


def features(sources):
    result = {
        name: []
        for name in (
            "ast",
            "numberless_ast",
            "conditions",
            "assignments",
            "helpers",
            "Move",
            "Release",
            "primitive_structure",
        )
    }
    result["parse_errors"] = []
    seen_ast = set()
    for unit in source_units(sources):
        try:
            tree = ast.parse(unit["code"])
        except SyntaxError as exc:
            result["parse_errors"].append(str(exc))
            continue
        normalized = ast.dump(tree, include_attributes=False)
        if normalized in seen_ast:
            continue
        seen_ast.add(normalized)
        result["ast"].append(normalized)
        result["numberless_ast"].append(
            ast.dump(MaskNumbers().visit(copy.deepcopy(tree)), include_attributes=False)
        )
        nodes = sorted(
            ast.walk(tree),
            key=lambda n: (getattr(n, "lineno", 0), getattr(n, "col_offset", 0)),
        )
        for node in nodes:
            if isinstance(node, (ast.If, ast.While)):
                result["conditions"].append(ast.unparse(node.test))
            elif isinstance(node, ast.Lambda):
                result["conditions"].append(ast.unparse(node))
            elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                result["assignments"].append(ast.unparse(node))
            elif isinstance(node, ast.FunctionDef):
                result["helpers"].append(ast.unparse(node))
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                name = node.func.id
                if name in ("Pick", "Move", "Release"):
                    result["primitive_structure"].append(name)
                if name in ("Move", "Release"):
                    result[name].append(ast.unparse(node))
    for key in ("ast", "numberless_ast"):
        result[key].sort()
    result["fingerprint"] = (
        digest(result["ast"]) if not result["parse_errors"] else None
    )
    return result


def source_changes(before, after):
    a, b = features(before), features(after)
    if not any(text.strip() for text in after.values()):
        return {
            "tags": ["no_replacement_source"],
            "before": a,
            "after": b,
            "diff": "# No attributable replacement source; not an edit deleting the policy.",
        }
    tags = []
    if a["parse_errors"] or b["parse_errors"]:
        tags.append("contains_unparseable_source")
    elif a["ast"] == b["ast"]:
        tags.append("same_ast")
    elif a["numberless_ast"] == b["numberless_ast"]:
        tags.append("numeric_literals_only")
    for key, tag in (
        ("Move", "move_call_changed"),
        ("Release", "release_call_changed"),
        ("conditions", "condition_or_selection_changed"),
        ("assignments", "assignment_changed"),
        ("helpers", "helper_definition_changed"),
        ("primitive_structure", "primitive_structure_changed"),
    ):
        if a[key] != b[key]:
            tags.append(tag)

    # Bundles retain unit labels and source versions; the diff is textual, not a proof.
    def text(sources):
        return "\n\n".join(
            f"# SOURCE UNIT: {Path(u['paths'][0]).name}\n{u['code']}"
            for u in source_units(sources)
        )

    patch = "\n".join(
        difflib.unified_diff(
            text(before).splitlines(),
            text(after).splitlines(),
            fromfile="feedback_source",
            tofile="next_attempt_source",
            lineterm="",
        )
    )
    return {"tags": tags, "before": a, "after": b, "diff": patch}


def outcome_class(result, events, rolled):
    if result.get("task_success", result.get("success")) is True:
        return "success"
    reason = result.get("reason", result.get("exit_reason", "")) or ""
    if not rolled:
        return "preparation_no_rollout"
    if result["status"] == "primitive_failed":
        ends = [
            e
            for e in events
            if e["kind"] == "primitive_end" and e.get("converged") is False
        ]
        return "primitive_" + (ends[-1]["primitive"] if ends else "unknown")
    if result["status"] == "budget_exhausted":
        return "budget_" + str(result.get("budget_exhausted_by") or "unknown")
    if "No object satisfies get condition" in reason:
        return "get_no_witness"
    for name in (
        "NameError",
        "SyntaxError",
        "TypeError",
        "UnboundLocalError",
        "RecursionError",
    ):
        if name in reason:
            return name
    return result["status"]


def load_attempt(result, index):
    directory = Path(result["run_dir"])
    root = directory / "artifacts/trial_001"
    events = []
    timeline = root / "timeline.jsonl"
    if timeline.exists():
        with timeline.open() as stream:
            for line in stream:
                events.append(json.loads(line))
    rolled = any(e["kind"] == "policy_rollout_start" for e in events)
    sources = {}
    for e in events:
        path = e.get("source_path")
        if path and (directory / path).is_file():
            sources[path] = (directory / path).read_text()
    checkpoint = result.get("active_checkpoint_id") or result.get("checkpoint_id")
    if checkpoint:
        for path in sorted((root / "checkpoints" / checkpoint).rglob("*.py")):
            sources[str(path.relative_to(directory))] = path.read_text()
    calls = []
    for call in sorted((root / "lmp").glob("call_*")):
        meta = (
            read(call / "prompt_metadata.json")
            if (call / "prompt_metadata.json").exists()
            else {}
        )
        response = (
            read(call / "response.json") if (call / "response.json").exists() else {}
        )
        prompt = (
            (call / "prompt.txt").read_text()
            if (call / "prompt.txt").exists()
            else None
        )
        request = (
            read(call / "request.json") if (call / "request.json").exists() else {}
        )
        calls.append(
            {
                "prompt_matches_request": prompt is not None
                and any(
                    m.get("content") == prompt for m in request.get("messages", [])
                ),
                "has_recovery_feedback": "OBSERVED FAILURE AND ACCUMULATED HISTORY"
                in (prompt or ""),
                "path": str(call),
                "role": meta.get("role"),
                "retry": meta.get("retry", 0),
                "finish_reasons": [
                    v.get("finish_reason") for v in response.get("choices", [])
                ],
                "usage": response.get("usage", {}),
                "response_text": (
                    (call / "response.txt").read_text()
                    if (call / "response.txt").exists()
                    else None
                ),
            }
        )
    primitive_ends = [
        {
            k: e.get(k)
            for k in (
                "event_id",
                "primitive",
                "steps",
                "converged",
                "position_error",
                "source_line",
                "source_path",
                "source_unit",
                "state_index",
            )
        }
        for e in events
        if e["kind"] == "primitive_end"
    ]
    targets = [
        {
            k: e.get(k)
            for k in (
                "event_id",
                "target",
                "source_line",
                "source_path",
                "source_unit",
                "state_index",
            )
        }
        for e in events
        if e["kind"] == "controller_phase" and e.get("phase") == "move"
    ]
    restoration = next(
        (e for e in events if e["kind"] == "initial_state_restoration"), None
    )
    final_output = None
    if (root / "final_output.json").exists():
        final_output = read(directory / read(root / "final_output.json")["artifact"])
    function_calls = Counter(
        (e.get("function"), e.get("source_path"))
        for e in events
        if e["kind"] == "python_call"
        and e.get("function") not in ("<lambda>", "<module>")
    )
    return {
        "executed_functions": [
            {"function": name, "source_path": path, "calls": count}
            for (name, path), count in function_calls.items()
        ],
        "round": index,
        "run_dir": str(directory),
        "result": result,
        "rolled_out": rolled,
        "failure_class": outcome_class(result, events, rolled),
        "sources": sources,
        "final_output": final_output,
        "features": features(sources),
        "calls": calls,
        "event_counts": dict(Counter(e["kind"] for e in events)),
        "primitive_ends": primitive_ends,
        "move_targets": targets,
        "restoration": restoration,
        "events": events,
    }


def parse_recovery_prompt(prompt):
    _, rest = prompt.split("CURRENT EXECUTED SOURCE AND BINDINGS\n", 1)
    source, rest = rest.split("\nOBSERVED FAILURE AND ACCUMULATED HISTORY\n", 1)
    evidence, _ = rest.split("\nOutput complete replacement", 1)
    return json.loads(source), json.loads(evidence)


def feedback_analysis(feedback, previous):
    evidence = feedback["evidence"]
    context = evidence.get("failure_context", []) + evidence.get(
        "supporting_context", []
    )
    available = previous["events"]
    kinds = Counter(e["kind"] for e in context)
    required = {
        "latest_status": evidence.get("latest", {}).get("status") is not None,
        "nonempty_reason": bool(evidence.get("latest", {}).get("reason")),
        "source_location": bool(evidence.get("last_source_location")),
        "goal_clauses": bool(evidence.get("derived_goal_evidence", {}).get("clauses")),
        "keyframes": bool(evidence.get("keyframes")),
        "primitive_arguments": any(
            e["kind"] == "primitive_start" and "arguments" in e for e in context
        ),
        "fixed_target": any(
            e["kind"] == "controller_phase" and e.get("target") is not None
            for e in context
        ),
        "controller_residual": any(
            e["kind"] == "primitive_end" and e.get("position_error") is not None
            for e in context
        ),
        "query_failure_candidates": any(
            e["kind"] == "query_failure" and "candidates" in e for e in context
        ),
        "loop_iteration": bool(kinds["loop_iteration"]),
        "branch": bool(kinds["branch"]),
        "predicate_or_query_result": bool(kinds["predicate"] + kinds["query_result"]),
        "explicit_execution_budget": any(
            k in canonical(evidence)
            for k in (
                '"execution_ticks"',
                '"budget_exhausted_by"',
                '"max_execution_ticks"',
            )
        ),
        "older_program_bodies": any(
            "sources" in e or "policy" in e
            for e in evidence.get("earlier_failures", [])
        ),
    }
    actual_loop_events = sum(e["kind"] == "loop_iteration" for e in available)
    top_locations = Counter(
        (e.get("source_path") or e.get("source_unit"), e.get("source_line"))
        for e in available
        if e["kind"] in ("python_line", "statement_start")
    )
    return {
        "presence": required,
        "selected_kind_counts": dict(kinds),
        "available_kind_counts": previous["event_counts"],
        "available_loop_iterations": actual_loop_events,
        "selected_loop_iterations": kinds["loop_iteration"],
        "selected_count": len(evidence.get("selected_event_ids", [])),
        "omitted_count": evidence.get("omitted_event_count"),
        "feedback_bound": feedback.get("feedback_token_bound"),
        "serialized_field_bytes": {
            k: len(canonical(v).encode()) for k, v in evidence.items()
        },
        "keyframe_indices": [f["state_index"] for f in evidence.get("keyframes", [])],
        "earlier_rounds": [f["round"] for f in evidence.get("earlier_failures", [])],
        "missing_fields": evidence.get("missing_fields", []),
        "most_repeated_statement_locations": [
            {"unit": key[0], "line": key[1], "count": count}
            for key, count in top_locations.most_common(3)
        ],
    }


def transition(parent, before, after):
    root = Path(after["run_dir"]) / "artifacts/trial_001"
    delivered = root / "recovery_feedback.json"
    prepared = (
        parent / f"artifacts/trial_001/rounds/{before['round']:03d}/feedback.json"
    )
    feedback = read(delivered) if delivered.exists() else None
    item = {
        "from_round": before["round"],
        "to_round": after["round"],
        "from_failure": before["failure_class"],
        "to_failure": after["failure_class"],
        "next_success": after["failure_class"] == "success",
        "next_rolled_out": after["rolled_out"],
        "prepared_path": str(prepared),
        "delivered_path": str(delivered),
        "issues": [],
    }
    if feedback is None:
        item["issues"].append("No delivered feedback artifact")
        return item
    item["feedback"] = feedback
    item["prepared_matches_delivered"] = (
        prepared.exists() and read(prepared) == feedback
    )
    item["history_length_correct"] = (
        len(feedback["evidence"]["earlier_failures"]) == before["round"]
    )
    item["source_matches_recorded_attempt"] = (
        feedback["current_sources"] == before["sources"]
    )
    item["feedback_coverage"] = feedback_analysis(feedback, before)
    calls = [c for c in after["calls"] if c["role"] == "recovery_policy"]
    item["recovery_policy_calls"] = len(calls)
    item["request_checks"] = []
    item["immediate_response_changes"] = []
    for call in calls:
        prompt_path = Path(call["path"]) / "prompt.txt"
        try:
            source, evidence = parse_recovery_prompt(prompt_path.read_text())
            match = evidence == feedback["evidence"] and source == {
                "sources": feedback["current_sources"],
                "bindings": feedback["execution_bindings"],
            }
        except (ValueError, KeyError):
            match = False
        item["request_checks"].append(
            {
                "path": str(prompt_path),
                "matches_prepared_feedback": match,
                "prompt_matches_http_request": call["prompt_matches_request"],
            }
        )
        item["immediate_response_changes"].append(
            source_changes(
                feedback["current_sources"],
                {"policy_response.py": call["response_text"] or ""},
            )
        )
    item["changes"] = source_changes(feedback["current_sources"], after["sources"])
    item["same_failure_class"] = before["failure_class"] == after["failure_class"]
    item["same_ast"] = "same_ast" in item["changes"]["tags"]
    # Helpers and static repair can intervene between a recovery response and rollout.
    item["additional_call_roles"] = dict(
        Counter(c["role"] for c in after["calls"] if c["role"] != "recovery_policy")
    )
    for flag in (
        "prepared_matches_delivered",
        "history_length_correct",
        "source_matches_recorded_attempt",
    ):
        if not item[flag]:
            item["issues"].append(flag)
    if not calls:
        item["issues"].append("No recovery-policy request")
    if any(
        not check["matches_prepared_feedback"]
        or not check["prompt_matches_http_request"]
        for check in item["request_checks"]
    ):
        item["issues"].append("Prompt differs from prepared feedback")
    return item


def relink(label, destination, output):
    return f"[{label}]({os.path.relpath(destination, output.parent)})"


def render_trial(trial, path):
    lines = [
        f"# {trial['id']}: {'recovered' if trial['recovered'] else 'not recovered'}",
        "",
        f"- Initial failure: `{trial['attempts'][0]['failure_class']}`; stop reason: `{trial['stop_reason']}`.",
        f"- {relink('Parent artifacts', trial['run_dir'], path)}; {relink('Machine-readable detail', path.with_suffix('.json'), path)}.",
        "",
        "## Attempt outcomes",
        "",
        "| Round | Outcome class | Rollout started | Tokens | Primitive calls completed |",
        "| --- | --- | --- | ---: | ---: |",
    ]
    for a in trial["attempts"]:
        lines.append(
            f"| {a['round']} | {a['failure_class']} | {a['rolled_out']} | {a['result'].get('known_tokens', 0)} | {len(a['primitive_ends'])} |"
        )
    for t in trial["transitions"]:
        after = trial["attempts"][t["to_round"]]
        lines += [
            "",
            f"## Feedback {t['from_round']} → replacement {t['to_round']}",
            "",
            f"- Outcome: `{t['from_failure']}` → `{t['to_failure']}`.",
            f"- {relink('Prepared feedback', t['prepared_path'], path)}; {relink('Delivered feedback', t['delivered_path'], path)}.",
        ]
        if "feedback" not in t:
            lines.append("- No delivered feedback was available.")
            continue
        f = t["feedback"]
        e = f["evidence"]
        lines += [
            f"- Code-edit tags (non-exclusive, not causal diagnoses): {', '.join(t['changes']['tags']) or 'none'}.",
            f"- Additional calls in this round: `{t['additional_call_roles']}`. These may change the candidate after its initial recovery response.",
            "",
            "### What the model received",
            "",
            "Selected fields below retain their actual saved values; complete payloads and prompts are linked above/below.",
            "",
            "```json",
            json.dumps(
                {
                    k: e[k]
                    for k in (
                        "latest",
                        "earlier_failures",
                        "last_source_location",
                        "derived_goal_evidence",
                        "keyframes",
                        "failure_context",
                        "missing_fields",
                    )
                    if k in e
                },
                indent=2,
            ),
            "```",
            "",
            f"- Supporting event kinds: `{t['feedback_coverage']['selected_kind_counts']}`; feedback byte bound: {f['feedback_token_bound']}.",
            "",
            "### Model response immediately after feedback",
        ]
        calls = [c for c in after["calls"] if c["role"] == "recovery_policy"]
        for c in calls:
            lines += [
                "",
                relink("Actual model prompt", Path(c["path"]) / "prompt.txt", path)
                + " · "
                + relink("Response", Path(c["path"]) / "response.txt", path),
                "",
                "```python",
                c["response_text"] or "# No response text",
                "```",
            ]
        lines += [
            "",
            "### Net source change before the next outcome",
            "",
            "This compares the source sent in feedback with the next attempt's attributable source bundle. A preparation failure may have no executed replacement. Helper generation and static repair are included in this net change.",
            "",
            "```diff",
            t["changes"]["diff"] or "# No textual change",
            "```",
            "",
            "### Next recorded execution",
            "",
            f"- Rollout entered: {after['rolled_out']}; status: `{after['result']['status']}`; reason: `{after['result'].get('reason','')}`.",
            f"- Primitive totals: `{dict(Counter(x['primitive'] for x in after['primitive_ends']))}`.",
            "- Last primitive records and fixed Move targets:",
            "",
            "```json",
            json.dumps(
                {
                    "primitive_ends": after["primitive_ends"][-5:],
                    "move_targets": after["move_targets"][-5:],
                },
                indent=2,
            ),
            "```",
            "",
            relink(
                "Timeline",
                Path(after["run_dir"]) / "artifacts/trial_001/timeline.jsonl",
                path,
            ),
        ]
    lines += [
        "",
        "## All attributable source bundles",
        "",
        "Source units are deduplicated by text; paths retain attribution. These are archived code, not executed by this analysis.",
    ]
    for a in trial["attempts"]:
        lines += [
            "",
            f"### Round {a['round']}",
            "",
            "Recorded named Python calls (Baseline only; full paths distinguish helper versions):",
            "",
            "```json",
            json.dumps(a["executed_functions"], indent=2),
            "```",
        ]
        for u in source_units(a["sources"]):
            lines += [
                "",
                " · ".join(
                    relink(Path(p).name, Path(a["run_dir"]) / p, path)
                    for p in u["paths"]
                ),
                "",
                "```python",
                u["code"],
                "```",
            ]
    path.write_text("\n".join(lines) + "\n")


def aggregate_details(trials, transitions):
    initial_failures = defaultdict(Counter)
    terminal_failures = Counter()
    preparation_stops, helper_calls, source_recurrence = [], [], []
    for trial in trials:
        initial = trial["attempts"][0]
        initial_failures[initial["failure_class"]]["trials"] += 1
        initial_failures[initial["failure_class"]]["recovered"] += trial["recovered"]
        if not trial["recovered"]:
            terminal_failures[trial["attempts"][-1]["failure_class"]] += 1
        previous = {}
        for attempt in trial["attempts"]:
            fp = attempt["features"]["fingerprint"]
            if fp and fp in previous:
                source_recurrence.append(
                    {
                        "trial": trial["id"],
                        "round": attempt["round"],
                        "previous_round": previous[fp],
                    }
                )
            if fp:
                previous[fp] = attempt["round"]
            if not attempt["rolled_out"]:
                preparation_stops.append(
                    {
                        "trial": trial["id"],
                        "round": attempt["round"],
                        "reason": attempt["result"].get("reason"),
                        "calls": [
                            {"role": c["role"], "finish_reasons": c["finish_reasons"]}
                            for c in attempt["calls"]
                        ],
                    }
                )
            if attempt["round"]:
                helper_calls.extend(
                    c | {"trial": trial["id"], "round": attempt["round"]}
                    for c in attempt["calls"]
                    if c["role"] == "helper"
                )
    bounds = [
        t["feedback_coverage"]["feedback_bound"]
        for t in transitions
        if "feedback_coverage" in t
    ]
    budgets = [
        t
        for t in transitions
        if t["from_failure"] == "budget_execution_ticks" and "feedback_coverage" in t
    ]
    return {
        "initial_failure_outcomes": dict(initial_failures),
        "terminal_failure_classes": dict(terminal_failures),
        "preparation_stops": preparation_stops,
        "source_recurrences": source_recurrence,
        "feedback_summary": {
            "total_feedbacks": len(bounds),
            "with_older_summaries": sum(
                bool(t.get("feedback_coverage", {}).get("earlier_rounds"))
                for t in transitions
            ),
            "with_older_program_bodies": sum(
                t.get("feedback_coverage", {})
                .get("presence", {})
                .get("older_program_bodies", False)
                for t in transitions
            ),
            "feedback_bound_min": min(bounds, default=0),
            "feedback_bound_median": statistics.median(bounds) if bounds else 0,
            "feedback_bound_max": max(bounds, default=0),
            "budget_feedbacks": len(budgets),
            "budget_logs_with_multiple_iterations": sum(
                t["feedback_coverage"]["available_loop_iterations"] > 1 for t in budgets
            ),
            "budget_feedback_with_multiple_iterations": sum(
                t["feedback_coverage"]["selected_loop_iterations"] > 1 for t in budgets
            ),
            "budget_feedback_with_any_iteration": sum(
                t["feedback_coverage"]["selected_loop_iterations"] > 0 for t in budgets
            ),
            "post_recovery_helper_calls": len(helper_calls),
            "helper_calls_with_recovery_feedback": sum(
                c["has_recovery_feedback"] for c in helper_calls
            ),
            "post_recovery_helper_rounds": len(
                {(c["trial"], c["round"]) for c in helper_calls}
            ),
        },
    }


def aggregate(trials):
    trans = [
        t
        | {
            "model": x["model"],
            "method": x["method"],
            "trial_id": x["id"],
            "trial_recovered": x["recovered"],
        }
        for x in trials
        for t in x["transitions"]
    ]
    groups = []
    for model, method in sorted({(x["model"], x["method"]) for x in trials}):
        rows = [x for x in trials if (x["model"], x["method"]) == (model, method)]
        groups.append(
            {
                "model": model,
                "method": method,
                "initially_failed": len(rows),
                "recovered": sum(x["recovered"] for x in rows),
                "not_recovered": sum(not x["recovered"] for x in rows),
                "initial_preparation_failures": sum(
                    not x["attempts"][0]["rolled_out"] for x in rows
                ),
                "transitions": sum(len(x["transitions"]) for x in rows),
            }
        )
    coverage = defaultdict(Counter)
    changes = defaultdict(Counter)
    outcomes = defaultdict(Counter)
    for t in trans:
        group = t["from_failure"]
        outcomes[group]["transitions"] += 1
        outcomes[group]["next_success"] += t["next_success"]
        for k, v in t.get("feedback_coverage", {}).get("presence", {}).items():
            coverage[group][k] += bool(v)
        for tag in t.get("changes", {}).get("tags", []):
            changes[tag]["transitions"] += 1
            changes[tag]["next_success"] += t["next_success"]
            changes[tag]["eventual_trial_success"] += t["trial_recovered"]
    return {
        "schema": VERSION,
        **aggregate_details(trials, trans),
        "groups": groups,
        "selected_trials": len(trials),
        "recovered": sum(x["recovered"] for x in trials),
        "not_recovered": sum(not x["recovered"] for x in trials),
        "transitions": len(trans),
        "next_success": sum(t["next_success"] for t in trans),
        "audit_issues": [
            {"trial": t["trial_id"], "round": t["to_round"], "issues": t["issues"]}
            for t in trans
            if t["issues"]
        ],
        "from_failure_outcomes": dict(outcomes),
        "feedback_coverage_by_failure": dict(coverage),
        "source_edit_tags": dict(changes),
        "same_failure_class_transitions": sum(
            t.get("same_failure_class", False) for t in trans
        ),
        "same_ast_transitions": sum(t.get("same_ast", False) for t in trans),
        "extra_call_roles": dict(
            sum((Counter(t.get("additional_call_roles", {})) for t in trans), Counter())
        ),
        "trials": [
            {
                "id": x["id"],
                "model": x["model"],
                "method": x["method"],
                "trial": x["trial"],
                "recovered": x["recovered"],
                "stop_reason": x["stop_reason"],
                "initial_failure": x["attempts"][0]["failure_class"],
                "final_failure": x["attempts"][-1]["failure_class"],
                "attempts": len(x["attempts"]),
                "analysis_path": x["analysis_path"],
                "run_dir": x["run_dir"],
                "transitions": [
                    {
                        "from_round": t["from_round"],
                        "to_round": t["to_round"],
                        "from_failure": t["from_failure"],
                        "to_failure": t["to_failure"],
                        "tags": t.get("changes", {}).get("tags", []),
                        "next_success": t["next_success"],
                        "same_failure_class": t.get("same_failure_class"),
                        "same_ast": t.get("same_ast"),
                        "coverage": t.get("feedback_coverage", {}),
                    }
                    for t in x["transitions"]
                ],
            }
            for x in trials
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch",
        nargs=3,
        action="append",
        required=True,
        metavar=("MODEL", "TRIAL_OFFSET", "DIRECTORY"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cohorts = []
    trials = []
    identities = set()
    for model, offset, folder in args.batch:
        batch = Path(folder).resolve()
        summary = read(batch / "summary.json")
        rows = summary["trials"]
        cohorts.append(
            {
                "model": model,
                "offset": int(offset),
                "batch": str(batch),
                "summary_sha256": hashlib.sha256(
                    (batch / "summary.json").read_bytes()
                ).hexdigest(),
                "total_trials": len(rows),
                "recovery_on": sum(bool(x.get("recovery_enabled")) for x in rows),
                "initial_success": sum(
                    bool(x.get("recovery_enabled") and x.get("first_attempt_success"))
                    for x in rows
                ),
            }
        )
        for row in rows:
            if not row.get("recovery_enabled") or row.get("first_attempt_success"):
                continue
            number = row["trial"] + int(offset)
            ident = f'{model}-{row["method"]}-{number:03d}'
            if ident in identities:
                raise ValueError(f"Duplicate trial identity: {ident}")
            identities.add(ident)
            parent = Path(row["run_dir"])
            attempts = [load_attempt(a, i) for i, a in enumerate(row["attempts"])]
            transitions = [
                transition(parent, a, b) for a, b in zip(attempts, attempts[1:])
            ]
            trial = {
                "id": ident,
                "model": model,
                "method": row["method"],
                "trial": number,
                "batch_trial": row["trial"],
                "batch": str(batch),
                "run_dir": str(parent),
                "recovered": bool(row["success"]),
                "stop_reason": row["stop_reason"],
                "attempts": attempts,
                "transitions": transitions,
                "analysis_path": str(output / "trials" / f"{ident}.md"),
            }
            # Do not duplicate full timelines; exact event files remain in each run.
            for a in attempts:
                a.pop("events")
            detail = output / "trials" / f"{ident}.json"
            write(detail, trial)
            render_trial(trial, detail.with_suffix(".md"))
            trials.append(trial)
    result = aggregate(trials)
    result["cohorts"] = cohorts
    result["analyzer_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    write(output / "analysis.json", result)
    lines = [
        "# Initially failed recovery-on trials",
        "",
        f"Selected trials: {result['selected_trials']}; recovered: {result['recovered']}; not recovered: {result['not_recovered']}; feedback transitions: {result['transitions']}.",
        "",
        "Edit tags are deterministic syntactic descriptions, not causal diagnoses. Failure classes derive from observed outcomes. No models or simulators were run.",
        "",
        "| Model | Method | Initially failed | Recovered | Not recovered | Initial preparation failures | Feedback transitions |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for g in result["groups"]:
        lines.append(
            f"| {g['model']} | {METHODS[g['method']]} | {g['initially_failed']} | {g['recovered']} | {g['not_recovered']} | {g['initial_preparation_failures']} | {g['transitions']} |"
        )
    lines += [
        "",
        "## Trial index",
        "",
        "| Model | Method | Trial | Initial failure | Final outcome | Stop reason | Trace |",
        "| --- | --- | ---: | --- | --- | --- | --- |",
    ]
    for x in sorted(
        result["trials"],
        key=lambda x: (x["recovered"], x["model"], x["method"], x["trial"]),
    ):
        lines.append(
            f"| {x['model']} | {METHODS[x['method']]} | {x['trial']} | {x['initial_failure']} | {x['final_failure']} | {x['stop_reason']} | {relink('Code, feedback, and diffs',x['analysis_path'],output/'index.md')} |"
        )
    (output / "index.md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("trials", "cohorts")},
            indent=2,
        )
    )
    return bool(result["audit_issues"])


if __name__ == "__main__":
    raise SystemExit(main())
