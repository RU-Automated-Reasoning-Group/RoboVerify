"""Deterministic, independently ablatable primitive diagnostics from actual traces."""

import math

VERSION = "primitive-feedback-v1"


def pair_actions(events):
    """Keep each dynamic call, including zero-step and incomplete calls."""
    actions = {}
    run_ids = {
        event.get("run_id")
        for event in events
        if event.get("kind") == "primitive_start"
    }
    if len(run_ids) > 1:
        raise ValueError("Mixed attempts in primitive action context")
    for event in events:
        kind = event.get("kind")
        if kind == "primitive_start":
            key = (event.get("run_id"), event["event_id"])
            if key in actions:
                raise ValueError("Duplicate primitive event identity")
            actions[key] = {"start": event, "end": None, "phases": []}
        elif kind in {"primitive_end", "controller_phase"}:
            key = (
                event.get("run_id"),
                (
                    event.get("parent_event_id")
                    if kind == "primitive_end"
                    else event.get("primitive_event_id")
                ),
            )
            action = actions.get(key)
            if action is None:
                continue
            if event.get("primitive") != action["start"].get("primitive"):
                raise ValueError("Primitive result identity mismatch")
            if kind == "primitive_end":
                if action["end"] is not None:
                    raise ValueError("Duplicate primitive end")
                action["end"] = event
            else:
                action["phases"].append(event)
    return sorted(actions.values(), key=lambda action: action["start"]["event_id"])


def source_location(event):
    return {
        key: event[key]
        for key in (
            "checkpoint_id",
            "source_unit",
            "source_path",
            "source_line",
            "helper",
        )
        if key in event
    }


def observation(record, index):
    trace = record.get("trace")
    if trace is None or type(index) is not int or not 0 <= index < len(trace.states):
        return None
    return trace.states[index]


def vector(values):
    return [float(v) for v in values]


def object_id(value, mapping):
    return mapping.get(value.get("object_handle")) if isinstance(value, dict) else None


def object_xyz(obs, ident):
    if obs is None or type(ident) is not int or ident < 0 or 13 + 12 * ident > len(obs):
        return None
    return vector(obs[10 + 12 * ident : 13 + 12 * ident])


def reference(value, mapping, obs, goals):
    """Block ID and XYZ of a Move/Release reference; goals use their fixed centre."""
    if isinstance(value, dict) and value.get("goal_handle") in goals:
        return None, vector(goals[value["goal_handle"]])
    ident = object_id(value, mapping)
    return ident, object_xyz(obs, ident)


def target_diagnostic(record, action, mapping, goals=None):
    goals = goals or {}
    start, end = action["start"], action["end"] or {}
    before = observation(record, start.get("state_index"))
    after = observation(record, end.get("state_index"))
    args, name = start.get("arguments", []), start["primitive"]
    result = {
        "action_id": start["event_id"],
        "primitive": name,
        "source": source_location(start),
        "entry_state": start.get("state_index"),
        "terminal_state": end.get("state_index"),
        "units": "world metres",
        "reference_sampling": "invocation",
        "axes": [],
    }
    phases = [p for p in action["phases"] if p.get("target") is not None]
    result["logged_fixed_target"] = phases[-1]["target"] if phases else None
    if before is not None:
        result["entry_gripper_xyz"] = vector(before[:3])
    if after is not None:
        result["terminal_gripper_xyz"] = vector(after[:3])
    if name == "Move" and len(args) == 4:
        for axis, ref in enumerate(args[:3]):
            ident, xyz = reference(ref, mapping, before, goals)
            offset = float(args[3][axis])
            result["axes"].append(
                {
                    "axis": "XYZ"[axis],
                    "reference_object": ident,
                    **goal_reference(ref, goals),
                    "reference_coordinate": xyz[axis] if xyz else None,
                    "offset": offset,
                    "computed_target": xyz[axis] + offset if xyz else None,
                }
            )
    elif name == "Release" and len(args) == 2:
        ident, xyz = reference(args[0], mapping, before, goals)
        result["frozen_xy_before_open"] = (
            vector(before[:2]) if before is not None else None
        )
        result["axes"] = [
            {
                "axis": "Z",
                "reference_object": ident,
                **goal_reference(args[0], goals),
                "reference_coordinate": xyz[2] if xyz else None,
                "offset": float(args[1]),
                "computed_target": xyz[2] + float(args[1]) if xyz else None,
            }
        ]
    elif name == "Pick":
        result["phase_targets"] = [
            {
                "phase": p["phase"],
                "state": p.get("state_index"),
                "fixed_target": p["target"],
            }
            for p in phases
        ]
        result["sampling_note"] = (
            "Approach uses entry block XY/gripper Z; descent retains XY and refreshes block Z after opening."
        )
    result["position_error"] = end.get("position_error")
    result["position_error_norm"] = "3D Euclidean for motion phases"
    return result


def phase_diagnostic(record, action):
    start, end = action["start"], action["end"] or {}
    config = start.get("controller_config", {})
    limit = start.get("step_limit")
    threshold = (
        config["gripper_threshold"] + config["gripper_tolerance"]
        if "gripper_threshold" in config and "gripper_tolerance" in config
        else None
    )
    result = {
        "action_id": start["event_id"],
        "primitive": start["primitive"],
        "step_limit_shared": limit,
        "total_steps": end.get("steps"),
        "phases": [],
        "criteria_source": (
            "recorded_instruction_configuration"
            if config
            else "unavailable_in_legacy_event"
        ),
    }
    phases = action["phases"]
    for i, phase in enumerate(phases):
        boundary = phases[i + 1] if i + 1 < len(phases) else end
        obs = observation(record, boundary.get("state_index"))
        complete = True if i + 1 < len(phases) else end.get("converged")
        row = {
            "phase": phase["phase"],
            "start_state": phase.get("state_index"),
            "end_state": boundary.get("state_index"),
            "completed": complete,
            "completion_evidence": (
                "next_phase_entered" if i + 1 < len(phases) else "primitive_result"
            ),
        }
        if "steps" in boundary and "steps" in phase:
            row["phase_steps"] = boundary["steps"] - phase["steps"]
            row["cumulative_steps"] = boundary["steps"]
        if phase["phase"] in {"open", "close"}:
            row.update(
                criterion="finger_state_sum",
                comparison=">" if phase["phase"] == "open" else "<=",
                threshold=threshold,
                measured_value=float(obs[3] + obs[4]) if obs is not None else None,
                position_error=None,
            )
        else:
            target = phase.get("target")
            row.update(
                criterion="3D_gripper_position_error",
                comparison="<=",
                threshold=config.get("position_tolerance"),
                measured_value=(
                    math.dist(vector(obs[:3]), target)
                    if obs is not None and target is not None
                    else None
                ),
            )
        result["phases"].append(row)
    expected = {
        "Pick": ["approach", "open", "descend", "close"],
        "Move": ["move"],
        "Release": ["open", "retreat"],
    }[start["primitive"]]
    result["phases_not_recorded"] = [
        p for p in expected if p not in {x["phase"] for x in phases}
    ]
    result["scope"] = (
        "Historical phase completion; no grasp/support certificate or recheck of earlier conditions."
    )
    return result


def ids(action, mapping):
    return {
        ident
        for arg in action["start"].get("arguments", [])
        if (ident := object_id(arg, mapping)) is not None
    }


def context_candidates(record, actions, failed, mapping, config):
    previous = [
        a
        for a in actions
        if a["end"]
        and a["end"].get("converged") is True
        and a["end"]["event_id"] < failed["start"]["event_id"]
    ]
    window = previous[-config.context_lookback :]
    selected = []

    def add(action, reason):
        if action and not any(a is action for a, _ in selected):
            selected.append((action, reason))

    add(previous[-1] if previous else None, "nearest_successful_predecessor")
    pick = None
    for action in reversed(window):
        if action["start"]["primitive"] == "Release":
            break
        if action["start"]["primitive"] == "Pick":
            pick = action
            break
    add(pick, "recent_selected_object_not_verified_grasp")
    relevant = ids(failed, mapping) | (ids(pick, mapping) if pick else set())
    for action in reversed(window):
        if ids(action, mapping) & relevant:
            add(action, "nearby_shared_object_reference")
    selected = selected[: config.context_actions]
    rows = []
    for action, reason in selected:
        start, end = action["start"], action["end"]
        relevant_ids = ids(action, mapping) | (ids(pick, mapping) if pick else set())
        states = {}
        for label, event in (("before", start), ("after", end)):
            obs = observation(record, event.get("state_index"))
            if obs is not None:
                states[label] = {
                    "state": event["state_index"],
                    "gripper_xyz": vector(obs[:3]),
                    "objects": {
                        str(i): object_xyz(obs, i) for i in sorted(relevant_ids)
                    },
                }
        args = [
            ({"object_id": object_id(arg, mapping)} if isinstance(arg, dict) else arg)
            for arg in start.get("arguments", [])
        ]
        targets = [
            {"phase": p["phase"], "target": p["target"]}
            for p in action["phases"]
            if p.get("target") is not None
        ]
        rows.append(
            {
                "action_id": start["event_id"],
                "primitive": start["primitive"],
                "selection_reason": reason,
                "source": source_location(start),
                "arguments": args,
                "converged": True,
                "steps": end.get("steps"),
                "position_error": end.get("position_error"),
                "phase_targets": targets,
                "states": states,
            }
        )
    return rows, {
        "lookback": config.context_lookback,
        "eligible_predecessors": len(previous),
        "candidate_action_ids_priority_order": [row["action_id"] for row in rows],
        "incomplete_predecessors": [
            a["start"]["event_id"]
            for a in actions
            if not a["end"] and a["start"]["event_id"] < failed["start"]["event_id"]
        ],
        "selected_object_is_grasp_evidence": False,
    }


def goal_reference(value, goals):
    """Name a goal reference; tower-task diagnostics keep their original keys."""
    if isinstance(value, dict) and value.get("goal_handle") in goals:
        return {"reference_goal": value["goal_handle"]}
    return {}


def build_diagnostics(record, mapping, config, goals=None):
    if not config.enabled or record["result"].get("status") != "primitive_failed":
        return None, []
    actions = pair_actions(record["events"])
    failures = [a for a in actions if a["end"] and a["end"].get("converged") is False]
    failed = (
        failures[-1]
        if failures
        else next((a for a in reversed(actions) if a["end"] is None), None)
    )
    if failed is None:
        return {
            "version": VERSION,
            "unavailable": "No attributable failed primitive invocation",
        }, []
    result = {"version": VERSION, "failed_action_id": failed["start"]["event_id"]}
    if config.target_derivation:
        result["target_derivation"] = target_diagnostic(record, failed, mapping, goals)
    if config.controller_phases:
        result["controller_phases"] = phase_diagnostic(record, failed)
    candidates = []
    if config.action_context:
        candidates, metadata = context_candidates(
            record, actions, failed, mapping, config
        )
        result["action_context"] = metadata | {
            "actions": [],
            "omitted_candidate_ids": [c["action_id"] for c in candidates],
        }
    return result, candidates
