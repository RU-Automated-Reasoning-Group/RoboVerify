"""RDP keyframes of a failed rollout for recovery feedback (Revision 7.2).

The recorded rollout is compacted exactly like a demonstration
(``demonstrations.compact``): event frames at the first and last state and at
every gripper-command switch, with block positions and DSL predicate changes,
plus Ramer-Douglas-Peucker corners of the gripper path between events. Frames
are rendered with the demonstration line format.

The keyframe item has its own byte budget. When the frames do not fit, the
first frame and the most recent frames (nearest the failure) are kept, and the
omitted span is declared; the first event kept after the gap lists every true
atom instead of changes relative to an omitted frame.
"""

import json
from dataclasses import replace

from .demonstrations import (
    Demonstration,
    compact,
    relations,
    render_frame,
    render_inputs,
)

VERSION = "recovery-keyframes-v1"
FORMAT = (
    "Recorded execution of the failed attempt, compacted like the demonstrations"
    " (demo-keyframes-v1). o<i> is physical block i (block_<i> elsewhere in this"
    " evidence; see object_handle_to_physical_id) and g<j> is goal j (goal_<j>)."
    " Positions are world-frame (x,y,z) centres in metres; t is the recorded"
    " control step. Gripper command: the latest open/closed command before step t;"
    " holding names the block within 0.030 m of a closed gripper, or none. event"
    " frames are the first and last step and every gripper-command switch; they"
    " list every block position and the DSL predicates of your API: holds lists"
    " every true atom, changes lists atoms that became true (+) or false (-) since"
    " the previous listed event. waypoint frames are corners of the gripper's 3D"
    " path between events (Ramer-Douglas-Peucker with waypoint_tolerance) and list"
    " only the gripper and the held block."
)


def demonstration_view(trace):
    """The fields of a recorded rollout that the demonstration compaction reads."""
    import numpy as np

    snapshot = trace.snapshots[0]
    actions = trace.actions[0] if isinstance(trace.actions, tuple) else trace.actions
    layout = getattr(snapshot, "goal_layout", None)
    bindings = {k: v for k, v in (snapshot.bindings or {}).items() if k == "b0"}
    return Demonstration(
        "rollout",
        None,
        trace.task,
        trace.num_blocks,
        np.asarray(trace.states, dtype=float),
        np.asarray(actions, dtype=float),
        bindings,
        None if layout is None else layout.as_dict(),
    )


def _size(value):
    return len(json.dumps(value, separators=(",", ":"), sort_keys=True).encode())


def _with_full_atoms(view, frame):
    """An event frame that lists every true atom (after an omitted span)."""
    if frame.kind != "event" or frame.relations is not None:
        return frame
    return replace(
        frame,
        relations=tuple(relations(view, frame.t)),
        added=None,
        removed=None,
    )


def _assemble(item, view, frames, tail):
    """Frame 0, then frames[tail:]; frames 1..tail-1 are declared omitted."""
    result = dict(item)
    if tail <= 1:
        result["frames"] = [render_frame(f) for f in frames]
        return result
    kept = [frames[0]] + list(frames[tail:])
    # Changes would refer to an omitted event: list every true atom instead.
    first_event = next((i for i, f in enumerate(kept) if i and f.kind == "event"), None)
    if first_event is not None:
        kept[first_event] = _with_full_atoms(view, kept[first_event])
    result["frames"] = [render_frame(f) for f in kept]
    result["omitted_frames"] = {
        "count": tail - 1,
        "from_t": frames[1].t,
        "to_t": frames[tail - 1].t,
        "note": "frames after the first are omitted up to the most recent frames"
        " that fit the keyframe budget",
    }
    return result


def rdp_keyframes(trace, waypoint_tolerance, budget):
    """Keyframe item within ``budget`` bytes, or None when even frame 0 cannot fit.

    Returns ``(item, delivered_frames, available_frames)``.
    """
    view = demonstration_view(trace)
    frames = compact(view, waypoint_tolerance)
    steps = len(view.states)
    item = {
        "method": "rdp",
        "version": VERSION,
        "waypoint_tolerance": waypoint_tolerance,
        "format": FORMAT,
        "summary": (
            f"{trace.task}, {view.num_blocks} blocks, {steps} states"
            f" (t=0..{steps - 1}), {sum(f.kind == 'event' for f in frames)} events,"
            f" {sum(f.kind == 'waypoint' for f in frames)} waypoints"
        ),
        "inputs": render_inputs(view),
    }
    for tail in range(1, len(frames) + 1):
        result = _assemble(item, view, frames, tail)
        if _size(result) <= budget:
            return result, len(result["frames"]), len(frames)
    return None, 0, len(frames)
