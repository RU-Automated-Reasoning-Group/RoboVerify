"""RDP keyframes of a failed rollout for recovery feedback (Revision 7.2).

The recorded rollout is compacted exactly like a demonstration
(``demonstrations.compact``): event frames at the first and last state and at
every gripper-command switch, with block positions and DSL predicate changes,
plus Ramer-Douglas-Peucker corners of the gripper path between events.

The result is a plain-text section laid out like one demonstration of the
demonstration block: a summary line, the inputs (and goals) lines and the
rendered frames (``recovery-keyframes-v2``). It follows the evidence object in
the recovery request, which carries only an index entry for it.

The section has its own byte budget. When the frames do not fit, the first
frame and the most recent frames (nearest the failure) are kept, a marker line
declares the omitted span, and the first event kept after the gap lists every
true atom instead of changes relative to an omitted frame.
"""

from dataclasses import replace

from .demonstrations import (
    Demonstration,
    compact,
    number,
    relations,
    render_frame,
    render_inputs,
    summary_line,
)

VERSION = "recovery-keyframes-v2"
TITLE = "FAILED EXECUTION KEYFRAMES"
END = "END OF FAILED EXECUTION"


def header(tolerance):
    return (
        f"{TITLE} ({VERSION})\n"
        "The latest attempt's recorded execution, compacted with the demonstration"
        " format (demo-keyframes-v1). It is debugging evidence of the failed"
        " attempt, not a demonstration.\n"
        "Format:\n"
        f"- Settings: waypoint_tolerance={number(tolerance)} m. Positions are"
        " world-frame (x,y,z) block or gripper centres in metres with 3 decimals;"
        " t is the recorded control step.\n"
        "- o<i> is physical block i: block_<i> in the evidence above (see"
        " object_handle_to_physical_id); g<j> is goal j (goal_<j>). The inputs line"
        " gives this execution's value of each input handle.\n"
        "- Gripper command: open or closed, the latest gripper command issued"
        " before step t. holding names the block within 0.030 m of a closed"
        " gripper, or none.\n"
        "- event frames are the first and last step and every step at which the"
        " gripper command switches between open and closed. They list every block"
        " position and the DSL predicates of your API: holds lists every true atom,"
        " and changes lists the atoms that became true (+) or false (-) since the"
        " previous listed event.\n"
        "- waypoint frames are corners of the gripper's 3D path between events"
        " (Ramer-Douglas-Peucker simplification with waypoint_tolerance). They list"
        " only the gripper and the held block.\n\n"
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


def _with_full_atoms(view, frame):
    """An event frame that lists every true atom (after an omitted span)."""
    if frame.kind != "event" or frame.relations is not None:
        return frame
    return replace(
        frame, relations=tuple(relations(view, frame.t)), added=None, removed=None
    )


def _render(view, frames, tail, round_index, tolerance):
    """Frame 0, then frames[tail:]; frames 1..tail-1 are declared omitted."""
    lines = [f"FAILED EXECUTION (round {round_index}): " + summary_line(view, frames)]
    lines += render_inputs(view)
    if tail <= 1:
        lines += [render_frame(f) for f in frames]
    else:
        kept = list(frames[tail:])
        # Changes would refer to an omitted event: list every true atom instead.
        first = next((i for i, f in enumerate(kept) if f.kind == "event"), None)
        if first is not None:
            kept[first] = _with_full_atoms(view, kept[first])
        lines.append(render_frame(frames[0]))
        lines.append(
            f"... {tail - 1} frames omitted (t={frames[1].t}..{frames[tail - 1].t})"
            " to fit the keyframe budget ..."
        )
        lines += [render_frame(f) for f in kept]
    return header(tolerance) + "\n".join(lines) + "\n" + END + "\n"


def rdp_keyframes(trace, round_index, waypoint_tolerance, budget):
    """Plain-text keyframe section within ``budget`` UTF-8 bytes.

    Returns ``(text, delivered_frames, available_frames)``; ``text`` is None when
    even the first frame cannot fit.
    """
    view = demonstration_view(trace)
    frames = compact(view, waypoint_tolerance)
    for tail in range(1, len(frames) + 1):
        text = _render(view, frames, tail, round_index, waypoint_tolerance)
        if len(text.encode("utf-8")) <= budget:
            delivered = len(frames) if tail <= 1 else len(frames) - tail + 1
            return text, delivered, len(frames)
    return None, 0, len(frames)
