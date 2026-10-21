"""Versioned optional API explanations; no changes to generated-code semantics."""

import hashlib
import inspect
import json
from dataclasses import asdict, is_dataclass

VERSION = "primitive-prompt-v1"
OPTION_NAMES = (
    "explain_target_offsets",
    "explain_convergence_criteria",
    "explain_partial_execution",
    "explain_motion_planning_limits",
)
BLOCKS = {
    "explain_target_offsets": """Target-to-offset conversion (mathematical notation, not new vector APIs):
For each axis: offset = desired_gripper_coordinate - reference_coordinate.
If a block-centre target is intended, first measure block_gripper_displacement =
block_coordinate - gripper_coordinate, then desired_gripper_coordinate =
desired_block_coordinate - block_gripper_displacement. Subtract the reference
coordinate exactly once when constructing the offset. Move itself adds it back.
Example: reference X=1.30 m and desired gripper X=1.40 m require dx=0.10 m.
Passing dx=1.40 m would command X=2.70 m. This numeric example is not scene data.
Read coordinates again after motion; measured attachment offsets may change.
""",
    "explain_convergence_criteria": """Controller phase-completion criteria:
Pick: approach -> open -> descend -> close. Approach/descend each require 3D
Euclidean gripper position error <= {pick_tolerance} m. After closing, Pick does
not recheck its earlier position criteria and does not automatically lift.
Move: completion checks only 3D gripper-to-fixed-target error <= {position_tolerance} m;
commanding closure during motion does not certify a grasp or closed-state test.
Release: open -> retreat. Retreat checks all XYZ axes against the target frozen
before opening, with 3D error <= {position_tolerance} m; reaching Z alone is insufficient.
Open/close use complementary controller finger-state predicates: open when the
sum of its two finger-state values > {gripper_threshold}, closed otherwise.
These internal tests do not expose a raw-state API or certify grasp/support.
Each primitive's phases share {step_limit} simulation steps; this is not a new
allowance per phase. A phase may complete in zero steps. Meeting its criterion
on the last allowed step is success; an unsatisfied criterion is nonconvergence.
""",
    "explain_partial_execution": """Partial execution and failure:
A primitive may change the scene before it fails; effects are not atomically
rolled back. Release can satisfy its open-state test and then fail in retreat.
The controller records nonconvergence; the CaP adapter raises an execution failure
and stops the current rollout, so later policy statements do not execute.
Completed motion or gripper phases do not prove secure grasp or stable placement.
These are general possibilities, not observations of a failure in this scene.
""",
    "explain_motion_planning_limits": """Motion-planning boundary:
Move drives toward its fixed gripper target. It does not plan obstacle avoidance
or insert clearance waypoints. Express any required lift, transfer, and descent
stages explicitly. A reachable target is not guaranteed to converge within the
primitive budget. No universal safe height, maximum displacement, or guaranteed
workspace box is supplied. An API example is not a certified safe task plan.
""",
}


def normalize(options=None):
    values = asdict(options) if is_dataclass(options) else dict(options or {})
    if set(values) - set(OPTION_NAMES):
        raise ValueError("Unknown primitive prompt option")
    resolved = {key: values.get(key, False) for key in OPTION_NAMES}
    if any(type(value) is not bool for value in resolved.values()):
        raise ValueError("Primitive prompt options must be Boolean")
    return resolved


def controller_values():
    from synthesis.api.control import (
        DEFAULT_CONTROL,
        DEFAULT_PICK_CONTROL,
        PrimitiveController,
    )

    return {
        "position_tolerance": DEFAULT_CONTROL.position_tolerance,
        "pick_tolerance": DEFAULT_PICK_CONTROL.position_tolerance,
        "gripper_threshold": format(
            DEFAULT_CONTROL.gripper_threshold + DEFAULT_CONTROL.gripper_tolerance,
            ".12g",
        ),
        "step_limit": inspect.signature(PrimitiveController.__init__)
        .parameters["limit"]
        .default,
    }


def render(options=None):
    flags = normalize(options)
    if not any(flags.values()):
        return ""  # Preserve all-off prompt bytes, including whitespace.
    values = controller_values() if flags["explain_convergence_criteria"] else {}
    return "\nAdditional primitive contract explanations:\n" + "\n".join(
        BLOCKS[key].format(**values) for key, enabled in flags.items() if enabled
    )


def provenance(options=None):
    flags = normalize(options)
    return {
        "primitive_prompt": flags,
        "primitive_prompt_version": VERSION,
        "primitive_prompt_package_sha256": hashlib.sha256(
            json.dumps(BLOCKS, sort_keys=True).encode()
        ).hexdigest(),
        "primitive_prompt_enabled_blocks": [
            key for key, enabled in flags.items() if enabled
        ],
        "primitive_prompt_rendered_sha256": hashlib.sha256(
            render(flags).encode()
        ).hexdigest(),
    }
