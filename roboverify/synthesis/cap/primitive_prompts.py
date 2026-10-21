"""Versioned optional API explanations; no changes to generated-code semantics."""

import hashlib
import inspect
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path

_PACKAGE = json.loads(
    (Path(__file__).with_name("prompts") / "primitive_v1.json").read_text(
        encoding="utf-8"
    )
)
VERSION = _PACKAGE["version"]
OPTION_NAMES = (
    "explain_target_offsets",
    "explain_convergence_criteria",
    "explain_partial_execution",
    "explain_motion_planning_limits",
)
BLOCKS = _PACKAGE["blocks"]


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
    return _PACKAGE["header"] + "\n".join(
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
