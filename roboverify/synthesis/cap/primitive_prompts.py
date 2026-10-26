"""Versioned optional API explanations; no changes to generated-code semantics."""

import hashlib
import inspect
import json
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path


def _load(name):
    path = Path(__file__).with_name("prompts") / name
    return json.loads(path.read_text(encoding="utf-8"))


# primitive-prompt-v2 (Revision 7) drops v1's motion-strategy sentence from the
# motion-planning block; v1 stays byte-identical for config_frozen_v1-v4.
PACKAGES = {
    package["version"]: package
    for package in map(_load, ("primitive_v1.json", "primitive_v2.json"))
}
DEFAULT_VERSION = "primitive-prompt-v1"
_PACKAGE = PACKAGES[DEFAULT_VERSION]
VERSION = _PACKAGE["version"]
OPTION_NAMES = (
    "explain_target_offsets",
    "explain_convergence_criteria",
    "explain_partial_execution",
    "explain_motion_planning_limits",
)
BLOCKS = _PACKAGE["blocks"]


def check_version(version):
    if version not in PACKAGES:
        raise ValueError(f"Unknown primitive prompt version: {version}")
    return version


@dataclass(frozen=True)
class PrimitivePrompt:
    """The Boolean options with the package version that renders them.

    Passed wherever the options alone used to go, so prompt builders render the
    version a run selected; bare options use ``DEFAULT_VERSION``.
    """

    options: object = None
    version: str = DEFAULT_VERSION


def _split(options, version):
    if isinstance(options, PrimitivePrompt):
        return options.options, options.version
    return options, version


def normalize(options=None):
    options, _ = _split(options, None)
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


def render(options=None, version=DEFAULT_VERSION):
    options, version = _split(options, version)
    flags = normalize(options)
    if not any(flags.values()):
        return ""  # Preserve all-off prompt bytes, including whitespace.
    package = PACKAGES[check_version(version)]
    values = controller_values() if flags["explain_convergence_criteria"] else {}
    return package["header"] + "\n".join(
        package["blocks"][key].format(**values)
        for key, enabled in flags.items()
        if enabled
    )


def provenance(options=None, version=DEFAULT_VERSION):
    options, version = _split(options, version)
    flags = normalize(options)
    blocks = PACKAGES[check_version(version)]["blocks"]
    return {
        "primitive_prompt": flags,
        "primitive_prompt_version": version,
        "primitive_prompt_package_sha256": hashlib.sha256(
            json.dumps(blocks, sort_keys=True).encode()
        ).hexdigest(),
        "primitive_prompt_enabled_blocks": [
            key for key, enabled in flags.items() if enabled
        ],
        "primitive_prompt_rendered_sha256": hashlib.sha256(
            render(flags, version).encode()
        ).hexdigest(),
    }
