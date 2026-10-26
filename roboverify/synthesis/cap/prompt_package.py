"""Versioned model-facing text; independent of simulator and checker semantics."""

import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path

from .contracts import GOAL_API_NAMES

# typed_v4 = typed_v3 plus the goal API, rendered only for goal tasks.
# typed_v5 = typed_v4 plus the demonstration block templates (Revision 6); with
# demonstrations off it renders typed_v4 text.
PROMPT_VERSIONS = ("legacy_v1", "typed_v2", "typed_v3", "typed_v4", "typed_v5")
PROMPT_ROOT = Path(__file__).with_name("prompts")


def validate_prompt_version(version):
    if version not in PROMPT_VERSIONS:
        raise ValueError(f"Unknown prompt version: {version}")
    return version


@lru_cache(maxsize=len(PROMPT_VERSIONS))
def package_bytes(version):
    validate_prompt_version(version)
    return (PROMPT_ROOT / f"{version}.json").read_bytes()


def package(version):
    return json.loads(package_bytes(version))


def prompt_provenance(version, primitive_prompt=None):
    from .primitive_prompts import provenance

    return {
        **provenance(primitive_prompt),
        "prompt_version": validate_prompt_version(version),
        "prompt_package_sha256": hashlib.sha256(package_bytes(version)).hexdigest(),
    }


def render_api(names, condition_mode, version, higher_tolerance, primitive_prompt=None):
    data = package(version)
    if version == "legacy_v1":
        description = "\n".join(
            line
            for line in data["api_description"].splitlines()
            if "ON_star_zero" in names or not line.startswith("ON_star_zero")
        )
    else:
        if not math.isfinite(higher_tolerance) or not 0 <= higher_tolerance < 0.025:
            raise ValueError("Higher tolerance must be finite and in [0, 0.025) metres")
        texts = dict(data)
        variant = data.get("goal_variant")
        if variant is not None and set(names) & GOAL_API_NAMES:
            texts |= {k: variant[k] for k in ("api_header", "api_footer", "examples")}
            entries = data["api_entries"] | variant["entry_overrides"]
        else:
            entries = data["api_entries"]
        description = texts["api_header"] + "\n".join(
            entry.format(higher_tolerance=repr(float(higher_tolerance)))
            for name, entry in entries.items()
            if name in names
        )
        description += texts["api_footer"] + texts["examples"]
        description = description.rstrip("\n")
    from .primitive_prompts import render

    return (
        description
        + render(primitive_prompt)
        + "\n"
        + data["condition_rules"][condition_mode].rstrip("\n")
    )
