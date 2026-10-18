"""Immutable, reconstructable source bundles, including failed generations."""

import difflib
import hashlib
from pathlib import Path

from .contracts import CHECKER_VERSION, OBSERVATION_API_VERSION, validate_condition_mode
from .prompt_package import prompt_provenance


class Checkpoints:
    def __init__(
        self, logger, *, method, condition_mode="dsl_only", prompt_version="legacy_v1"
    ):
        self.provenance = prompt_provenance(prompt_version) | {
            "method": method,
            "condition_mode": validate_condition_mode(condition_mode),
        }
        self.prompt_version = prompt_version
        if logger.prompt_provenance and logger.prompt_provenance != self.provenance:
            raise ValueError("Cannot mix prompt packages within a trial")
        logger.prompt_provenance = dict(self.provenance)
        self.condition_mode = validate_condition_mode(condition_mode)
        self.logger, self.method = logger, method
        self.counter = 0
        self.previous = None
        self.previous_text = ""
        self.texts = {}
        self.active = None
        self.checker_hash = (
            hashlib.sha256(
                Path(__file__).with_name("validation.py").read_bytes()
            ).hexdigest()
            if method == "cap_method_1"
            else None
        )

    def save(
        self,
        source,
        helpers=None,
        *,
        reason,
        validation="pending",
        diagnostics=(),
        parent=None,
        **metadata,
    ):
        self.counter += 1
        cid = f"cp_{self.counter:04d}"
        directory = f"checkpoints/{cid}"
        sources = {"policy.py": source}
        for name, definition in sorted((helpers or {}).items()):
            if not name.isidentifier():
                raise ValueError("Invalid helper name")
            sources[f"helpers/{name}.py"] = definition
        paths = {}
        for name, text in sources.items():
            path = self.logger.artifact(f"{directory}/{name}", text, print_content=True)
            paths[name] = {
                "path": self.logger.relative(path),
                "sha256": hashlib.sha256(text.encode()).hexdigest(),
                "generation_matches": list(
                    self.logger.source_generations.get(
                        hashlib.sha256(text.strip().encode()).hexdigest(), []
                    )
                ),
            }
        combined = "\n".join(f"# {name}\n{text}" for name, text in sources.items())
        parent_id = parent or self.previous
        parent_text = self.texts[parent_id] if parent_id else ""
        self.logger.artifact(
            f"{directory}/diff.patch",
            "".join(
                difflib.unified_diff(
                    parent_text.splitlines(True),
                    combined.splitlines(True),
                    fromfile=parent_id or "empty",
                    tofile=cid,
                )
            ),
        )
        self.logger.artifact(
            f"{directory}/validation.json",
            {
                "status": validation,
                "diagnostics": [d.to_dict() for d in diagnostics],
            },
        )
        self.logger.artifact(
            f"{directory}/manifest.json",
            {
                **self.provenance,
                "checkpoint_id": cid,
                "parent_checkpoint_id": parent_id,
                "method": self.method,
                "condition_mode": self.condition_mode,
                "observation_api_version": OBSERVATION_API_VERSION,
                "checker_version": (
                    CHECKER_VERSION if self.method == "cap_method_1" else None
                ),
                "refinement_enabled": self.method == "cap_method_1",
                "checker_sha256": self.checker_hash,
                "reason": reason,
                "validation": validation,
                "validation_enabled": self.method == "cap_method_1",
                "execution_eligible": validation in {"valid", "not_run"},
                "sources": paths,
                **metadata,
            },
        )
        self.previous, self.previous_text = cid, combined
        self.texts[cid] = combined
        self.logger.emit(
            "checkpoint", checkpoint_id=cid, reason=reason, validation=validation
        )
        if reason == "execution_bundle":
            self.active = cid
            self.logger.context["checkpoint_id"] = cid
        return cid
