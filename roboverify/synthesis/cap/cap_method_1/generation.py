"""Source-only recursive generation; no generated Python executes here."""

import ast
from pathlib import Path

from ..contracts import (
    ACTIONS,
    API_NAMES,
    Bundle,
    ConstraintFailure,
    Diagnostic,
    api_description,
)
from ..validation import definitions, missing_calls, parse, validate, validate_helper

PROMPTS = Path(__file__).with_name("prompts")


def prompt(name, *, api_names=API_NAMES, **fields):
    return (
        (PROMPTS / f"{name}.txt")
        .read_text()
        .format(api=api_description(api_names), **fields)
    ).rstrip("\n")


class Generator:
    def __init__(
        self, model, logger, checkpoints, budget, max_attempts=3, api_names=API_NAMES
    ):
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        self.model, self.logger, self.checkpoints, self.budget = (
            model,
            logger,
            checkpoints,
            budget,
        )
        self.max_attempts = max_attempts
        self.api_names = api_names

    def generate(self, instruction, context="b0 is the input block."):
        source = self.model.complete(
            prompt(
                "policy",
                api_names=self.api_names,
                instruction=instruction,
                context=context,
            ),
            purpose="policy",
        )
        bundle = self.prepare(source, {}, "body", root_source=source)
        bundle.checkpoint_id = self.checkpoints.save(
            bundle.source, bundle.helpers, reason="execution_bundle", validation="valid"
        )
        return bundle

    def prepare(
        self,
        source,
        known,
        role,
        *,
        root_source,
        unit="policy",
        ancestry=(),
        pending=None,
    ):
        self.budget.depth(len(ancestry))
        pending = dict(pending or {})
        helpers = dict(known)
        for attempt in range(self.max_attempts):
            self.budget.remaining()
            assembled = pending | helpers | ({unit: source} if unit != "policy" else {})
            cp = self.checkpoints.save(
                source if unit == "policy" else root_source,
                assembled,
                reason="initial_source" if attempt == 0 else "refined_source",
                unit=unit,
                role=role,
                attempt=attempt,
            )
            event = self.logger.emit(
                "validation_start", checkpoint_id=cp, helper=unit, role=role
            )
            try:
                tree = parse(source, unit)
                for node in ast.walk(tree):
                    if (
                        isinstance(node, ast.Call)
                        and node.func.id in API_NAMES - self.api_names
                    ):
                        raise ConstraintFailure(
                            [
                                Diagnostic(
                                    "API_UNAVAILABLE",
                                    node.func.id,
                                    node.lineno,
                                    node.col_offset,
                                    unit,
                                )
                            ]
                        )
                if unit != "policy":
                    definitions(Bundle("", {unit: source}))
                if role == "condition":
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Call) and node.func.id in ACTIONS:
                            raise ConstraintFailure(
                                [
                                    Diagnostic(
                                        "ACTION_IN_CONDITION",
                                        node.func.id,
                                        node.lineno,
                                        node.col_offset,
                                        unit,
                                    )
                                ]
                            )
                if unit == "policy":
                    validate(Bundle(source, helpers), complete=False, role=role)
                else:
                    validate_helper(source, unit, helpers, role)
                self.logger.emit(
                    "local_validation_pass", checkpoint_id=cp, parent_event_id=event
                )
                for call in missing_calls(source, helpers, role):
                    if call.name in ancestry or call.name == unit:
                        raise ConstraintFailure(
                            [Diagnostic("DEPENDENCY_CYCLE", call.name, unit=unit)]
                        )
                    if call.name in helpers:
                        continue
                    self.logger.emit(
                        "helper_missing",
                        helper=call.name,
                        signature=call.signature,
                        role=call.role,
                        depth=len(ancestry),
                        checkpoint_id=cp,
                    )
                    definition = self.model.complete(
                        prompt(
                            "helper",
                            api_names=self.api_names,
                            signature=call.signature,
                            role=call.role,
                        ),
                        purpose="helper",
                        helper=call.name,
                        parent_event_id=event,
                    )
                    child = self.prepare(
                        definition,
                        helpers,
                        call.role,
                        root_source=source if unit == "policy" else root_source,
                        unit=call.name,
                        ancestry=ancestry + (unit,),
                        pending=pending | ({unit: source} if unit != "policy" else {}),
                    )
                    helpers.update(child.helpers)
                    helpers[call.name] = child.source
                if unit == "policy":
                    validate(Bundle(source, helpers), complete=True, role=role)
                # Parent-level complete checking uses actual argument types and value flow.
                self.checkpoints.save(
                    source if unit == "policy" else root_source,
                    pending | helpers | ({unit: source} if unit != "policy" else {}),
                    reason="validation_result",
                    validation="valid" if unit == "policy" else "local_pass",
                    parent=cp,
                    unit=unit,
                )
                self.logger.emit(
                    "validation_pass",
                    helper=unit,
                    scope="complete" if unit == "policy" else "local",
                    checkpoint_id=cp,
                )
                return Bundle(source, helpers)
            except ConstraintFailure as exc:
                self.checkpoints.save(
                    source if unit == "policy" else root_source,
                    pending | helpers | ({unit: source} if unit != "policy" else {}),
                    reason="validation_result",
                    validation="invalid",
                    diagnostics=exc.diagnostics,
                    parent=cp,
                    unit=unit,
                )
                self.logger.emit(
                    "validation_failed",
                    helper=unit,
                    checkpoint_id=cp,
                    diagnostics=[d.to_dict() for d in exc.diagnostics],
                )
                if attempt + 1 == self.max_attempts:
                    raise
                self.logger.emit(
                    "refine_start", helper=unit, checkpoint_id=cp, attempt=attempt + 1
                )
                source = self.model.complete(
                    prompt(
                        "repair",
                        api_names=self.api_names,
                        source=source,
                        role=role,
                        diagnostics=str(exc),
                    ),
                    purpose="repair",
                    helper=unit,
                    parent_event_id=event,
                )
                # Discard speculative descendants of a failed candidate.
                helpers = dict(known)
        raise AssertionError("Unreachable preparation state")
