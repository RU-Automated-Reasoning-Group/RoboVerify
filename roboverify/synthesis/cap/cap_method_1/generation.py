"""Source-only recursive generation; no generated Python executes here."""

import ast

from ..checker import require_check, require_helper_check
from ..contracts import (
    ACTIONS,
    API_NAMES,
    Bundle,
    ConstraintFailure,
    Diagnostic,
    api_description,
    validate_condition_mode,
)
from ..prompt_package import package, validate_prompt_version
from ..timing import timed
from ..validation import (
    Value,
    definitions,
    missing_calls,
    parse,
    validate,
    validate_helper,
)


def prompt(
    name,
    *,
    api_names=API_NAMES,
    condition_mode="dsl_only",
    prompt_version="legacy_v1",
    higher_tolerance=0.001,
    unit="policy",
    primitive_prompt=None,
    **fields,
):
    data = package(prompt_version)
    if prompt_version != "legacy_v1":
        fields["output_rule"] = data["output_rules"][
            "policy" if unit == "policy" else "helper"
        ]
    return (
        data["method1_templates"][name]
        .format(
            api=api_description(
                api_names,
                condition_mode,
                prompt_version=prompt_version,
                higher_tolerance=higher_tolerance,
                primitive_prompt=primitive_prompt,
            ),
            **fields,
        )
        .rstrip("\n")
    )


class Generator:
    def __init__(
        self,
        model,
        logger,
        checkpoints,
        budget,
        max_attempts=3,
        api_names=API_NAMES,
        condition_mode="dsl_only",
        prompt_version="legacy_v1",
        higher_tolerance=0.001,
        inputs=(("b0", "object"),),
    ):
        self.prompt_version = validate_prompt_version(prompt_version)
        self.higher_tolerance = higher_tolerance
        # (name, kind) pairs of the bound input handles (goal tasks bind goals).
        self.inputs = {name: Value(kind) for name, kind in inputs}
        self.primitive_prompt = checkpoints.primitive_prompt
        if checkpoints.prompt_version != prompt_version:
            raise ValueError("Checkpoint and generator prompt versions differ")
        self.condition_mode = validate_condition_mode(condition_mode)
        if checkpoints.condition_mode != condition_mode:
            raise ValueError("Checkpoint and generator condition modes differ")
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

    @timed("generation_control")
    def generate(self, instruction, context="b0 is the input block."):
        source = self.model.complete(
            prompt(
                "policy",
                prompt_version=self.prompt_version,
                higher_tolerance=self.higher_tolerance,
                primitive_prompt=self.primitive_prompt,
                api_names=self.api_names,
                condition_mode=self.condition_mode,
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

    @timed("generation_control")
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
                tree = self.budget.timings.call("validation", parse, source, unit)
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
                    self.budget.timings.call(
                        "validation",
                        require_check,
                        Bundle(source, helpers),
                        complete=False,
                        role=role,
                        condition_mode=self.condition_mode,
                        api_names=self.api_names,
                        inputs=self.inputs,
                        logger=self.logger,
                    )
                else:
                    self.budget.timings.call(
                        "validation",
                        require_helper_check,
                        source,
                        unit,
                        helpers,
                        role,
                        condition_mode=self.condition_mode,
                        api_names=self.api_names,
                        inputs=self.inputs,
                        logger=self.logger,
                    )
                self.logger.emit(
                    "local_validation_pass", checkpoint_id=cp, parent_event_id=event
                )
                for call in missing_calls(
                    source, helpers, role, api_names=self.api_names
                ):
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
                            prompt_version=self.prompt_version,
                            higher_tolerance=self.higher_tolerance,
                            primitive_prompt=self.primitive_prompt,
                            unit=call.name,
                            api_names=self.api_names,
                            condition_mode=self.condition_mode,
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
                    self.budget.timings.call(
                        "validation",
                        require_check,
                        Bundle(source, helpers),
                        complete=True,
                        role=role,
                        condition_mode=self.condition_mode,
                        api_names=self.api_names,
                        inputs=self.inputs,
                        logger=self.logger,
                    )
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
                return Bundle(source, helpers, condition_mode=self.condition_mode)
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
                        prompt_version=self.prompt_version,
                        higher_tolerance=self.higher_tolerance,
                        primitive_prompt=self.primitive_prompt,
                        unit=unit,
                        api_names=self.api_names,
                        condition_mode=self.condition_mode,
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
