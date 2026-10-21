"""Complete policy generation and interface-only whole-candidate repair."""

from ..checker import AnalysisFailure, DSLContract, check_bundle
from ..contracts import Bundle, ConstraintFailure, Diagnostic, api_description
from ..prompt_package import package
from ..runtime import Runtime
from ..timing import timed


class Method2:
    def __init__(
        self,
        model,
        adapter,
        logger,
        checkpoints,
        budget,
        max_attempts=3,
        *,
        condition_mode="dsl_only",
        prompt_version="typed_v3",
        higher_tolerance=0.001,
    ):
        self.templates = package(prompt_version).get("method2_templates")
        if self.templates is None:
            raise ValueError(
                "Method 2 requires a prompt package with method2_templates (typed_v3)"
            )
        if (
            checkpoints.prompt_version != prompt_version
            or checkpoints.condition_mode != condition_mode
        ):
            raise ValueError("Checkpoint and method configuration differ")
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        self.model, self.adapter, self.logger = model, adapter, logger
        self.checkpoints, self.budget, self.max_attempts = (
            checkpoints,
            budget,
            max_attempts,
        )
        self.contract = DSLContract(condition_mode, tuple(sorted(adapter.api_names)))
        self.api = api_description(
            adapter.api_names,
            condition_mode,
            prompt_version=prompt_version,
            higher_tolerance=higher_tolerance,
            primitive_prompt=checkpoints.primitive_prompt,
        )
        self.runtime = Runtime(adapter, budget, logger, condition_mode)
        self.generator = self  # Same runner generate-only interface as Method 1.

    @timed("generation_control")
    def generate(self, instruction, context="b0 is the input block."):
        source = self.model.complete(
            self.templates["policy"]
            .format(api=self.api, instruction=instruction, context=context)
            .rstrip("\n"),
            purpose="policy",
        )
        for attempt in range(self.max_attempts):
            self.budget.remaining()
            bundle = Bundle(source, condition_mode=self.contract.condition_mode)
            cp = self.checkpoints.save(
                source,
                reason="initial_source" if attempt == 0 else "refined_source",
                attempt=attempt,
            )
            report = self.budget.timings.call(
                "validation", check_bundle, bundle, self.contract
            )
            self.logger.artifact(
                f"checkpoints/{cp}/check_report.json", report.to_dict()
            )
            self.logger.emit(
                "validation_result",
                checkpoint_id=cp,
                status=report.status,
                attempt=attempt,
            )
            if report.status == "pass":
                bundle.checkpoint_id = self.checkpoints.save(
                    source, reason="execution_bundle", validation="valid"
                )
                return bundle
            if report.status == "error":
                raise AnalysisFailure(report.reason)
            if attempt + 1 == self.max_attempts:
                raise ConstraintFailure([Diagnostic(**d) for d in report.diagnostics])
            self.logger.emit(
                "refine_start",
                checkpoint_id=cp,
                attempt=attempt + 1,
                scope="whole_candidate",
            )
            source = self.model.complete(
                self.templates["repair"]
                .format(api=self.api, source=source, diagnostics=report.reason)
                .rstrip("\n"),
                purpose="repair",
            )
        raise AssertionError("Unreachable generation state")

    def __call__(self, instruction, context="b0 is the input block."):
        bundle = self.generate(instruction, context)
        return self.runtime.execute(bundle)[0]
