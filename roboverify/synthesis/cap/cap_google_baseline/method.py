"""Run the original LMP/LMPFGen objects against the shared RoboVerify namespace."""

from ..contracts import validate_condition_mode
from .compatibility import load_core, module_config


class GoogleBaseline:
    def __init__(
        self,
        model,
        adapter,
        logger,
        checkpoints,
        budget,
        *,
        condition_mode="dsl_only",
        prompt_version="legacy_v1",
        higher_tolerance=0.001,
        **kwargs,
    ):
        self.condition_mode = validate_condition_mode(condition_mode)
        if checkpoints.condition_mode != condition_mode:
            raise ValueError("Checkpoint and method condition modes differ")
        if checkpoints.prompt_version != prompt_version:
            raise ValueError("Checkpoint and method prompt versions differ")
        self.logger = logger
        core = load_core(model, logger, checkpoints, budget)
        fixed = {}
        variable = adapter.namespace()

        class ObservedFGen(core.LMPFGen):
            depth = 0

            def create_new_fs_from_code(self, *args, **kwargs):
                self.depth += 1
                try:
                    budget.depth(self.depth)
                    event = logger.emit("helper_resolution_start", depth=self.depth)
                    with logger.scoped(parent_event_id=event, depth=self.depth):
                        result = super().create_new_fs_from_code(*args, **kwargs)
                        if self.depth == 1:
                            core.final_output.ready = True
                            generated = (
                                result[0] if isinstance(result, tuple) else result
                            )
                            core.final_output.capture(variable | generated)
                        return result
                finally:
                    self.depth -= 1

            def create_f_from_sig(self, name, signature, *args, **kwargs):
                event = logger.emit(
                    "helper_generation_start", helper=name, signature=signature
                )
                with logger.scoped(helper=name, parent_event_id=event):
                    return super().create_f_from_sig(name, signature, *args, **kwargs)

        fgen = ObservedFGen(
            module_config(
                helper=True,
                api_names=adapter.api_names,
                condition_mode=condition_mode,
                prompt_version=prompt_version,
                higher_tolerance=higher_tolerance,
            ),
            fixed,
            variable,
        )
        self.policy = core.LMP(
            "policy",
            module_config(
                api_names=adapter.api_names,
                condition_mode=condition_mode,
                prompt_version=prompt_version,
                higher_tolerance=higher_tolerance,
            ),
            fgen,
            fixed,
            variable,
        )

    def __call__(self, instruction, context="b0 is the input block."):
        self.logger.emit(
            "baseline_start",
            validation_enabled=False,
            refinement_enabled=False,
            condition_mode=self.condition_mode,
        )
        result = self.policy(instruction, context=context)
        self.logger.emit("baseline_end", result=result)
        return result
