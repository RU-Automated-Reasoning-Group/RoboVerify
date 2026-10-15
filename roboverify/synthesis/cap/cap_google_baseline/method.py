"""Run the original LMP/LMPFGen objects against the shared RoboVerify namespace."""

from .compatibility import load_core, module_config


class GoogleBaseline:
    def __init__(self, model, adapter, logger, checkpoints, budget, **kwargs):
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
                        return super().create_new_fs_from_code(*args, **kwargs)
                finally:
                    self.depth -= 1

            def create_f_from_sig(self, name, signature, *args, **kwargs):
                event = logger.emit(
                    "helper_generation_start", helper=name, signature=signature
                )
                with logger.scoped(helper=name, parent_event_id=event):
                    return super().create_f_from_sig(name, signature, *args, **kwargs)

        fgen = ObservedFGen(
            module_config(helper=True, api_names=adapter.api_names), fixed, variable
        )
        self.policy = core.LMP(
            "policy", module_config(api_names=adapter.api_names), fgen, fixed, variable
        )

    def __call__(self, instruction, context="b0 is the input block."):
        self.logger.emit(
            "baseline_start", validation_enabled=False, refinement_enabled=False
        )
        result = self.policy(instruction, context=context)
        self.logger.emit("baseline_end", result=result)
        return result
