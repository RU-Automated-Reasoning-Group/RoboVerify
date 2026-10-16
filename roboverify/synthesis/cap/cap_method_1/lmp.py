"""Plan 1 policy invocation, separated from provider and simulator details."""

from ..runtime import Runtime
from .generation import Generator


class Method1:
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
    ):
        self.generator = Generator(
            model,
            logger,
            checkpoints,
            budget,
            max_attempts,
            adapter.api_names,
            condition_mode,
        )
        self.runtime = Runtime(adapter, budget, logger, condition_mode)

    def __call__(self, instruction, context="b0 is the input block."):
        bundle = self.generator.generate(instruction, context)
        return self.runtime.execute(bundle)[0]
