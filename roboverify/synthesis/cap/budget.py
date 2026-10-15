"""One shared budget for generation, repair, queries, and physical execution."""

import time
from dataclasses import dataclass, field

from .contracts import BudgetExceeded


@dataclass
class Budget:
    timeout: float = 60.0
    max_model_calls: int = 30
    max_steps: int = 10000
    max_depth: int = 12
    max_tokens: int = 100000
    calls: int = 0
    steps: int = 0
    tokens: int = 0
    unknown_usage_calls: int = 0
    started: float = field(default_factory=time.monotonic)

    def __post_init__(self):
        if self.timeout <= 0 or any(
            v <= 0
            for v in (
                self.max_model_calls,
                self.max_steps,
                self.max_depth,
                self.max_tokens,
            )
        ):
            raise ValueError("Budget limits must be positive")

    def remaining(self):
        remaining = self.timeout - (time.monotonic() - self.started)
        if remaining <= 0:
            raise BudgetExceeded("Wall-clock budget exhausted")
        return remaining

    def tick(self):
        self.remaining()
        self.steps += 1
        if self.steps > self.max_steps:
            raise BudgetExceeded("Execution/query step budget exhausted")

    def model_call(self):
        self.remaining()
        if self.calls >= self.max_model_calls:
            raise BudgetExceeded("Model-call budget exhausted")
        self.calls += 1

    def account_tokens(self, count):
        if count is None:
            self.unknown_usage_calls += 1
        else:
            self.tokens += int(count)
            if self.tokens > self.max_tokens:
                raise BudgetExceeded("Token budget exhausted")

    def depth(self, depth):
        self.remaining()
        if depth > self.max_depth:
            raise BudgetExceeded("Dependency/call depth budget exhausted")
