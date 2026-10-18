"""One shared budget for generation, repair, queries, and physical execution."""

import time
from dataclasses import dataclass, field

from .contracts import BudgetExceeded
from .timing import PhaseTimings


@dataclass
class Budget:
    timeout: float = 600.0
    max_model_calls: int = 30
    max_steps: int = 10000
    max_depth: int = 12
    max_tokens: int = 100000
    max_sim_steps: int = 10000
    simulation_steps: int = 0
    max_observed_depth: int = 0
    exhausted_by: str | None = None
    timings: PhaseTimings = field(default_factory=PhaseTimings, repr=False)
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
                self.max_sim_steps,
            )
        ):
            raise ValueError("Budget limits must be positive")

    def exhaust(self, kind, message):
        self.exhausted_by = self.exhausted_by or kind
        raise BudgetExceeded(message)

    def remaining(self):
        remaining = self.timeout - (time.monotonic() - self.started)
        if remaining <= 0:
            self.exhaust("wall_clock", "Wall-clock budget exhausted")
        return remaining

    def tick(self):
        self.remaining()
        self.steps += 1
        if self.steps > self.max_steps:
            self.exhaust("execution_ticks", "Execution/query step budget exhausted")

    def model_call(self):
        self.remaining()
        if self.calls >= self.max_model_calls:
            self.exhaust("model_calls", "Model-call budget exhausted")
        self.calls += 1

    def account_tokens(self, count):
        if count is None:
            self.unknown_usage_calls += 1
        else:
            self.tokens += int(count)
            if self.tokens > self.max_tokens:
                self.exhaust("tokens", "Token budget exhausted")

    def simulation_step(self):
        self.remaining()
        if self.simulation_steps >= self.max_sim_steps:
            self.exhaust("simulation_steps", "Simulation-step budget exhausted")
        self.simulation_steps += 1

    def depth(self, depth):
        self.remaining()
        self.max_observed_depth = max(self.max_observed_depth, depth)
        if depth > self.max_depth:
            self.exhaust("helper_depth", "Dependency/call depth budget exhausted")
