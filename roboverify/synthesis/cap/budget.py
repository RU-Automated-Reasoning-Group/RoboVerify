"""One shared budget for generation, repair, queries, and physical execution."""

import time
from dataclasses import dataclass, field

from .contracts import BudgetExceeded
from .timing import PhaseTimings


def effort_tokens(count, demonstration_tokens=0):
    """Provider total tokens of one request, excluding its demonstration block.

    The block is measured once per trial with the same tokenizer; joining it to
    the method prompt can change the count at the boundary by a token, so the
    difference is clamped at zero.
    """
    return max(0, int(count) - int(demonstration_tokens))


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
    # Revision 6: tokens of the demonstration block, charged separately from the
    # effort budget (``tokens``/``max_tokens``), once per request that carries it.
    demonstration_tokens: int = 0
    # Revision 6, §3.6: server-reported cached prompt tokens and the serving
    # instance per request. Diagnostics only; no budget reads them.
    cached_prompt_tokens: int = 0
    cache_unreported_calls: int = 0
    served_by: dict = field(default_factory=dict)
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

    def account_tokens(self, count, demonstration_tokens=0):
        """Charge provider-reported total tokens, minus the demonstration block."""
        self.demonstration_tokens += demonstration_tokens
        if count is None:
            self.unknown_usage_calls += 1
        else:
            self.tokens += effort_tokens(count, demonstration_tokens)
            if self.tokens > self.max_tokens:
                self.exhaust("tokens", "Token budget exhausted")

    def record_prefix_cache(self, cached, backend=None):
        if cached is None:
            self.cache_unreported_calls += 1
        else:
            self.cached_prompt_tokens += cached
        if backend:
            self.served_by[backend] = self.served_by.get(backend, 0) + 1

    def prefix_cache_summary(self):
        return {
            "cached_prompt_tokens": self.cached_prompt_tokens,
            "cache_unreported_calls": self.cache_unreported_calls,
            "served_by": dict(self.served_by),
        }

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
