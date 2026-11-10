"""Immutable per-trial budgets for the matched comparison."""

import hashlib
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class TrialConfig:
    task: str
    method: str
    seed: int = 0
    wall_seconds: float = 48.0
    structure_seconds: float = 20.0
    max_rollouts: int = 45
    max_length: int = 8
    max_transition_trials: int = 100000
    beam_width: int = 24
    max_expansions: int = 100000
    shortlist: int = 12
    cem_iterations: int = 1
    cem_samples: int = 2
    cem_elites: int = 1
    cem_std: float = 0.004
    parameter_radius: float = 0.04
    temperature: float = 0.01
    decay: float = 0.999
    temperature_floor: float = 0.001
    solver_timeout_ms: int = 250

    def __post_init__(self):
        if self.task not in ("stack", "unstack") or self.method not in (
            "beam",
            "matched_mcmc",
        ):
            raise ValueError("Unsupported comparison task/method")
        if not 0 < self.structure_seconds < self.wall_seconds <= 49:
            raise ValueError(
                "Structure budget must leave fitting time within the 49-second maximum"
            )
        if (
            self.max_rollouts < 1
            or self.max_length < 1
            or self.max_transition_trials < 1
        ):
            raise ValueError("Positive work budgets required")
        if (
            self.cem_iterations < 0
            or not 1 <= self.cem_elites <= self.cem_samples
            or self.cem_std <= 0
        ):
            raise ValueError("Invalid common CEM configuration")

    def as_dict(self):
        return asdict(self)

    def digest(self):
        return hashlib.sha256(
            json.dumps(self.as_dict(), sort_keys=True).encode()
        ).hexdigest()


def candidate_seed(seed, rows):
    """Same proposal and trial seed receive the same CEM noise in either arm."""
    payload = json.dumps(
        {"seed": int(seed), "instructions": rows}, sort_keys=True, separators=(",", ":")
    )
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:4], "big")
