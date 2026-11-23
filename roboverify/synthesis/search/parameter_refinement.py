"""Bounded coordinate trials around an already evaluated program's floats.

The caller owns feasibility screens, simulation, imitation and hard acceptance.
This helper changes no instructions, operands or controllers and never evaluates
the original again. It supplies deterministic trials and a common deadline.
"""

import math
from copy import deepcopy
from dataclasses import dataclass
from time import perf_counter

from synthesis.cfg.program_source import program_fingerprint


@dataclass(frozen=True)
class ParameterRefinementBudget:
    step: float = 0.001
    max_trials: int = 40
    seconds: float = 600.0
    coordinate_multipliers: tuple = (1, 2)
    deadline: float | None = None

    def __post_init__(self):
        if type(self.max_trials) is not int or self.max_trials < 1:
            raise ValueError("max_trials must be a positive integer")
        for name in ("step", "seconds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        multipliers = tuple(self.coordinate_multipliers)
        if not multipliers or any(
            isinstance(value, bool) or not math.isfinite(value) or value <= 0
            for value in multipliers
        ):
            raise ValueError("Coordinate multipliers must be positive and finite")
        if len(set(multipliers)) != len(multipliers):
            raise ValueError("Coordinate multipliers must be distinct")
        object.__setattr__(self, "coordinate_multipliers", multipliers)
        if self.deadline is not None and (
            isinstance(self.deadline, bool) or not math.isfinite(self.deadline)
        ):
            raise ValueError("Absolute deadline must be finite")


@dataclass(frozen=True)
class ParameterRefinementResult:
    program: object | None
    report: dict | None
    status: str
    history: tuple
    seconds: float
    original_fingerprint: str

    @property
    def accepted(self):
        return self.program is not None

    def __bool__(self):
        return self.accepted


class _LocalDeadline(TimeoutError):
    pass


def refine_parameters(program, evaluate, *, budget=None, on_trial=None):
    """Try every registered coordinate at +step, -step, +2step, -2step.

    The outer order is ``coordinate_multipliers``, then registration order,
    then positive/negative sign. Every trial starts from the original vector;
    failed trials never become a new centre. Registration occurs on a copy.

    ``on_trial(trial, index, deadline)`` runs before evaluation so the caller can
    freeze the exact executable. ``evaluate(trial, index, deadline)`` returns a
    dictionary. Only explicit ``accepted=True`` AND ``complete=True`` can be
    promoted; a ``budget_exhausted=True`` report stops the search. Callbacks must
    leave executable parameters and structure unchanged. Their other exceptions,
    including external TimeoutError, propagate. The original is never evaluated.
    """
    budget = budget or ParameterRefinementBudget()
    started = perf_counter()
    deadline = min(
        value
        for value in (budget.deadline, started + budget.seconds)
        if value is not None
    )
    original_fingerprint = program_fingerprint(program)
    template = deepcopy(program)
    parameters = tuple(
        float(value) for value in template.register_trainable_parameter()
    )
    if not all(math.isfinite(value) for value in parameters):
        raise ValueError("Refinement requires finite original parameters")
    history = []
    active = None

    def finish(status, accepted=None, report=None):
        return ParameterRefinementResult(
            accepted,
            report,
            status,
            tuple(history),
            perf_counter() - started,
            original_fingerprint,
        )

    def check_deadline():
        if perf_counter() >= deadline:
            raise _LocalDeadline()

    try:
        check_deadline()
        if not parameters:
            return finish("no_parameters")
        for multiplier in budget.coordinate_multipliers:
            for coordinate in range(len(parameters)):
                for sign in (1, -1):
                    check_deadline()
                    if len(history) >= budget.max_trials:
                        return finish("trial_budget_exhausted")
                    delta = sign * multiplier * budget.step
                    values = list(parameters)
                    values[coordinate] += delta
                    if not math.isfinite(values[coordinate]):
                        raise ValueError(
                            "Refinement step produced a nonfinite parameter"
                        )
                    trial = deepcopy(template)
                    trial.update_trainable_parameter(values)
                    fingerprint = program_fingerprint(trial)
                    index = len(history)
                    began = perf_counter()
                    row = dict(
                        index=index,
                        coordinate=coordinate,
                        delta=delta,
                        parameters=values,
                        fingerprint=fingerprint,
                        evaluated=False,
                        accepted=False,
                        report=None,
                        seconds=0.0,
                    )
                    history.append(row)
                    active = (row, began)
                    if on_trial is not None:
                        on_trial(trial, index, deadline)
                    if program_fingerprint(trial) != fingerprint:
                        raise ValueError("on_trial changed the executable being frozen")
                    check_deadline()
                    report = evaluate(trial, index, deadline)
                    if not isinstance(report, dict):
                        raise TypeError("Refinement evaluate must return a dictionary")
                    row.update(
                        evaluated=True, report=report, seconds=perf_counter() - began
                    )
                    if program_fingerprint(trial) != fingerprint:
                        raise ValueError("evaluate changed the trial executable")
                    check_deadline()
                    if report.get("budget_exhausted") is True:
                        return finish("budget_exhausted")
                    if (
                        report.get("accepted") is True
                        and report.get("complete") is True
                    ):
                        row["accepted"] = True
                        return finish("accepted", trial, report)
                    active = None
        return finish("space_exhausted")
    except _LocalDeadline:
        if active is not None:
            row, began = active
            row["accepted"] = False
            row["deadline_exhausted"] = True
            row["seconds"] = perf_counter() - began
        return finish("deadline_exhausted")
