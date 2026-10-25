"""Shared bounded, source-only DSL analysis for generation and post-hoc audit."""

import hashlib
import json
import math
import signal
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .contracts import (
    API_NAMES,
    CHECKER_VERSION,
    OBSERVATION_API_VERSION,
    BudgetExceeded,
    CapError,
    ConstraintFailure,
    Diagnostic,
    task_api_names,
    task_inputs,
    validate_condition_mode,
)
from .validation import OBJECT, Value, validate, validate_helper


class AnalysisFailure(CapError):
    pass


@dataclass(frozen=True)
class DSLContract:
    condition_mode: str = "dsl_only"
    api_names: tuple = tuple(sorted(API_NAMES))
    input_types: tuple = (("b0", "object"),)
    checker_version: str = CHECKER_VERSION
    observation_api_version: str = OBSERVATION_API_VERSION

    def __post_init__(self):
        validate_condition_mode(self.condition_mode)
        if not set(self.api_names) <= API_NAMES:
            raise ValueError("Unknown contract API")
        if (
            self.checker_version != CHECKER_VERSION
            or self.observation_api_version != OBSERVATION_API_VERSION
        ):
            raise ValueError("Unsupported executable contract version")

    @classmethod
    def for_task(cls, task, mode="dsl_only"):
        return cls(
            mode,
            tuple(sorted(task_api_names(task))),
            tuple((name, "object") for name in sorted(task_inputs(task))),
        )

    def identity(self):
        data = asdict(self)
        data["sha256"] = digest(data)
        data["checker_source_sha256"] = hashlib.sha256(
            Path(__file__).read_bytes()
            + Path(__file__).with_name("validation.py").read_bytes()
        ).hexdigest()
        return data


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass
class CheckReport:
    status: str
    scope: str
    role: str
    contract: dict
    bundle_sha256: str
    diagnostics: list = field(default_factory=list)
    unresolved: list = field(default_factory=list)
    coverage: str = "complete"
    elapsed_seconds: float = 0.0
    reason: str = ""

    def to_dict(self):
        return asdict(self)


@contextmanager
def analysis_deadline(seconds):
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("Analysis timeout must be finite and positive")
    if threading.current_thread() is not threading.main_thread():
        raise AnalysisFailure("Bounded analysis requires a process main thread")
    handler = signal.getsignal(signal.SIGALRM)
    timer = signal.getitimer(signal.ITIMER_REAL)
    started = time.monotonic()
    outer_first = timer[0] > 0 and timer[0] <= seconds

    def expired(signum, frame):
        if outer_first and callable(handler):
            handler(signum, frame)
        raise AnalysisFailure("DSL analysis deadline exhausted")

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(
        signal.ITIMER_REAL, min(timer[0], seconds) if timer[0] else seconds
    )
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, handler)
        if timer[0]:
            signal.setitimer(
                signal.ITIMER_REAL,
                max(0.000001, timer[0] - (time.monotonic() - started)),
                timer[1],
            )


def check_bundle(
    bundle, contract, *, scope="complete", role="body", focus=None, timeout=10.0
):
    """Local reports are never permission to execute. No code binding or model calls."""
    if scope not in {"local", "complete"} or role not in {"body", "condition"}:
        raise ValueError("Invalid check scope or role")
    if focus is not None and scope != "local":
        raise ValueError("A focused helper check must be local")
    started = time.monotonic()
    report = CheckReport(
        "error",
        scope,
        role,
        contract.identity(),
        digest({"source": bundle.source, "helpers": bundle.helpers}),
        coverage="partial",
    )
    try:
        with analysis_deadline(timeout):
            if focus is None:
                pending = validate(
                    bundle,
                    complete=scope == "complete",
                    role=role,
                    inputs={name: Value(kind) for name, kind in contract.input_types},
                    condition_mode=contract.condition_mode,
                    api_names=contract.api_names,
                )
            else:
                if bundle.condition_mode not in (None, contract.condition_mode):
                    raise ConstraintFailure(
                        [
                            Diagnostic(
                                "CONDITION_MODE", "Bundle and checker modes differ"
                            )
                        ]
                    )
                pending = validate_helper(
                    bundle.helpers[focus],
                    focus,
                    {k: v for k, v in bundle.helpers.items() if k != focus},
                    role,
                    condition_mode=contract.condition_mode,
                    api_names=contract.api_names,
                )
            report.unresolved = [asdict(call) for call in pending]
            report.status = (
                "pass"
                if scope == "complete"
                else ("pending" if pending else "local_pass")
            )
            report.coverage = "complete" if scope == "complete" else "local_generic"
    except ConstraintFailure as exc:
        report.status = "fail"
        report.diagnostics = [d.to_dict() for d in exc.diagnostics]
        report.unresolved = [
            d.to_dict() for d in exc.diagnostics if d.rule == "UNRESOLVED_HELPER"
        ]
        report.reason = str(exc)
    except BudgetExceeded:
        raise  # An enclosing trial budget is not a checker-internal error.
    except Exception as exc:
        report.reason = f"{type(exc).__name__}: {exc}"
    report.elapsed_seconds = time.monotonic() - started
    return report


def require_check(
    bundle,
    *,
    condition_mode="dsl_only",
    api_names=API_NAMES,
    inputs=None,
    complete=True,
    role="body",
    focus=None,
    logger=None,
    timeout=10.0,
):
    contract = DSLContract(
        condition_mode,
        tuple(sorted(api_names)),
        tuple(
            sorted(
                (name, value.kind) for name, value in (inputs or {"b0": OBJECT}).items()
            )
        ),
    )
    report = check_bundle(
        bundle,
        contract,
        scope="complete" if complete else "local",
        role=role,
        focus=focus,
        timeout=timeout,
    )
    if logger:
        logger.artifact(
            f"checks/check_{logger.sequence + 1:06d}.json", report.to_dict()
        )
    if report.status == "fail":
        raise ConstraintFailure([Diagnostic(**d) for d in report.diagnostics])
    if report.status == "error":
        raise AnalysisFailure(report.reason)
    return report


def require_helper_check(source, name, known, role, **options):
    from .contracts import Bundle

    return require_check(
        Bundle("", known | {name: source}),
        complete=False,
        focus=name,
        role=role,
        **options,
    )
