"""Scoped options for symbolic solvers, including rebuilt finite contexts."""

from contextlib import contextmanager
from contextvars import ContextVar

_minimize_unsat_core = ContextVar("minimize_unsat_core", default=False)


def get_minimize_unsat_core():
    return _minimize_unsat_core.get()


@contextmanager
def using_unsat_core_minimization(enabled=None):
    """Configure newly created symbolic solvers; None inherits the outer scope.

    Core production and premise-consistency checks remain enabled. This only
    controls Z3's optional effort to shrink the returned core.
    """
    if enabled is None:
        enabled = get_minimize_unsat_core()
    if not isinstance(enabled, bool):
        raise ValueError("Unsat-core minimization must be a boolean")
    token = _minimize_unsat_core.set(enabled)
    try:
        yield
    finally:
        _minimize_unsat_core.reset(token)
