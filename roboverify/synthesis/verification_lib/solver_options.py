"""Scoped options for symbolic solvers, including rebuilt finite contexts."""

from contextlib import contextmanager
from contextvars import ContextVar

_minimize_unsat_core = ContextVar("minimize_unsat_core", default=False)


def get_minimize_unsat_core():
    return _minimize_unsat_core.get()


def finite_sat_witness(solver, context, *, restore_timeout_ms, witness_timeout_ms=250):
    """Try a finite model of the SAME assertions; only SAT is useful evidence.

    Domain closure restricts the search, so its SAT answer is also a model of
    the unrestricted assertions. Finite UNSAT/UNKNOWN proves nothing here.
    The original assertions, timeout and unbounded proof obligations survive.
    """
    import z3
    from synthesis.util.symbols import fresh_const

    if restore_timeout_ms <= 0 or witness_timeout_ms <= 0:
        raise ValueError("Finite witness query budgets must be positive")
    if context.verification_mode != "box" or context.mode != "declare":
        return False
    surroundings = tuple(solver.assertions())
    objects = [
        fresh_const(context.BoxSort, "witness_object", avoid=surroundings)
        for _ in range(4)
    ]
    if context.use_tbl:
        objects.append(context.get_consts("tbl"))
    obj = fresh_const(
        context.BoxSort, "witness_domain", avoid=(*surroundings, *objects)
    )
    solver.push()
    try:
        solver.set(timeout=min(witness_timeout_ms, restore_timeout_ms))
        solver.add(z3.ForAll([obj], z3.Or(*(obj == other for other in objects))))
        return solver.check() == z3.sat
    finally:
        solver.pop()
        solver.set(timeout=restore_timeout_ms)


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
