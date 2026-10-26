"""Scoped choice of the height vocabulary used by inference and verification."""

from contextlib import contextmanager
from contextvars import ContextVar

_height_predicate = ContextVar("height_predicate", default="Higher")


def get_height_predicate():
    return _height_predicate.get()


def validate_height_predicate(name):
    if name not in ("Higher", "EqualHeight"):
        raise ValueError("Height predicate must be Higher or EqualHeight")
    return name


@contextmanager
def using_height_predicate(name):
    """Select the algebra for newly constructed contexts within a pipeline run."""
    token = _height_predicate.set(validate_height_predicate(name))
    try:
        yield
    finally:
        _height_predicate.reset(token)
