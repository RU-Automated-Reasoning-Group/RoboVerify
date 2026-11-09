"""Scoped selection of a WP transformer and its matching symbolic backend."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, replace


@dataclass(frozen=True)
class WPOptions:
    wp_system: str = "old"
    support_complete: bool = False
    reduction_max_slots: int = 64
    reduction_max_instances: int = 200000

    def __post_init__(self):
        if self.wp_system not in ("old", "new", "reduction"):
            raise ValueError("WP system must be old, new, or reduction")
        if self.support_complete and self.wp_system != "new":
            raise ValueError("Support-completeness axioms require --wp-system new")
        if self.reduction_max_slots < 1 or self.reduction_max_instances < 1:
            raise ValueError("Reduction compilation budgets must be positive")

    def kwargs(self):
        return asdict(self)


_options = ContextVar("wp_options", default=WPOptions())


def wp_options(**overrides):
    return replace(
        _options.get(), **{k: v for k, v in overrides.items() if v is not None}
    )


@contextmanager
def using_wp_system(wp_system="old", **options):
    token = _options.set(WPOptions(wp_system=wp_system, **options))
    try:
        yield
    finally:
        _options.reset(token)
