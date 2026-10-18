"""Exclusive host-side phase timings; nested work is never double counted."""

import time
from collections import defaultdict
from contextlib import contextmanager
from functools import wraps


class PhaseTimings:
    def __init__(self, clock=None):
        self.clock = clock or time.monotonic
        self.started = self.last = self.clock()
        self.stack = []
        self.seconds = defaultdict(float)

    def _charge(self):
        now = self.clock()
        self.seconds[self.stack[-1] if self.stack else "overhead"] += now - self.last
        self.last = now
        return now

    @contextmanager
    def measure(self, phase):
        self._charge()
        self.stack.append(phase)
        try:
            yield
        finally:
            self._charge()
            self.stack.pop()

    def call(self, phase, function, /, *args, **kwargs):
        with self.measure(phase):
            return function(*args, **kwargs)

    def snapshot(self):
        now = self._charge()
        return {
            "active_wall_seconds": now - self.started,
            "phase_seconds": dict(sorted(self.seconds.items())),
            "phase_accounting": "exclusive; model request time includes network/provider wait",
        }


def timed(phase):
    def decorate(function):
        @wraps(function)
        def wrapped(self, *args, **kwargs):
            timer = getattr(self, "timings", None)
            if timer is None:
                timer = getattr(getattr(self, "budget", None), "timings", None)
            if timer is None:
                return function(self, *args, **kwargs)
            return timer.call(phase, function, self, *args, **kwargs)

        return wrapped

    return decorate
