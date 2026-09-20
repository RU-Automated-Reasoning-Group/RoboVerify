from dataclasses import dataclass, field


@dataclass
class BlockRegion:
    symbolic: tuple
    physical: tuple = ()
    bindings: frozenset = frozenset()


@dataclass
class LoopRegion:
    guard: object
    exists_vars: tuple
    body: tuple
    init: tuple = ()
    update: tuple = ()
    invariant: object = None
    iteration_counts: tuple = ()
    postconditions: tuple = ()
    body_demos: tuple = ()
    require_unique_guard: bool = False

    @property
    def max_iters(self):
        if not self.iteration_counts or min(self.iteration_counts) < 0:
            raise ValueError("Loop lowering requires recorded iteration counts")
        return max(self.iteration_counts)
