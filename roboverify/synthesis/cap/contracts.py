"""Simulator-independent contracts shared by generation, checking, and execution."""

from dataclasses import asdict, dataclass, field


class CapError(Exception):
    """An explicitly classified experiment failure."""


class BudgetExceeded(CapError):
    pass


class AdapterGap(CapError):
    pass


class EnvironmentUnavailable(CapError):
    pass


class PrimitiveFailure(CapError):
    pass


@dataclass(frozen=True)
class Diagnostic:
    rule: str
    message: str
    line: int = 0
    column: int = 0
    unit: str = "policy"

    def to_dict(self):
        return asdict(self)


class ConstraintFailure(CapError):
    def __init__(self, diagnostics):
        self.diagnostics = list(diagnostics)
        super().__init__(
            "; ".join(f"{d.unit}:{d.line}: {d.rule}: {d.message}" for d in diagnostics)
        )


@dataclass
class Bundle:
    source: str
    helpers: dict = field(default_factory=dict)
    checkpoint_id: str | None = None


@dataclass(frozen=True)
class ObjectHandle:
    """Opaque policy identity; physical IDs stay in the adapter's private map."""

    token: str
    trial: str = field(repr=False)

    def __repr__(self):
        return f"ObjectHandle({self.token!r})"


PREDICATES = frozenset({"ON", "ON_star", "Higher", "Scattered", "ON_star_zero", "eq"})
ACTIONS = frozenset({"Pick", "Move", "Release"})
QUERIES = frozenset({"exists", "forall", "get"})
API_NAMES = PREDICATES | ACTIONS | QUERIES
API_DESCRIPTION = """Available APIs (all object operands are opaque block handles):
Pick(b) -> None
Move(x_ref, y_ref, z_ref, [dx, dy, dz]) -> None # note that Move has 4 arguments.
Release(b, height_offset) -> None
ON(a, b), ON_star(a, b), Higher(a, b), Scattered(a, b), eq(a, b) -> bool
ON_star_zero(a, b) -> bool (frozen task-entry geometry)
exists(lambda x: condition), forall(lambda x: condition) -> bool
get(lambda x: condition) -> ObjectHandle (first matching witness; fails if absent)
Quantifiers may have multiple lambda parameters; get requires exactly one.
Objects are selected only by these queries. b0 is the input block handle.
Move uses current X(x_ref)+dx, Y(y_ref)+dy, Z(z_ref)+dz, in metres.
Release opens the gripper and retreats to current Z(b)+height_offset; it is not a placement macro.
ON_star(a,b) means horizontal alignment with a at least as high as b; it is reflexive.
ON(a,b) means distinct objects with ON_star(a,b) and no intervening object.
Higher(a,b) means Z(a) >= Z(b) minus the configured tolerance (default 0.001 m).
Higher is non-strict and reflexive. Scattered uses the shared 0.10 m XY separation rule.
Do not import modules, inspect object attributes, read coordinates/images/env,
or invent perception APIs. Use only predicate-derived Boolean conditions and
object identity comparisons. Finite numerical arithmetic is for action offsets.
Use assignment, def/return, if/else, while, break/continue, Boolean logic,
and these APIs. Lambda expressions are only query arguments. No for loops,
comprehensions, subscripting, decorators, dynamic calls, or function aliases.
Helpers may compose the APIs. Condition helpers and query lambdas must not act.
Use explicit finite offsets. Do not return controller feedback as a condition.
Emit top-level statements that invoke the policy; defining a function alone does not execute it.
"""


def task_api_names(task):
    return API_NAMES if task == "reverse" else API_NAMES - {"ON_star_zero"}


def api_description(names):
    return "\n".join(
        line
        for line in API_DESCRIPTION.splitlines()
        if "ON_star_zero" in names or not line.startswith("ON_star_zero")
    )
