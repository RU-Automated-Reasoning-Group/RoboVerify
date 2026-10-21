"""Simulator-independent contracts shared by generation, checking, and execution."""

import math
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
    condition_mode: str | None = None


@dataclass(frozen=True)
class ObjectHandle:
    """Opaque policy identity; physical IDs stay in the adapter's private map."""

    token: str
    trial: str = field(repr=False)

    def __repr__(self):
        return f"ObjectHandle({self.token!r})"


CONDITION_MODES = ("dsl_only", "coordinate_predicates")
OBSERVATION_API_VERSION = "positions-v1"
CHECKER_VERSION = "shared-contract-v3"


def validate_condition_mode(mode):
    if mode not in CONDITION_MODES:
        raise ValueError(f"Unknown condition mode: {mode}")
    return mode


class Position3(tuple):
    """Immutable snapshot of world-frame XYZ, in metres."""

    def __new__(cls, values):
        values = tuple(values)
        if len(values) != 3 or any(
            type(value) not in (int, float) or not math.isfinite(value)
            for value in values
        ):
            raise CapError("Position3 requires three finite numeric coordinates")
        return super().__new__(cls, (float(value) for value in values))


PREDICATES = frozenset({"ON", "ON_star", "Higher", "Scattered", "ON_star_zero", "eq"})
ACTIONS = frozenset({"Pick", "Move", "Release"})
QUERIES = frozenset({"exists", "forall", "get"})
POSITION_QUERIES = frozenset({"get_gripper_position", "get_block_position"})
API_NAMES = PREDICATES | ACTIONS | QUERIES | POSITION_QUERIES
API_DESCRIPTION = """Available APIs (all object operands are opaque block handles):
Pick(b) -> None
Move(x_ref, y_ref, z_ref, [dx, dy, dz]) -> None # note that Move has 4 arguments.
Release(b, height_offset) -> None
get_gripper_position() -> Position3
get_block_position(b) -> Position3
Position3 is a copied immutable XYZ triple in world-frame metres. Block positions are centres.
Use only literal indices 0, 1, 2 on Position3 to read X, Y, Z. Read again after actions;
saved positions do not update. Getters read current state without stepping physics.
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
Do not import modules, inspect object attributes, read images/env or raw observation arrays,
or invent perception APIs. Read coordinates only through the position getters above.
Object equality compares identities; object handles are not numerical IDs.
Finite numerical arithmetic may compute action offsets.
Use assignment, def/return, if/else, while, break/continue, Boolean logic,
and these APIs. Lambda expressions are only query arguments. No for loops,
comprehensions, general subscripting, decorators, dynamic calls, or function aliases.
Helpers may compose the APIs. Condition helpers and query lambdas must not act.
Use explicit finite offsets. Do not return controller feedback as a condition.
Emit top-level statements that invoke the policy; defining a function alone does not execute it.
"""


def task_api_names(task):
    return API_NAMES if task == "reverse" else API_NAMES - {"ON_star_zero"}


CONDITION_RULES = {
    "dsl_only": """Condition mode: dsl_only.
Loop, branch and selection conditions must use only DSL predicates, object identity,
Boolean logic, quantifiers, and pure helpers composed of those operations.
Position reads and numerical computations may parameterize actions, but numerical
comparisons (including coordinate, counter and constant comparisons) are forbidden.
These rules apply transitively through assignments, helper arguments and returns.
""",
    "coordinate_predicates": """Condition mode: coordinate_predicates.
Loop, branch and selection conditions may use DSL predicates and pure generated
numeric predicates. Coordinate comparisons, counter guards, constant comparisons,
and numeric helper arguments are allowed; numbers need not originate from coordinates.
Numeric <, <=, >, >=, == and != are allowed. Conditions must return Booleans.
All condition helpers and query lambdas must remain free of physical actions.
""",
}


def api_description(
    names,
    condition_mode="dsl_only",
    *,
    prompt_version="legacy_v1",
    higher_tolerance=0.001,
    primitive_prompt=None,
):
    validate_condition_mode(condition_mode)
    from .prompt_package import render_api

    return render_api(
        names, condition_mode, prompt_version, higher_tolerance, primitive_prompt
    )
