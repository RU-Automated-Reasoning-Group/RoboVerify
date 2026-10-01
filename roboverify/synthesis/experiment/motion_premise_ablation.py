"""Scoped removal of extra motion premises for fixed-program diagnostics.

Keep the primitive semantics and every proof obligation. Auxiliary loop facts
are omitted only when used as assumptions, never when checked as conclusions.
This is intentionally separate from production verification configuration.
"""

from collections import Counter
from contextlib import ExitStack, contextmanager
from unittest.mock import patch

import z3
from synthesis.cfg import verification
from synthesis.util.symbols import fresh_const
from synthesis.verification_lib import motion_verification, tower_geometry

AUXILIARY_GEOMETRY_CHECKS = frozenset(
    f"{name}_{stage}"
    for name in ("arm_clearance", "column_alignment", "tower_height")
    for stage in ("entry", "preserved")
)


def quantifier_count(expressions):
    """Count unique quantifier nodes, including nested bodies, for query audits."""
    seen = set()

    def visit(expr):
        if expr.get_id() in seen:
            return 0
        seen.add(expr.get_id())
        if z3.is_quantifier(expr):
            return 1 + visit(expr.body())
        return sum(visit(child) for child in expr.children())

    return sum(visit(expr) for expr in expressions)


@contextmanager
def without_extra_motion_premises():
    """Use ordinary translated premises; retain exact semantics in VC goals.

    The caller must additionally omit initial_arm. Cube size, table height as a
    release-contract constant, program guards, and motion transitions remain.
    Exact copies of the relational assumptions are also omitted from effect
    checks: those copies strengthen the ordinary finite assumption translation.
    """
    omissions = Counter()
    original_effects = motion_verification.check_abstract_effects

    def table_reference_only(problem, table_surface_height):
        # Keep the reference needed to *check* height entry/preservation, without
        # assuming those facts or reasserting exact quantified initial conditions.
        problem.table_center_height = (
            fresh_const(z3.RealSort(), "table_center_height")
            if table_surface_height is None
            else z3.RealVal(str(table_surface_height)) + problem.context.L / 2
        )
        omissions["supported_height_and_exact_conditions"] += 1

    def checked_only(original):
        def condition(problem, *, fields=None):
            # CFG fresh contexts omit fields when asserting an auxiliary fact;
            # entry/preservation checks explicitly pass the current fields.
            if fields is None:
                omissions[original.__name__] += 1
                return z3.BoolVal(True)
            return original(problem, fields=fields)

        return condition

    def no_input_alignment(problem, roots):
        omissions["input_tower_alignment"] += 1

    def effects_from_translated_premises(problem):
        omissions["exact_effect_premise_reassertion"] += 1
        # check_abstract_effects uses initial_condition only to reassert exact
        # premises; its expected/actual effects remain unchanged and quantified.
        with patch.object(problem, "initial_condition", ()):
            return original_effects(problem)

    with ExitStack() as scope:
        scope.enter_context(
            patch.object(tower_geometry, "assume_tower_geometry", table_reference_only)
        )
        for name in ("arm_clearance", "column_alignment"):
            scope.enter_context(
                patch.object(
                    tower_geometry, name, checked_only(getattr(tower_geometry, name))
                )
            )
        # Patch both the definition and the CFG's imported aliases. Alignment is
        # otherwise assumed again when Get/Assign exposes an input root.
        for module in (motion_verification, verification):
            scope.enter_context(
                patch.object(module, "assume_input_alignment", no_input_alignment)
            )
            scope.enter_context(
                patch.object(
                    module, "check_abstract_effects", effects_from_translated_premises
                )
            )
        yield omissions
