"""Capture-safe normalization to the function-free existential/universal fragment.

This is a conservative recognizer, not a decision procedure for fragment
definability. Unsupported alternation is reported instead of Skolemizing it
into object-valued functions or silently bounding it.
"""

from dataclasses import dataclass

import z3

from synthesis.util.symbols import open_quantifier


class ReductionUnsupported(ValueError):
    pass


@dataclass(frozen=True)
class EPR:
    exists: tuple
    forall: tuple
    matrix: object


def object_constants(expression, context):
    """Validate the assertion vocabulary and return its free object constants."""
    relations = (
        context.ON_star,
        context.ON_star_zero,
        context.Higher,
        context.Scattered,
    )
    constants, seen = {}, set()

    def term(node):
        if node.sort() != context.BoxSort:
            raise ReductionUnsupported("Reduction requires block-valued terms")
        if z3.is_var(node):
            return
        if z3.is_const(node) and node.decl().kind() in (
            z3.Z3_OP_UNINTERPRETED,
            z3.Z3_OP_DT_CONSTRUCTOR,
        ):
            constants[node.decl()] = node
            return
        raise ReductionUnsupported("Reduction does not support object-valued functions")

    def visit(node):
        if node.get_id() in seen:
            return
        seen.add(node.get_id())
        if z3.is_quantifier(node):
            if not (node.is_forall() or node.is_exists()) or any(
                node.var_sort(i) != context.BoxSort for i in range(node.num_vars())
            ):
                raise ReductionUnsupported(
                    "Reduction supports only first-order block quantifiers"
                )
            visit(node.body())
        elif z3.is_app(node):
            kind = node.decl().kind()
            if node.decl() in relations:
                for child in node.children():
                    term(child)
            elif (
                kind in (z3.Z3_OP_EQ, z3.Z3_OP_DISTINCT)
                and node.arg(0).sort() == context.BoxSort
            ):
                for child in node.children():
                    term(child)
            elif z3.is_true(node) or z3.is_false(node):
                return
            elif kind in (
                z3.Z3_OP_AND,
                z3.Z3_OP_OR,
                z3.Z3_OP_NOT,
                z3.Z3_OP_IMPLIES,
                z3.Z3_OP_ITE,
                z3.Z3_OP_XOR,
                z3.Z3_OP_EQ,
            ) and all(z3.is_bool(child) for child in node.children()):
                for child in node.children():
                    visit(child)
            else:
                raise ReductionUnsupported(
                    "Reduction supports only ON_star, ON_star_zero, Higher, Scattered, and equality"
                )
        else:
            raise ReductionUnsupported("Unsupported reduction formula")

    visit(expression)
    return tuple(constants.values())


def to_epr(expression, context, tick=lambda: None):
    expression = z3.simplify(expression)
    object_constants(expression, context)

    def combine(parts, constructor):
        return EPR(
            tuple(v for p in parts for v in p.exists),
            tuple(v for p in parts for v in p.forall),
            constructor(*(p.matrix for p in parts)),
        )

    def visit(node, negative=False):
        tick()
        if z3.is_quantifier(node):
            variables, body = open_quantifier(node, avoid=(expression,))
            result = visit(body, negative)
            used = {v.decl() for v in object_constants(result.matrix, context)}
            variables = tuple(v for v in variables if v.decl() in used)
            existential = node.is_exists() != negative
            if existential:
                return EPR(variables + result.exists, result.forall, result.matrix)
            if variables and result.exists:
                raise ReductionUnsupported(
                    "Counterexample formula is outside the supported exists-forall fragment "
                    "(a universal quantifier precedes an existential); use --wp-system new"
                )
            return EPR(result.exists, variables + result.forall, result.matrix)
        if z3.is_not(node):
            return visit(node.arg(0), not negative)
        if z3.is_implies(node):
            return visit(z3.Or(z3.Not(node.arg(0)), node.arg(1)), negative)
        if z3.is_and(node) or z3.is_or(node):
            conjunction = z3.is_and(node) != negative
            return combine(
                [visit(c, negative) for c in node.children()],
                z3.And if conjunction else z3.Or,
            )
        if z3.is_app(node) and node.decl().kind() == z3.Z3_OP_ITE:
            condition, yes, no = node.children()
            return visit(
                z3.Or(z3.And(condition, yes), z3.And(z3.Not(condition), no)), negative
            )
        if z3.is_app(node) and (
            node.decl().kind() == z3.Z3_OP_XOR
            or (z3.is_eq(node) and z3.is_bool(node.arg(0)))
        ):
            left, right = node.children()
            equivalent = z3.Or(z3.And(left, right), z3.And(z3.Not(left), z3.Not(right)))
            return visit(equivalent, negative != (node.decl().kind() == z3.Z3_OP_XOR))
        return EPR((), (), z3.Not(node) if negative else node)

    return visit(expression)
