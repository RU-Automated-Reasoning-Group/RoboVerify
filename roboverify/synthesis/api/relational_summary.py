"""Demonic WPs for framed, nondeterministic Higher/Scattered action summaries.

An endpoint atom is not a deterministic placement update. Relations involving
the moved object may change arbitrarily, subject to the atom and the explicit
successor domain. Other pairs, equality and frozen entry ON* are preserved.
Boolean relation arrays are first-class quantified values; this is deliberately
outside the paper's claimed AFR fragment.
"""

import itertools

import z3

from synthesis.api.instructions import Instruction
from synthesis.util.symbols import fresh_const, open_quantifier


class SuccessorRelations(dict):
    """Relation expressions and exactly the fresh values that parameterize them."""

    def __init__(self, expressions, variables):
        super().__init__(expressions)
        self.variables = tuple(variables)


class RelationalSummary(Instruction):
    """Verification-only contract; operand order is independent of the mover."""

    def __init__(self, relation, left, right, *, moved=None):
        if relation not in ("Higher", "EqualHeight", "Scattered"):
            raise ValueError(
                "Relational summaries support Higher, EqualHeight or Scattered"
            )
        self.relation = relation
        self.left = left
        self.right = right
        self.moved = left if moved is None else moved
        if self.moved not in (left, right):
            raise ValueError("The endpoint atom must mention the moved object")

    @property
    def reference(self):
        return self.right if self.left == self.moved else self.left

    def __str__(self):
        return (
            f"establish {self.relation}({self.left}, {self.right}) "
            f"moving {self.moved} [framed-domain-havoc]"
        )

    def get_operand(self):
        return [dict(type="BoxName", val=n) for n in (self.left, self.right)]

    def eval(self, env, traj, return_image=False):
        raise RuntimeError(
            "RelationalSummary is verification-only; execute its primitives"
        )

    def atom(self, context):
        return getattr(context, self.relation)(
            context.get_consts(self.left), context.get_consts(self.right)
        )


def domain_axioms(context):
    """Collect actual current-state axioms without solver tracking implications.

    Frozen-entry axioms need not be regenerated: that interpretation never changes.
    The same returned obligations are checked at the motion endpoint, not assumed.
    """

    class Collector:
        def __init__(self):
            self.rows = []

        def assert_and_track(self, expression, label):
            self.rows.append((str(label), expression))

    collector = Collector()
    context.add_axiom(collector)
    context.add_axiom_height(collector)
    context.add_axiom_scattered(collector)
    return tuple(collector.rows)


def rewrite_relations(expression, context, replacements):
    """Simultaneously substitute mutable relations, under arbitrary binders.

    substitute_funs uses de Bruijn formal arguments and avoids capture. In
    particular, it never rewrites ON_star_zero or reprocesses inserted old facts.
    """
    x, y = z3.Var(0, context.BoxSort), z3.Var(1, context.BoxSort)
    return z3.substitute_funs(
        expression,
        *(
            (getattr(context, name), z3.Select(z3.Select(array, x), y))
            for name, array in replacements.items()
        ),
    )


def expand_finite_objects(expression, context):
    """Exactly expand Box quantifiers only in an exhaustive EnumSort context.

    Open every binder before recursion, including retained Boolean/relation
    binders, so outer references and nested shadowing remain capture-free.
    """
    if context.mode != "enum":
        return expression

    def visit(expr):
        if z3.is_quantifier(expr):
            variables, body = open_quantifier(expr)
            finite = [v for v in variables if v.sort() == context.BoxSort]
            retained = [v for v in variables if v.sort() != context.BoxSort]
            parts = [
                visit(z3.substitute(body, *zip(finite, values)))
                for values in itertools.product(context.enum_blocks, repeat=len(finite))
            ]
            expanded = (z3.And if expr.is_forall() else z3.Or)(*parts)
            if retained:
                expanded = (z3.ForAll if expr.is_forall() else z3.Exists)(
                    retained, expanded
                )
            return expanded
        if z3.is_app(expr) and expr.num_args():
            return expr.decl()(*(visit(child) for child in expr.children()))
        return expr

    return z3.simplify(visit(expression), blast_select_store=True)


def successor_contract(summary, context, *, avoid=(), admissibility=None):
    """Return fresh successor relation arrays and their framed contract.

    The default admissibility is the context's current-state domain axioms. A
    caller may supply a different current-vocabulary formula for direct API use;
    it is an explicit part of that contract, not an assumed motion premise.
    """
    if not isinstance(summary, RelationalSummary):
        raise TypeError("Expected a RelationalSummary")
    if context.wp_system == "reduction":
        raise ValueError(
            "RelationalSummary is outside the reduction vocabulary; use old or new WP"
        )
    if summary.relation not in context.mutable_relations:
        raise ValueError("Summary height relation differs from the selected algebra")
    relation_sort = z3.ArraySort(
        context.BoxSort, z3.ArraySort(context.BoxSort, z3.BoolSort())
    )
    domain = (
        z3.And(*(expr for _, expr in domain_axioms(context)))
        if admissibility is None
        else admissibility
    )
    context.validate_height_formula(domain)
    surroundings = (*avoid, domain, summary.atom(context))
    if context.mode == "enum":
        # A finite context has an exact Boolean encoding, avoiding quantified
        # array incompleteness. This is not a bound on the unbounded WP rule.
        after, variables = {}, []
        for name in context.mutable_relations:
            matrix = z3.K(context.BoxSort, z3.K(context.BoxSort, z3.BoolVal(False)))
            for x in context.enum_blocks:
                row = z3.K(context.BoxSort, z3.BoolVal(False))
                for y in context.enum_blocks:
                    value = fresh_const(
                        z3.BoolSort(), "summary_" + name, avoid=surroundings
                    )
                    variables.append(value)
                    row = z3.Store(row, y, value)
                matrix = z3.Store(matrix, x, row)
            after[name] = matrix
        after = SuccessorRelations(after, variables)
    else:
        after = {
            name: fresh_const(relation_sort, "summary_" + name, avoid=surroundings)
            for name in context.mutable_relations
        }
        after = SuccessorRelations(after, after.values())
    x = fresh_const(context.BoxSort, "frame_x", avoid=surroundings)
    y = fresh_const(context.BoxSort, "frame_y", avoid=(*surroundings, x))
    source = context.get_consts(summary.moved)
    frame = z3.ForAll(
        [x, y],
        z3.Implies(
            z3.And(x != source, y != source),
            z3.And(
                *(
                    z3.Select(z3.Select(after[name], x), y)
                    == getattr(context, name)(x, y)
                    for name in context.mutable_relations
                )
            ),
        ),
    )
    physical = source != context.get_consts("tbl") if context.use_tbl else True
    contract = z3.And(
        physical,
        frame,
        rewrite_relations(summary.atom(context), context, after),
        rewrite_relations(domain, context, after),
    )
    return after, contract


def summary_wp(summary, postcondition, context, *, admissibility=None):
    """Exact total/demonic WP of the stated relational contract.

    Existence prevents an impossible endpoint from proving every postcondition.
    Every allowed successor, not one convenient placement, must satisfy Q.
    """
    context.validate_height_formula(postcondition)
    after, contract = successor_contract(
        summary, context, avoid=(postcondition,), admissibility=admissibility
    )
    variables = list(after.variables)
    contract = expand_finite_objects(contract, context)
    rewritten = expand_finite_objects(
        rewrite_relations(postcondition, context, after), context
    )
    return z3.And(
        z3.Exists(variables, contract),
        z3.ForAll(
            variables,
            z3.Implies(contract, rewritten),
        ),
    )


def check_summary_realization(problem, summary, context, *, entry_bindings=None):
    """Check endpoint relation, full frame and successor admissibility in motion.

    All obligations use exact quantified translation. The current-state axioms
    are conclusions here; adding them as endpoint assumptions would be circular.
    """
    problem.solver.push()
    try:
        # Initial finite instantiation may discard quantified premises needed
        # by exact domain conclusions. Reassert only supplied ENTRY premises.
        problem.solver.add(
            *(
                problem.context.translate_exact(c, problem.constants)
                for c in problem.initial_condition
            )
        )
        problem.check("summary_premise_consistency")
        _check_summary_endpoint(
            problem, summary, context, entry_bindings=entry_bindings
        )
    finally:
        problem.solver.pop()


def _check_summary_endpoint(problem, summary, context, *, entry_bindings=None):
    bindings = problem.bindings if entry_bindings is None else entry_bindings
    summary = RelationalSummary(
        summary.relation,
        bindings[summary.left],
        bindings[summary.right],
        moved=bindings[summary.moved],
    )
    source = summary.moved
    problem.check(
        "summary_endpoint",
        z3.Not(
            z3.And(
                problem.released,
                problem.physical(source),
                problem.at(summary.atom(context), resolve=False),
            )
        ),
    )
    low_x = fresh_const(problem.context.BoxSort, "summary_frame_object")
    low_source = problem.constants[source]
    # Coordinate preservation is stronger than relation-pair preservation
    # and covers every unnamed object as well as all named aliases.
    frame = z3.ForAll(
        [low_x],
        z3.Implies(
            z3.Not(problem.context.lowlevel_box_equal(low_x, low_source)),
            z3.And(
                *(
                    z3.substitute_vars(a, low_x) == z3.substitute_vars(b, low_x)
                    for a, b in zip(problem.effect_entry_fields, problem.fields)
                )
            ),
        ),
    )
    problem.check("summary_frame", z3.Not(frame))
    for label, axiom in domain_axioms(context):
        problem.check(
            "summary_domain_" + label, z3.Not(problem.at(axiom, resolve=False))
        )
