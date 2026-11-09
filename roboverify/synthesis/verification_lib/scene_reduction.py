"""Exact finite-scene reduction for supported relational counterexample queries.

GroundEPR.lean enumerates finite scene diagrams. Here Boolean circuits encode
the SAME finite choices (active objects, roots, natural levels, and names).
No unbounded scene axioms, arithmetic theory, or object functions are sent to
the solver. The result is propositional, hence a special case of EPR.

For DeclareSort queries we require an exists-forall counterexample and use the
proved k*(k+1) bound. EnumSort queries are exact only at their given size.
Compilation limits are effort limits: exceeding one yields UNKNOWN, never
UNSAT from a smaller domain. This Python compiler is not kernel-certified.
"""

import itertools
import math
from functools import lru_cache
from time import monotonic

import z3

from synthesis.util.symbols import fresh_const, open_quantifier
from synthesis.verification_lib.epr_normalize import (
    ReductionUnsupported,
    object_constants,
    to_epr,
)


class ReductionLimit(ValueError):
    pass


def simplify_scene_formula(expression, context):
    """Use only reflexivity and table isolation, valid in every exact scene.

    Removing these atoms before prenexing can avoid irrelevant existential
    witnesses and a much larger small-model bound. This does not use user
    premises or the weaker relational axiom backend.
    """
    cache, binders = {}, {}
    relations = (
        context.ON_star,
        context.ON_star_zero,
        context.Higher,
        context.Scattered,
    )
    table = context.get_consts("tbl")

    def visit(node, depth=0):
        key = (node, depth)
        if key in cache:
            return cache[key]
        if z3.is_quantifier(node):
            variables, body = open_quantifier(node, avoid=(expression,))
            canonical = []
            for i, variable in enumerate(variables):
                binder_key = (depth, i)
                if binder_key not in binders:
                    binders[binder_key] = fresh_const(
                        variable.sort(), "scene_binder", avoid=(expression,)
                    )
                canonical.append(binders[binder_key])
            # Reuse binder names only at the same nesting depth. Alpha-equal
            # preconditions can then cancel before the countermodel bound is
            # computed, without capturing any user or enclosing variable.
            body = z3.substitute(body, *zip(variables, canonical))
            constructor = z3.ForAll if node.is_forall() else z3.Exists
            result = constructor(canonical, visit(body, depth + 1))
        elif z3.is_app(node) and node.decl() in relations:
            left, right = node.children()
            if left.eq(right):
                result = z3.BoolVal(node.decl() != context.Scattered)
            elif context.use_tbl and (left.eq(table) or right.eq(table)):
                result = (
                    z3.BoolVal(False)
                    if node.decl() == context.Scattered
                    else left == right
                )
            else:
                result = node
        elif z3.is_app(node) and node.num_args():
            result = node.decl()(*(visit(c, depth) for c in node.children()))
        else:
            result = node
        cache[key] = result
        return result

    return z3.simplify(visit(expression))


def _word(width, prefix):
    return tuple(z3.FreshBool(prefix) for _ in range(width))


@lru_cache(maxsize=1024)
def _number(value, width):
    return tuple(z3.BoolVal(bool(value & (1 << i))) for i in range(width))


def _equal(left, right):
    return z3.And(*(a == b for a, b in zip(left, right)))


def _ge(left, right):
    result = z3.BoolVal(True)
    for a, b in zip(left, right):
        result = z3.Or(z3.And(a, z3.Not(b)), z3.And(a == b, result))
    return result


def _increment(word):
    carry, result = z3.BoolVal(True), []
    for bit in word:
        result.append(z3.Xor(bit, carry))
        carry = z3.And(carry, bit)
    return tuple(result)


class SceneReduction:
    def __init__(self, expression, context, *, deadline):
        self.context, self.deadline = context, deadline
        self.work = 0
        self.constraints, self.names = [], {}
        self.eval_cache, self.dependencies = {}, {}
        # Validate before applying scene identities: unsupported vocabulary must
        # not accidentally acquire an interpretation through simplification.
        self.input_names = object_constants(expression, context)
        self.expression = simplify_scene_formula(expression, context)
        self.tick()
        try:
            # Propagate Boolean context through alpha-identical quantified
            # clauses (e.g. Legal and not (Legal and Q)) before prenexing.
            simplify = z3.TryFor(
                z3.Then("ctx-simplify", "simplify"),
                max(1, int((deadline - monotonic()) * 1000)),
            )
            self.expression = simplify(self.expression).as_expr()
        except z3.Z3Exception as exc:
            raise ReductionLimit("Reduction preprocessing failed: " + str(exc)) from exc
        self.tick()
        self.original_names = object_constants(self.expression, context)
        self.epr = None
        if context.mode == "declare":
            self.epr = to_epr(self.expression, context, self.tick)
            # Physical-only contexts are the restriction of the Lean object
            # model to non-table objects. One additional seed witnesses their
            # required nonempty physical domain.
            table = context.get_consts("tbl")
            free = [
                c for c in self.original_names if not (context.use_tbl and c.eq(table))
            ]
            self.seed_count = len(free) + len(self.epr.exists) + (not context.use_tbl)
            self.limit = self.seed_count * (self.seed_count + 1)
        else:
            self.seed_count = None
            self.limit = len(context.enum_blocks) - int(context.use_tbl)
        if self.limit > context.wp_options.reduction_max_slots:
            raise ReductionLimit(
                f"Reduction needs {self.limit} physical slots; compilation limit is "
                f"{context.wp_options.reduction_max_slots}. No smaller bound was used."
            )
        self.width = max(1, math.ceil(math.log2(self.limit + 1)))
        self.active = [z3.FreshBool("scene_active") for _ in range(self.limit)]
        if context.mode == "enum":
            self.constraints.extend(self.active)
        else:
            self.constraints.extend(
                z3.Implies(self.active[i], self.active[i - 1])
                for i in range(1, self.limit)
            )
        if not context.use_tbl:
            self.constraints.append(z3.Or(*self.active))
        self.domain = tuple(range(self.limit + int(context.use_tbl)))
        if self.epr is not None:
            count = len(self.domain) ** len(self.epr.forall)
            if count > context.wp_options.reduction_max_instances:
                raise ReductionLimit(
                    f"Reduction requires {count} universal instances; compilation budget exceeded"
                )
        self.current = self.scene("current")
        self.frozen = self.scene("frozen")
        self.views = {}
        for name in self.input_names:
            self.add_name(name)
        constructors = [
            self.names[c.decl()]
            for c in self.input_names
            if c.decl().kind() == z3.Z3_OP_DT_CONSTRUCTOR
        ]
        self.constraints.extend(
            z3.Not(_equal(a, b)) for a, b in itertools.combinations(constructors, 2)
        )
        if self.epr is None:
            condition = self.evaluate(self.expression)
        else:
            for name in self.epr.exists:
                self.add_name(name)
            clauses = []
            for values in itertools.product(self.domain, repeat=len(self.epr.forall)):
                self.tick()
                bindings = {v.decl(): i for v, i in zip(self.epr.forall, values)}
                clauses.append(
                    z3.Implies(
                        z3.And(*(self.present(i) for i in values)),
                        self.evaluate(self.epr.matrix, bindings=bindings),
                    )
                )
            condition = z3.And(*clauses)
        self.query = z3.And(*self.constraints, condition)

    def tick(self):
        self.work += 1
        if self.work > self.context.wp_options.reduction_max_instances:
            raise ReductionLimit("Reduction formula-construction budget exceeded")
        if monotonic() >= self.deadline:
            raise ReductionLimit("Reduction compilation timeout")

    def present(self, index):
        return z3.BoolVal(True) if index == self.limit else self.active[index]

    def scene(self, prefix):
        roots = [_word(self.width, prefix + "_root") for _ in self.active]
        heights = [_word(self.width, prefix + "_level") for _ in self.active]
        zero = _number(0, self.width)
        root_conditions = [
            z3.And(
                self.active[j],
                _equal(roots[j], _number(j, self.width)),
                _equal(heights[j], zero),
            )
            for j in range(self.limit)
        ]
        increments = [_increment(height) for height in heights]
        for i in range(self.limit):
            self.tick()
            self.constraints.append(
                z3.Implies(
                    self.active[i],
                    z3.And(
                        _ge(_number(self.limit - 1, self.width), heights[i]),
                        z3.Or(
                            *(
                                z3.And(
                                    _equal(roots[i], _number(j, self.width)),
                                    root_conditions[j],
                                )
                                for j in range(self.limit)
                            )
                        ),
                    ),
                )
            )
            # In a finite natural-level encoding, a predecessor at level h-1
            # for each positive h is equivalent to all lower levels existing.
            self.constraints.append(
                z3.Implies(
                    z3.And(self.active[i], z3.Not(_equal(heights[i], zero))),
                    z3.Or(
                        *(
                            z3.And(
                                self.active[j],
                                _equal(roots[i], roots[j]),
                                _equal(heights[i], increments[j]),
                            )
                            for j in range(self.limit)
                        )
                    ),
                )
            )
            for j in range(i):
                self.tick()
                self.constraints.append(
                    z3.Implies(
                        z3.And(self.active[i], self.active[j]),
                        z3.Not(
                            z3.And(
                                _equal(roots[i], roots[j]),
                                _equal(heights[i], heights[j]),
                            )
                        ),
                    )
                )
        return roots, heights

    def add_name(self, name):
        if name.decl() in self.names:
            return
        if self.context.use_tbl and name.eq(self.context.get_consts("tbl")):
            self.names[name.decl()] = _number(self.limit, self.width)
            return
        word = _word(self.width, "scene_name")
        self.names[name.decl()] = word
        self.constraints.append(
            z3.Or(
                *(
                    z3.And(self.present(i), _equal(word, _number(i, self.width)))
                    for i in self.domain
                )
            )
        )

    def term(self, expression, environment, bindings):
        if z3.is_var(expression):
            return environment[z3.get_var_index(expression)]
        if expression.decl() in bindings:
            return bindings[expression.decl()]
        return self.names[expression.decl()]

    def word(self, value):
        return _number(value, self.width) if isinstance(value, int) else value

    def view(self, value, frozen=False):
        key = (value, frozen)
        if key in self.views:
            return self.views[key]
        roots, heights = self.frozen if frozen else self.current
        if isinstance(value, int):
            result = (
                z3.BoolVal(value == self.limit),
                roots[value] if value < self.limit else _number(0, self.width),
                heights[value] if value < self.limit else _number(0, self.width),
            )
        else:
            choices = [_equal(value, _number(i, self.width)) for i in range(self.limit)]
            result = (
                (
                    _equal(value, _number(self.limit, self.width))
                    if self.context.use_tbl
                    else z3.BoolVal(False)
                ),
                tuple(
                    z3.Or(*(z3.And(choices[i], roots[i][b]) for i in range(self.limit)))
                    for b in range(self.width)
                ),
                tuple(
                    z3.Or(
                        *(z3.And(choices[i], heights[i][b]) for i in range(self.limit))
                    )
                    for b in range(self.width)
                ),
            )
        self.views[key] = result
        return result

    def atom(self, relation, left, right):
        lt, lr, lh = self.view(left, relation == self.context.ON_star_zero)
        rt, rr, rh = self.view(right, relation == self.context.ON_star_zero)
        physical = z3.And(z3.Not(lt), z3.Not(rt))
        if relation == self.context.Scattered:
            return z3.And(physical, z3.Not(_equal(lr, rr)))
        same = z3.BoolVal(True) if relation == self.context.Higher else _equal(lr, rr)
        return z3.Or(z3.And(lt, rt), z3.And(physical, same, _ge(lh, rh)))

    def free_dependencies(self, expression):
        key = expression.get_id()
        if key not in self.dependencies:
            if z3.is_const(expression) and expression.sort() == self.context.BoxSort:
                result = {expression.decl()}
            else:
                result = set().union(
                    *(self.free_dependencies(c) for c in expression.children())
                )
            self.dependencies[key] = tuple(sorted(result, key=lambda d: d.get_id()))
        return self.dependencies[key]

    def evaluate(self, expression, environment=(), bindings=None):
        bindings = {} if bindings is None else bindings
        # QF subformulas depend only on the universal variables they actually
        # mention. Share their Boolean circuits across the finite expansion.
        if self.epr is not None:
            key = (
                expression.get_id(),
                tuple(
                    bindings[d]
                    for d in self.free_dependencies(expression)
                    if d in bindings
                ),
            )
            if key not in self.eval_cache:
                self.eval_cache[key] = self._evaluate(expression, environment, bindings)
            return self.eval_cache[key]
        return self._evaluate(expression, environment, bindings)

    def _evaluate(self, expression, environment, bindings):
        self.tick()
        if z3.is_quantifier(expression):
            count = len(self.domain) ** expression.num_vars()
            if count > self.context.wp_options.reduction_max_instances:
                raise ReductionLimit(
                    "Finite quantifier expansion exceeds compilation budget"
                )
            terms = []
            for values in itertools.product(self.domain, repeat=expression.num_vars()):
                guard = z3.And(*(self.present(i) for i in values))
                body = self.evaluate(
                    expression.body(), tuple(reversed(values)) + environment, bindings
                )
                terms.append(
                    z3.Implies(guard, body)
                    if expression.is_forall()
                    else z3.And(guard, body)
                )
            return (z3.And if expression.is_forall() else z3.Or)(*terms)
        if expression.decl() in (
            self.context.ON_star,
            self.context.ON_star_zero,
            self.context.Higher,
            self.context.Scattered,
        ):
            left, right = (
                self.term(c, environment, bindings) for c in expression.children()
            )
            return self.atom(expression.decl(), left, right)
        if (
            expression.decl().kind() in (z3.Z3_OP_EQ, z3.Z3_OP_DISTINCT)
            and expression.arg(0).sort() == self.context.BoxSort
        ):
            terms = [
                self.word(self.term(c, environment, bindings))
                for c in expression.children()
            ]
            if z3.is_eq(expression):
                return _equal(*terms)
            return z3.And(
                *(z3.Not(_equal(a, b)) for a, b in itertools.combinations(terms, 2))
            )
        return expression.decl()(
            *(self.evaluate(c, environment, bindings) for c in expression.children())
        )

    @staticmethod
    def number(model, word):
        return sum(
            1 << i
            for i, bit in enumerate(word)
            if z3.is_true(model.eval(bit, model_completion=True))
        )

    def decode(self, model):
        """Reconstruct a native Z3 relational model for existing CE consumers."""
        count = sum(
            z3.is_true(model.eval(a, model_completion=True)) for a in self.active
        )
        slots = list(range(count)) + ([self.limit] if self.context.use_tbl else [])
        context = self.context
        if context.mode == "enum":
            assigned = {
                self.number(model, self.names[c.decl()]): c
                for c in self.input_names
                if c.decl().kind() == z3.Z3_OP_DT_CONSTRUCTOR
            }
            unused = iter(
                c
                for c in context.enum_blocks
                if not any(c.eq(a) for a in assigned.values())
            )
            objects = {i: assigned[i] if i in assigned else next(unused) for i in slots}
        else:
            objects = {
                i: (
                    context.get_consts("tbl")
                    if i == self.limit
                    else fresh_const(
                        context.BoxSort, "reduced_object", avoid=self.input_names
                    )
                )
                for i in slots
            }
        solver = z3.Solver()
        solver.set(timeout=max(1, int((self.deadline - monotonic()) * 1000)))
        solver.add(z3.Distinct(*objects.values()))
        obj = fresh_const(context.BoxSort, "reduced_domain", avoid=self.input_names)
        solver.add(
            z3.ForAll([obj], z3.Or(*(obj == value for value in objects.values())))
        )
        if context.use_tbl:
            solver.add(context.get_consts("tbl") == objects[self.limit])
        for name in self.input_names:
            solver.add(name == objects[self.number(model, self.names[name.decl()])])
        for relation in (
            context.ON_star,
            context.ON_star_zero,
            context.Higher,
            context.Scattered,
        ):
            for i, j in itertools.product(slots, repeat=2):
                self.tick()
                truth = z3.is_true(
                    model.eval(self.atom(relation, i, j), model_completion=True)
                )
                solver.add(relation(objects[i], objects[j]) == truth)
        answer = solver.check()
        if answer != z3.sat:
            raise ReductionLimit(
                "Reduced model reconstruction: "
                + (
                    solver.reason_unknown()
                    if answer == z3.unknown
                    else "inconsistent diagram"
                )
            )
        return solver.model()


class ReductionSolver:
    """Incremental solver facade; each check compiles the current assertions.

    Tracked cores conservatively contain all tracked inputs. That is a valid
    nonminimal core, so callers must retain their premise-consistency check.
    """

    def __init__(self, context, timeout_ms):
        self.context, self.timeout_ms = context, timeout_ms
        self.rows, self.scopes = [], []
        self.last_reduction = None
        self._answer, self._model, self._reason = None, None, ""

    def add(self, *expressions):
        self._invalidate()
        for expression in expressions:
            if isinstance(expression, (list, tuple, z3.AstVector)):
                self.add(*expression)
            else:
                self.rows.append(
                    (
                        (
                            z3.BoolVal(expression)
                            if isinstance(expression, bool)
                            else expression
                        ),
                        None,
                    )
                )

    def assert_and_track(self, expression, label):
        self._invalidate()
        self.rows.append(
            (expression, z3.Bool(label) if isinstance(label, str) else label)
        )

    def push(self):
        self.scopes.append(len(self.rows))

    def pop(self, num=1):
        self._invalidate()
        for _ in range(num):
            del self.rows[self.scopes.pop() :]

    def assertions(self):
        return [expression for expression, _ in self.rows]

    def _invalidate(self):
        self._answer, self._model, self.last_reduction = None, None, None

    def set(self, *args, **kwargs):
        options = dict(zip(args[::2], args[1::2]))
        options.update(kwargs)
        if "timeout" in options:
            if options["timeout"] <= 0:
                raise ValueError("Reduction timeout must be positive")
            self.timeout_ms = options["timeout"]

    def check(self, *assumptions):
        deadline = monotonic() + self.timeout_ms / 1000
        self._model, self._reason, self.last_reduction = None, "", None
        self._assumptions = assumptions
        try:
            reduced = SceneReduction(
                z3.And(*self.assertions(), *assumptions),
                self.context,
                deadline=deadline,
            )
            self.last_reduction = reduced
            solver = z3.Solver()
            solver.set(timeout=max(1, int((deadline - monotonic()) * 1000)))
            solver.add(reduced.query)
            self._answer = solver.check()
            if self._answer == z3.sat:
                self._model = reduced.decode(solver.model())
            elif self._answer == z3.unknown:
                self._reason = solver.reason_unknown()
        except (ReductionUnsupported, ReductionLimit) as exc:
            self._answer, self._reason = z3.unknown, str(exc)
        return self._answer

    def model(self):
        if self._model is None:
            raise z3.Z3Exception("No reduced SAT model available")
        return self._model

    def unsat_core(self):
        if self._answer != z3.unsat:
            return []
        return [label for _, label in self.rows if label is not None] + list(
            self._assumptions
        )

    def reason_unknown(self):
        return self._reason

    def sexpr(self):
        if self.last_reduction is None:
            raise ValueError("Call check() before exporting a reduced query")
        solver = z3.Solver()
        solver.add(self.last_reduction.query)
        return solver.sexpr()
