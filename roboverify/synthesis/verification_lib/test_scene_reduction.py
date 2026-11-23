"""Independent finite-scene oracle and scope tests for exact reduction."""

import itertools
import unittest

import z3

from synthesis.api.instructions import Put
from synthesis.api.program import put_precondition, wp
from synthesis.api.relational_summary import expand_finite_objects
from synthesis.verification_lib.epr_normalize import to_epr
from synthesis.verification_lib.highlevel_verification_lib import HighLevelContext
from synthesis.verification_lib.scene_reduction import simplify_scene_formula
from synthesis.verification_lib.symbolic_verify import VC, discharge_vc
from synthesis.verification_lib.test_wp_backends import context, facts, towers, truth


class ReductionTests(unittest.TestCase):
    def test_valid_invalid_vacuous_and_incremental_queries(self):
        c = context("reduction")
        a, b, tbl = [c.get_consts(n) for n in ("a", "b", "tbl")]
        cases = [
            (z3.Implies(True, c.ON_star(a, a)), "valid"),
            (z3.Implies(z3.Not(c.ON_star(a, a)), False), "vacuous"),
            (z3.Implies(z3.Distinct(a, b, tbl), c.ON_star(a, b)), "invalid"),
        ]
        for formula, expected in cases:
            result = discharge_vc(VC("body", None, formula), c, 10000)
            self.assertEqual(result.status, expected, result.reason)
        solver = c.new_solver(10000)
        solver.add(z3.Distinct(a, b, tbl), c.ON_star(a, b))
        self.assertEqual(solver.check(), z3.sat, solver.reason_unknown())
        self.assertTrue(z3.is_true(solver.model().eval(c.ON_star(a, b))))
        solver.push()
        solver.add(z3.Not(c.Higher(a, b)))
        self.assertEqual(solver.check(), z3.unsat)
        solver.pop()
        self.assertEqual(solver.check(), z3.sat)

    def test_compiled_query_is_propositional_and_bound_is_derived(self):
        c = context("reduction")
        a, b = [c.get_consts(n) for n in ("a", "b")]
        solver = c.new_solver(10000)
        solver.add(a != b, c.ON_star(a, b))
        self.assertEqual(solver.check(), z3.sat, solver.reason_unknown())
        reduction = solver.last_reduction
        self.assertEqual(reduction.seed_count, 2)
        self.assertEqual(reduction.limit, 6)
        pending, seen = [reduction.query], set()
        while pending:
            node = pending.pop()
            if node.get_id() in seen:
                continue
            seen.add(node.get_id())
            self.assertFalse(z3.is_quantifier(node))
            self.assertTrue(z3.is_bool(node))
            if node.decl().kind() == z3.Z3_OP_UNINTERPRETED:
                self.assertEqual(node.num_args(), 0)
            pending.extend(node.children())

    def test_resource_budget_never_proves_a_smaller_bound(self):
        c = context("reduction", reduction_max_slots=1)
        a = c.get_consts("a")
        solver = c.new_solver(5000)
        solver.add(a != c.get_consts("tbl"))
        self.assertEqual(solver.check(), z3.unknown)
        self.assertIn("No smaller bound", solver.reason_unknown())
        c = context("reduction", reduction_max_instances=1)
        solver = c.new_solver(5000)
        solver.add(c.ON_star(c.get_consts("a"), c.get_consts("b")))
        self.assertEqual(solver.check(), z3.unknown)
        self.assertIn("budget", solver.reason_unknown())

    def test_unsupported_alternation_functions_and_predicates_fail_closed(self):
        c = context("reduction")
        x, y = z3.Consts("x y", c.BoxSort)
        function = z3.Function("f", c.BoxSort, c.BoxSort)
        predicate = z3.Function("P", c.BoxSort, z3.BoolSort())
        formulas = [
            z3.ForAll([x], z3.Exists([y], c.ON_star(y, x))),
            c.Higher(x, function(x)),
            predicate(x),
        ]
        for formula in formulas:
            solver = c.new_solver(5000)
            solver.add(formula)
            self.assertEqual(solver.check(), z3.unknown)
            self.assertTrue(solver.reason_unknown())

    def test_empty_physical_domain_and_frozen_scene_independence(self):
        c = context("reduction")
        x = c.get_consts("x")
        solver = c.new_solver(5000)
        solver.add(z3.ForAll([x], x == c.get_consts("tbl")))
        self.assertEqual(solver.check(), z3.sat, solver.reason_unknown())
        self.assertEqual(solver.last_reduction.limit, 0)
        c = context("reduction", count=2)
        a, b, tbl = [c.get_consts(n) for n in ("a", "b", "tbl")]
        solver = c.new_solver(5000)
        solver.add(z3.Distinct(a, b, tbl), c.ON_star(a, b), c.ON_star_zero(b, a))
        self.assertEqual(solver.check(), z3.sat, solver.reason_unknown())
        model = solver.model()
        self.assertTrue(z3.is_true(model.eval(c.ON_star(a, b))))
        self.assertTrue(z3.is_true(model.eval(c.ON_star_zero(b, a))))

    def test_physical_only_domain_is_nonempty_and_model_keeps_simplified_names(self):
        c = HighLevelContext(wp_system="reduction")
        a, x = [c.get_consts(n) for n in ("a", "x")]
        solver = c.new_solver(5000)
        solver.add(c.ON_star(a, a))
        self.assertEqual(solver.check(), z3.sat, solver.reason_unknown())
        self.assertEqual(solver.last_reduction.seed_count, 1)
        self.assertTrue(z3.is_true(solver.model().eval(c.ON_star(a, a))))
        solver.add(z3.ForAll([x], x != a))
        self.assertEqual(solver.check(), z3.unsat, solver.reason_unknown())

    def test_finite_solver_matches_all_two_block_scene_pairs(self):
        c = context("reduction", count=2)
        a, b, table = c.enum_blocks
        objects = {"a": a, "b": b, "tbl": table}
        frozen_scenes = towers("ab")
        for columns, frozen_columns in itertools.product(towers("ab"), frozen_scenes):
            current, frozen = facts(columns), facts(frozen_columns)
            solver = c.new_solver(5000)
            solver.add(c.get_consts("tbl") == table)
            for relation in ("ON_star", "ON_star_zero", "Higher", "Scattered"):
                for left, right in itertools.product(objects, repeat=2):
                    solver.add(
                        getattr(c, relation)(objects[left], objects[right])
                        == truth(relation, left, right, current, frozen)
                    )
            self.assertEqual(solver.check(), z3.sat, solver.reason_unknown())
        # Exhaust all height/ON tables on two physical blocks, rather than only
        # asking the compiler to accept inputs generated by its own constraints.
        possible = {
            tuple(
                truth(r, x, y, facts(cols), facts(cols))
                for r in ("ON_star", "Higher", "Scattered")
                for x, y in (("a", "b"), ("b", "a"))
            )
            for cols in towers("ab")
        }
        for values in itertools.product((False, True), repeat=6):
            solver = c.new_solver(5000)
            solver.add(c.get_consts("tbl") == table)
            for (relation, (left, right)), value in zip(
                itertools.product(("ON_star", "Higher", "Scattered"), ((a, b), (b, a))),
                values,
            ):
                solver.add(getattr(c, relation)(left, right) == value)
            self.assertEqual(
                solver.check(),
                z3.sat if values in possible else z3.unsat,
                (values, solver.reason_unknown()),
            )

    def test_enum_constructors_can_alias_table_and_remain_distinct(self):
        c = context("reduction", count=1)
        first, second = c.enum_blocks
        for chosen in (first, second):
            solver = c.new_solver(5000)
            solver.add(c.get_consts("tbl") == chosen, first != second)
            self.assertEqual(solver.check(), z3.sat, solver.reason_unknown())
            self.assertTrue(
                z3.is_true(solver.model().eval(c.get_consts("tbl") == chosen))
            )

    def test_boolean_normalization_and_quantifier_shadowing(self):
        c = context("reduction", count=2)
        x, y = z3.Consts("x y", c.BoxSort)
        p = z3.Exists([x], c.Higher(x, y))
        q = z3.ForAll([x], c.ON_star(x, x))
        for expression in (
            z3.Implies(q, p),
            z3.Xor(p, q),
            p == q,
            z3.If(q, p, z3.Not(p)),
            z3.Or(p, z3.Exists([x], z3.Not(c.Higher(x, x)))),
            z3.Exists([x], z3.ForAll([y], c.Higher(x, y))),
        ):
            normal = to_epr(expression, c)
            restored = normal.matrix
            if normal.forall:
                restored = z3.ForAll(normal.forall, restored)
            if normal.exists:
                restored = z3.Exists(normal.exists, restored)
            solver = z3.Solver()
            solver.set(timeout=5000)
            solver.add(
                expand_finite_objects(expression, c)
                != expand_finite_objects(restored, c)
            )
            self.assertEqual(solver.check(), z3.unsat)

    def test_shared_put_vc_uses_reduction(self):
        c = context("reduction", count=2)
        a, b, tbl = [c.get_consts(n) for n in ("a", "b", "tbl")]
        x = c.get_consts("x")
        clear = lambda obj: z3.ForAll([x], z3.Implies(c.ON_star(x, obj), x == obj))
        pre = z3.And(z3.Distinct(a, b, tbl), clear(a), clear(b))
        vc = VC("body", None, z3.Implies(pre, wp(Put("a", "b"), c.ON_star(a, b), c)))
        result = discharge_vc(vc, c, 10000)
        self.assertEqual(result.status, "valid", result.reason)

    def test_unbounded_table_put_vc_and_frozen_frame(self):
        c = context("reduction")
        a, b = [c.get_consts(n) for n in ("a", "b")]
        action = Put("a", "tbl")
        for post, premise in (
            (c.ON_star(a, a), z3.BoolVal(True)),
            (c.ON_star_zero(a, b), c.ON_star_zero(a, b)),
        ):
            formula = z3.Implies(
                z3.And(put_precondition(action, c), premise), wp(action, post, c)
            )
            result = discharge_vc(VC("body", None, formula), c, 10000)
            self.assertEqual(result.status, "valid", result.reason)

    def test_unbounded_block_put_and_table_alias_counterexample(self):
        c = context("reduction")
        a, b, tbl = [c.get_consts(n) for n in ("a", "b", "tbl")]
        action = Put("a", "b")
        for post, extra, expected in (
            (c.ON_star(a, b), b != tbl, "valid"),
            (c.ON_star_zero(a, b), c.ON_star_zero(a, b), "valid"),
            (c.ON_star(a, b), z3.BoolVal(True), "invalid"),
        ):
            premise = z3.And(put_precondition(action, c), extra)
            formula = z3.Implies(premise, wp(action, post, c))
            result = discharge_vc(VC("body", None, formula), c, 10000)
            self.assertEqual(result.status, expected, result.reason)

    def test_scene_simplification_preserves_shadowed_binders_and_free_names(self):
        c = context("reduction", count=2)
        x, y, a = [c.get_consts(n) for n in ("x", "y", "scene_binder!0")]
        expression = z3.And(
            z3.Exists([x], z3.And(x == a, z3.ForAll([y], c.Higher(x, y)))),
            z3.ForAll([x], z3.Exists([y], z3.Or(x == y, c.Scattered(x, a)))),
            z3.ForAll([y], z3.Exists([x], c.ON_star(x, y))),
        )
        simplified = simplify_scene_formula(expression, c)
        # No scene identities apply here; alpha normalization must preserve
        # the formula even for arbitrary relation interpretations.
        solver = z3.Solver()
        solver.add(
            expand_finite_objects(expression, c) != expand_finite_objects(simplified, c)
        )
        self.assertEqual(solver.check(), z3.unsat)


if __name__ == "__main__":
    unittest.main()
