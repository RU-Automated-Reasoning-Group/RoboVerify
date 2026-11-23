"""Adversarial checks for weak action summaries, not deterministic Put rewrites."""

import itertools
import unittest

import z3

from synthesis.api.instructions import Assign, Seq
from synthesis.api.program import wp
from synthesis.api.relational_summary import (
    RelationalSummary,
    expand_finite_objects,
    rewrite_relations,
    successor_contract,
    summary_wp,
)
from synthesis.verification_lib.highlevel_verification_lib import HighLevelContext

_sorts = itertools.count()


class RelationalSummaryTests(unittest.TestCase):
    def context(self, count=2, *, table=False):
        context = HighLevelContext(
            mode="enum",
            num_blocks=count,
            use_tbl=table,
            sort_name=f"RelationalSummaryTest{next(_sorts)}",
        )
        names = ["a", "b", "c"][:count]
        if table:
            names[-1] = "tbl"
        values = [context.get_consts(name) for name in names]
        self.context_ = context
        self.bindings = [x == y for x, y in zip(values, context.enum_blocks)]
        return context, values

    def check(self, formula, expected, *premises):
        solver = self.context_.new_solver(10000)
        solver.add(*self.bindings, *premises, formula)
        result = solver.check()
        self.assertEqual(result, expected, solver.reason_unknown())

    def separated(self, context):
        x, y = z3.Consts("x y", context.BoxSort)
        return z3.ForAll(
            [x, y],
            z3.And(
                context.ON_star(x, y) == (x == y),
                context.Higher(x, y),
                context.Scattered(x, y) == (x != y),
            ),
        )

    def test_endpoint_is_established_in_both_operand_orders(self):
        context, (a, b) = self.context()
        for relation, left, right in (
            ("Higher", "a", "b"),
            ("Higher", "b", "a"),
            ("Scattered", "a", "b"),
            ("Scattered", "b", "a"),
        ):
            with self.subTest(relation=relation, left=left):
                summary = RelationalSummary(relation, left, right, moved="a")
                self.check(
                    z3.Not(wp(summary, summary.atom(context), context)), z3.unsat
                )

    def test_higher_does_not_imply_on_or_strict_height(self):
        context, (a, b) = self.context()
        summary = RelationalSummary("Higher", "a", "b")
        for goal in (context.ON_star(a, b), z3.Not(context.Higher(b, a))):
            self.check(wp(summary, goal, context), z3.unsat)

    def test_scattered_does_not_choose_a_height(self):
        context, (a, b) = self.context()
        summary = RelationalSummary("Scattered", "a", "b")
        for goal in (context.Higher(a, b), context.Higher(b, a)):
            self.check(wp(summary, goal, context), z3.unsat)

    def test_scattered_from_one_block_does_not_scatter_from_every_block(self):
        context, (a, b, c) = self.context(3)
        summary = RelationalSummary("Scattered", "a", "b")
        self.check(
            wp(summary, context.Scattered(a, c), context),
            z3.unsat,
            self.separated(context),
        )

    def test_nonmover_frame_is_preserved_but_incident_facts_are_not(self):
        context, (a, b, c) = self.context(3)
        summary = RelationalSummary("Higher", "a", "b")
        self.check(
            z3.Not(wp(summary, context.Scattered(b, c), context)),
            z3.unsat,
            self.separated(context),
        )
        self.check(
            wp(summary, context.Scattered(a, c), context),
            z3.unsat,
            self.separated(context),
        )

    def test_impossible_endpoint_and_empty_admissibility_do_not_verify(self):
        context, (a, b) = self.context()
        impossible = RelationalSummary("Scattered", "a", "a")
        self.check(wp(impossible, z3.BoolVal(True), context), z3.unsat)
        possible = RelationalSummary("Higher", "a", "b")
        self.check(
            summary_wp(
                possible, z3.BoolVal(True), context, admissibility=z3.BoolVal(False)
            ),
            z3.unsat,
        )
        self.check(wp(possible, z3.BoolVal(False), context), z3.unsat)

    def test_table_is_an_isolated_marker_not_a_movable_block(self):
        context, (a, table) = self.context(table=True)
        for summary in (
            RelationalSummary("Higher", "tbl", "a"),
            RelationalSummary("Scattered", "a", "tbl"),
        ):
            self.check(wp(summary, z3.BoolVal(True), context), z3.unsat)

    def test_frozen_geometry_and_shadowed_quantifiers_survive(self):
        context, (a, b) = self.context()
        summary = RelationalSummary("Higher", "a", "b")
        x = z3.Const("summary_ON_star", context.BoxSort)
        y = z3.Const("summary_Higher", context.BoxSort)
        goal = z3.And(
            z3.Not(context.ON_star_zero(a, b)),
            z3.ForAll([x], z3.Exists([y], z3.And(x == y, context.ON_star_zero(x, y)))),
            z3.ForAll([x], z3.Exists([x], context.Higher(x, x))),
        )
        self.check(
            z3.Not(wp(summary, goal, context)),
            z3.unsat,
            z3.Not(context.ON_star_zero(a, b)),
        )

    def test_summary_composes_with_alias_assignment(self):
        context, (a, b) = self.context()
        c = context.get_consts("c")
        summary = RelationalSummary("Scattered", "a", "b")
        sequence = Seq(summary, Assign("c", "a"))
        self.check(z3.Not(wp(sequence, context.Scattered(c, b), context)), z3.unsat)

    def test_two_summaries_havoc_the_first_incident_fact(self):
        context, (a, b) = self.context()
        sequence = Seq(
            RelationalSummary("Scattered", "a", "b"),
            RelationalSummary("Higher", "a", "b"),
        )
        self.check(z3.Not(wp(sequence, context.Higher(a, b), context)), z3.unsat)
        self.check(wp(sequence, context.Scattered(a, b), context), z3.unsat)
        self.check(wp(sequence, z3.BoolVal(False), context), z3.unsat)

    def test_finite_expansion_preserves_mixed_and_nested_binders(self):
        context, (a, b) = self.context()
        flag = z3.Bool("same_name")
        x = z3.Const("same_name", context.BoxSort)
        array = z3.Array("array", context.BoxSort, z3.BoolSort())
        formulas = (
            z3.ForAll([flag], z3.Exists([x], flag == context.Higher(x, a))),
            z3.ForAll([flag, x], z3.Exists([flag, x], z3.And(flag, x == a))),
        )
        for formula in formulas:
            expanded = expand_finite_objects(formula, context)
            self.check(formula != expanded, z3.unsat)
        # Comparing quantified arrays with SMT is itself incomplete. Check the
        # exact expected bound-variable AST for this independently built case.
        formula = z3.Exists([array], z3.ForAll([x], array[x] == context.Higher(x, a)))
        expected = z3.Exists(
            [array],
            z3.And(
                *(
                    array[value] == context.Higher(value, a)
                    for value in context.enum_blocks
                )
            ),
        )
        expanded = expand_finite_objects(formula, context)
        expected = z3.simplify(expected, blast_select_store=True)
        self.assertTrue(expanded.is_exists())
        self.assertEqual(expanded.num_vars(), 1)
        self.assertEqual(expanded.var_sort(0), array.sort())
        # Binder display names deliberately differ; their de Bruijn bodies agree.
        self.assertTrue(expanded.body().eq(expected.body()))

    def test_finite_encoding_quantifies_booleans_unbounded_quantifies_relations(self):
        context, _ = self.context()
        finite, _ = successor_contract(RelationalSummary("Higher", "a", "b"), context)
        self.assertEqual(len(finite.variables), 3 * 2 * 2)
        self.assertTrue(all(z3.is_bool(v) for v in finite.variables))
        unbounded_context = HighLevelContext()
        symbolic, _ = successor_contract(
            RelationalSummary("Higher", "a", "b"), unbounded_context
        )
        self.assertEqual(len(symbolic.variables), 3)
        self.assertTrue(all(z3.is_array(v) for v in symbolic.variables))

    def test_rewrite_is_simultaneous_and_retains_frozen_relation(self):
        context, (a, b) = self.context()
        after, _ = successor_contract(RelationalSummary("Higher", "a", "b"), context)
        rewritten = rewrite_relations(
            z3.And(context.Higher(a, b), context.ON_star_zero(a, b)), context, after
        )
        self.assertIn("ON_star_zero", rewritten.sexpr())
        self.assertNotIn("(Higher ", rewritten.sexpr())


if __name__ == "__main__":
    unittest.main()
