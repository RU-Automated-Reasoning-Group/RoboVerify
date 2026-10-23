"""Put dispatch follows support values and preserves frozen entry geometry."""

import unittest
from uuid import uuid4

import z3
from synthesis.api.instructions import Assign, Put, Seq
from synthesis.api.program import wp
from synthesis.verification_lib.highlevel_verification_lib import HighLevelContext


class PutAliasTests(unittest.TestCase):
    def setUp(self):
        self.context = HighLevelContext(
            mode="enum",
            enum_names=["a", "old", "dest", "tbl"],
            sort_name="PutAlias" + uuid4().hex,
            use_tbl=True,
        )
        self.a, self.old, self.dest, self.tbl = [
            self.context.get_consts(n) for n in ("a", "old", "dest", "tbl")
        ]
        self.b = self.context.get_consts("b")

    def test_table_alias_matches_literal_for_all_relations_and_quantifiers(self):
        c = self.context
        x, y = z3.Consts("x y", c.BoxSort)
        for relation in (c.ON_star, c.ON_star_zero, c.Higher, c.Scattered):
            for post in (
                relation(self.a, self.old),
                z3.ForAll([x], z3.Exists([y], relation(x, y))),
            ):
                solver = z3.Solver()
                solver.add(self.b == self.tbl)
                solver.add(wp(Put("a", "b"), post, c) != wp(Put("a", "tbl"), post, c))
                self.assertEqual(solver.check(), z3.unsat)

    def test_first_iteration_assignment_dispatches_to_table(self):
        c = self.context
        post = z3.And(
            c.Scattered(self.a, self.old), z3.Not(c.ON_star(self.a, self.old))
        )
        solver = z3.Solver()
        solver.add(
            wp(Seq(Assign("b", "tbl"), Put("a", "b")), post, c)
            != wp(Put("a", "tbl"), post, c)
        )
        self.assertEqual(solver.check(), z3.unsat)

    def test_table_branch_does_not_change_frozen_entry_order(self):
        c = self.context
        solver = z3.Solver()
        x = z3.FreshConst(c.BoxSort)
        solver.add(
            self.b == self.tbl,
            self.a != self.tbl,
            c.ON_star_zero(self.a, self.old),
            z3.ForAll([x], z3.Implies(c.ON_star(x, self.a), x == self.a)),
        )
        solver.add(z3.Not(wp(Put("a", "b"), c.ON_star_zero(self.a, self.old), c)))
        self.assertEqual(solver.check(), z3.unsat)

    def test_block_put_detaches_source_before_attachment(self):
        c = self.context
        names = ("a", "old", "dest", "tbl")
        objects = {n: c.get_consts(n) for n in names}
        solver = z3.Solver()
        solver.add(*(objects[n] == e for n, e in zip(names, c.enum_blocks)))
        for left in names:
            for right in names:
                value = left == right or (left, right) == ("a", "old")
                solver.add(c.ON_star(objects[left], objects[right]) == value)
                solver.add(c.ON_star_zero(objects[left], objects[right]) == value)
        post = z3.And(
            c.ON_star(self.a, self.dest),
            z3.Not(c.ON_star(self.a, self.old)),
            c.ON_star_zero(self.a, self.old),
        )
        solver.add(z3.Not(wp(Put("a", "dest"), post, c)))
        self.assertEqual(solver.check(), z3.unsat)

    def test_public_put_rejects_occupied_source_or_destination(self):
        c = self.context
        for occupied in (self.a, self.dest):
            solver = z3.Solver()
            solver.add(z3.Distinct(self.a, self.old, self.dest, self.tbl))
            solver.add(c.ON_star(self.old, occupied))
            solver.add(wp(Put("a", "dest"), z3.BoolVal(True), c))
            self.assertEqual(solver.check(), z3.unsat)


if __name__ == "__main__":
    unittest.main()
