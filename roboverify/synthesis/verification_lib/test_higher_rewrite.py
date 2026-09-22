"""Put Higher effects against the agreed discrete physical height model."""

import unittest

import z3

from synthesis.api.instructions import Put
from synthesis.api.program import wp
from synthesis.verification_lib.highlevel_verification_lib import HighLevelContext


class HigherPutRuleTwo(unittest.TestCase):
    def test_matches_placement_on_all_integer_height_levels(self):
        high = HighLevelContext()
        a, b, c = [high.get_consts(name) for name in ("a", "b", "c")]
        # Normalize L to one and the common bottom-center height to zero.
        # The source's old height is arbitrary and must not affect this result.
        za = z3.Real("old_source_height")
        zb, zc = z3.Ints("base_level other_level")
        solver = z3.Solver()
        solver.add(z3.Distinct(a, b, c), z3.Not(high.ON_star(b, a)))
        for left, zl in ((a, za), (b, zb), (c, zc)):
            for right, zr in ((a, za), (b, zb), (c, zc)):
                solver.add(high.Higher(left, right) == (zl >= zr))
        self.assertEqual(solver.check(), z3.sat)
        predicted = wp(Put("a", "b"), high.Higher(c, a), high)
        solver.add(predicted != (zc >= zb + 1))
        self.assertEqual(solver.check(), z3.unsat)

    def test_preserves_table_isolation_when_source_is_level_with_base(self):
        high = HighLevelContext(use_tbl=True)
        a, b, tbl = [high.get_consts(name) for name in ("a", "b", "tbl")]
        solver = z3.Solver()
        high.add_axiom_higher(solver)
        solver.add(
            z3.Distinct(a, b, tbl),
            z3.Not(high.ON_star(b, a)),
            high.Higher(a, b),
            high.Higher(b, a),
        )
        self.assertEqual(solver.check(), z3.sat)
        solver.add(wp(Put("a", "b"), high.Higher(tbl, a), high))
        self.assertEqual(solver.check(), z3.unsat)
