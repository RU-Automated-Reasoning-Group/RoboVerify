import unittest

import z3

from synthesis.api.instructions import Assign, Get, Put, While
from synthesis.api.program import Program
from synthesis.api.guard_eval import evaluate_z3
from synthesis.cfg.demos import DemoSegment, DemoTrace
from synthesis.cfg.graph import Edge, RelationalCFG
from synthesis.cfg.lower import lower
from synthesis.cfg.region import BlockRegion, LoopRegion
from synthesis.cfg.scope import scope
from synthesis.cfg.validate import validate_split
from synthesis.predicates.scene import Scene
from synthesis.predicates.term import atom, boolean, conjunction, ref
from synthesis.verification_lib.highlevel_verification_lib import HighLevelContext


class IRTests(unittest.TestCase):
    def test_absolute_split_indices_survive_repeated_splits(self):
        segment = DemoSegment(3, 10, 19, DemoTrace(tuple(range(20))))
        left, right = segment.split(15)
        inner, end = right.split(17)
        self.assertEqual((inner.t_start, inner.t_end), (15, 17))
        self.assertEqual(end.states, (17, 18, 19))
        self.assertIs(inner.parent, right)
        self.assertFalse(validate_split({3: 15}, {3: 15}))
        self.assertTrue(validate_split({3: 16}, {3: 15}))

    def test_lowering_preserves_program_vcs_and_recorded_iteration_bound(self):
        ctx = HighLevelContext()
        guard = atom("eq", ref("b_prime"), ref("b"))
        region = LoopRegion(guard, ("b_prime",), (BlockRegion((Put("b_prime", "tbl"),)),),
                            init=(("b", "b0"),), update=(("b", "b_prime"),),
                            invariant=z3.BoolVal(True), iteration_counts=(2, 3))
        graph = RelationalCFG.initial([], boolean(True), boolean(True))
        graph.nodes["v0"].region = region
        actual = lower(graph, ctx)
        expected = Program(2, [Assign("b", "b0"), While(ctx.get_consts("b_prime") == ctx.get_consts("b"),
                            [ctx.get_consts("b_prime")], [Put("b_prime", "tbl"), Assign("b", "b_prime")], z3.BoolVal(True), max_iters=3)])
        self.assertEqual(actual.instructions[1].max_iters, 3)
        for a, b in zip(actual.VC_gen(z3.BoolVal(True), z3.BoolVal(True), ctx), expected.VC_gen(z3.BoolVal(True), z3.BoolVal(True), ctx)):
            self.assertTrue(a.expr.eq(b.expr))

    def test_scope_intersects_guard_and_bypass_paths(self):
        graph = RelationalCFG.initial([], boolean(True), boolean(True), ("b0",))
        graph.edges = [Edge("entry", "v0", boolean(True), frozenset(["x"])),
                       Edge("entry", "exit", boolean(True)), Edge("v0", "exit", boolean(True))]
        values = scope(graph)
        self.assertIn("x", values["v0"])
        self.assertNotIn("x", values["exit"])

    def test_get_requires_witness_and_quantifies_all_choices(self):
        ctx = HighLevelContext()
        x, b = ctx.get_consts("x"), ctx.get_consts("b")
        p = Program(1, [Get("x", x == b, [x])])
        self.assertTrue(p.highlevel_verification(z3.BoolVal(True), x == b, context=ctx))
        empty = Program(1, [Get("x", z3.BoolVal(False), [x])])
        self.assertFalse(empty.highlevel_verification(z3.BoolVal(True), z3.BoolVal(True), context=ctx))

    def test_legacy_nested_guards_and_frozen_geometry(self):
        ctx = HighLevelContext()
        x, y, b = (ctx.get_consts(v) for v in ("x", "y", "b"))
        scene = Scene({0: (0,0,.425), 1: (.2,0,.425)}, {"b": 0}, {0: (0,0,.425), 1: (0,0,.475)})
        expr = z3.ForAll([x], z3.Exists([y], x == y))
        self.assertTrue(evaluate_z3(expr, scene))
        self.assertTrue(evaluate_z3(z3.Exists([x], z3.And(x != b, ctx.ON_star_zero(x,b))), scene))
        self.assertFalse(evaluate_z3(z3.Exists([x], z3.And(x != b, ctx.ON_star(x,b))), scene))


if __name__ == "__main__": unittest.main()


class UnstackLoweringTests(unittest.TestCase):
    def test_existing_unstack_fixture_and_lowered_cfg_have_same_verdicts(self):
        from synthesis.entry.verify_unstack_with_learned_invariant import build_unstack_programs
        from synthesis.predicates.term import forall, disjunction, implies, negate
        from synthesis.verification_lib.symbolic_verify import discharge_vc
        ctx=HighLevelContext(mode="enum",num_blocks=3,use_tbl=True,sort_name="UnstackLowering")
        bp,b0,c,tbl=(ref(n) for n in ("b_prime","b0","c","tbl"))
        guard=conjunction(negate(atom("eq",bp,tbl)),forall(["c"],implies(
            conjunction(negate(atom("eq",c,tbl)),disjunction(atom("ON_star",b0,bp),atom("ON_star",c,b0))),atom("ON_star",bp,c))))
        region=LoopRegion(guard,("b_prime",),(BlockRegion((Put("b_prime","tbl"),)),),
                          init=(("b","b0"),),update=(("b","b_prime"),),invariant=z3.BoolVal(True),iteration_counts=(10,))
        cfg=RelationalCFG.initial([],boolean(True),boolean(True));cfg.nodes["v0"].region=region
        actual=lower(cfg,ctx);expected,_=build_unstack_programs(ctx,z3.BoolVal(True))
        for a,b in zip(actual.VC_gen(z3.BoolVal(True),z3.BoolVal(True),ctx),expected.VC_gen(z3.BoolVal(True),z3.BoolVal(True),ctx)):
            solver=z3.Solver();solver.set(timeout=1000);solver.add(a.expr != b.expr)
            self.assertEqual(solver.check(),z3.unsat)
            self.assertEqual(discharge_vc(a,ctx,1000).status,discharge_vc(b,ctx,1000).status)
