import unittest

import z3

from synthesis.api.guard_eval import evaluate_z3
from synthesis.api.instructions import Assign, Get, Put, While
from synthesis.api.program import Program
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

    def test_lowering_preserves_vcs_without_using_demo_count_as_execution_limit(self):
        ctx = HighLevelContext()
        guard = atom("eq", ref("b_prime"), ref("b"))
        region = LoopRegion(
            guard,
            ("b_prime",),
            (BlockRegion((Put("b_prime", "tbl"),)),),
            init=(("b", "b0"),),
            update=(("b", "b_prime"),),
            invariant=z3.BoolVal(True),
            iteration_counts=(2, 3),
        )
        graph = RelationalCFG.initial([], boolean(True), boolean(True))
        graph.nodes["v0"].region = region
        actual = lower(graph, ctx)
        expected = Program(
            2,
            [
                Assign("b", "b0"),
                While(
                    ctx.get_consts("b_prime") == ctx.get_consts("b"),
                    [ctx.get_consts("b_prime")],
                    [Put("b_prime", "tbl"), Assign("b", "b_prime")],
                    z3.BoolVal(True),
                    max_iters=None,
                ),
            ],
        )
        self.assertIsNone(actual.instructions[1].max_iters)
        for a, b in zip(
            actual.VC_gen(z3.BoolVal(True), z3.BoolVal(True), ctx),
            expected.VC_gen(z3.BoolVal(True), z3.BoolVal(True), ctx),
        ):
            self.assertTrue(a.expr.eq(b.expr))

    def test_loop_budget_exhaustion_is_not_normal_exit(self):
        from types import SimpleNamespace
        from unittest.mock import Mock, patch

        from synthesis.api.instructions import LoopBudgetExceeded

        env = SimpleNamespace(num_blocks=1, symbolic_name_to_box_id={})
        scene = Scene({0: (0, 0, 0)}, {})
        body = Mock()
        body.eval.return_value = []
        loop = While(z3.BoolVal(True), [], [body], z3.BoolVal(True), max_iters=2)
        with patch(
            "synthesis.predicates.scene.scene_from_obs", return_value=scene
        ), patch.object(loop, "_find_and_bind_guard_exists", return_value=True):
            with self.assertRaises(LoopBudgetExceeded):
                loop.eval(env, [object()])
        self.assertEqual(body.eval.call_count, 2)
        body.reset_mock()
        loop.max_iters = None
        with patch(
            "synthesis.predicates.scene.scene_from_obs", return_value=scene
        ), patch.object(
            loop, "_find_and_bind_guard_exists", side_effect=[True] * 4 + [False]
        ):
            loop.eval(env, [object()])
        self.assertEqual(body.eval.call_count, 4)

    def test_scope_intersects_guard_and_bypass_paths(self):
        graph = RelationalCFG.initial([], boolean(True), boolean(True), ("b0",))
        graph.edges = [
            Edge("entry", "v0", boolean(True), frozenset(["x"])),
            Edge("entry", "exit", boolean(True)),
            Edge("v0", "exit", boolean(True)),
        ]
        values = scope(graph)
        self.assertIn("x", values["v0"])
        self.assertNotIn("x", values["exit"])

    def test_get_requires_witness_and_quantifies_all_choices(self):
        ctx = HighLevelContext()
        x, b = ctx.get_consts("x"), ctx.get_consts("b")
        p = Program(1, [Get("x", x == b, [x])])
        self.assertTrue(p.highlevel_verification(z3.BoolVal(True), x == b, context=ctx))
        empty = Program(1, [Get("x", z3.BoolVal(False), [x])])
        self.assertFalse(
            empty.highlevel_verification(
                z3.BoolVal(True), z3.BoolVal(True), context=ctx
            )
        )

    def test_learned_guard_rejects_multiple_witnesses_without_mutating_bindings(self):
        from types import SimpleNamespace
        from unittest.mock import patch

        from synthesis.api.guard_eval import AmbiguousGuardWitness, find_and_bind

        ctx = HighLevelContext()
        inst = While(z3.BoolVal(True), [ctx.get_consts("x")], [], z3.BoolVal(True))
        inst.require_unique_guard = True
        inst._get_num_blocks = lambda env, obs: 2
        env = SimpleNamespace(symbolic_name_to_box_id={"b0": 0})
        scene = Scene({0: (0, 0, 0.425), 1: (0.2, 0, 0.425)}, {"b0": 0})
        with patch("synthesis.api.guard_eval.scene_from_obs", return_value=scene):
            with self.assertRaises(AmbiguousGuardWitness):
                find_and_bind(inst, env, [None])
        self.assertEqual(env.symbolic_name_to_box_id, {"b0": 0})

    def test_learned_guard_uniqueness_is_a_verification_obligation(self):
        from synthesis.verification_lib.symbolic_verify import discharge_vc

        context = HighLevelContext(mode="enum", num_blocks=2, sort_name="UniqueGuard")
        x, b = context.get_consts("x"), context.get_consts("b")
        for guard, expected in [(z3.BoolVal(True), "invalid"), (x == b, "valid")]:
            loop = While(guard, [x], [Assign("b", "b")], z3.BoolVal(True))
            loop.require_unique_guard = True
            obligations = Program(1, [loop]).VC_gen(
                z3.BoolVal(True), z3.BoolVal(True), context
            )
            check = next(vc for vc in obligations if vc.kind == "guard_unique")
            self.assertEqual(
                discharge_vc(check, context, timeout_ms=1000).status, expected
            )

    def test_legacy_nested_guards_and_frozen_geometry(self):
        ctx = HighLevelContext()
        x, y, b = (ctx.get_consts(v) for v in ("x", "y", "b"))
        scene = Scene(
            {0: (0, 0, 0.425), 1: (0.2, 0, 0.425)},
            {"b": 0},
            {0: (0, 0, 0.425), 1: (0, 0, 0.475)},
        )
        expr = z3.ForAll([x], z3.Exists([y], x == y))
        self.assertTrue(evaluate_z3(expr, scene))
        self.assertTrue(
            evaluate_z3(z3.Exists([x], z3.And(x != b, ctx.ON_star_zero(x, b))), scene)
        )
        self.assertFalse(
            evaluate_z3(z3.Exists([x], z3.And(x != b, ctx.ON_star(x, b))), scene)
        )


if __name__ == "__main__":
    unittest.main()


class UnstackLoweringTests(unittest.TestCase):
    def test_existing_unstack_fixture_and_lowered_cfg_have_same_verdicts(self):
        from synthesis.entry.verify_unstack_with_learned_invariant import (
            build_unstack_programs,
        )
        from synthesis.predicates.term import disjunction, forall, implies, negate
        from synthesis.verification_lib.symbolic_verify import discharge_vc

        ctx = HighLevelContext(
            mode="enum", num_blocks=3, use_tbl=True, sort_name="UnstackLowering"
        )
        bp, b0, c, tbl = (ref(n) for n in ("b_prime", "b0", "c", "tbl"))
        guard = conjunction(
            negate(atom("eq", bp, tbl)),
            forall(
                ["c"],
                implies(
                    conjunction(
                        negate(atom("eq", c, tbl)),
                        disjunction(atom("ON_star", b0, bp), atom("ON_star", c, b0)),
                    ),
                    atom("ON_star", bp, c),
                ),
            ),
        )
        region = LoopRegion(
            guard,
            ("b_prime",),
            (BlockRegion((Put("b_prime", "tbl"),)),),
            init=(("b", "b0"),),
            update=(("b", "b_prime"),),
            invariant=z3.BoolVal(True),
            iteration_counts=(10,),
        )
        cfg = RelationalCFG.initial([], boolean(True), boolean(True))
        cfg.nodes["v0"].region = region
        actual = lower(cfg, ctx)
        expected, _ = build_unstack_programs(ctx, z3.BoolVal(True))
        for a, b in zip(
            actual.VC_gen(z3.BoolVal(True), z3.BoolVal(True), ctx),
            expected.VC_gen(z3.BoolVal(True), z3.BoolVal(True), ctx),
        ):
            solver = z3.Solver()
            solver.set(timeout=1000)
            solver.add(a.expr != b.expr)
            self.assertEqual(solver.check(), z3.unsat)
            self.assertEqual(
                discharge_vc(a, ctx, 1000).status, discharge_vc(b, ctx, 1000).status
            )
