import unittest

from synthesis.cfg.kleene import (
    anti_unify,
    carried_bindings,
    encode_fragment,
    match_template,
)
from synthesis.cfg.quotient import find_repetition
from synthesis.predicates.term import atom, ref


class QuotientTests(unittest.TestCase):
    def test_three_iterations_derive_carried_update_and_rebound_source(self):
        labels = [
            atom("ON", ref(a), ref(b))
            for a, b in [("b1", "b0"), ("b2", "b1"), ("b3", "b2")]
        ]
        result = find_repetition(labels)
        self.assertIsNotNone(result)
        self.assertEqual(
            (result.start, result.width, len(result.substitutions)), (0, 1, 3)
        )
        carry, rebound = carried_bindings(result.template)
        self.assertEqual(carry, {"p1": "p0"})
        self.assertEqual(rebound, ("p0",))
        self.assertEqual(result.template.first["p1"], ref("b0"))
        # p1 -> b, p0 -> b_prime yields Assign(b, b_prime).

    def test_length_two_repeated_units_are_kept(self):
        a = lambda rel, x, y: atom(rel, ref(x), ref(y))
        labels = [
            a("ON", "b1", "b0"),
            a("Higher", "b1", "b0"),
            a("ON", "b2", "b1"),
            a("Higher", "b2", "b1"),
        ]
        result = find_repetition(labels)
        self.assertEqual(result.width, 2)

    def test_noninjective_first_substitution_is_not_composed(self):
        labels = [atom("ON", ref("a"), ref("a")), atom("ON", ref("b"), ref("c"))]
        template = anti_unify(encode_fragment(labels[:1]), encode_fragment(labels[1:]))
        with self.assertRaises(ValueError):
            carried_bindings(template)


if __name__ == "__main__":
    unittest.main()


class FullQuotientTests(unittest.TestCase):
    def test_full_flat_fold_learns_unique_guard_and_lowers_carried_assignment(self):
        import z3

        from synthesis.api.instructions import Assign, While
        from synthesis.cfg.demos import DemoAssignment, DemoSegment, DemoTrace
        from synthesis.cfg.graph import Edge, Node, RelationalCFG
        from synthesis.cfg.lower import lower
        from synthesis.cfg.quotient import quotient
        from synthesis.cfg.region import BlockRegion
        from synthesis.predicates.scene import Scene
        from synthesis.predicates.term import boolean
        from synthesis.verification_lib.highlevel_verification_lib import (
            HighLevelContext,
        )

        positions = {i: (0, 0, 0.425 + 0.05 * i) for i in range(4)}
        scene = Scene(positions, {f"b{i}": i for i in range(4)})
        trace = DemoTrace((scene, scene, scene, scene))
        names = ["v0", "v1", "v2"]
        labels = [atom("ON", ref(f"b{i+1}"), ref(f"b{i}")) for i in range(3)]
        nodes = {
            name: Node(
                name,
                BlockRegion(
                    (Assign("cursor", f"b{i+1}"),), (Assign("cursor", f"b{i+1}"),)
                ),
            )
            for i, name in enumerate(names)
        }
        edges = [
            Edge("entry", "v0", boolean(True)),
            Edge("v0", "v1", labels[0]),
            Edge("v1", "v2", labels[1]),
            Edge("v2", "exit", labels[2]),
        ]
        demos = DemoAssignment(
            {
                name: [DemoSegment(0, i, i + 1, trace, scene.bindings)]
                for i, name in enumerate(names)
            }
        )
        cfg = RelationalCFG(nodes, edges, names, demos, initial_scope=frozenset(["b0"]))
        self.assertTrue(
            quotient(cfg, infer_invariant=lambda rows, g, scope: z3.BoolVal(True))
        )
        cfg.validate_structure()
        program = lower(cfg, HighLevelContext())
        self.assertEqual(
            (program.instructions[0].left, program.instructions[0].right), ("b", "b0")
        )
        loop = program.instructions[1]
        self.assertIsInstance(loop, While)
        self.assertEqual((loop.body[-1].left, loop.body[-1].right), ("b", "b_prime"))
        self.assertEqual(loop.max_iters, 3)
