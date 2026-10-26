"""EqualHeight agreement and placement against independently updated towers."""

import contextlib
import io
import itertools
import unittest
from uuid import uuid4

import z3
from synthesis.api.guard_eval import evaluate_z3
from synthesis.api.instructions import Put
from synthesis.api.program import put_effect, put_precondition, wp
from synthesis.api.relational_summary import (
    RelationalSummary,
    expand_finite_objects,
    successor_contract,
)
from synthesis.cfg.tasks import symbolic_task_spec, task_identity
from synthesis.inference_lib.demo_store import LoopHeadState
from synthesis.inference_lib.inference import (
    compute_data,
    compute_data_with_function,
    compute_omega_k,
)
from synthesis.predicates.scene import Scene, evaluate
from synthesis.predicates.term import atom, from_z3, ref, to_z3
from synthesis.util import on
from synthesis.verification_lib.counterexamples import model_to_loop_head, state_holds
from synthesis.verification_lib.height_predicate import using_height_predicate
from synthesis.verification_lib.highlevel_verification_lib import HighLevelContext
from synthesis.verification_lib.lowlevel_verification_lib import LowLevelContext


class EqualHeightSemanticsTests(unittest.TestCase):
    def test_all_evaluation_paths_and_table(self):
        high = HighLevelContext(height_predicate="EqualHeight", use_tbl=True)
        a, b = [high.get_consts(n) for n in ("a", "b")]
        tolerance = 1 / 1024  # Exact inclusive floating-point boundary.
        cases = [
            ([0, 0, 0.5], [0.2, 0, 0.5], True),
            ([0, 0, 0.5], [0.2, 0, 0.5 + tolerance], True),
            ([0, 0, 0.5], [0.2, 0, 0.5 + tolerance * 1.01], False),
            ([0, 0, 0.5], [0.2, 0, 0.55], False),
            (on.TABLE, [0, 0, 0.5], False),
            (on.TABLE, on.TABLE, True),
        ]
        with on.using_higher_tolerance(tolerance):
            for left, right, expected in cases:
                for p, q in ((left, right), (right, left)):
                    with self.subTest(p=p, q=q):
                        positions = {
                            "tbl" if pos is on.TABLE else name: pos
                            for name, pos in (("p", p), ("q", q))
                        }
                        bindings = {
                            "a": "tbl" if p is on.TABLE else "p",
                            "b": "tbl" if q is on.TABLE else "q",
                        }
                        if "tbl" in positions:
                            bindings["tbl"] = "tbl"
                        scene = Scene(positions, bindings)
                        row = LoopHeadState("loop", positions, positions, bindings)
                        formula = high.EqualHeight(a, b)
                        self.assertEqual(on.equal_height_implementation(p, q), expected)
                        self.assertEqual(
                            evaluate(atom("EqualHeight", ref("a"), ref("b")), scene),
                            expected,
                        )
                        self.assertEqual(evaluate_z3(formula, scene), expected)
                        self.assertEqual(state_holds(formula, row), expected)
                        with contextlib.redirect_stdout(io.StringIO()):
                            data = compute_data(
                                positions,
                                positions,
                                [formula],
                                {},
                                {a: bindings["a"], b: bindings["b"]},
                            )
                            function_data = compute_data_with_function(
                                positions,
                                positions,
                                [formula],
                                {},
                                {a: bindings["a"], b: bindings["b"]},
                                [],
                                [],
                            )
                        self.assertEqual(bool(data[0]), expected)
                        self.assertEqual(data, function_data)
                        low = LowLevelContext(use_tbl=True)
                        la, lb, tbl = [low.get_consts(n) for n in ("a", "b", "tbl")]
                        solver = z3.Solver()
                        for obj, pos in ((la, p), (lb, q)):
                            solver.add(obj == tbl if pos is on.TABLE else obj != tbl)
                            if pos is not on.TABLE:
                                solver.add(low.Z(obj) == z3.RealVal(repr(pos[2])))
                        solver.add(low.lowlevel_equal_height(la, lb) != expected)
                        self.assertEqual(solver.check(), z3.unsat)
                        exact = low.translate_exact(formula, {"a": la, "b": lb})
                        self.assertTrue(z3.eq(exact, low.lowlevel_equal_height(la, lb)))

    def test_algebra_scope_serialization_and_archive_identity(self):
        identity = task_identity("stack")
        with using_height_predicate("EqualHeight"):
            high = HighLevelContext()
            self.assertEqual(high.height_predicate, "EqualHeight")
            self.assertEqual(identity, task_identity("stack"))
            self.assertIn("EqualHeight", str(symbolic_task_spec("stack", high)[0]))
        self.assertEqual(HighLevelContext().height_predicate, "Higher")
        solver = high.new_solver(5000)
        self.assertNotIn("Higher", solver.sexpr())
        x, y, z = [high.get_consts(n) for n in ("x", "y", "z")]
        solver.add(
            high.EqualHeight(x, y),
            high.EqualHeight(y, z),
            z3.Not(high.EqualHeight(x, z)),
        )
        self.assertEqual(solver.check(), z3.unsat)
        formula = z3.ForAll([x], high.EqualHeight(x, y))
        restored = high.spec_to_expr(
            high.expr_to_spec(formula), known_const_names=["y"]
        )
        solver = z3.Solver()
        solver.add(z3.Or(formula != restored, formula != to_z3(from_z3(formula), high)))
        self.assertEqual(solver.check(), z3.unsat)
        vocabulary, _ = compute_omega_k(2, [high.EqualHeight], [y], high)
        self.assertTrue(vocabulary)
        self.assertTrue(all(p.decl() == high.EqualHeight for p in vocabulary))
        self.assertEqual(len(vocabulary), 3)
        self.assertTrue(all(str(p.arg(0)) < str(p.arg(1)) for p in vocabulary))

    def test_tolerant_equality_requires_separated_levels(self):
        a, b, c = ([0, 0, z] for z in (0, 0.00075, 0.0015))
        self.assertTrue(on.equal_height_implementation(a, b))
        self.assertTrue(on.equal_height_implementation(b, c))
        self.assertFalse(on.equal_height_implementation(a, c))

    def test_summary_havocs_the_selected_height_relation(self):
        high = HighLevelContext(
            mode="enum",
            num_blocks=2,
            sort_name="EqualSummary",
            height_predicate="EqualHeight",
        )
        a, b = [high.get_consts(n) for n in ("a", "b")]
        action = RelationalSummary("EqualHeight", "a", "b")
        after, _ = successor_contract(action, high)
        self.assertEqual(set(after), {"ON_star", "EqualHeight", "Scattered"})
        solver = high.new_solver(5000)
        solver.add(a == high.enum_blocks[0], b == high.enum_blocks[1])
        solver.add(z3.Not(wp(action, high.EqualHeight(b, a), high)))
        self.assertEqual(solver.check(), z3.unsat)
        with self.assertRaisesRegex(ValueError, "selected algebra"):
            successor_contract(RelationalSummary("Higher", "a", "b"), high)
        with self.assertRaisesRegex(ValueError, "selected algebra"):
            to_z3(atom("Higher", ref("a"), ref("b")), high)
        with self.assertRaisesRegex(ValueError, "selected algebra"):
            wp(Put("a", "b"), high.Higher(a, b), high)
        with self.assertRaisesRegex(ValueError, "selected algebra"):
            wp(action, high.Higher(a, b), high)

    def test_realization_uses_equal_height_without_higher(self):
        high = HighLevelContext(
            mode="enum",
            num_blocks=2,
            sort_name="EqualRealize",
            height_predicate="EqualHeight",
        )
        a, b = high.enum_blocks
        solver = high.new_solver(5000)
        for x, y in itertools.product((a, b), repeat=2):
            solver.add(
                high.ON_star(x, y) == x.eq(y),
                high.ON_star_zero(x, y) == x.eq(y),
                high.EqualHeight(x, y),
                high.Scattered(x, y) == (not x.eq(y)),
            )
        solver.add(high.get_consts("a") == a, high.get_consts("b") == b)
        self.assertEqual(solver.check(), z3.sat)
        row = model_to_loop_head(high, solver.model(), "loop", ("a", "b"))
        self.assertTrue(
            state_holds(
                high.EqualHeight(high.get_consts("a"), high.get_consts("b")), row
            )
        )


class EqualHeightPlacementTests(unittest.TestCase):
    def test_all_four_block_towers_and_enabled_relocations(self):
        # Enumerate every partition into ordered columns (bottom to top).
        scenes = set()
        for ordering in itertools.permutations("abcd"):
            for cuts in itertools.product((False, True), repeat=3):
                towers, start = [], 0
                for i, cut in enumerate((*cuts, True)):
                    if cut:
                        towers.append(ordering[start : i + 1])
                        start = i + 1
                scenes.add(tuple(sorted(towers)))
        high = HighLevelContext(
            mode="enum",
            enum_names=list("abcd") + ["tbl"],
            sort_name="EqualTower_" + uuid4().hex,
            use_tbl=True,
            height_predicate="EqualHeight",
        )
        objects = dict(zip(list("abcd") + ["tbl"], high.enum_blocks))
        solver = z3.Solver()
        solver.set(timeout=5000)
        for name, obj in objects.items():
            solver.add(high.get_consts(name) == obj)
        solver.add(high.get_consts("alias") == objects["tbl"])
        effects = {}
        for towers in sorted(scenes):
            heights = {n: level for column in towers for level, n in enumerate(column)}
            columns = {n: col for col, column in enumerate(towers) for n in column}
            solver.push()
            for left, right in itertools.product(objects, repeat=2):
                physical = "tbl" not in (left, right)
                solver.add(
                    high.EqualHeight(objects[left], objects[right])
                    == (heights[left] == heights[right] if physical else left == right)
                )
                solver.add(
                    high.ON_star(objects[left], objects[right])
                    == (
                        columns[left] == columns[right]
                        and heights[left] >= heights[right]
                        if physical
                        else left == right
                    )
                )
            for source in (c[-1] for c in towers):
                targets = [c[-1] for c in towers if c[-1] != source] + ["tbl", "alias"]
                for target in targets:
                    with self.subTest(towers=towers, source=source, target=target):
                        instruction = Put(source, target)
                        key = (source, target)
                        if key not in effects:
                            effects[key] = (
                                expand_finite_objects(
                                    put_precondition(instruction, high), high
                                ),
                                {
                                    (left, right): expand_finite_objects(
                                        put_effect(
                                            instruction,
                                            high.EqualHeight(
                                                objects[left], objects[right]
                                            ),
                                            high,
                                        ),
                                        high,
                                    )
                                    for left, right in itertools.product(
                                        objects, repeat=2
                                    )
                                },
                            )
                        enabled, predictions = effects[key]
                        after = dict(heights)
                        after[source] = (
                            0 if target in ("tbl", "alias") else heights[target] + 1
                        )
                        violations = []
                        for left, right in itertools.product(objects, repeat=2):
                            physical = "tbl" not in (left, right)
                            expected = (
                                after[left] == after[right]
                                if physical
                                else left == right
                            )
                            prediction = predictions[left, right]
                            violations.append(prediction != expected)
                        solver.push()
                        solver.add(z3.Or(z3.Not(enabled), *violations))
                        self.assertEqual(solver.check(), z3.unsat)
                        solver.pop()
            solver.pop()

    def test_occupied_source_rejected_and_quantifiers_do_not_capture(self):
        high = HighLevelContext(height_predicate="EqualHeight")
        a, b, c, t = [high.get_consts(n) for n in ("a", "b", "c", "height_support")]
        renamed = z3.FreshConst(high.BoxSort)
        lhs = wp(Put("a", "b"), z3.ForAll([t], high.EqualHeight(a, t)), high)
        rhs = wp(
            Put("a", "b"), z3.ForAll([renamed], high.EqualHeight(a, renamed)), high
        )
        solver = z3.Solver()
        solver.set(timeout=5000)
        solver.add(lhs != rhs)
        self.assertEqual(solver.check(), z3.unsat)
        solver = z3.Solver()
        solver.add(
            z3.Distinct(a, b, c),
            high.ON_star(c, a),
            wp(Put("a", "b"), z3.BoolVal(True), high),
        )
        self.assertEqual(solver.check(), z3.unsat)


if __name__ == "__main__":
    unittest.main()
