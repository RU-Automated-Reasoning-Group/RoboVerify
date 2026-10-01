"""Concrete diagnostic witnesses for the Higher ablation, without saved demos."""

import unittest

import z3

from synthesis.api.instructions import Assign, Put
from synthesis.api.program import to_seq, wp
from synthesis.experiment.ablate_higher import structural_invariant, uses_higher
from synthesis.inference_lib.demo_store import LoopHeadState
from synthesis.util.on import using_higher_tolerance
from synthesis.verification_lib.counterexamples import stacks_to_positions, state_holds
from synthesis.verification_lib.highlevel_verification_lib import HighLevelContext


class HigherAblationTests(unittest.TestCase):
    def setUp(self):
        self.context = HighLevelContext()
        self.invariant = structural_invariant(self.context)

    def test_structural_control_covers_stack_heads_and_exit_without_higher(self):
        blocks = [f"x{i}" for i in range(4)]
        entry = stacks_to_positions([[b] for b in blocks])
        self.assertFalse(uses_higher(self.invariant))
        for top in range(4):
            state = LoopHeadState(
                "1",
                stacks_to_positions(
                    [blocks[: top + 1]] + [[b] for b in blocks[top + 1 :]]
                ),
                entry,
                {"b0": blocks[0], "b": blocks[top]},
            )
            self.assertTrue(state_holds(self.invariant, state))

    def test_second_tower_exposes_attachment_preservation_mismatch(self):
        # This is the three-clause invariant observed in the no-Higher run:
        # clear top, root, and separation. It omits the single-tower clause.
        weak = z3.And(
            self.invariant.arg(0), self.invariant.arg(1), self.invariant.arg(3)
        )
        positions = stacks_to_positions([["a"], ["d", "c"]])
        state = LoopHeadState(
            "1", positions, positions, {"b": "a", "b0": "a", "b_prime": "c"}
        )
        selected, b = self.context.get_consts("b_prime"), self.context.get_consts("b")
        n = z3.Const("guard_object", self.context.BoxSort)
        guard = z3.And(
            selected != b,
            z3.ForAll(
                [n], z3.Implies(self.context.ON_star(n, selected), n == selected)
            ),
        )
        body = to_seq([Put("b_prime", "b"), Assign("b", "b_prime")])
        self.assertTrue(state_holds(weak, state))
        self.assertTrue(state_holds(guard, state))
        self.assertFalse(state_holds(wp(body, weak, self.context), state))
        self.assertFalse(state_holds(self.invariant, state))
        # Real detachment loses ON*(c,d); the abstract attachment WP retains it.
        successor = LoopHeadState(
            "1",
            stacks_to_positions([["a", "c"], ["d"]]),
            positions,
            {"b": "c", "b0": "a", "b_prime": "c"},
        )
        self.assertTrue(state_holds(weak, successor))

    def test_relational_structure_alone_does_not_bound_singleton_height(self):
        positions = stacks_to_positions([["a"], ["c"]])
        before = LoopHeadState(
            "1", positions, positions, {"b": "a", "b0": "a", "c": "c"}
        )
        raised = {name: list(point) for name, point in positions.items()}
        raised["c"][2] += 0.5
        after = LoopHeadState("1", raised, positions, before.constants)
        self.assertTrue(state_holds(self.invariant, before))
        self.assertTrue(state_holds(self.invariant, after))
        for name in ("ON_star", "Scattered"):
            for a in ("b", "c"):
                for b in ("b", "c"):
                    predicate = getattr(self.context, name)(
                        self.context.get_consts(a), self.context.get_consts(b)
                    )
                    self.assertEqual(
                        state_holds(predicate, before), state_holds(predicate, after)
                    )
        higher = self.context.Higher(
            self.context.get_consts("b"), self.context.get_consts("c")
        )
        with using_higher_tolerance(0.001):
            self.assertTrue(state_holds(higher, before))
            self.assertFalse(state_holds(higher, after))


if __name__ == "__main__":
    unittest.main()
