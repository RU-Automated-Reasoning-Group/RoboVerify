import contextlib
import io
import itertools
import unittest
from unittest.mock import Mock

import z3

from synthesis.api.instructions import Assign, While
from synthesis.api.program import Program
from synthesis.experiment.invariant_learning.tasks import StackExperiment
from synthesis.experiment.invariant_learning.witness import (
    WitnessQuery,
    find_witness,
    finite_formula,
    missing_head_query,
)
from synthesis.inference_lib.demo_store import DemoStore, InvInference, LoopHeadState
from synthesis.verification_lib.counterexamples import stacks_to_positions
from synthesis.verification_lib.highlevel_verification_lib import HighLevelContext


class WitnessTests(unittest.TestCase):
    ids = itertools.count()

    def context(self, n):
        return HighLevelContext(
            mode="enum", num_blocks=n, sort_name=f"WitnessTest{next(self.ids)}"
        )

    def test_finite_expansion_preserves_nested_shadowed_quantifiers(self):
        ctx = self.context(2)
        x, y = ctx.get_consts("x"), ctx.get_consts("y")
        formula = z3.ForAll(
            [x],
            z3.Exists(
                [y],
                z3.And(
                    x == y,
                    z3.ForAll([x], z3.Or(x == y, ctx.Higher(x, y))),
                ),
            ),
        )
        expanded = finite_formula(formula, ctx)
        solver = z3.Solver()
        solver.add(z3.Xor(formula, expanded))
        self.assertEqual(solver.check(), z3.unsat)

    def test_first_id_binding_excludes_other_matching_witness(self):
        ctx = self.context(3)
        a, choice, base = [ctx.get_consts(s) for s in ("a", "choice", "base")]
        loop = While(
            choice != a, [choice], [Assign("a", "choice")], a != ctx.enum_blocks[2]
        )
        program = Program(2, [Assign("a", "base"), loop])
        solver = ctx.new_solver(1000)
        solver.add(base == ctx.enum_blocks[0], missing_head_query(program, ctx, 1))
        self.assertEqual(solver.check(), z3.unsat)
        # With base=1, the first selected object is 0, which is uncovered here.
        program.instructions[1].invariant = a != ctx.enum_blocks[0]
        solver = ctx.new_solver(1000)
        solver.add(base == ctx.enum_blocks[1], missing_head_query(program, ctx, 1))
        self.assertEqual(solver.check(), z3.sat)

    def test_zero_iteration_head_is_included(self):
        ctx = self.context(1)
        loop = While(z3.BoolVal(False), [], [], z3.BoolVal(False))
        solver = ctx.new_solver(1000)
        solver.add(missing_head_query(Program(1, [loop]), ctx, 0))
        self.assertEqual(solver.check(), z3.sat)

    def test_guard_false_successor_is_included(self):
        ctx = self.context(2)
        a, b = ctx.get_consts("a"), ctx.get_consts("b")
        loop = While(a != b, [], [Assign("a", "b")], a != b)
        solver = ctx.new_solver(1000)
        solver.add(a != b, missing_head_query(Program(1, [loop]), ctx, 1))
        self.assertEqual(solver.check(), z3.sat)

    def test_unknown_stops_before_larger_size(self):
        solver = Mock()
        solver.check.side_effect = [z3.sat, z3.unknown]
        solver.reason_unknown.return_value = "timeout"
        build = Mock(return_value=WitnessQuery(solver, z3.BoolVal(True), None, 0))
        result = find_witness(build, 4)
        self.assertEqual(result.status, "unknown")
        build.assert_called_once_with(1)

    def test_empty_domain_and_unsat_are_recorded_before_smallest_witness(self):
        def build(n):
            solver = z3.Solver()
            if n == 1:
                solver.add(z3.BoolVal(False))
            return WitnessQuery(solver, z3.BoolVal(n == 3), lambda model: n, 0)

        result = find_witness(build, 4)
        self.assertEqual(result.size, 3)
        self.assertEqual(
            [r["status"] for r in result.attempts],
            ["empty_initial_domain", "unsat", "sat"],
        )

    def test_two_block_invariant_requires_reachable_three_block_head(self):
        task = StackExperiment("synthesis.examples.stack:build_program")
        import numpy as np

        task._geometry = (np.array([0.0, 0.0]), np.array([0.6, 0.0, 0.5]), 0.425)
        store = DemoStore()
        for size in (1, 2):
            blocks = [f"x{i+1}" for i in range(size)]
            entry = stacks_to_positions([[b] for b in blocks])
            for top in range(size):
                store.add(
                    LoopHeadState(
                        task.loop_id,
                        stacks_to_positions(
                            [blocks[: top + 1]] + [[b] for b in blocks[top + 1 :]]
                        ),
                        entry,
                        {"b0": blocks[0], "b": blocks[top]},
                    )
                )
        with contextlib.redirect_stdout(io.StringIO()):
            invariant, _ = InvInference(
                store, task.loop_id, task.vocabulary, task.context
            )
        task.set_invariant(invariant)
        result = find_witness(lambda size: task.search_query(size, 10000), 3)
        self.assertEqual(result.status, "found", result.reason)
        self.assertEqual(result.size, 3)
        self.assertEqual(
            [r["status"] for r in result.attempts], ["unsat", "unsat", "sat"]
        )
        self.assertEqual(len(result.witness.positions), 3)


if __name__ == "__main__":
    unittest.main()
