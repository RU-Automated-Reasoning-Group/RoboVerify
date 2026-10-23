"""Current tower programs, supplied invariant controls, and proof scope."""

import unittest

import z3
from synthesis.api.instructions import Put, While
from synthesis.cfg.lower import lower
from synthesis.cfg.program_adapter import program_to_cfg
from synthesis.cfg.program_source import load_program
from synthesis.cfg.tasks import task_spec
from synthesis.cfg.verification import propose_summaries, symbolic_problem
from synthesis.examples.tower_invariants import (
    reverse_paper_final,
    reverse_paper_table,
    unstack_paper_base,
    unstack_paper_physical_base,
)
from synthesis.inference_lib.demo_store import LoopHeadState
from synthesis.util.on import TABLE
from synthesis.verification_lib.counterexamples import state_holds
from synthesis.verification_lib.highlevel_verification_lib import HighLevelContext


class TowerVerificationTests(unittest.TestCase):
    def test_unstack_paper_needs_its_base_to_remain_a_physical_block(self):
        context = HighLevelContext(use_tbl=True, exists_top=True)
        b0, tbl = [context.get_consts(n) for n in ("b0", "tbl")]
        obj = z3.FreshConst(context.BoxSort)
        solver = context.new_solver(1000)
        solver.add(z3.ForAll([obj], obj == tbl), b0 == tbl)
        solver.add(unstack_paper_base(context))
        self.assertEqual(solver.check(), z3.sat)
        solver.add(unstack_paper_physical_base(context))
        self.assertEqual(solver.check(), z3.unsat)

    def graph(self, task):
        context = HighLevelContext(use_tbl=True, exists_top=True)
        definition = load_program(
            f"synthesis.examples.{task}:build_program",
            context,
            4,
            allow_observation_references=True,
        )
        cfg = program_to_cfg(definition, [], *task_spec(task))
        propose_summaries(cfg, context)
        return context, cfg

    def test_current_programs_propose_the_requested_put_targets(self):
        for task, target in (("unstack", "tbl"), ("reverse", "b")):
            with self.subTest(task=task):
                context, cfg = self.graph(task)
                # Lowering requires an invariant, but summary construction does not.
                for node in cfg.nodes.values():
                    if hasattr(node.region, "invariant"):
                        node.region.invariant = z3.BoolVal(True)
                program = lower(cfg, context)
                loop = next(i for i in program.instructions if isinstance(i, While))
                puts = [i for i in loop.body if isinstance(i, Put)]
                self.assertEqual(
                    [(i.upper_block, i.base_block) for i in puts], [("b_prime", target)]
                )

    def test_finite_checks_count_physical_blocks_in_addition_to_table(self):
        context, cfg = self.graph("unstack")
        cfg.nodes["v0"].region.invariant = z3.BoolVal(True)
        _, pre, _, finite = symbolic_problem(cfg, context, 2)
        self.assertTrue(finite.exists_top)
        self.assertEqual(len(finite.enum_blocks), 3)
        a, b, tbl = [finite.get_consts(n) for n in ("a", "b", "tbl")]
        solver = finite.new_solver()
        solver.add(pre, z3.Distinct(a, b, tbl))
        self.assertEqual(solver.check(), z3.sat)

    def test_paper_repair_admits_the_actual_final_reverse_state(self):
        context = HighLevelContext(use_tbl=True)
        state = LoopHeadState(
            "1",
            {"base": [0.4, -0.15, 0.475], "top": [0.4, -0.15, 0.425], "tbl": TABLE},
            {"base": [0.6, 0.2, 0.425], "top": [0.6, 0.2, 0.475], "tbl": TABLE},
            {"b0": "base", "b": "base", "tbl": "tbl"},
        )
        self.assertFalse(state_holds(reverse_paper_table(context), state))
        self.assertTrue(state_holds(reverse_paper_final(context), state))


if __name__ == "__main__":
    unittest.main()
