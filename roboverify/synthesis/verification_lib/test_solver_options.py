import argparse
import contextlib
import io
import unittest
from unittest.mock import patch

import z3
from synthesis.entry.verification_options import add_verification_options
from synthesis.verification_lib.highlevel_verification_lib import HighLevelContext
from synthesis.verification_lib.solver_options import (
    finite_sat_witness,
    get_minimize_unsat_core,
    using_unsat_core_minimization,
)
from synthesis.verification_lib.symbolic_verify import VC, discharge_vc


class UnsatCoreOptionsTests(unittest.TestCase):
    def test_finite_unsat_does_not_refute_unbounded_consistency(self):
        context = HighLevelContext()
        objects = [z3.FreshConst(context.BoxSort) for _ in range(6)]
        solver = z3.Solver()
        solver.add(z3.Distinct(*objects))
        self.assertFalse(finite_sat_witness(solver, context, restore_timeout_ms=1000))
        self.assertEqual(solver.check(), z3.sat)

    def test_finite_sat_witness_preserves_assertions_and_scope(self):
        context = HighLevelContext(use_tbl=True, exists_top=True)
        solver = context.new_solver(1000)
        a, b = [context.get_consts(n) for n in ("a", "b")]
        solver.add(a != b)
        assertions = list(solver.assertions())
        self.assertTrue(finite_sat_witness(solver, context, restore_timeout_ms=1000))
        self.assertEqual(solver.num_scopes(), 0)
        self.assertEqual(
            [a.sexpr() for a in solver.assertions()], [a.sexpr() for a in assertions]
        )

    def test_inconclusive_redundancy_keeps_the_candidate(self):
        from synthesis.inference_lib.inference import check_redundancy

        context = HighLevelContext()
        a, b = [context.get_consts(n) for n in ("a", "b")]
        candidate = context.ON_star(a, b)
        with patch(
            "synthesis.inference_lib.inference.z3.Solver", spec=z3.Solver
        ) as solver, contextlib.redirect_stdout(io.StringIO()):
            solver.return_value.check.return_value = z3.unknown
            solver.return_value.reason_unknown.return_value = "timeout"
            kept = check_redundancy([candidate, candidate], context, timeout_ms=1)
        self.assertEqual(len(kept), 1)
        self.assertTrue(kept[0].eq(candidate))

    def test_consistency_unknown_can_use_sat_but_not_finite_unsat(self):
        original_check = z3.Solver.check
        for count, expected in ((2, "valid"), (6, "unknown")):
            with self.subTest(count=count):
                context = HighLevelContext(use_tbl=True, exists_top=True)
                objects = [z3.FreshConst(context.BoxSort) for _ in range(count)]
                vc = VC("exit", "loop", z3.Implies(z3.Distinct(*objects), True))
                calls = 0

                def unknown_consistency(solver, *args):
                    nonlocal calls
                    calls += 1
                    return z3.unknown if calls == 2 else original_check(solver, *args)

                with patch.object(
                    z3.Solver, "check", unknown_consistency
                ), patch.object(
                    z3.Solver, "reason_unknown", return_value="incomplete quantifiers"
                ):
                    result = discharge_vc(vc, context, 1000)
                self.assertEqual(result.status, expected)
                self.assertEqual(result.queries, 3)

    def test_cli_default_and_both_switches(self):
        parser = argparse.ArgumentParser()
        add_verification_options(parser)
        self.assertFalse(parser.parse_args([]).minimize_unsat_core)
        self.assertTrue(
            parser.parse_args(["--minimize-unsat-core"]).minimize_unsat_core
        )
        self.assertFalse(
            parser.parse_args(["--no-minimize-unsat-core"]).minimize_unsat_core
        )

    def test_nested_scope_restores_after_exception(self):
        self.assertFalse(get_minimize_unsat_core())
        with using_unsat_core_minimization(True):
            with using_unsat_core_minimization():
                self.assertTrue(get_minimize_unsat_core())
            with self.assertRaisesRegex(RuntimeError, "stop"):
                with using_unsat_core_minimization(False):
                    self.assertFalse(get_minimize_unsat_core())
                    raise RuntimeError("stop")
            self.assertTrue(get_minimize_unsat_core())
        self.assertFalse(get_minimize_unsat_core())
        with self.assertRaises(ValueError):
            with using_unsat_core_minimization("false"):
                pass

    def test_option_reaches_new_finite_and_unbounded_solvers(self):
        settings = []
        original_set = z3.Solver.set

        def record(solver, *args, **kwargs):
            if args and args[0] == "smt.core.minimize":
                settings.append(args[1])
            return original_set(solver, *args, **kwargs)

        with patch.object(z3.Solver, "set", record):
            HighLevelContext().new_solver(1000)
            with using_unsat_core_minimization(True):
                HighLevelContext().new_solver(1000)
                HighLevelContext(
                    mode="enum", num_blocks=2, sort_name="CoreOptionsFinite"
                ).new_solver(1000)
            HighLevelContext().new_solver(1000)
        self.assertEqual(settings, [False, True, True, False])

    def test_both_modes_preserve_valid_invalid_and_vacuous_verdicts(self):
        context = HighLevelContext()
        a = context.get_consts("a")
        reflexive = context.ON_star(a, a)
        cases = (
            (z3.Implies(True, reflexive), "valid"),
            (z3.Implies(True, False), "invalid"),
            (z3.Implies(z3.Not(reflexive), False), "vacuous"),
        )
        for enabled in (False, True):
            with using_unsat_core_minimization(enabled):
                for formula, expected in cases:
                    with self.subTest(enabled=enabled, expected=expected):
                        result = discharge_vc(VC("body", None, formula), context, 1000)
                        self.assertEqual(result.status, expected)
                        if expected == "valid":
                            self.assertEqual(result.queries, 2)

    def test_cfg_cli_scopes_and_records_option(self):
        from synthesis.entry.synthesize_cfg import main

        for flags, expected in (
            ([], False),
            (["--minimize-unsat-core"], True),
            (["--no-minimize-unsat-core"], False),
        ):
            with self.subTest(flags=flags), patch(
                "synthesis.entry.synthesize_cfg.RunLogger"
            ) as logger, patch(
                "synthesis.entry.synthesize_cfg._run"
            ) as run, contextlib.redirect_stdout(
                io.StringIO()
            ):

                def check_scope(args, logger):
                    self.assertEqual(get_minimize_unsat_core(), expected)
                    return 0

                run.side_effect = check_scope
                self.assertEqual(main(["--demos", "unused.npz", *flags]), 0)
                run.assert_called_once()
                self.assertEqual(
                    logger.call_args.args[2]["minimize_unsat_core"], expected
                )
                self.assertFalse(get_minimize_unsat_core())
