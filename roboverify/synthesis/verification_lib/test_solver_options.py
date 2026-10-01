import argparse
import contextlib
import io
import unittest
from unittest.mock import patch

import z3

from synthesis.entry.verification_options import add_verification_options
from synthesis.verification_lib.highlevel_verification_lib import HighLevelContext
from synthesis.verification_lib.solver_options import (
    get_minimize_unsat_core,
    using_unsat_core_minimization,
)
from synthesis.verification_lib.symbolic_verify import VC, discharge_vc


class UnsatCoreOptionsTests(unittest.TestCase):
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
