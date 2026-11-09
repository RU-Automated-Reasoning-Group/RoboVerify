"""Backend selection, relational assumptions, and shared WP scene regressions."""

import argparse
import contextlib
import io
import itertools
import unittest
from unittest.mock import patch
from uuid import uuid4

import z3

from synthesis.api.instructions import Assign, Put, Seq, Skip, While
from synthesis.api.program import Program, put_effect, put_precondition, wp
from synthesis.api.relational_summary import expand_finite_objects
from synthesis.entry.verification_options import (
    add_verification_options,
    wp_options_from_args,
)
from synthesis.verification_lib.highlevel_verification_lib import HighLevelContext
from synthesis.verification_lib.relational_axioms import bottom, maximum, next_rep
from synthesis.verification_lib.symbolic_verify import VC, discharge_vc
from synthesis.verification_lib.wp_options import using_wp_system


def context(system="new", count=None, **kwargs):
    return HighLevelContext(
        wp_system=system,
        mode="declare" if count is None else "enum",
        enum_names=(
            None if count is None else [f"obj{i}" for i in range(count)] + ["table"]
        ),
        use_tbl=True,
        sort_name="WP_" + uuid4().hex,
        **kwargs,
    )


def towers(names):
    result = set()
    for order in itertools.permutations(names):
        for cuts in itertools.product((False, True), repeat=len(names) - 1):
            columns, start = [], 0
            for i, cut in enumerate((*cuts, True)):
                if cut:
                    columns.append(order[start : i + 1])
                    start = i + 1
            result.add(tuple(sorted(columns)))
    return sorted(result)


def facts(columns):
    return (
        {name: i for i, column in enumerate(columns) for name in column},
        {name: j for column in columns for j, name in enumerate(column)},
    )


def truth(relation, left, right, current, frozen):
    if "tbl" in (left, right):
        return relation != "Scattered" and left == right
    roots, levels = frozen if relation == "ON_star_zero" else current
    if relation == "Higher":
        return levels[left] >= levels[right]
    if relation == "Scattered":
        return roots[left] != roots[right]
    return roots[left] == roots[right] and levels[left] >= levels[right]


class WPSelectionTests(unittest.TestCase):
    def test_default_scopes_and_explicit_override(self):
        self.assertEqual(HighLevelContext().wp_system, "old")
        with using_wp_system("new", support_complete=True):
            self.assertEqual(HighLevelContext().wp_system, "new")
            self.assertTrue(HighLevelContext().wp_options.support_complete)
            with using_wp_system("reduction"):
                self.assertEqual(HighLevelContext().wp_system, "reduction")
            self.assertEqual(HighLevelContext().wp_system, "new")
            self.assertEqual(
                HighLevelContext(wp_system="old", support_complete=False).wp_system,
                "old",
            )
        self.assertEqual(HighLevelContext().wp_system, "old")

    def test_flags_and_incompatible_vocabulary(self):
        parser = argparse.ArgumentParser()
        add_verification_options(parser)
        self.assertEqual(wp_options_from_args(parser.parse_args([])).wp_system, "old")
        args = parser.parse_args(["--wp-system", "new", "--wp-support-complete"])
        self.assertTrue(wp_options_from_args(args).support_complete)
        with self.assertRaisesRegex(ValueError, "require"):
            HighLevelContext(wp_system="new", height_predicate="EqualHeight")
        with self.assertRaisesRegex(ValueError, "require"):
            HighLevelContext(wp_system="new", verification_mode="goals")
        with self.assertRaisesRegex(ValueError, "requires|require"):
            HighLevelContext(wp_system="reduction", support_complete=True)

    def test_cfg_rebuild_preserves_backend_options(self):
        from synthesis.cfg.test_tower_verification import TowerVerificationTests
        from synthesis.cfg.verification import symbolic_problem

        original, cfg = TowerVerificationTests().graph("unstack")
        cfg.nodes["v0"].region.invariant = z3.BoolVal(True)
        selected = HighLevelContext(
            wp_system="new", use_tbl=True, exists_top=True, support_complete=True
        )
        for size in (2, None):
            _, _, _, rebuilt = symbolic_problem(cfg, selected, size)
            self.assertEqual(rebuilt.wp_options, selected.wp_options)

    def test_cli_scopes_and_records_selected_backend(self):
        from synthesis.entry.synthesize_cfg import main

        for system in ("old", "new", "reduction"):
            with self.subTest(system=system), patch(
                "synthesis.entry.synthesize_cfg.RunLogger"
            ) as logger, patch(
                "synthesis.entry.synthesize_cfg._run"
            ) as run, contextlib.redirect_stdout(
                io.StringIO()
            ):

                def check_scope(args, logger):
                    self.assertEqual(HighLevelContext().wp_system, system)
                    return 0

                run.side_effect = check_scope
                self.assertEqual(
                    main(["--demos", "unused.npz", "--wp-system", system]), 0
                )
                run.assert_called_once()
                self.assertEqual(logger.call_args.args[2]["wp_system"], system)
            self.assertEqual(HighLevelContext().wp_system, "old")


class RelationalAxiomTests(unittest.TestCase):
    def test_representatives_cover_physical_blocks_including_table_isolation(self):
        c = context()
        a, b, u, tbl = [c.get_consts(n) for n in ("a", "b", "u", "tbl")]
        claims = [
            z3.Implies(
                z3.And(a != tbl, b != tbl), z3.Or(c.Higher(a, b), c.Higher(b, a))
            ),
            z3.Implies(a != tbl, z3.Exists([u], z3.And(u != tbl, bottom(c, u)))),
            z3.Implies(
                b != tbl,
                z3.Or(
                    maximum(c, b), z3.Exists([u], z3.And(u != tbl, next_rep(c, b, u)))
                ),
            ),
            z3.Implies(
                a != tbl, z3.And(z3.Not(c.Higher(a, tbl)), z3.Not(c.Higher(tbl, a)))
            ),
        ]
        for claim in claims:
            solver = c.new_solver(5000)
            solver.add(z3.Not(claim))
            self.assertEqual(solver.check(), z3.unsat, solver.reason_unknown())

    def test_new_axioms_exclude_incomparable_singletons_and_false_wp(self):
        for mode in ("old", "new"):
            c = context(mode, 2)
            a, b, tbl = [c.get_consts(n) for n in ("a", "b", "tbl")]
            solver = c.new_solver(5000)
            solver.add(
                z3.Distinct(a, b, tbl), z3.Not(c.Higher(a, b)), z3.Not(c.Higher(b, a))
            )
            self.assertEqual(solver.check(), z3.sat if mode == "old" else z3.unsat)
        c = context()
        for target in ("b", "tbl"):
            solver = c.new_solver(5000)
            solver.add(wp(Put("a", target), z3.BoolVal(False), c))
            self.assertEqual(solver.check(), z3.unsat, solver.reason_unknown())

    def test_optional_completeness_and_independent_frozen_scene(self):
        for complete in (False, True):
            c = context(count=2, support_complete=complete)
            a, b, tbl = [c.get_consts(n) for n in ("a", "b", "tbl")]
            x, y = z3.Consts("x y", c.BoxSort)
            solver = c.new_solver(5000)
            solver.add(
                z3.Distinct(a, b, tbl),
                c.Higher(b, a),
                z3.Not(c.Higher(a, b)),
                z3.ForAll([x, y], c.ON_star(x, y) == (x == y)),
                c.ON_star_zero(b, a),
            )
            self.assertEqual(solver.check(), z3.unsat if complete else z3.sat)


class SharedWPTests(unittest.TestCase):
    def test_all_three_block_scenes_against_independent_relocation(self):
        c = context(count=3)
        objects = dict(zip(("a", "b", "c", "tbl"), c.enum_blocks))
        solver = z3.Solver()
        solver.set(timeout=5000)
        solver.add(*(c.get_consts(n) == obj for n, obj in objects.items()))
        solver.add(c.get_consts("alias") == objects["tbl"])
        frozen = facts((("c", "a", "b"),))
        predictions = {}
        for columns in towers("abc"):
            current = facts(columns)
            solver.push()
            for rel in ("ON_star", "ON_star_zero", "Higher", "Scattered"):
                for left, right in itertools.product(objects, repeat=2):
                    solver.add(
                        getattr(c, rel)(objects[left], objects[right])
                        == truth(rel, left, right, current, frozen)
                    )
            self.assertEqual(solver.check(), z3.sat)
            for source in (column[-1] for column in columns):
                for target in [
                    column[-1] for column in columns if column[-1] != source
                ] + ["tbl", "alias"]:
                    key = source, target
                    if key not in predictions:
                        action = Put(source, target)
                        predictions[key] = (
                            expand_finite_objects(put_precondition(action, c), c),
                            {
                                (rel, left, right): expand_finite_objects(
                                    put_effect(
                                        action,
                                        getattr(c, rel)(objects[left], objects[right]),
                                        c,
                                    ),
                                    c,
                                )
                                for rel in (
                                    "ON_star",
                                    "ON_star_zero",
                                    "Higher",
                                    "Scattered",
                                )
                                for left, right in itertools.product(objects, repeat=2)
                            },
                        )
                    roots, heights = map(dict, current)
                    roots[source] = (
                        max(roots.values()) + 1
                        if target in ("tbl", "alias")
                        else roots[target]
                    )
                    heights[source] = (
                        0 if target in ("tbl", "alias") else heights[target] + 1
                    )
                    legal, effects = predictions[key]
                    solver.push()
                    solver.add(
                        z3.Or(
                            z3.Not(legal),
                            *(
                                effect
                                != truth(rel, left, right, (roots, heights), frozen)
                                for (rel, left, right), effect in effects.items()
                            ),
                        )
                    )
                    self.assertEqual(
                        solver.check(), z3.unsat, (columns, source, target)
                    )
                    solver.pop()
            solver.pop()

    def test_quantifier_capture_negation_and_table_alias(self):
        c = context(count=2)
        a, b, tbl = [c.get_consts(n) for n in ("a", "b", "tbl")]
        x, y = z3.Consts("height_rep maximum", c.BoxSort)
        posts = [
            z3.Not(c.Higher(a, b)),
            z3.ForAll(
                [x], z3.Exists([y], z3.And(c.Higher(x, y), c.ON_star_zero(x, b)))
            ),
            z3.Exists(
                [x], z3.And(c.Higher(a, x), c.Higher(x, a), c.ON_star_zero(x, b))
            ),
        ]
        old = HighLevelContext(wp_system="old", use_tbl=True, sort_name=c.sort_name)
        # Reuse exactly the same finite declarations for the legacy traversal.
        old.BoxSort = c.BoxSort
        for name in ("ON_star", "ON_star_zero", "Higher", "EqualHeight", "Scattered"):
            setattr(old, name, getattr(c, name))
        for columns in towers("ab"):
            solver = z3.Solver()
            solver.set(timeout=5000)
            objects = dict(zip(("a", "b", "tbl"), c.enum_blocks))
            solver.add(*(c.get_consts(n) == obj for n, obj in objects.items()))
            solver.add(c.get_consts("alias") == tbl)
            current = facts(columns)
            for rel in ("ON_star", "ON_star_zero", "Higher", "Scattered"):
                for left, right in itertools.product(objects, repeat=2):
                    solver.add(
                        getattr(c, rel)(objects[left], objects[right])
                        == truth(rel, left, right, current, current)
                    )
            for post in posts:
                for target in ("b", "tbl"):
                    solver.push()
                    solver.add(
                        wp(Put("a", target), post, c) != wp(Put("a", target), post, old)
                    )
                    self.assertEqual(solver.check(), z3.unsat)
                    solver.pop()
                solver.push()
                solver.add(
                    wp(Seq(Assign("alias", "tbl"), Put("a", "alias")), post, c)
                    != wp(Put("a", "tbl"), post, c)
                )
                self.assertEqual(solver.check(), z3.unsat)
                solver.pop()

    def test_quantified_unstack_guard_remains_supported(self):
        from synthesis.examples.unstack import build_program

        c = context(count=2, exists_top=True)
        original = build_program(c, num_blocks=2).instructions[0]
        loop = While(
            original.instantiated_cond,
            original.guard_exists_vars,
            [Put("b_prime", "tbl")],
            z3.BoolVal(True),
        )
        vcs = Program(1, [loop]).VC_gen(z3.BoolVal(True), z3.BoolVal(True), c)
        self.assertEqual([vc.kind for vc in vcs], ["establish", "preserve", "exit"])
        self.assertEqual(discharge_vc(vcs[-1], c, 5000).status, "valid")

    def test_public_legality_still_rejects_occupied_or_table_source(self):
        c = context()
        a, b, other, tbl = [c.get_consts(n) for n in ("a", "b", "other", "tbl")]
        for premise in (
            a == tbl,
            a == b,
            z3.And(other != a, c.ON_star(other, a)),
            z3.And(other != b, c.ON_star(other, b)),
        ):
            solver = z3.Solver()
            solver.add(premise, b != tbl, wp(Put("a", "b"), z3.BoolVal(True), c))
            self.assertEqual(solver.check(), z3.unsat)


if __name__ == "__main__":
    unittest.main()
