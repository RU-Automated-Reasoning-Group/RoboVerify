"""Contract cuts preserve milestones, pre-update roles and full trace coverage."""

import unittest
from dataclasses import replace
from unittest.mock import patch

from synthesis.cfg.boundaries import BoundaryBudget, refine_iterations
from synthesis.cfg.demos import DemoSegment, DemoTrace
from synthesis.cfg.iterations import Iterations
from synthesis.cfg.quotient import quotient, retry_boundary_fold
from synthesis.cfg.refine import scene_at
from synthesis.cfg.test_quotient import FullQuotientTests
from synthesis.cfg.test_validate import C, example_cfg
from synthesis.cfg.validate import validate_cfg
from synthesis.predicates.scene import Scene, evaluate
from synthesis.predicates.term import atom, ref


class PartitionTests(unittest.TestCase):
    def test_split_preserves_or_clamps_outgoing_milestone(self):
        cfg = example_cfg(
            [
                (False, False, False),
                (False, True, False),
                (True, False, False),
                (True, False, False),
                (False, False, True),
            ]
        )
        row = replace(cfg.demos.for_node("left")[0], t_end=4, milestone_index=2)
        before, after = row.split(1)
        self.assertIsNone(before.milestone_index)
        self.assertEqual(after.milestone_index, 2)
        _, after = row.split(3)
        self.assertEqual(after.milestone_index, 3)

    def test_final_fold_preserves_long_terminal_suffix(self):
        from synthesis.cfg.graph import Edge
        from synthesis.predicates.term import block_id

        cfg = FullQuotientTests().continuation_cfg()
        cfg.order.remove("tail")
        del cfg.nodes["tail"], cfg.demos.segments["tail"]
        cfg.edges = cfg.edges[:3] + [
            Edge("v2", "exit", atom("ON", block_id(3), block_id(2)))
        ]
        row = cfg.demos.for_node("v2")[0]
        row.trace.states = (*row.trace.states, *([row.trace.states[-1]] * 200))
        end = len(row.trace.states) - 1
        cfg.demos.segments["v2"] = [replace(row, t_end=end)]
        cfg._task_demos = [replace(cfg.demos.for_node("v0")[0], t_end=end)]
        with patch(
            "synthesis.cfg.boundaries.propose_handoffs",
            return_value=(
                [(2, 4, 87)],
                {"status": "found"},
            ),
        ):
            self.assertTrue(quotient(cfg, boundary_budget=BoundaryBudget()))
        tail = cfg.demos.for_node(cfg.order[-1])[0]
        self.assertEqual((tail.t_start, tail.t_end), (87, end))
        self.assertTrue(validate_cfg(cfg))
        cfg.demos.segments[cfg.order[-1]] = [replace(tail, t_end=end - 1)]
        self.assertFalse(validate_cfg(cfg))

    def test_early_loop_search_can_name_unresolved_tail_without_claiming_completion(
        self,
    ):
        from synthesis.api.instructions import Skip
        from synthesis.cfg.region import BlockRegion, LoopRegion
        from synthesis.cfg.synthesize import synthesize_cfg

        cfg = FullQuotientTests().continuation_cfg()
        cfg.nodes["tail"].region = None
        visited = []

        def realize(node, rows, post):
            visited.append((node.name, node.synthesis_approach))
            return (
                node.region
                if isinstance(node.region, LoopRegion)
                else BlockRegion(None, (Skip(0),))
            ), True

        result = synthesize_cfg(
            cfg,
            realize,
            lambda cfg: [],
            quotient=quotient,
            boundary_budget=BoundaryBudget(),
            max_refinements=1,
        )
        self.assertTrue(result)
        self.assertTrue(all(approach == "relational" for name, approach in visited))
        self.assertIn("tail", [name for name, _ in visited])
        self.assertIn("body0", [name for name, _ in visited])

    def test_final_id_closure_still_rejects_unresolved_region(self):
        from synthesis.cfg.id_first import close_id_candidate

        cfg = FullQuotientTests().continuation_cfg()
        cfg.nodes["tail"].region = None
        with self.assertRaisesRegex(ValueError, "unresolved"):
            close_id_candidate(cfg)
        close_id_candidate(cfg, allow_unresolved=True)
        self.assertIsNone(cfg.nodes["tail"].region)

    def test_refined_cut_requires_explicit_persistent_truth_certificate(self):
        cfg = example_cfg(
            [
                (False, False, False),
                (True, False, False),
                (True, False, False),
                (False, False, True),
            ]
        )
        left = cfg.demos.for_node("left")[0]
        right = cfg.demos.for_node("right")[0]
        cfg.demos.segments["left"] = [replace(left, t_end=2)]
        cfg.demos.segments["right"] = [replace(right, t_start=2)]
        self.assertFalse(validate_cfg(cfg))
        cfg.demos.segments["left"] = [replace(left, t_end=2, milestone_index=1)]
        self.assertTrue(validate_cfg(cfg))

    def test_certificate_cannot_bridge_false_milestone_or_drop_transition(self):
        cfg = example_cfg(
            [
                (False, False, False),
                (True, False, False),
                (False, False, False),
                (True, False, False),
                (False, False, True),
            ]
        )
        left = cfg.demos.for_node("left")[0]
        right = cfg.demos.for_node("right")[0]
        cfg.demos.segments["left"] = [replace(left, t_end=3, milestone_index=1)]
        cfg.demos.segments["right"] = [replace(right, t_start=3)]
        self.assertFalse(validate_cfg(cfg))
        cfg.demos.segments["left"] = [replace(left, t_end=1, milestone_index=1)]
        self.assertFalse(validate_cfg(cfg))  # gap from one to three

    def test_iteration_shift_keeps_roles_and_terminal_update_separate(self):
        cfg = FullQuotientTests().continuation_cfg()
        with patch(
            "synthesis.cfg.boundaries.propose_handoffs",
            return_value=(
                [(3, 5, 8)],
                {"status": "found"},
            ),
        ):
            self.assertTrue(quotient(cfg, boundary_budget=BoundaryBudget()))
        loop = cfg.nodes[cfg.order[0]].region
        rows = loop.body_demos[0]
        self.assertEqual([(r.t_start, r.t_end) for r in rows], [(0, 3), (3, 5), (5, 8)])
        self.assertEqual([r.milestone_index for r in rows], [2, 4, 7])
        for row in rows:
            self.assertTrue(evaluate(loop.postconditions[0], scene_at(row, row.t_end)))
        terminal = loop.exit_demos[0]
        self.assertEqual((terminal.t_start, terminal.t_end), (8, 8))
        self.assertNotEqual(rows[-1].bindings["b"], terminal.bindings["b"])
        self.assertEqual(terminal.bindings["b"], rows[-1].bindings["b_prime"])
        self.assertEqual(cfg.demos.for_node("tail")[0].t_start, 8)
        self.assertEqual(cfg.demos.for_node("tail")[0].t_end, 8)
        self.assertTrue(validate_cfg(cfg))
        self.assertTrue(validate_cfg(loop.body_cfg))

    def test_failed_body_can_try_second_vector_without_losing_loop(self):
        cfg = FullQuotientTests().continuation_cfg()
        with patch(
            "synthesis.cfg.boundaries.propose_handoffs",
            return_value=(
                [(3, 5, 8), (3, 6, 8)],
                {"status": "found"},
            ),
        ):
            self.assertTrue(quotient(cfg, boundary_budget=BoundaryBudget()))
            name = cfg.order[0]
            self.assertTrue(retry_boundary_fold(cfg, name))
            loop = cfg.nodes[name].region
            self.assertEqual([r.t_end for r in loop.body_demos[0]], [3, 6, 8])
            self.assertFalse(retry_boundary_fold(cfg, name))
        self.assertTrue(validate_cfg(cfg))

    def test_retry_snapshot_survives_mutating_id_closure(self):
        from synthesis.api.instructions import Release, ReleaseByName
        from synthesis.cfg.id_first import close_id_candidate

        cfg = FullQuotientTests().continuation_cfg()
        trace = cfg.demos.for_node("tail")[0].trace
        with patch(
            "synthesis.cfg.boundaries.propose_handoffs",
            return_value=(
                [(3, 5, 8), (3, 6, 8)],
                {"status": "found"},
            ),
        ):
            self.assertTrue(quotient(cfg, boundary_budget=BoundaryBudget()))
            name = cfg.order[0]
            source = cfg._boundary_folds[name].source
            close_id_candidate(cfg, allow_unresolved=True)
            cfg.synthesis_approach = "relational"
            self.assertIsInstance(cfg.nodes["tail"].region.physical[0], ReleaseByName)
            self.assertIsInstance(source.nodes["tail"].region.physical[0], Release)
            self.assertNotIn("block_3", source.initial_scope)
            self.assertTrue(retry_boundary_fold(cfg, name))
        self.assertIs(cfg.demos.for_node("tail")[0].trace, trace)
        self.assertIsInstance(cfg.nodes["tail"].region.physical[0], ReleaseByName)
        self.assertIn("block_3", cfg.initial_scope)
        self.assertTrue(validate_cfg(cfg))

    def test_zero_milestone_origin_is_retained(self):
        scene = Scene({0: (0, 0, 0), 1: (0, 0, 0.05)}, {"a": 1, "b": 0})
        trace = DemoTrace((scene,) * 4, num_blocks=2)
        row = DemoSegment(0, 0, 1, trace, scene.bindings, milestone_index=0)
        original = Iterations([(row,)], DemoSegment(0, 1, 1, trace, scene.bindings))
        with patch(
            "synthesis.cfg.boundaries.propose_handoffs",
            return_value=(
                [(3,)],
                {"status": "found"},
            ),
        ):
            changed, _ = refine_iterations([original], [C])
        self.assertEqual(changed[0].bodies[0][0].milestone_index, 0)

    def test_invalid_budgets_rejected(self):
        for kwargs in (
            {"max_shift_steps": 0},
            {"max_proposals": -1},
            {"seconds": float("nan")},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                BoundaryBudget(**kwargs)


class StableGuardTests(unittest.TestCase):
    def test_role_prior_is_explicit_and_renaming_respects_roles(self):
        from synthesis.predicates.guard import loop_guard_synthesis
        from synthesis.predicates.language import Language

        for base, updated, witness in (("base", "a", "new"), ("z", "b", "x")):
            head = Scene(
                {0: (0, 0, 0), 1: (0.2, 0, 0), 2: (0, 0, 0.05)}, {base: 0, updated: 2}
            )
            end = Scene(
                {0: (0, 0, 0), 1: (0, 0, 0.1), 2: (0, 0, 0.05)}, {base: 0, updated: 1}
            )
            learned = loop_guard_synthesis(
                [(head, {witness: 1})],
                [end],
                (witness,),
                (base, updated),
                language=Language(relations=("Scattered",)),
                updated_names={updated},
            )
            self.assertTrue(learned)
            self.assertEqual(learned.term, atom("Scattered", ref(base), ref(witness)))


if __name__ == "__main__":
    unittest.main()
