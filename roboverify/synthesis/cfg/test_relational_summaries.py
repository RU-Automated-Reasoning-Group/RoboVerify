"""Real CFG summary inference, symbolic WPs and geometric refinement checks."""

import unittest
from dataclasses import replace
from unittest.mock import patch

import z3

from synthesis.api.instructions import Assign, MoveByName, Put, ReleaseByName, Skip
from synthesis.api.relational_summary import RelationalSummary
from synthesis.cfg.graph import RelationalCFG
from synthesis.cfg.region import BlockRegion
from synthesis.cfg.test_verification import placement_cfg
from synthesis.cfg.verification import (
    propose_summaries,
    symbolic_problem,
    verify_cfg_motion,
)
from synthesis.predicates.term import atom, conjunction, disjunction, forall, ref
from synthesis.verification_lib.highlevel_verification_lib import HighLevelContext
from synthesis.verification_lib.motion_verification import (
    MotionContract,
    prepare_alignment,
)
from synthesis.verification_lib.primitive_motion import PrimitiveMotionProblem
from synthesis.verification_lib.root_selection import RootContext, RootSelection
from synthesis.verification_lib.symbolic_verify import symbolic_verify
from synthesis.verification_lib.test_primitive_motion import primitives


def with_post(cfg, post):
    cfg.edges[-1] = replace(cfg.edges[-1], label=post)
    cfg.postcondition = post
    return cfg


def three_block_cfg(post, *, final_x=0.0, assignment=False):
    a, b, c, x = map(ref, ("a", "b", "c", "x"))
    pre = conjunction(
        forall(["x"], disjunction(*(atom("eq", x, n) for n in (a, b, c)))),
        atom("Scattered", a, b),
        atom("Scattered", a, c),
        atom("Scattered", b, c),
    )
    cfg = RelationalCFG.initial([], pre, post, ("a", "b", "c"))
    body = primitives()
    body[-2] = MoveByName("b", "b", "b", target_offset=[final_x, 0, 0.05])
    if assignment:
        body.append(Assign("c", "a"))
    cfg.nodes["v0"].region = BlockRegion(None, tuple(body))
    return cfg


class RelationalSummaryIntegrationTests(unittest.TestCase):
    def verify(self, cfg, *, positions=None):
        context = HighLevelContext()
        propose_summaries(cfg, context)
        result = verify_cfg_motion(
            cfg,
            context,
            initial_positions=positions
            or {
                "a": [0.3, 0, -0.1],
                "b": [0, 0, 0],
                "b0": [0, 0, 0],
            },
            initial_arm=[0.3, 0, 0.2],
        )
        return context, result

    def status(self, result, suffix):
        return [c.status for c in result.checks if c.obligation.endswith("/" + suffix)]

    def finite_symbolic(self, cfg, context, size):
        result = symbolic_verify(
            lambda count: symbolic_problem(cfg, context, count),
            min_blocks=size,
            max_blocks=size,
            prove_unbounded=False,
            timeout_ms=10000,
        )
        self.assertTrue(result, str(result))

    def test_higher_inference_reaches_both_verifiers_including_split_transfer(self):
        for split in (False, True):
            with self.subTest(split=split):
                cfg = with_post(
                    placement_cfg(split), atom("Higher", ref("a"), ref("b"))
                )
                context, result = self.verify(cfg)
                self.assertTrue(result, str(result))
                last = cfg.nodes[cfg.order[-1]].region.symbolic[0]
                self.assertIsInstance(last, RelationalSummary)
                if split:
                    self.assertIsInstance(cfg.nodes["v0"].region.symbolic[0], Skip)
                self.finite_symbolic(cfg, context, 2)

    def test_scattered_inference_reaches_both_verifiers(self):
        cfg = three_block_cfg(atom("Scattered", ref("a"), ref("c")))
        context, result = self.verify(
            cfg,
            positions={
                "a": [0.3, 0, -0.1],
                "b": [0, 0, 0],
                "c": [0.6, 0, 0],
            },
        )
        self.assertTrue(result, str(result))
        self.finite_symbolic(cfg, context, 3)

    def test_reverse_higher_order_is_preserved_and_checked(self):
        cfg = three_block_cfg(atom("Higher", ref("c"), ref("a")))
        _, result = self.verify(
            cfg,
            positions={
                "a": [0.3, 0, -0.1],
                "b": [0, 0, 0],
                "c": [0.6, 0, 0.1],
            },
        )
        self.assertTrue(result, str(result))
        self.assertEqual(cfg.nodes["v0"].region.symbolic[0].moved, "a")
        bad = with_post(placement_cfg(), atom("Higher", ref("b"), ref("a")))
        _, result = self.verify(bad)
        self.assertFalse(result)
        self.assertEqual(self.status(result, "summary_endpoint"), ["refuted"])

    def test_scattered_endpoint_is_not_inferred_from_a_put(self):
        cfg = with_post(placement_cfg(), atom("Scattered", ref("a"), ref("b")))
        _, result = self.verify(cfg)
        self.assertFalse(result)
        self.assertEqual(self.status(result, "summary_endpoint"), ["refuted"])

    def test_assignment_after_release_cannot_change_summary_operands(self):
        cfg = three_block_cfg(atom("Higher", ref("c"), ref("a")), assignment=True)
        _, result = self.verify(
            cfg,
            positions={
                "a": [0.3, 0, -0.1],
                "b": [0, 0, 0],
                "c": [0.6, 0, 0],
            },
        )
        # c <- a makes Higher(c,a) reflexively true afterward, but the action
        # summary refers to the original c and must still fail refinement.
        self.assertFalse(result)
        self.assertEqual(self.status(result, "summary_endpoint"), ["refuted"])

    def test_endpoint_does_not_assume_successor_domain(self):
        cfg = three_block_cfg(atom("Higher", ref("a"), ref("b")), final_x=0.024)
        _, result = self.verify(
            cfg,
            positions={
                "a": [0.3, 0, -0.1],
                "b": [0, 0, 0],
                "c": [0.11, 0, 0],
            },
        )
        self.assertEqual(self.status(result, "summary_endpoint"), ["valid"])
        self.assertEqual(self.status(result, "summary_domain_scattered3"), ["refuted"])
        self.assertFalse(result)

    def test_nonmover_disturbance_is_rejected_even_when_endpoint_holds(self):
        cfg = three_block_cfg(atom("Higher", ref("a"), ref("b")))
        execute = PrimitiveMotionProblem.execute

        def disturb(problem, body):
            execute(problem, body)
            if any(isinstance(i, ReleaseByName) for i in body):
                point = problem.current["c"]
                problem.transport("c", (point[0] + 0.1, point[1], point[2]))

        with patch.object(PrimitiveMotionProblem, "execute", disturb):
            _, result = self.verify(
                cfg,
                positions={
                    "a": [0.3, 0, -0.1],
                    "b": [0, 0, 0],
                    "c": [0.6, 0, 0],
                },
            )
        self.assertEqual(self.status(result, "summary_endpoint"), ["valid"])
        self.assertEqual(self.status(result, "summary_frame"), ["refuted"])
        self.assertFalse(result)

    def test_later_put_does_not_reassume_input_alignment(self):
        from types import SimpleNamespace

        context = HighLevelContext()
        history = [RelationalSummary("Higher", "a", "b"), Assign("c", "a")]
        root = RootContext(context, z3.BoolVal(True), history=history)
        problem = SimpleNamespace(contract=MotionContract("a", "b"), checks=[])
        with patch.object(
            root, "initial_roots", side_effect=AssertionError("reassumed input")
        ), patch.object(root, "select", return_value=RootSelection("unproved")):
            prepare_alignment(problem, root, ["a", "b"], continuation=(Put("a", "b"),))
        self.assertEqual(problem.checks[0].status, "unsupported")

    def test_global_edge_preserves_put_fallback_and_unrelated_atom_fails_closed(self):
        cfg = with_post(
            placement_cfg(),
            conjunction(
                atom("ON_star", ref("a"), ref("b")), atom("Higher", ref("a"), ref("b"))
            ),
        )
        propose_summaries(cfg, HighLevelContext())
        self.assertIsInstance(cfg.nodes["v0"].region.symbolic[0], Put)
        cfg = with_post(placement_cfg(), atom("Higher", ref("b"), ref("b0")))
        with self.assertRaisesRegex(ValueError, "moved object"):
            propose_summaries(cfg, HighLevelContext())


if __name__ == "__main__":
    unittest.main()
