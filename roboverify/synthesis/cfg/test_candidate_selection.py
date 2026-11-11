import unittest

from synthesis.api.instructions import Pick, Skip
from synthesis.cfg.candidate_selection import select_validated_candidate
from synthesis.cfg.demos import DemoAssignment
from synthesis.cfg.graph import Node, RelationalCFG
from synthesis.cfg.id_first import close_id_candidate
from synthesis.cfg.region import BlockRegion, LoopRegion
from synthesis.cfg.test_id_first import ground_cfg
from synthesis.predicates.term import boolean


def block(value):
    return BlockRegion(None, (Skip(value),))


def graph(*nodes):
    return RelationalCFG(
        {n.name: n for n in nodes}, [], [n.name for n in nodes], DemoAssignment()
    )


class CandidateSelectionTests(unittest.TestCase):
    def nested(self):
        node = Node("body", block(0))
        node._contract_alternatives = (block(0), block(1), block(2))
        body = graph(node)
        loop = LoopRegion(boolean(True), (), (node.region,), body_cfg=body)
        return graph(Node("loop", loop)), node, loop

    def test_later_accepted_alternative_commits_nested_body(self):
        cfg, node, loop = self.nested()
        alternatives = node._contract_alternatives
        observed = []

        def validate(candidate, index):
            self.assertIs(loop.body[0], node.region)
            observed.append(node.region)
            return {"accepted": index == 1, "program_fingerprint": str(index)}

        report = select_validated_candidate(cfg, validate)
        self.assertTrue(report["accepted"])
        self.assertIs(node.region, alternatives[1])
        self.assertIs(loop.body[0], node.region)
        self.assertEqual(len(observed), 2)
        self.assertEqual(report["candidate_selection"]["block_option_counts"], [3])
        self.assertEqual(report["candidate_selection"]["selected_ranks"], [1])

    def test_rejection_restores_original_and_reports_its_fingerprint(self):
        cfg, node, loop = self.nested()
        original = node.region
        report = select_validated_candidate(
            cfg,
            lambda candidate, index: {
                "accepted": False,
                "program_fingerprint": str(index),
            },
        )
        self.assertIs(node.region, original)
        self.assertIs(loop.body[0], original)
        self.assertEqual(report["program_fingerprint"], "0")
        self.assertEqual(report["candidate_selection"]["attempted"], 3)
        self.assertIsNone(report["candidate_selection"]["selected_ranks"])

    def test_exception_restores_original_after_later_trial(self):
        cfg, node, loop = self.nested()
        original = node.region

        def validate(candidate, index):
            if index:
                raise TimeoutError("outer deadline")
            return {"accepted": False}

        with self.assertRaisesRegex(TimeoutError, "outer deadline"):
            select_validated_candidate(cfg, validate)
        self.assertIs(node.region, original)
        self.assertIs(loop.body[0], original)

    def test_bounded_combinations_visit_all_single_changes_before_joint_changes(self):
        first, second = Node("a", block(0)), Node("b", block(3))
        first._contract_alternatives = (block(1), block(2))
        second._contract_alternatives = (block(4),)
        originals = first.region, second.region
        report = select_validated_candidate(
            graph(first, second), lambda candidate, index: {"accepted": False}, limit=4
        )
        ranks = [r["ranks"] for r in report["candidate_selection"]["trials"]]
        self.assertEqual(ranks[:3], [[0, 0], [0, 1], [1, 0]])
        self.assertEqual(len(ranks), 4)
        self.assertEqual((first.region, second.region), originals)

    def test_shared_budget_exhaustion_stops_selection(self):
        cfg, node, loop = self.nested()
        report = select_validated_candidate(
            cfg,
            lambda candidate, index: {
                "accepted": False,
                "selection_budget_exhausted": True,
            },
        )
        self.assertEqual(report["candidate_selection"]["attempted"], 1)

    def test_no_alternatives_still_requires_one_complete_validation(self):
        calls = []
        report = select_validated_candidate(
            graph(Node("only", block(0))),
            lambda candidate, index: calls.append(index) or {"accepted": False},
        )
        self.assertEqual(calls, [0])
        self.assertFalse(report["accepted"])

    def test_alternative_cannot_change_the_cfg_binding_interface(self):
        node = Node("body", block(0))
        option = block(1)
        option.bindings = frozenset({"new_export"})
        node._contract_alternatives = (option,)
        report = select_validated_candidate(
            graph(node), lambda candidate, index: {"accepted": False}
        )
        self.assertEqual(report["candidate_selection"]["attempted"], 1)

    def test_id_naming_covers_ids_only_used_by_an_alternative(self):
        cfg = ground_cfg()
        for node in cfg.nodes.values():
            node.region = block(0)
        target = cfg.nodes[cfg.order[0]]
        target._contract_alternatives = (BlockRegion(None, (Pick(3),)),)
        close_id_candidate(cfg)
        instruction = target._contract_alternatives[0].physical[0]
        operand = instruction.get_operand()[0]
        self.assertEqual(operand["type"], "BoxName")
        self.assertEqual(cfg._fixed_id_bindings[operand["val"]], 3)
        self.assertIn(operand["val"], target.available_scope)


if __name__ == "__main__":
    unittest.main()
