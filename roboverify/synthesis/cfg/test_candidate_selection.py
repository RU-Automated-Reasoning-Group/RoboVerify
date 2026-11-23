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


def complete_report(*drifts):
    return {
        "accepted": True,
        "complete": True,
        "passed": len(drifts),
        "total": len(drifts),
        "evaluated": len(drifts),
        "demos": [
            {"accepted": True, "max_nonmover_drift_m": drift} for drift in drifts
        ],
    }


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
        self.assertEqual(str(node.region.physical[0]), str(alternatives[1].physical[0]))
        self.assertIs(loop.body[0], node.region)
        self.assertEqual(len(observed), 2)
        self.assertEqual(report["candidate_selection"]["block_option_counts"], [3])
        self.assertEqual(report["candidate_selection"]["selected_ranks"], [1])

    def test_default_first_keeps_early_return_and_does_not_require_frame_schema(self):
        cfg, node, _ = self.nested()
        seen = []

        def validate(candidate, index):
            seen.append(index)
            return {"accepted": True}

        report = select_validated_candidate(cfg, validate)
        self.assertEqual(seen, [0])
        self.assertTrue(report["accepted"])
        self.assertEqual(report["candidate_selection"]["policy"], "first")
        self.assertEqual(report["candidate_selection"]["stop_reason"], "first_accepted")
        self.assertEqual(node.region.physical[0].skip_steps, 0)

    def test_frame_ranks_worst_case_not_mean_and_retains_complete_snapshot(self):
        cfg, node, loop = self.nested()
        singleton = Node("singleton", block(9))
        cfg.nodes[singleton.name] = singleton
        cfg.order.append(singleton.name)
        originals = node.region, singleton.region
        observed = []
        reports = [complete_report(0.0001, 0.004), complete_report(0.002, 0.002)]

        def validate(candidate, index):
            self.assertEqual(node.region.physical[0].skip_steps, index)
            self.assertEqual(singleton.region.physical[0].skip_steps, 9)
            self.assertIs(loop.body[0], node.region)
            # Simulate numeric fitting that changes a singleton and mutates a
            # selected body in place. Later trials must start from fresh values.
            node.region.physical[0].skip_steps += 100
            singleton.region.physical[0].skip_steps = 10 + index
            observed.append((node.region, singleton.region))
            return reports[index] if index < 2 else {"accepted": False}

        report = select_validated_candidate(cfg, validate, policy="frame-margin")
        self.assertTrue(report["accepted"])
        self.assertEqual(node.region.physical[0].skip_steps, 101)
        self.assertEqual(singleton.region.physical[0].skip_steps, 11)
        self.assertIs(loop.body[0], node.region)
        self.assertEqual(originals[0].physical[0].skip_steps, 0)
        self.assertEqual(originals[1].physical[0].skip_steps, 9)
        observed[1][0].physical[0].skip_steps = 999
        reports[1]["demos"][0]["max_nonmover_drift_m"] = 999
        self.assertEqual(node.region.physical[0].skip_steps, 101)
        self.assertEqual(report["demos"][0]["max_nonmover_drift_m"], 0.002)
        selection = report["candidate_selection"]
        self.assertEqual(selection["selected_ranks"], [1])
        self.assertEqual(selection["selected_max_nonmover_drift_m"], 0.002)
        self.assertEqual(selection["attempted"], 3)

    def test_reported_budget_exhaustion_keeps_prior_validated_incumbent(self):
        cfg, node, loop = self.nested()
        singleton = Node("singleton", block(9))
        cfg.nodes[singleton.name] = singleton
        cfg.order.append(singleton.name)

        def validate(candidate, index):
            singleton.region = block(40 + index)
            if index == 0:
                return complete_report(0.003)
            node.region = block(999)
            return {"accepted": False, "selection_budget_exhausted": True}

        report = select_validated_candidate(cfg, validate, policy="frame-margin")
        self.assertTrue(report["accepted"])
        self.assertEqual(node.region.physical[0].skip_steps, 0)
        self.assertEqual(singleton.region.physical[0].skip_steps, 40)
        self.assertIs(loop.body[0], node.region)
        self.assertEqual(report["candidate_selection"]["attempted"], 2)
        self.assertEqual(
            report["candidate_selection"]["stop_reason"], "budget_exhausted"
        )

    def test_late_accepted_report_cannot_replace_incumbent(self):
        cfg, node, _ = self.nested()

        def validate(candidate, index):
            report = complete_report(0.003 if index == 0 else 0.0001)
            if index:
                report["budget_exhausted"] = True
            return report

        report = select_validated_candidate(cfg, validate, policy="frame-margin")
        self.assertTrue(report["accepted"])
        self.assertEqual(node.region.physical[0].skip_steps, 0)
        self.assertFalse(report["candidate_selection"]["trials"][1]["accepted"])

    def test_incomplete_or_missing_frame_evidence_never_promotes(self):
        variants = []
        for value in (None, float("nan"), float("inf"), -0.001, True):
            variants.append(complete_report(value))
        for fields in (
            {"complete": False},
            {"complete": "yes"},
            {"passed": 0},
            {"evaluated": 0},
            {"total": 2},
            {"unevaluated": [{"seed": 2}]},
            {"demos": [{"accepted": False, "max_nonmover_drift_m": 0.001}]},
            {"demos": [{"accepted": True}]},
        ):
            variants.append({**complete_report(0.001), **fields})
        variants.append({"accepted": True})
        for report in variants:
            with self.subTest(report=report):
                cfg, node, _ = self.nested()
                original = node.region
                result = select_validated_candidate(
                    cfg, lambda *args: report, policy="frame-margin", limit=1
                )
                self.assertFalse(result["accepted"])
                self.assertIs(node.region, original)
                self.assertIsNone(result["candidate_selection"]["selected_ranks"])

    def test_complete_rejection_cannot_win_for_having_less_drift(self):
        cfg, node, _ = self.nested()
        reports = [complete_report(0.003), {**complete_report(0.0), "accepted": False}]
        result = select_validated_candidate(
            cfg, lambda candidate, index: reports[index], policy="frame-margin", limit=2
        )
        self.assertTrue(result["accepted"])
        self.assertEqual(node.region.physical[0].skip_steps, 0)

    def test_frame_ties_keep_earlier_and_candidate_limit_is_honored(self):
        cfg, node, _ = self.nested()
        seen = []

        def validate(candidate, index):
            seen.append(index)
            return complete_report(0.001)

        result = select_validated_candidate(
            cfg, validate, policy="frame-margin", limit=2
        )
        self.assertEqual(seen, [0, 1])
        self.assertEqual(node.region.physical[0].skip_steps, 0)
        self.assertEqual(
            result["candidate_selection"]["stop_reason"], "candidate_limit"
        )

    def test_exception_after_incumbent_restores_all_entry_regions(self):
        cfg, node, loop = self.nested()
        singleton = Node("singleton", block(9))
        cfg.nodes[singleton.name] = singleton
        cfg.order.append(singleton.name)
        originals = node.region, singleton.region

        def validate(candidate, index):
            singleton.region.physical[0].skip_steps = 99
            if index:
                raise TimeoutError("external deadline")
            return complete_report(0.001)

        with self.assertRaisesRegex(TimeoutError, "external deadline"):
            select_validated_candidate(cfg, validate, policy="frame-margin")
        self.assertIs(node.region, originals[0])
        self.assertIs(singleton.region, originals[1])
        self.assertEqual(singleton.region.physical[0].skip_steps, 9)
        self.assertIs(loop.body[0], originals[0])

    def test_invalid_policy_is_rejected_before_any_validation(self):
        cfg, _, _ = self.nested()
        with self.assertRaises(ValueError):
            select_validated_candidate(cfg, lambda *args: self.fail(), policy="unknown")

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
