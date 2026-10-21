"""Trial, rollout, and transition denominators must remain distinct."""

import unittest

from .revision42_report import aggregate


def attempt(status, *, primitive=None, rolled=True, success=False):
    return {
        "status": status,
        "primitive": primitive,
        "nonconvergence": primitive is not None,
        "rolled_out": rolled,
        "success": success,
    }


def trial(attempts, variant="full", stop_reason="max_recovery_rounds"):
    return {
        "stop_reason": stop_reason,
        "method": "cap_method_1",
        "variant": variant,
        "attempts": attempts,
        "success": attempts[-1]["success"],
        "calls": len(attempts),
        "total_tokens": 123,
        "issues": [],
    }


class ReportDenominatorTests(unittest.TestCase):
    def test_terminal_failure_is_not_a_transition_and_other_failure_is_not_success(
        self,
    ):
        row = trial(
            [
                attempt("primitive_failed", primitive="Move"),
                attempt("goal_failed"),
                attempt("primitive_failed", primitive="Release"),
            ]
        )
        g = aggregate([row])[("cap_method_1", "full")]
        self.assertEqual(g["nonconv_transitions"], 1)
        self.assertEqual(g["transition_other_failure"], 1)
        self.assertEqual(g["repeat_nonconv"], 0)
        self.assertEqual(g["terminal_nonconv_without_next"], 1)
        self.assertEqual(g["nonconv_rollouts"], 2)
        self.assertEqual(g["any_nonconv"], 1)
        self.assertEqual(g["success"], 0)

    def test_primitive_changes_and_preparation_stops_keep_correct_denominators(self):
        first = trial(
            [
                attempt("primitive_failed", primitive="Pick"),
                attempt("primitive_failed", primitive="Move"),
                attempt("generation_failed", rolled=False),
            ]
        )
        second = trial(
            [
                attempt("primitive_failed", primitive="Move"),
                attempt("completed", success=True),
            ],
            "no_target",
        )
        g = aggregate([first, second])
        full = g[("cap_method_1", "full")]
        self.assertEqual(full["nonconv_transitions"], 2)
        self.assertEqual(full["repeat_nonconv"], 1)
        self.assertEqual(full["same_primitive"], 0)
        self.assertEqual(full["transition_preparation"], 1)
        self.assertEqual(full["rollouts"], 2)
        self.assertEqual(full["final_nonconv"], 0)
        self.assertEqual(g[("cap_method_1", "no_target")]["success"], 1)

    def test_terminal_paths_distinguish_rollout_and_preparation_stops(self):
        rows = [
            trial([attempt("budget_exhausted")]),
            trial(
                [attempt("budget_exhausted", rolled=False)],
                stop_reason="preparation_failure_no_rollout",
            ),
            trial([attempt("goal_failed")]),
            trial(
                [attempt("primitive_failed", primitive="Pick")],
                stop_reason="required_feedback_exceeds_token_limit",
            ),
        ]
        g = aggregate(rows)[("cap_method_1", "full")]
        self.assertEqual(g["final_budget_stop"], 1)
        self.assertEqual(g["preparation_stop"], 1)
        self.assertEqual(g["final_goal_failure"], 1)
        self.assertEqual(g["feedback_size_stop"], 1)
        self.assertEqual(g["final_nonconv"], 1)


if __name__ == "__main__":
    unittest.main()
