"""Revision 9 driver, program extraction and report helpers (no simulator)."""

import json
import tempfile
import unittest
from pathlib import Path

from synthesis.cap.experiments import revision9_generalization as r9
from synthesis.cap.generalization import programs
from synthesis.cap.reports import generalization_report as report

SOURCE = '''
def top(base):
    """Highest block on base."""
    return get(lambda t: ON_star(t, base))


def policy():
    # carry every block
    x = top(b0)
    Move(x, x, tbl, [0.0, 0.0, 0.05])


policy()
'''
RENAMED = """
def highest(column):
    return get(lambda z: ON_star(z, column))


def policy():
    y = highest(b0)
    Move(y, y, tbl, [0.0, 0.0, 0.05])


policy()
"""


class BudgetTests(unittest.TestCase):
    def test_budgets_scale_above_the_trained_count_only(self):
        self.assertEqual(r9.budgets("stack", 2)["max_steps"], 10_000)
        self.assertEqual(r9.budgets("stack", 4)["max_sim_steps"], 10_000)
        scaled = r9.budgets("stack", 8)
        self.assertEqual(scaled["max_steps"], 80_000)
        self.assertEqual(scaled["max_sim_steps"], 20_000)
        self.assertEqual(r9.budgets("grid", 12)["max_steps"], 80_000)

    def test_outcome_classification(self):
        self.assertEqual(r9.classify(True, "completed", ""), "success")
        self.assertEqual(
            r9.classify(False, "failed", "MujocoException: contact buffer full"),
            "simulator_failure",
        )
        self.assertEqual(
            r9.classify(False, "budget_exhausted", "", "max_steps"), "budget_exhausted"
        )
        self.assertEqual(
            r9.classify(False, "primitive_failed", "Move did not converge"),
            "policy_failure",
        )


class ProgramTests(unittest.TestCase):
    def test_alpha_hash_ignores_names_comments_and_docstrings(self):
        self.assertEqual(programs.alpha_hash(SOURCE), programs.alpha_hash(RENAMED))

    def test_alpha_hash_keeps_constants_and_api_names(self):
        self.assertNotEqual(
            programs.alpha_hash(SOURCE),
            programs.alpha_hash(SOURCE.replace("0.05", "0.06")),
        )
        self.assertNotEqual(
            programs.alpha_hash(SOURCE),
            programs.alpha_hash(SOURCE.replace("tbl", "b0")),
        )

    def test_trial_programs_reads_the_successful_round_checkpoint(self):
        with tempfile.TemporaryDirectory() as root:
            trial = Path(root, "stack", "trials", "t-001")
            round_dir = trial / "cap" / "r1"
            checkpoint = (
                round_dir / "artifacts" / "trial_001" / "checkpoints" / "cp_0002"
            )
            checkpoint.mkdir(parents=True)
            (checkpoint / "policy.py").write_text(SOURCE)
            (trial / "job.json").write_text(json.dumps({"run_config": {"seed": 7}}))
            recovery = trial / "cap" / "x-recovery-1"
            recovery.mkdir(parents=True)
            failed = {"run_dir": "unused", "checkpoint_id": "cp_0001"}
            succeeded = {"run_dir": str(round_dir), "active_checkpoint_id": "cp_0002"}
            (recovery / "result.json").write_text(
                json.dumps(
                    {
                        "task_spec_success": True,
                        "task_spec_success_round": 1,
                        "attempts": [failed, succeeded],
                    }
                )
            )
            found = programs.trial_programs("pool", [root])
            self.assertEqual(len(found), 1)
            self.assertEqual(
                (found[0].task, found[0].seed, found[0].round), ("stack", 7, 1)
            )
            self.assertEqual(
                programs.trial_programs("pool", [root], exclude={("stack", 7)}), []
            )


class ReportTests(unittest.TestCase):
    def test_failure_kinds(self):
        self.assertEqual(report.failure_kind({"success": True}), "success")
        self.assertEqual(
            report.failure_kind(
                {
                    "status": "primitive_failed",
                    "reason": "PrimitiveFailure: Pick did not converge",
                }
            ),
            "pick_not_converged",
        )
        self.assertEqual(
            report.failure_kind(
                {
                    "status": "incomplete",
                    "reason": "Controller step limit at 0.3: {'converged': False, 'phase': 'descend'}",
                }
            ),
            "pick_not_converged",
        )
        self.assertEqual(
            report.failure_kind(
                {
                    "status": "incomplete",
                    "reason": "Loop 3 still has a guard witness after 10 iterations",
                }
            ),
            "loop_limit",
        )
        self.assertEqual(
            report.failure_kind(
                {
                    "status": "goal_failed",
                    "task_spec_postcondition": True,
                    "task_spec_settled_postcondition": False,
                }
            ),
            "toppled_while_settling",
        )

    def test_largest_generalizing_count_stops_at_the_first_failure(self):
        case_n = {"a": 2, "b": 3, "c": 3, "d": 4}
        policy = {
            "rows": {
                "a": {"success": True},
                "b": {"success": True},
                "c": {"success": False},
                "d": {"success": True},
            }
        }
        self.assertEqual(report.largest_general_count(policy, case_n, list(case_n)), 2)
        policy["rows"]["c"]["success"] = True
        self.assertEqual(report.largest_general_count(policy, case_n, list(case_n)), 4)


if __name__ == "__main__":
    unittest.main()
