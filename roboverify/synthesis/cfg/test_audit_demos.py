"""Archive auditing checks current identity, exact replay, and final stability."""

import copy
import tempfile
import unittest
from pathlib import Path

from synthesis.cfg.collection import record_execution, validate_trace
from synthesis.cfg.program_source import load_program
from synthesis.cfg.recordings import save_traces
from synthesis.entry.audit_demos import audit_archive
from synthesis.verification_lib.highlevel_verification_lib import HighLevelContext


class AuditDemosTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        definition = load_program(
            "synthesis.examples.stack:build_program", HighLevelContext(), 2
        )
        cls.trace = record_execution(definition, seed=0, task="stack", num_blocks=2)
        if not validate_trace(cls.trace):
            raise AssertionError(cls.trace.metadata)
        from synthesis.cfg.tasks import task_identity

        cls.trace.metadata["task_spec"] = task_identity("stack")

    def test_valid_archive_replays_and_stays_successful_after_settling(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "demo.npz"
            save_traces(path, [self.trace])
            result = audit_archive(path, seeds=[0])
            self.assertTrue(result["valid"], result)
            self.assertTrue(result["trajectories"][0]["settled_physical_success"])
            self.assertLessEqual(result["trajectories"][0]["replay_max_error"], 1e-8)
            with self.assertRaisesRegex(ValueError, "seeds"):
                audit_archive(path, seeds=[0, 1])

    def test_stale_program_and_corrupted_actions_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "demo.npz"
            stale = copy.deepcopy(self.trace)
            stale.metadata["fingerprint"] = "wrong"
            save_traces(path, [stale])
            result = audit_archive(path, seeds=[0])
            self.assertFalse(result["valid"])
            self.assertIn("current factory", result["trajectories"][0]["reason"])
            corrupt = copy.deepcopy(self.trace)
            corrupt.actions[0][-1][:3] += 0.25
            save_traces(path, [corrupt])
            result = audit_archive(path, seeds=[0])
            self.assertFalse(result["valid"])


if __name__ == "__main__":
    unittest.main()
