"""Wall limits remain effective when a native callback consumes an alarm."""

import contextlib
import io
import json
import signal
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from synthesis.api.program import Program
from synthesis.cfg.collection import check_execution_deadline, execution_deadline
from synthesis.cfg.straightline import segment_rollout


class StickyDeadlineTests(unittest.TestCase):
    def test_swallowed_constructor_alarm_is_rechecked_before_using_environment(self):
        now = [0.0]
        env = Mock()

        def factory(trace):
            now[0] = 11.0
            # MuJoCo's Cython model wrapper can ignore an exception raised from
            # its NumPy callback. Emulate consumption of that one-shot alarm.
            alarm = signal.getsignal(signal.SIGALRM)
            signal.setitimer(signal.ITIMER_REAL, 0)
            try:
                alarm(signal.SIGALRM, None)
            except TimeoutError:
                pass
            return env

        with patch(
            "synthesis.cfg.collection.time.monotonic", side_effect=lambda: now[0]
        ), patch("synthesis.cfg.straightline.reset_segment") as reset:
            with self.assertRaisesRegex(TimeoutError, "outer budget"):
                with execution_deadline(10, message="outer budget"):
                    segment_rollout(
                        Program(0, []),
                        SimpleNamespace(trace=SimpleNamespace(seed=0)),
                        factory,
                    )
        reset.assert_not_called()
        env.close.assert_called_once()
        check_execution_deadline()  # the expired context is cleaned up

    def test_nested_shorter_limit_does_not_reset_outer_absolute_budget(self):
        now = [0.0]
        with patch(
            "synthesis.cfg.collection.time.monotonic", side_effect=lambda: now[0]
        ):
            with self.assertRaisesRegex(TimeoutError, "outer"):
                with execution_deadline(10, message="outer"):
                    with self.assertRaisesRegex(TimeoutError, "inner"):
                        with execution_deadline(1, message="inner"):
                            now[0] = 2.0
                            check_execution_deadline()
                    now[0] = 9.0
                    check_execution_deadline()
                    now[0] = 11.0
                    check_execution_deadline()

    def test_longer_inner_limit_cannot_extend_outer_after_alarm_was_consumed(self):
        now = [0.0]
        with patch(
            "synthesis.cfg.collection.time.monotonic", side_effect=lambda: now[0]
        ):
            with self.assertRaisesRegex(TimeoutError, "outer"):
                with execution_deadline(10, message="outer"):
                    with execution_deadline(20, message="inner"):
                        now[0] = 11.0
                        signal.setitimer(signal.ITIMER_REAL, 0)
                        check_execution_deadline()

    def test_cli_cannot_finish_success_after_a_consumed_alarm(self):
        from synthesis.entry.synthesize_cfg import main

        now = [0.0]

        def late_run(args, logger):
            now[0] = 11.0
            signal.setitimer(signal.ITIMER_REAL, 0)
            return 0

        with tempfile.TemporaryDirectory() as directory, patch(
            "synthesis.cfg.collection.time.monotonic", side_effect=lambda: now[0]
        ), patch(
            "synthesis.entry.synthesize_cfg.run", side_effect=late_run
        ), contextlib.redirect_stdout(
            io.StringIO()
        ):
            code = main(
                [
                    "--mode",
                    "synthesize",
                    "--demos",
                    "unused.npz",
                    "--wall-timeout-seconds",
                    "10",
                    "--output-dir",
                    directory,
                ]
            )
            self.assertEqual(code, 2)
            result = json.loads(
                next(Path(directory).glob("cfg/*/result.json")).read_text()
            )
            self.assertEqual(result["status"], "budget_exhausted")
            self.assertIn("End-to-end", result["reason"])


if __name__ == "__main__":
    unittest.main()
