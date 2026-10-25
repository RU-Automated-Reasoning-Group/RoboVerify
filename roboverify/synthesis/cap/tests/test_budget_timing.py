"""Resource caps and exclusive timing semantics, independent of server speed."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from synthesis.cfg.collection import STACK_SETTLING_STEPS

from ..budget import Budget
from ..config import ModelConfig, RunConfig
from ..contracts import BudgetExceeded
from ..llm import LoggedModel, Response, ScriptedModel, TransportError
from ..logger import CapLogger
from ..runner import run
from ..timing import PhaseTimings
from .test_adapter import RobotFixture


class BudgetTimingTests(unittest.TestCase):
    def test_nested_phase_timings_are_exclusive_even_on_failure(self):
        now = [0.0]
        timer = PhaseTimings(clock=lambda: now[0])
        now[0] = 1
        with timer.measure("execution"):
            now[0] = 3
            with self.assertRaises(ValueError):
                with timer.measure("simulation"):
                    now[0] = 7
                    raise ValueError("simulated failure")
            now[0] = 10
        now[0] = 12
        result = timer.snapshot()
        self.assertEqual(
            result["phase_seconds"], {"execution": 5, "simulation": 4, "overhead": 3}
        )
        self.assertEqual(
            sum(result["phase_seconds"].values()), result["active_wall_seconds"]
        )
        self.assertEqual(timer.snapshot(), result)

    def test_timer_forwards_logging_field_names_without_collision(self):
        timer = PhaseTimings()
        result = timer.call(
            "execution", lambda **fields: fields, function="policy", phase="body"
        )
        self.assertEqual(result, {"function": "policy", "phase": "body"})

    def test_resource_caps_do_not_depend_on_wall_deadline(self):
        self.assertEqual(Budget().timeout, 600)
        self.assertEqual(RunConfig().timeout, 600)
        cases = [
            (
                Budget(max_model_calls=1),
                lambda b: (b.model_call(), b.model_call()),
                "model_calls",
            ),
            (Budget(max_steps=1), lambda b: (b.tick(), b.tick()), "execution_ticks"),
            (Budget(max_tokens=5), lambda b: b.account_tokens(6), "tokens"),
            (Budget(max_depth=1), lambda b: b.depth(2), "helper_depth"),
            (
                Budget(max_sim_steps=1),
                lambda b: (b.simulation_step(), b.simulation_step()),
                "simulation_steps",
            ),
        ]
        for budget, action, expected in cases:
            with self.subTest(expected=expected), self.assertRaises(BudgetExceeded):
                action(budget)
            self.assertEqual(budget.exhausted_by, expected)
            self.assertGreater(budget.remaining(), 0)
        with self.assertRaises(ValueError):
            Budget(max_sim_steps=0)

    def test_first_exhaustion_reason_survives_cleanup(self):
        b = Budget(max_model_calls=1)
        b.model_call()
        with self.assertRaises(BudgetExceeded):
            b.model_call()
        b.started -= 1000
        with self.assertRaises(BudgetExceeded):
            b.remaining()
        self.assertEqual(b.exhausted_by, "model_calls")

    def test_failed_requests_and_retry_backoff_are_timed(self):
        now = [0.0]
        timer = PhaseTimings(clock=lambda: now[0])
        budget = Budget(timings=timer)

        class Backend:
            count = 0

            def payload(self, prompt):
                return {"prompt": prompt}

            def generate(self, prompt, timeout):
                self.count += 1
                now[0] += self.count
                if self.count == 1:
                    raise TransportError("retry fixture", retryable=True)
                return Response(
                    "pass",
                    {"total_tokens": 11},
                    "stop",
                    {"usage": {"total_tokens": 11}},
                )

        with tempfile.TemporaryDirectory() as directory:
            logger = CapLogger(directory, console=False, timings=timer)
            logger.context["helper"] = "helper_fixture"
            model = LoggedModel(Backend(), logger, budget, retries=1)
            with patch(
                "synthesis.cap.llm.time.sleep",
                side_effect=lambda delay: now.__setitem__(0, now[0] + delay),
            ):
                self.assertEqual(
                    model.complete("Generate", purpose="google_generation"), "pass\n"
                )
            phases = timer.snapshot()["phase_seconds"]
            self.assertEqual(phases["model_helper"], 3)
            self.assertEqual(phases["transport_backoff"], 1)
            self.assertEqual(budget.calls, 2)
            self.assertEqual(budget.tokens, 11)
            logger.close()

    def test_runner_queue_time_is_separate_and_simulation_cap_is_enforced(self):
        for method in ("cap_google_baseline", "cap_method_1"):
            with (
                self.subTest(method=method),
                tempfile.TemporaryDirectory() as directory,
            ):
                env = RobotFixture(1, success=True)
                with contextlib.redirect_stdout(io.StringIO()):
                    result = run(
                        RunConfig(
                            method=method,
                            num_blocks=1,
                            max_sim_steps=2,
                            save_video=False,
                            output_root=directory,
                        ),
                        ModelConfig(),
                        backend=ScriptedModel(["Pick(b0)"]),
                        environment_factory=lambda config, snapshot: env,
                        queue_wait_seconds=123.0,
                        batch_context={"max_workers": 2},
                        capture_stdout=False,
                    )
                self.assertEqual(result["status"], "budget_exhausted")
                self.assertEqual(result["budget_exhausted_by"], "simulation_steps")
                self.assertFalse(result["success"])
                # The settled diagnostic steps the environment after the rollout,
                # outside the policy budget.
                self.assertEqual(len(env.actions), 2 + STACK_SETTLING_STEPS)
                self.assertEqual(result["simulation_steps"], 2)
                self.assertEqual(result["queue_wait_seconds"], 123)
                self.assertAlmostEqual(
                    sum(result["phase_seconds"].values()),
                    result["active_wall_seconds"],
                    places=7,
                )
                for phase in (
                    "setup",
                    "model_policy",
                    "execution",
                    "simulation",
                    "recording",
                    "logging_io",
                ):
                    self.assertIn(phase, result["phase_seconds"])
                timing = Path(result["run_dir"]) / "artifacts/trial_001/timings.json"
                self.assertEqual(
                    json.loads(timing.read_text())["queue_wait_seconds"], 123
                )

    def test_unstack_uses_the_configured_timeout(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            result = run(
                RunConfig(task="unstack", timeout=600, output_root=directory),
                ModelConfig(),
                backend=ScriptedModel(["pass"]),
                generate_only=True,
                capture_stdout=False,
            )
            from synthesis.experiment.report import RunSummary

            self.assertEqual(
                RunSummary(Path(result["run_dir"])).config["config"]["timeout"], 600
            )

    def test_cli_defaults_and_simulation_limit(self):
        from synthesis.entry.run_cap import main

        with (
            tempfile.TemporaryDirectory() as directory,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            responses = Path(directory) / "responses.json"
            responses.write_text('["pass"]')
            with patch(
                "synthesis.entry.run_cap.run",
                return_value={
                    "status": "generated",
                    "success": None,
                    "run_dir": directory,
                },
            ) as mocked:
                self.assertEqual(
                    main(
                        [
                            "--method",
                            "cap_method_1",
                            "--generate-only",
                            "--responses-file",
                            str(responses),
                            "--max-sim-steps",
                            "123",
                        ]
                    ),
                    0,
                )
                config = mocked.call_args.args[0]
                self.assertEqual(config.timeout, 600)
                self.assertEqual(config.max_sim_steps, 123)


if __name__ == "__main__":
    unittest.main()
