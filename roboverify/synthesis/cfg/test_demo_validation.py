import unittest

from synthesis.cfg.demo_validation import validate_demonstrations
from synthesis.cfg.demos import DemoSegment, DemoTrace
from synthesis.predicates.scene import Scene
from synthesis.predicates.term import atom, boolean, ref


class DemoValidationTests(unittest.TestCase):
    def setUp(self):
        self.low = Scene({0: (0, 0, 0), 1: (0.2, 0, 0)}, {"a": 1, "b": 0})
        self.on = Scene({0: (0, 0, 0), 1: (0, 0, 0.05)}, {"a": 1, "b": 0})
        self.post = atom("ON", ref("a"), ref("b"))

    def segment(self, *states):
        return DemoSegment(7, 0, len(states) - 1, DemoTrace(states))

    def test_transient_success_does_not_validate_demo(self):
        result = validate_demonstrations(
            [self.segment(self.low, self.on, self.low)], boolean(True), self.post
        )
        self.assertFalse(result)
        self.assertEqual((result.post_reached, result.post_at_end), (1, 0))
        self.assertEqual(result.issues[0]["demo"], 7)
        self.assertTrue(result.issues[0]["post_reached_transiently"])

    def test_initial_condition_and_empty_input_are_checked(self):
        self.assertFalse(validate_demonstrations([], boolean(True), self.post))
        result = validate_demonstrations(
            [self.segment(self.low, self.on)], self.post, self.post
        )
        self.assertFalse(result)
        self.assertEqual(result.issues[0]["reason"], "Initial task condition is false")

    def test_task_correct_demo_is_accepted_without_oracle(self):
        result = validate_demonstrations(
            [self.segment(self.low, self.on)], boolean(True), self.post
        )
        self.assertTrue(result)
        self.assertEqual(result.post_at_end, 1)

    def test_settled_criterion_keeps_the_last_state_post_as_a_diagnostic(self):
        result = validate_demonstrations(
            [self.segment(self.on, self.low)],
            boolean(True),
            self.post,
            require_final_post=False,
        )
        self.assertTrue(result)
        self.assertEqual((result.post_at_end, result.post_reached), (0, 1))

    def test_cli_requires_stored_settled_results_when_requested(self):
        from types import SimpleNamespace
        from unittest.mock import Mock, patch

        from synthesis.cfg.task_success import MissingSettledValidation
        from synthesis.cfg.tasks import task_identity
        from synthesis.entry.synthesize_cfg import run

        def trace(settled=None):
            metadata = {
                "task_spec": task_identity("stack"),
                "initial_bindings": {"b0": 0},
            }
            if settled is not None:
                metadata["validation"] = {"settled": settled}
            # The last-state post is false: only the settled result decides.
            return DemoTrace(
                (self.low, self.on, self.low),
                task="stack",
                num_blocks=2,
                metadata=metadata,
            )

        args = SimpleNamespace(
            demos="missing.npz", task="stack", num_blocks=2, post_settling_steps=50
        )
        with patch(
            "synthesis.entry.synthesize_cfg.load_traces", return_value=[trace()]
        ), patch("synthesis.entry.synthesize_cfg.verified_synthesis") as search:
            with self.assertRaisesRegex(MissingSettledValidation, "revalidate_settled"):
                run(args, Mock())
        search.assert_not_called()
        failed = dict(post_settling_steps=50, success=False, issues=["Settled"])
        logger = Mock()
        with patch(
            "synthesis.entry.synthesize_cfg.load_traces", return_value=[trace(failed)]
        ), patch("synthesis.entry.synthesize_cfg.verified_synthesis") as search:
            self.assertEqual(run(args, logger), 2)
        search.assert_not_called()
        checks = logger.finish.call_args.kwargs["demonstration_checks"]
        self.assertEqual(checks["criterion"], "task-spec-v2")
        self.assertEqual(checks["issues"][0]["reason"], "Settled task check failed")

    def test_cli_rejects_bad_demo_before_search(self):
        from types import SimpleNamespace
        from unittest.mock import Mock, patch

        from synthesis.cfg.tasks import task_identity
        from synthesis.entry.synthesize_cfg import run

        trace = DemoTrace(
            (self.low, self.on, self.low),
            task="stack",
            num_blocks=2,
            metadata={
                "task_spec": task_identity("stack"),
                "initial_bindings": {"b0": 0},
            },
        )
        args = SimpleNamespace(demos="unused.npz", task="stack", num_blocks=2)
        logger = Mock()
        with patch(
            "synthesis.entry.synthesize_cfg.load_traces", return_value=[trace]
        ), patch("synthesis.entry.synthesize_cfg.verified_synthesis") as search:
            self.assertEqual(run(args, logger), 2)
        search.assert_not_called()
        self.assertEqual(logger.finish.call_args.args[0], "invalid_demonstrations")
