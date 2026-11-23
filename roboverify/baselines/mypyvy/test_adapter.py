"""Fast regressions for the presentation adapter and fail-closed runner."""

import argparse
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from baselines.mypyvy.run import bounded_process, classify, run
from baselines.mypyvy.worker import (
    MODEL_PREFIX,
    ModelLines,
    certificate_text,
    scoped_check,
)


class ModelPresentationTests(unittest.TestCase):
    def test_multiple_sorts_and_multiline_nested_terms(self):
        lines = [
            "(\n",
            "; cardinality of block is 2\n",
            "; rep: (as @block_0 block)\n",
            "; rep: (as @block_1 block)\n",
            "; cardinality of node is 1\n",
            "; rep: (as @node_0 node)\n",
            "(define-fun f ((x block)) Bool\n",
            "(\n",
            "ite (= x (as @block_0 block)) true false))\n",
            ")\n",
        ]
        parser = ModelLines()
        output = [parser.feed(line) for line in lines]
        self.assertEqual(output[0], "(model\n")
        self.assertEqual(output[7], "(\n")
        self.assertIn("(declare-sort node 0)", "".join(output))
        self.assertIn(MODEL_PREFIX + "block_0", output[8])
        self.assertNotIn("@", "".join(output))
        self.assertEqual(parser.depth, 0)
        self.assertEqual(parser.feed("(\n"), "(model\n")

    def test_unsupported_root_and_quoted_tokens_fail(self):
        for value in ["", "unknown\n", "(error bad)\n", ")\n"]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                ModelLines().feed(value)
        parser = ModelLines()
        parser.feed("(\n")
        with self.assertRaises(ValueError):
            parser.feed("(define-fun |quoted| () Bool true)\n")

    def test_immutable_certificate_clauses_remain_obligations(self):
        result = certificate_text("mutable relation done()\n", ["frozen", "!frozen"])
        self.assertIn("invariant [inferred_0] (done | !done) & (frozen)", result)
        self.assertIn("invariant [inferred_1] (done | !done) & (!frozen)", result)
        self.assertNotIn("axiom", result)


class AssumptionTests(unittest.TestCase):
    def test_temporary_scope_is_restored_on_success_and_failure(self):
        class Stack:
            def __init__(self):
                self.values = ["permanent"]

            def push(self):
                self.saved = self.values.copy()

            def add(self, *items):
                self.values.extend(items)

            def pop(self):
                self.values = self.saved

        instance = SimpleNamespace(use_cvc4=True, z3solver=Stack())

        def check(obj, assumptions):
            self.assertEqual(obj.z3solver.values, ["permanent", "temporary"])
            self.assertEqual(assumptions, [])
            return "sat"

        self.assertEqual(scoped_check(check, instance, ["temporary"]), "sat")
        self.assertEqual(instance.z3solver.values, ["permanent"])

        def failure(*args):
            raise RuntimeError("solver crashed")

        with self.assertRaises(RuntimeError):
            scoped_check(failure, instance, ["temporary"])
        self.assertEqual(instance.z3solver.values, ["permanent"])


class OutcomeTests(unittest.TestCase):
    def test_wall_deadline_kills_worker_and_solver_descendant(self):
        with tempfile.TemporaryDirectory(prefix="rv-budget-test-") as tmp:
            marker = Path(tmp) / "survived"
            child = f"import time; from pathlib import Path; time.sleep(1); Path({str(marker)!r}).touch()"
            parent = f"import subprocess,sys,time; subprocess.Popen([sys.executable, '-c', {child!r}]); time.sleep(10)"
            observed = bounded_process(
                [sys.executable, "-c", parent],
                Path(tmp) / "run.log",
                time.monotonic() + 0.3,
            )
            self.assertTrue(observed["timed_out"])
            self.assertLess(observed["elapsed_seconds"], 2)
            time.sleep(1)
            self.assertFalse(marker.exists(), "solver descendant survived the deadline")

    def test_no_success_from_exit_or_mixed_failure_markers(self):
        self.assertEqual(classify("updr", 0, ""), "inconclusive")
        for failure in [
            "error: bad",
            "Traceback (most recent call last)",
            "not inductive",
            "failed",
        ]:
            self.assertEqual(classify("verify", 0, "all ok!\n" + failure), "error")
        self.assertEqual(classify("verify", 0, "unknown\nall ok!"), "unknown")
        self.assertEqual(classify("verify", 1, "all ok!"), "error")
        self.assertEqual(classify("verify", 0, "all ok!", timed_out=True), "timeout")
        self.assertEqual(
            classify("updr", 0, "updr found abstract counterexample!"),
            "abstract_counterexample",
        )

    def test_nonfinite_or_excessive_unstack_budget_rejected_before_work(self):
        for budget in [float("nan"), float("inf"), -1, 0, 51]:
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                run(argparse.Namespace(task="unstack", timeout_seconds=budget))


if __name__ == "__main__":
    unittest.main()
