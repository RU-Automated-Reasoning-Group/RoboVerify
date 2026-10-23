"""Optional real-solver checks; no simulator or saved demonstrations required."""

import itertools
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from baselines.mypyvy.run import MODELS, PROJECT, probe_input
from baselines.mypyvy.worker import certificate_text

ROOT = os.environ.get("ROBOVERIFY_MYPYVY_ROOT")
CVC5 = os.environ.get("ROBOVERIFY_CVC5")
PYTHON = os.environ.get("ROBOVERIFY_BASELINE_PYTHON", sys.executable)


def relation_table(name, blocks, predicate):
    # fixture() closes the universe. Ground literals give the complete table
    # without asking the finite-model finder to rediscover a disjunction.
    return "".join(
        f"assert {'' if predicate(x, y) else '!'}{name}({x},{y})\n"
        for x in blocks
        for y in blocks
    )


def fixture(n):
    blocks = [f"c{i}" for i in range(n)]
    symbols = blocks + ["tbl"]
    declarations = "\n".join(f"immutable constant {b}: block" for b in blocks) + "\n"
    initial = "assert init\nassert b0 = c0\n"
    initial += (
        "assert "
        + " & ".join(f"{a} != {b}" for a, b in itertools.combinations(symbols, 2))
        + "\n"
    )
    initial += "assert forall X. " + " | ".join(f"X = {b}" for b in symbols) + "\n"
    return declarations, initial, blocks


def expected_state(blocks, positions):
    symbols = blocks + ["tbl"]

    def on(x, y):
        return x == y or (
            x != "tbl"
            and y != "tbl"
            and positions[x][0] == positions[y][0]
            and positions[x][1] >= positions[y][1]
        )

    def higher(x, y):
        return x == y or (
            x != "tbl" and y != "tbl" and positions[x][1] >= positions[y][1]
        )

    def scattered(x, y):
        return x != "tbl" and y != "tbl" and positions[x][0] != positions[y][0]

    return "".join(
        relation_table(name, symbols, fn)
        for name, fn in [("on_star", on), ("higher", higher), ("scattered", scattered)]
    )


@unittest.skipUnless(
    ROOT and CVC5,
    "Set ROBOVERIFY_MYPYVY_ROOT and ROBOVERIFY_CVC5 for real solver regressions",
)
class ModelIntegrationTests(unittest.TestCase):
    def invoke(
        self, source, mode="trace", expected_success=True, solver="cvc5", binary=None
    ):
        with tempfile.TemporaryDirectory(prefix="rv-mypyvy-test-") as tmp:
            path = Path(tmp) / "test.pyv"
            path.write_text(source)
            cmd = [PYTHON, "-m", "baselines.mypyvy.worker", "--mypyvy-root", ROOT]
            if solver == "cvc5":
                cmd += ["--cvc5", binary or CVC5]
            cmd += ["--", mode, "--timeout", "3000", str(path)]
            result = subprocess.run(
                cmd, cwd=PROJECT, text=True, capture_output=True, timeout=20
            )
            output = result.stdout + result.stderr
            if expected_success:
                self.assertEqual(result.returncode, 0, output[-4000:])
                if mode == "trace":
                    self.assertRegex(output, r"checked \d+ traces and found 0 errors")
                elif mode == "verify":
                    self.assertIn("all ok!", output)
            else:
                self.assertNotEqual(result.returncode, 0, output)
            return output

    def test_carrier_is_asserted_and_initial_states_exist(self):
        for task in ("stack", "unstack", "reverse"):
            with self.subTest(task=task):
                self.invoke(probe_input((MODELS / f"{task}.pyv").read_text()))

    def test_complete_transfers_match_independent_column_height_oracle(self):
        for task in ("unstack", "reverse"):
            for n in (1, 2, 3, 4):
                with self.subTest(task=task, n=n):
                    declarations, initial, blocks = fixture(n)
                    positions = {b: (0, i) for i, b in enumerate(blocks)}
                    trace = initial + expected_state(blocks, positions)
                    moving = (
                        blocks[::-1]
                        if task == "reverse"
                        else (blocks[:0:-1] if n > 1 else blocks * 2)
                    )
                    for i, block in enumerate(moving):
                        positions[block] = (1, i) if task == "reverse" else (i + 1, 0)
                        trace += f"{task}_step({block})\n" + expected_state(
                            blocks, positions
                        )
                        if task == "reverse":
                            trace += f"assert b = {block}\n"
                    if task == "reverse" or n > 1:
                        trace += "finish\nassert done\n"
                        if task == "reverse":
                            trace += "assert forall X, Y. X != tbl & Y != tbl -> (on_star_zero(X,Y) <-> on_star(Y,X))\n"
                    else:
                        # Exact executable Unstack guard remains enabled for a singleton.
                        trace += "assert eligible(c0)\n"
                    source = (MODELS / f"{task}.pyv").read_text() + declarations
                    self.invoke(source + "sat trace {\n" + trace + "}\n")

    def test_bad_actions_and_stale_reverse_support_are_impossible(self):
        declarations, initial, _ = fixture(3)
        for task in ("unstack", "reverse"):
            source = (MODELS / f"{task}.pyv").read_text() + declarations
            traces = [
                f"unsat trace {{ {initial} {task}_step(c0) }}",
                f"unsat trace {{ {initial} finish }}",
                f"unsat trace {{ {initial} {task}_step(c2) assert on_star(c2,c1) }}",
            ]
            self.invoke(source + "\n".join(traces))
        declarations, initial, _ = fixture(1)
        self.invoke(
            (MODELS / "unstack.pyv").read_text()
            + declarations
            + f"unsat trace {{ {initial} finish }}"
        )

    def test_reverse_cannot_finish_with_original_order(self):
        declarations, initial, _ = fixture(3)
        source = (MODELS / "reverse.pyv").read_text() + declarations
        self.invoke(
            source
            + f"unsat trace {{ {initial} reverse_step(c2) reverse_step(c1) reverse_step(c0) finish assert on_star(c2,c0) }}"
        )

    def test_immutable_inferred_facts_are_checked_not_assumed(self):
        source = "sort block\nimmutable relation frozen()\nmutable relation done()\ninit !done & frozen\nsafety !done\n"
        self.invoke(certificate_text(source, ["frozen"]), mode="verify")
        self.invoke(
            certificate_text(source, ["!frozen"]), mode="verify", expected_success=False
        )

    def test_reserved_model_symbol_rejected(self):
        source = "sort block\nimmutable constant __rv_cvc5_model_block_0: block\nmutable relation done()\ninit !done\nsafety !done\n"
        self.invoke(source, mode="typecheck", expected_success=False)

    def test_solver_unknown_and_crash_never_succeed(self):
        source = "sort block\nmutable relation done()\ninit !done\nsafety !done\nsat trace { assert init }\n"
        with tempfile.TemporaryDirectory(prefix="rv-solver-failure-") as tmp:
            for name, body in [
                ("unknown", "print('unknown', flush=True)\ninput()\n"),
                ("crash", "raise SystemExit(7)\n"),
            ]:
                binary = Path(tmp) / name
                binary.write_text(f"#!{PYTHON}\n" + body)
                binary.chmod(0o755)
                with self.subTest(response=name):
                    output = self.invoke(
                        source, expected_success=False, binary=str(binary)
                    )
                    self.assertNotIn("all ok!", output)


if __name__ == "__main__":
    unittest.main()
