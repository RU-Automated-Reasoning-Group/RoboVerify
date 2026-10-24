import unittest

from .revision43_feedback_report import paired


def row(seed, profile, recovered, rounds=1):
    return {
        "seed": seed,
        "profile": profile,
        "recovered": recovered,
        "round1": recovered and rounds == 1,
    }


class PairedTest(unittest.TestCase):
    def test_discordant_counts_and_difference(self):
        rows = [
            row("a", "full", True),
            row("b", "full", True),
            row("c", "full", False),
            row("d", "full", True),
            row("a", "minimal", False),
            row("b", "minimal", True, rounds=2),
            row("c", "minimal", True),
            row("d", "minimal", False),
        ]
        table = paired(rows, "recovered")
        self.assertEqual(
            {k: table["minimal"][k] for k in ("n", "k", "b", "c")},
            {"n": 4, "k": 2, "b": 2, "c": 1},
        )
        self.assertAlmostEqual(table["minimal"]["diff"], -0.25)
        lo, hi = table["minimal"]["ci95"]
        self.assertLess(lo, -0.25)
        self.assertGreater(hi, -0.25)
        self.assertEqual(paired(rows, "round1")["minimal"]["k"], 1)

    def test_reference_has_zero_width_interval(self):
        rows = [row("a", "full", True), row("b", "full", False)]
        table = paired(rows, "recovered")
        self.assertEqual(table["full"]["ci95"], (0.0, 0.0))

    def test_unpaired_seeds_are_ignored(self):
        rows = [
            row("a", "full", True),
            row("a", "minimal", True),
            row("b", "minimal", False),
        ]
        self.assertEqual(paired(rows, "recovered")["minimal"]["n"], 1)


if __name__ == "__main__":
    unittest.main()
