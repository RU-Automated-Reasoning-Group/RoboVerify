"""Frozen configuration v6: v5 with general task instructions (Revision 7.1)."""

import re
import unittest
from dataclasses import asdict

from ..tests import prompt_parity
from . import config_frozen_v5 as v5
from . import config_frozen_v6 as v6


class FrozenV6Tests(unittest.TestCase):
    def test_v6_changes_only_the_instructions(self):
        for task in v6.TASKS:
            for demonstrations in (True, False):
                with self.subTest(task=task, demonstrations=demonstrations):
                    self.assertEqual(
                        asdict(v6.run_config(task, demonstrations, seed=3)),
                        asdict(v5.run_config(task, demonstrations, seed=3)),
                    )
                    self.assertNotEqual(v6.instruction(task), v5.instruction(task))
        self.assertEqual(v6.TASKS, v5.TASKS)
        self.assertEqual(v6.model_config(), v5.model_config())
        self.assertEqual(v6.SERVING, v5.SERVING)

    def test_tower_instructions_add_only_the_generality_paragraph(self):
        for task in ("stack", "unstack", "reverse", "restack"):
            with self.subTest(task=task):
                self.assertEqual(
                    v6.instruction(task).replace(v6._TOWER_GENERALITY, ""),
                    v5.instruction(task),
                )

    def test_instructions_state_no_block_or_goal_count(self):
        for task in v6.TASKS:
            with self.subTest(task=task):
                text = v6.instruction(task)
                self.assertIn("Do not assume or depend on a particular number", text)
                # Remaining numbers: physical constants, "3D", R, C, n >= 1 and
                # the symbolic layer sizes "n, n-1, ..., 2, 1".
                numbers = set(re.findall(r"(?<![\w.])\d+(?:\.\d+)?", text))
                self.assertLessEqual(numbers, {"0.001", "0.05", "0.02", "1", "2", "3"})
        self.assertNotIn("6 goal", v6.instruction("grid"))
        self.assertNotIn("3 rows", v6.instruction("grid"))
        self.assertIn("R rows of C goals", v6.instruction("grid"))
        self.assertNotIn("3, 2 and 1", v6.instruction("pyramid"))
        self.assertIn("n, n-1, ..., 2, 1 goals", v6.instruction("pyramid"))

    def test_rendered_prompts_change_only_the_instruction(self):
        for task in ("stack", "pyramid"):
            with self.subTest(task=task):
                runs = []
                for frozen in (v5, v6):
                    settings = frozen.run_settings(
                        task, demonstrations=False, method="cap_method_2"
                    )
                    overrides = {
                        k: settings[k]
                        for k in (
                            "prompt_version",
                            "demonstrations",
                            "recovery",
                            "primitive_prompt",
                            "primitive_prompt_version",
                        )
                    }
                    prompts, _ = prompt_parity.rendered_prompts(
                        "cap_method_2",
                        "dsl_only",
                        task,
                        overrides,
                        instruction=frozen.instruction(task),
                    )
                    runs.append(prompts)
                old, new = runs
                # The first request ends with the instruction, without its final
                # newline; static repair requests do not repeat it.
                was, now = v5.instruction(task).strip(), v6.instruction(task).strip()
                self.assertEqual(len(old), len(new))
                self.assertTrue(old[0].endswith(was))
                self.assertTrue(new[0].endswith(now))
                for before, after in zip(old, new):
                    self.assertEqual(before.replace(was, now), after)


if __name__ == "__main__":
    unittest.main()
