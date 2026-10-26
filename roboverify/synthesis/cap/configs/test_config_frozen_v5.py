"""Frozen configuration v5: v4 with primitive-prompt-v2 (Revision 7)."""

import unittest
from dataclasses import asdict

from ..primitive_prompts import PACKAGES, render
from ..tests import prompt_parity
from . import config_frozen_v4 as v4
from . import config_frozen_v5 as v5

SENTENCE = "Express any required lift, transfer, and descent\nstages explicitly."


class FrozenV5Tests(unittest.TestCase):
    def test_v5_is_v4_with_the_v2_primitive_prompt(self):
        for task in v5.TASKS:
            for demonstrations in (True, False):
                with self.subTest(task=task, demonstrations=demonstrations):
                    config = v5.run_config(task, demonstrations, seed=3)
                    base = v4.run_config(task, demonstrations, seed=3)
                    self.assertEqual(
                        config.primitive_prompt_version, "primitive-prompt-v2"
                    )
                    self.assertEqual(
                        base.primitive_prompt_version, "primitive-prompt-v1"
                    )
                    changed = {
                        k for k, v in asdict(config).items() if asdict(base)[k] != v
                    }
                    self.assertEqual(changed, {"primitive_prompt_version"})
                    self.assertEqual(v5.instruction(task), v4.instruction(task))
        self.assertEqual(v5.model_config(), v4.model_config())
        self.assertEqual(v5.SERVING, v4.SERVING)

    def test_v2_package_differs_from_v1_only_by_the_sentence(self):
        v1, v2 = PACKAGES["primitive-prompt-v1"], PACKAGES["primitive-prompt-v2"]
        self.assertEqual(v1["header"], v2["header"])
        for key, text in v1["blocks"].items():
            expected = text.replace(" " + SENTENCE, "")
            self.assertEqual(v2["blocks"][key], expected)
        self.assertIn(SENTENCE, v1["blocks"]["explain_motion_planning_limits"])
        flags = asdict(v5.run_config("stack").primitive_prompt)
        self.assertNotEqual(render(flags), render(flags, "primitive-prompt-v2"))

    def test_v5_prompts_drop_only_the_sentence(self):
        for task in ("stack", "grid"):
            with self.subTest(task=task):
                runs = []
                for frozen in (v4, v5):
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
                        )
                    } | {
                        "primitive_prompt_version": settings.get(
                            "primitive_prompt_version", "primitive-prompt-v1"
                        )
                    }
                    prompts, _ = prompt_parity.rendered_prompts(
                        "cap_method_2", "dsl_only", task, overrides
                    )
                    runs.append(prompts)
                old, new = runs
                self.assertEqual(len(old), len(new))
                for before, after in zip(old, new):
                    self.assertIn(SENTENCE, before)
                    self.assertNotIn("lift, transfer, and descent", after)
                    self.assertEqual(before.replace(" " + SENTENCE, ""), after)


if __name__ == "__main__":
    unittest.main()
