"""Final configuration v1: v7 with the final release check (Revision 7.4)."""

import unittest
from dataclasses import asdict

from ..tasks import success_criterion
from ..tests import prompt_parity
from . import config_final_v1 as final
from . import config_frozen_v7 as v7


class FinalV1Tests(unittest.TestCase):
    def test_run_config_adds_only_the_release_check(self):
        for task in final.TASKS:
            for demonstrations in (True, False):
                with self.subTest(task=task, demonstrations=demonstrations):
                    new = asdict(final.run_config(task, demonstrations, seed=3))
                    old = asdict(v7.run_config(task, demonstrations, seed=3))
                    changed = {k: (old[k], new[k]) for k in new if old[k] != new[k]}
                    self.assertEqual(changed, {"final_release_check": (False, True)})
        self.assertEqual(final.TASKS, v7.TASKS)
        self.assertEqual(final.model_config(), v7.model_config())
        self.assertEqual(final.MODELS, v7.MODELS)
        self.assertEqual(final.ARCHIVES, v7.ARCHIVES)

    def test_success_criterion_names_the_release_check(self):
        config = final.run_config("stack")
        self.assertEqual(success_criterion(config), final.SUCCESS_CRITERION)
        self.assertEqual(success_criterion(v7.run_config("stack")), "task-spec-v2")

    def test_instructions_add_only_the_release_sentence(self):
        for task in final.TASKS:
            with self.subTest(task=task):
                text = final.instruction(task)
                self.assertEqual(text.count(final.RELEASE_SENTENCE), 1)
                self.assertEqual(
                    text.replace(final.RELEASE_SENTENCE, ""), v7.instruction(task)
                )
                self.assertTrue(text.endswith(final.RELEASE_SENTENCE + final._REQUEST))

    def test_rendered_prompts_change_only_the_instruction(self):
        for task in ("stack", "pyramid"):
            with self.subTest(task=task):
                runs = []
                for frozen in (v7, final):
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
                was, now = v7.instruction(task).strip(), final.instruction(task).strip()
                self.assertEqual(len(old), len(new))
                self.assertTrue(new[0].endswith(now))
                for before, after in zip(old, new):
                    self.assertEqual(before.replace(was, now), after)


if __name__ == "__main__":
    unittest.main()
