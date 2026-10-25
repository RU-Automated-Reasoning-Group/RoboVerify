import unittest
from dataclasses import asdict

from ..experiments.revision43_feedback_experiment import profile_settings, prompt_flags
from . import config_frozen_v1 as frozen


class FrozenConfigTests(unittest.TestCase):
    def test_matches_the_selected_components(self):
        config = frozen.run_config("stack", seed=3)
        self.assertEqual(config.method, "cap_method_2")
        self.assertEqual(asdict(config.primitive_prompt), prompt_flags("0101"))
        full = profile_settings("full")
        enrichments = asdict(config.primitive_feedback)
        self.assertEqual(
            {k: enrichments[k] for k in full["primitive_feedback"]},
            full["primitive_feedback"],
        )
        feedback = asdict(config.recovery.feedback)
        self.assertEqual(
            {k: feedback[k] for k in full["recovery"]["feedback"]},
            full["recovery"]["feedback"],
        )
        self.assertEqual(config.recovery.history.mode, "executed_programs")
        self.assertEqual(config.recovery.success_field, "task_spec_success")
        self.assertEqual((config.num_blocks, config.seed), (4, 3))

    def test_every_tasks_settings_validate(self):
        for task in frozen.TASKS:
            with self.subTest(task=task):
                frozen.run_config(task)
        self.assertEqual(frozen.run_config("restack").initial_tower_blocks, 3)
        self.assertEqual(frozen.model_config().max_output_tokens, 32768)

    def test_missing_instructions_fail_closed(self):
        self.assertIn("ON_star(x,b0)", frozen.instruction("stack"))
        with self.assertRaises(NotImplementedError):
            frozen.configuration("unstack")

    def test_settings_are_copies(self):
        frozen.run_settings("stack")["recovery"]["feedback"]["keyframes"] = False
        self.assertTrue(frozen.RUN["recovery"]["feedback"]["keyframes"])


if __name__ == "__main__":
    unittest.main()
