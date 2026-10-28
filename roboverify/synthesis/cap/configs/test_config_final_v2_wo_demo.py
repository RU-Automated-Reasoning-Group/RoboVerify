"""Final configuration v2 without demonstrations."""

import unittest
from dataclasses import asdict

from . import config_final_v2 as v2
from . import config_final_v2_wo_demo as wo_demo


class FinalV2WithoutDemonstrationsTests(unittest.TestCase):
    def test_only_the_demonstrations_are_off(self):
        for task in wo_demo.TASKS:
            with self.subTest(task=task):
                config = wo_demo.run_config(task, seed=3)
                self.assertFalse(config.demonstrations.enabled)
                self.assertEqual(
                    asdict(config), asdict(v2.run_config(task, False, seed=3))
                )
                self.assertEqual(wo_demo.instruction(task), v2.instruction(task))
        self.assertEqual(wo_demo.model_config(), v2.model_config())
        self.assertEqual(wo_demo.model_config().prompt_cache, "5m")
        self.assertEqual(wo_demo.BATCH, v2.BATCH)
        self.assertEqual(wo_demo.TASKS, v2.TASKS)
        self.assertFalse(wo_demo.DEMONSTRATION_TOKENS_ONCE)

    def test_demonstrations_cannot_be_turned_on(self):
        with self.assertRaises(ValueError):
            wo_demo.configuration("stack", True)
        config, model, instruction = wo_demo.configuration("grid", seed=1)
        self.assertEqual(config.seed, 1)
        self.assertEqual(instruction, v2.instruction("grid"))


if __name__ == "__main__":
    unittest.main()
