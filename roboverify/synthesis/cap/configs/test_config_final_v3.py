"""Final configuration v3: final v2 (both arms) with a 1,000,000-step analysis budget."""

import unittest
from dataclasses import asdict

from . import config_final_v2 as v2
from . import config_final_v2_wo_demo as wo_demo
from . import config_final_v3 as v3


class FinalV3Tests(unittest.TestCase):
    def test_only_the_analysis_budget_changes_in_both_arms(self):
        for task in v3.TASKS:
            for demonstrations, base in ((True, v2), (False, wo_demo)):
                with self.subTest(task=task, demonstrations=demonstrations):
                    config = v3.run_config(task, demonstrations, seed=3)
                    self.assertEqual(config.max_analysis_steps, 1_000_000)
                    self.assertEqual(config.demonstrations.enabled, demonstrations)
                    self.assertEqual(
                        asdict(config) | {"max_analysis_steps": None},
                        asdict(base.run_config(task, demonstrations, seed=3)),
                    )
                    self.assertEqual(v3.instruction(task), base.instruction(task))
        self.assertEqual(v3.model_config(), v2.model_config())
        self.assertEqual(v3.BATCH, v2.BATCH)
        self.assertTrue(v3.DEMONSTRATION_TOKENS_ONCE)

    def test_demonstrations_are_on_by_default(self):
        config, model, instruction = v3.configuration("grid", seed=1)
        self.assertTrue(config.demonstrations.enabled)
        self.assertEqual(config.seed, 1)
        self.assertEqual(model.prompt_cache, "5m")
        self.assertEqual(instruction, v2.instruction("grid"))


if __name__ == "__main__":
    unittest.main()
