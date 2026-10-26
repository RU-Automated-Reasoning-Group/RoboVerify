"""Frozen configuration v3: v2 plus demonstration input (Revision 6)."""

import hashlib
import json
import unittest
from dataclasses import asdict

from ..tests import prompt_parity
from . import config_frozen_v2 as v2
from . import config_frozen_v3 as v3


class FrozenV3Tests(unittest.TestCase):
    def test_v3_is_v2_plus_demonstrations_and_the_context_window(self):
        for task in v3.TASKS:
            with self.subTest(task=task):
                config = v3.run_config(task, seed=2)
                base = v2.run_config(task, seed=2)
                self.assertEqual(config.prompt_version, "typed_v5")
                self.assertEqual(config.recovery.context_tokens, 262144)
                settings = config.demonstrations
                self.assertTrue(settings.enabled)
                self.assertEqual(settings.waypoint_tolerance, 0.01)
                self.assertEqual(settings.max_tokens, 131072)
                self.assertEqual(settings.token_counter, "server_tokenize")
                self.assertEqual(settings.archive, v3.ARCHIVES[task][0])
                self.assertFalse(
                    v3.run_config(task, demonstrations=False).demonstrations.enabled
                )
                changed = {k for k, v in asdict(config).items() if asdict(base)[k] != v}
                self.assertEqual(
                    changed, {"prompt_version", "recovery", "demonstrations"}
                )
                self.assertEqual(
                    asdict(config.recovery) | {"context_tokens": 200000},
                    asdict(base.recovery),
                )
                self.assertEqual(v3.instruction(task), v2.instruction(task))
        self.assertEqual(v3.model_config(), v2.model_config())
        self.assertEqual(v3.SERVING["max_model_len"], 262144)
        self.assertTrue(v3.SERVING["enable_prefix_caching"])
        self.assertNotIn("max_model_len", v2.SERVING)
        self.assertTrue(set(v3.DEMONSTRATION_SEEDS).isdisjoint(range(20)))

    def test_v3_settings_are_copies(self):
        v3.run_settings("stack")["demonstrations"]["max_tokens"] = 1
        self.assertEqual(v3.DEMONSTRATIONS["max_tokens"], 131072)

    def test_config_v3_off_renders_the_v2_prompts(self):
        golden = json.loads(prompt_parity.GOLDEN.read_text())
        for case in golden:
            if case["task"] not in {"stack", "grid"}:
                continue
            settings = v3.run_settings(
                case["task"],
                demonstrations=False,
                method=case["method"],
                condition_mode=case["mode"],
            )
            overrides = {
                k: settings[k] for k in ("prompt_version", "demonstrations", "recovery")
            }
            prompts, _ = prompt_parity.rendered_prompts(
                case["method"], case["mode"], case["task"], overrides
            )
            self.assertEqual(
                [hashlib.sha256(p.encode()).hexdigest() for p in prompts],
                case["sha256"],
            )


if __name__ == "__main__":
    unittest.main()
