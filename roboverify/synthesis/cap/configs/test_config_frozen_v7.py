"""Frozen configuration v7: hosted Claude models, larger budgets, RDP recovery."""

import unittest
from dataclasses import asdict

from . import config_frozen_v6 as v6
from . import config_frozen_v7 as v7


def flat(value, prefix=""):
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            out |= flat(item, f"{prefix}{key}.")
        return out
    return {prefix[:-1]: value}


class FrozenV7Tests(unittest.TestCase):
    def test_run_config_changes_only_budgets_and_keyframes(self):
        expected = {
            "max_tokens": (100000, 200000),
            "recovery.round_tokens": (200000, 300000),
            "recovery.total_tokens": (900000, 1400000),
            "recovery.context_tokens": (262144, 524288),
            "recovery.feedback.keyframe_method": ("legacy", "rdp"),
            "recovery.feedback.keyframe_bytes": (8192, 65536),
            "recovery.feedback.evidence_bytes": (32768, 98304),
            # Hosted models have no local tokenizer (no effect when off).
            "demonstrations.token_counter": ("server_tokenize", "provider_usage"),
        }
        for task in v7.TASKS:
            for demonstrations in (True, False):
                with self.subTest(task=task, demonstrations=demonstrations):
                    new = flat(asdict(v7.run_config(task, demonstrations, seed=3)))
                    old = flat(asdict(v6.run_config(task, demonstrations, seed=3)))
                    changed = {k: (old[k], new[k]) for k in new if old[k] != new[k]}
                    self.assertEqual(changed, expected)
                    self.assertEqual(v7.instruction(task), v6.instruction(task))

    def test_models_are_hosted_claude_with_64k_output(self):
        haiku = v7.model_config()
        opus = v7.model_config(**v7.MODELS["claude-opus-5.5"])
        for model in (haiku, opus):
            self.assertEqual(model.provider, "openrouter")
            self.assertEqual(model.api_key_env, "OPENROUTER_API_KEY")
            self.assertEqual(model.max_output_tokens, 65536)
            self.assertEqual(model.extra_body, {"reasoning": {"effort": "medium"}})
        self.assertEqual(haiku.model, "anthropic/claude-haiku-5.5")
        self.assertEqual(opus.model, "anthropic/claude-opus-5.5")
        self.assertLessEqual(
            65536, v7.SERVING["provider_limits"]["max_completion_tokens"]
        )

    def test_largest_reservations_fit_the_budgets(self):
        # Round 0: prompt without demonstrations (~25 KB) plus the output cap,
        # twice (generation and one repair), fits max_tokens; a recovery request
        # with the full evidence cap and ~60 KB of other text fits a round.
        config = v7.run_config("restack")
        output = v7.MODEL["max_output_tokens"]
        self.assertLessEqual(2 * (25000 + output), config.max_tokens + output)
        recovery = config.recovery
        request = 60000 + recovery.feedback.evidence_bytes + output
        self.assertLessEqual(request, recovery.round_tokens)
        demo_block = 90000  # largest measured block (ReStack ~72k) with margin
        self.assertLessEqual(request + demo_block, recovery.context_tokens)


if __name__ == "__main__":
    unittest.main()
