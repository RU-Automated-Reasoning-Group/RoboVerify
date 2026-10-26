"""Frozen configuration v4: v3 plus the settled criterion and re-collected archives."""

import hashlib
import json
import unittest
from dataclasses import asdict
from pathlib import Path

from ..tasks import success_criterion
from ..tests import prompt_parity
from . import config_frozen_v3 as v3
from . import config_frozen_v4 as v4

ROOT = Path(__file__).resolve().parents[3]


class FrozenV4Tests(unittest.TestCase):
    def test_v4_is_v3_plus_final_settling_and_two_archives(self):
        for task in v4.TASKS:
            with self.subTest(task=task):
                config = v4.run_config(task, seed=2)
                base = v3.run_config(task, seed=2)
                self.assertEqual(config.post_settling_steps, 50)
                self.assertEqual(base.post_settling_steps, 0)
                self.assertEqual(success_criterion(config), "task-spec-v2")
                self.assertEqual(success_criterion(base), "task-spec-v1")
                changed = {k for k, v in asdict(config).items() if asdict(base)[k] != v}
                expected = {"post_settling_steps"}
                if task in {"unstack", "reverse"}:
                    expected.add("demonstrations")
                    self.assertIn("cap-revision6-v4", config.demonstrations.archive)
                self.assertEqual(changed, expected)
                self.assertEqual(
                    asdict(config.demonstrations)
                    | {
                        "archive": base.demonstrations.archive,
                        "archive_sha256": base.demonstrations.archive_sha256,
                    },
                    asdict(base.demonstrations),
                )
                self.assertEqual(v4.instruction(task), v3.instruction(task))
                self.assertFalse(
                    v4.run_config(task, demonstrations=False).demonstrations.enabled
                )
        self.assertEqual(v4.model_config(), v3.model_config())
        self.assertEqual(v4.SERVING, v3.SERVING)
        self.assertEqual(v4.DEMONSTRATION_SEEDS, v3.DEMONSTRATION_SEEDS)

    def test_v4_settings_are_copies(self):
        v4.run_settings("stack")["recovery"]["context_tokens"] = 1
        self.assertEqual(v4.RUN["recovery"]["context_tokens"], 262144)
        self.assertEqual(v3.RUN.get("post_settling_steps", 0), 0)

    def test_pinned_archives_match_when_present(self):
        for task, (archive, sha256) in v4.ARCHIVES.items():
            path = ROOT / archive
            if not path.exists():
                continue  # archives are not committed
            with self.subTest(task=task):
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), sha256)

    def test_config_v4_off_renders_the_v3_prompts(self):
        golden = json.loads(prompt_parity.GOLDEN.read_text())
        for case in golden:
            if case["task"] not in {"stack", "grid"}:
                continue
            settings = v4.run_settings(
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
