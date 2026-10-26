"""RDP keyframes of a failed rollout in recovery feedback (Revision 7.2)."""

import json
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ..config import RecoveryConfig, RecoveryFeedbackConfig
from ..recovery_keyframes import TITLE, VERSION, rdp_keyframes
from . import test_recovery


def synthetic_trace(cycles=6):
    """Two blocks; the gripper repeatedly closes, lifts block 1, and opens."""
    states, actions = [], []
    gripper = np.array([1.0, 0.5, 0.6])
    block0, block1 = np.array([1.0, 0.7, 0.425]), np.array([1.2, 0.7, 0.425])
    for cycle in range(cycles):
        for step in range(10):
            target = block1 + [0, 0, 0.05 * (step % 5)]
            gripper = gripper + 0.5 * (target - gripper)
            state = np.zeros(10 + 12 * 2)
            state[:3] = gripper
            state[10:13] = block0
            state[22:25] = block1 + [0.01 * cycle, 0, 0]
            states.append(state)
            actions.append([0.0, 0.0, 0.0, -0.2 if step < 5 else 0.2])
    actions = np.array(actions[:-1])
    return SimpleNamespace(
        task="stack",
        num_blocks=2,
        states=np.array(states),
        actions=(actions, tuple(range(len(states)))),
        snapshots=[SimpleNamespace(bindings={"b0": 0}, goal_layout=None)],
    )


def size(text):
    return len(text.encode("utf-8"))


class KeyframeConfigTests(unittest.TestCase):
    def test_legacy_is_the_default_and_keeps_the_default_profile(self):
        self.assertEqual(RecoveryFeedbackConfig().keyframe_method, "legacy")
        self.assertTrue(RecoveryConfig().default_profile)

    def test_rdp_requires_fixed_items_and_valid_sizes(self):
        RecoveryFeedbackConfig(keyframe_method="rdp", evidence_layout="fixed_items")
        for bad in (
            {"keyframe_method": "rdp"},
            {"keyframe_method": "all", "evidence_layout": "fixed_items"},
            {"keyframe_bytes": 0},
            {"keyframe_waypoint_tolerance": 0.0},
        ):
            with self.subTest(settings=bad), self.assertRaises(ValueError):
                RecoveryFeedbackConfig(**bad)


class RdpKeyframeTests(unittest.TestCase):
    def test_complete_section_uses_the_demonstration_layout(self):
        text, delivered, available = rdp_keyframes(synthetic_trace(), 2, 0.01, 10**6)
        self.assertEqual(delivered, available)
        lines = text.splitlines()
        self.assertEqual(lines[0], f"{TITLE} ({VERSION})")
        body = next(
            i for i, line in enumerate(lines) if line.startswith("FAILED EXECUTION (")
        )
        self.assertRegex(
            lines[body], r"^FAILED EXECUTION \(round 2\): stack, 2 blocks o0\.\.o1, "
        )
        self.assertEqual(lines[body + 1], "inputs: b0=o0")
        self.assertTrue(lines[body + 2].startswith("t=0 event "))
        self.assertTrue(lines[body + 3].startswith("  blocks: o0="))
        self.assertTrue(lines[body + 4].startswith("  holds: "))
        self.assertEqual(lines[-1], "END OF FAILED EXECUTION")
        self.assertNotIn("\\n", text)
        self.assertTrue(any(" waypoint " in line for line in lines))
        self.assertTrue(any(line.startswith("  changes: ") for line in lines))

    def test_frames_match_the_demonstration_rendering(self):
        from ..demonstrations import compact, render_frame
        from ..recovery_keyframes import demonstration_view

        trace = synthetic_trace()
        text, _, _ = rdp_keyframes(trace, 0, 0.01, 10**6)
        rendered = "\n".join(
            render_frame(f) for f in compact(demonstration_view(trace), 0.01)
        )
        self.assertIn(rendered, text)

    def test_budget_keeps_first_and_latest_frames_with_full_atoms(self):
        trace = synthetic_trace()
        full, _, available = rdp_keyframes(trace, 0, 0.01, 10**6)
        budget = size(full) * 2 // 3
        text, delivered, _ = rdp_keyframes(trace, 0, 0.01, budget)
        self.assertLessEqual(size(text), budget)
        self.assertLess(delivered, available)
        lines = text.splitlines()
        marker = next(i for i, line in enumerate(lines) if line.startswith("... "))
        self.assertIn(f"{available - delivered} frames omitted", lines[marker])
        self.assertEqual(lines[-2], full.splitlines()[-2])
        after = [line for line in lines[marker + 1 :] if line.startswith("  ")]
        self.assertTrue(after[1].startswith("  holds: "))

    def test_no_frame_fits(self):
        text, delivered, available = rdp_keyframes(synthetic_trace(), 0, 0.01, 10)
        self.assertIsNone(text)
        self.assertEqual(delivered, 0)
        self.assertGreater(available, 0)


class ExperimentVariantTests(unittest.TestCase):
    def test_rdp_variant_changes_only_the_keyframe_settings(self):
        from dataclasses import asdict

        from ..batch import variant_config
        from ..config import RunConfig
        from ..configs import config_frozen_v6 as v6
        from ..experiments.revision72_keyframe_experiment import RDP

        for task in v6.TASKS:
            base = asdict(v6.run_config(task))
            merged = asdict(RunConfig(**variant_config(base, RDP)))
            changed = {
                key
                for key, value in merged["recovery"]["feedback"].items()
                if base["recovery"]["feedback"][key] != value
            }
            self.assertEqual(
                changed, {"keyframe_method", "keyframe_bytes", "evidence_bytes"}
            )
            self.assertEqual(
                merged | {"recovery": None}, base | {"recovery": None}, task
            )


class RdpRecoveryTests(unittest.TestCase):
    # Reuse the recovery fixture without rerunning RecoveryTests' own tests.
    fixture = test_recovery.RecoveryTests
    setUp, tearDown = fixture.setUp, fixture.tearDown
    factory, execute = fixture.factory, fixture.execute

    def test_recovery_request_carries_rdp_keyframes(self):
        result, backend = self.execute(
            "cap_method_2",
            ["Pick(b0)\nx=get(lambda x: not ON_star(x,x))", "pass"],
            enabled=True,
            feedback={"evidence_layout": "fixed_items", "keyframe_method": "rdp"},
        )
        self.assertTrue(result["success"], result)
        prompt = backend.prompts[1]
        self.assertIn(f"\n\n{TITLE} ({VERSION})\n", prompt)
        self.assertIn("\nt=0 event ", prompt)
        self.assertLess(prompt.index('"keyframes":{'), prompt.index(TITLE + " ("))
        root = Path(result["attempts"][1]["run_dir"]) / "artifacts/trial_001"
        feedback = json.loads((root / "recovery_feedback.json").read_text())
        self.assertEqual(feedback["keyframe_method"], "rdp")
        self.assertEqual(feedback["evidence"]["keyframes"]["method"], "rdp")
        self.assertIn(feedback["keyframe_text"], prompt)
        self.assertEqual(feedback["item_delivery"]["keyframes"], "complete")


if __name__ == "__main__":
    unittest.main()
