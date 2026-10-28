"""Revision 7.4: a block left touching the gripper fails the attempt."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from synthesis.cfg.reset import restore

from ..config import ModelConfig, RecoveryConfig, RunConfig, stored_run_config
from ..configs import config_frozen_v1 as v1
from ..configs import config_frozen_v7 as v7
from ..llm import ScriptedModel
from ..recovery import SUCCESS_CHECK_NOTE, success_check, summarize
from ..runner import run
from ..tasks import FINAL_RELEASE_VERSION, final_release_outcome, success_criterion
from .test_adapter import RobotFixture

GEOMS = ("table0", "robot0:l_gripper_finger_link", "robot0:r_gripper_finger_link")


def install_contacts(env, pairs):
    """Give a fixture the contact interface of mujoco_py: geom id pairs."""
    names = list(GEOMS) + [f"object{i}" for i in range(env.num_blocks)]
    env.sim.model = SimpleNamespace(geom_id2name=lambda i: names[i])
    env.sim.data.contact = [
        SimpleNamespace(geom1=names.index(a), geom2=names.index(b)) for a, b in pairs
    ]
    env.sim.data.ncon = len(pairs)


class ConfigTests(unittest.TestCase):
    def test_option_is_on_by_default(self):
        self.assertTrue(RunConfig().final_release_check)
        self.assertEqual(
            success_criterion(RunConfig()), "task-spec-v1+" + FINAL_RELEASE_VERSION
        )
        with self.assertRaises(ValueError):
            RunConfig(final_release_check=1)
        config = RunConfig(post_settling_steps=50)
        self.assertEqual(
            success_criterion(config), "task-spec-v2+" + FINAL_RELEASE_VERSION
        )
        off = RunConfig(final_release_check=False, post_settling_steps=50)
        self.assertEqual(success_criterion(off), "task-spec-v2")

    def test_recorded_configs_without_the_option_keep_it_off(self):
        self.assertFalse(stored_run_config({"task": "stack"}).final_release_check)
        recorded = stored_run_config({"final_release_check": True})
        self.assertTrue(recorded.final_release_check)
        self.assertFalse(v1.run_config("stack").final_release_check)
        self.assertFalse(v7.run_config("stack").final_release_check)


class OutcomeTests(unittest.TestCase):
    def outcome(self, pairs, fingers=(0.04, 0.04), ledger=None):
        env = RobotFixture(2)
        env.obs[3:5] = fingers
        env.obs[10:13] = env.obs[:3]  # block 0 at the gripper centre
        install_contacts(env, pairs)
        return final_release_outcome(env, SimpleNamespace(ledger_held=ledger))

    def test_no_gripper_contact_is_released(self):
        result = self.outcome([("table0", "object0"), ("object0", "object1")])
        self.assertTrue(result["released"])
        self.assertEqual(result["unreleased_blocks"], [])
        self.assertEqual(result["issues"], [])

    def test_one_finger_contact_is_unreleased(self):
        result = self.outcome([("object1", "robot0:l_gripper_finger_link")])
        self.assertFalse(result["released"])
        self.assertEqual(result["unreleased_blocks"], [1])
        self.assertEqual(
            result["contacts"],
            [{"block": 1, "geoms": ["robot0:l_gripper_finger_link"]}],
        )
        self.assertIn("block_1 is still in contact", result["issues"][0])

    def test_grasp_reports_closed_fingers_and_ledger(self):
        result = self.outcome(
            [
                ("robot0:l_gripper_finger_link", "object0"),
                ("object0", "robot0:r_gripper_finger_link"),
            ],
            fingers=(0.02, 0.02),
            ledger=0,
        )
        self.assertFalse(result["released"])
        self.assertEqual(result["contacts"][0]["geoms"], sorted(GEOMS[1:]))
        self.assertTrue(result["fingers_closed"])
        self.assertEqual(result["near_closed_gripper"], [0])
        self.assertEqual(result["ledger_held"], "0")

    def test_evidence_names_unreleased_blocks(self):
        result = {
            "success_criterion": "task-spec-v2+final-release-v1",
            "task_spec_postcondition": True,
            "task_spec_settled_postcondition": True,
            "final_release": {"released": False, "unreleased_blocks": [3]},
            "status": "unreleased_block",
            "task_spec_success": False,
        }
        evidence = success_check(result)
        self.assertEqual(evidence["unreleased_blocks"], ["block_3"])
        self.assertTrue(evidence["postcondition_after_settling"])
        self.assertEqual(evidence["note"], SUCCESS_CHECK_NOTE)
        summary = summarize({"result": result, "candidate_sha256": "c"}, 0)
        self.assertEqual(summary["status"], "unreleased_block")
        self.assertFalse(summary["released"])
        self.assertFalse(summary["task_spec_success"])
        del result["final_release"]
        summary = summarize({"result": result, "candidate_sha256": "c"}, 0)
        self.assertNotIn("released", summary)
        self.assertNotIn("task_spec_success", summary)


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.contacts = []

    def tearDown(self):
        self.tmp.cleanup()

    def factory(self, config, snapshot):
        env = RobotFixture(config.num_blocks, success=True)
        if snapshot:
            restore(env, snapshot)
        # Each new environment takes the next scripted contact set.
        install_contacts(env, self.contacts.pop(0) if self.contacts else [])
        return env

    def execute(self, responses, contacts, check=True, recovery=None):
        self.contacts = list(contacts)
        config = RunConfig(
            method="cap_method_2",
            num_blocks=1,
            save_video=False,
            output_root=self.tmp.name,
            prompt_version="typed_v3",
            post_settling_steps=50,
            final_release_check=check,
            recovery=recovery or RecoveryConfig(),
        )
        backend = ScriptedModel(responses)
        with contextlib.redirect_stdout(io.StringIO()):
            result = run(
                config,
                ModelConfig(max_output_tokens=16384),
                backend=backend,
                environment_factory=self.factory,
                capture_stdout=False,
            )
        return result, backend

    def test_unreleased_block_fails_an_otherwise_successful_attempt(self):
        held = [("robot0:l_gripper_finger_link", "object0")]
        result, _ = self.execute(["pass"], [held])
        self.assertEqual(result["status"], "unreleased_block")
        self.assertIn("block_0 is still in contact", result["reason"])
        self.assertTrue(result["success"])  # physical evaluator, unchanged
        self.assertTrue(result["task_success"]["success"])  # shared checker
        self.assertFalse(result["task_spec_success"])
        self.assertFalse(result["final_release"]["released"])
        self.assertEqual(result["success_criterion"], "task-spec-v2+final-release-v1")

    def test_released_attempt_keeps_its_status(self):
        result, _ = self.execute(["pass"], [[]])
        self.assertEqual(result["status"], "completed")
        self.assertTrue(result["task_spec_success"])
        self.assertTrue(result["final_release"]["released"])

    def test_option_off_writes_no_release_fields(self):
        held = [("robot0:l_gripper_finger_link", "object0")]
        result, _ = self.execute(["pass"], [held], check=False)
        self.assertEqual(result["status"], "completed")
        self.assertTrue(result["task_spec_success"])
        self.assertNotIn("final_release", result)
        self.assertEqual(result["success_criterion"], "task-spec-v2")

    def test_ledger_tracks_the_last_unreleased_pick(self):
        result, _ = self.execute(["Pick(b0)"], [[]])
        self.assertEqual(result["final_release"]["ledger_held"], "0")
        result, _ = self.execute(["Pick(b0)\nRelease(b0, 0.1)"], [[]])
        self.assertIsNone(result["final_release"]["ledger_held"])

    def test_recovery_request_carries_success_check(self):
        held = [("robot0:l_gripper_finger_link", "object0")]
        result, backend = self.execute(
            ["pass", "pass"],
            [held, []],
            recovery=RecoveryConfig(
                enabled=True,
                success_field="task_spec_success",
                feedback={"evidence_layout": "fixed_items"},
            ),
        )
        self.assertTrue(result["task_spec_success"], result)
        self.assertEqual(result["task_spec_success_round"], 1)
        self.assertEqual(result["attempts"][0]["status"], "unreleased_block")
        prompt = backend.prompts[1]
        self.assertIn('"unreleased_blocks":["block_0"]', prompt)
        self.assertIn('"status":"unreleased_block"', prompt)
        root = Path(result["attempts"][1]["run_dir"]) / "artifacts/trial_001"
        feedback = json.loads((root / "recovery_feedback.json").read_text())
        self.assertEqual(feedback["item_delivery"]["success_check"], "complete")
        self.assertFalse(feedback["evidence"]["success_check"]["released"])


class SimulatorTests(unittest.TestCase):
    """Two-block Stack in MuJoCo: a policy that ends holding the block fails."""

    PICK = (
        "x = get(lambda x: not eq(x, b0))\n"
        "Pick(x)\n"
        "Move(b0, b0, b0, [0.0, 0.0, 0.10])\n"
        "Move(b0, b0, b0, [0.0, 0.0, 0.055])\n"
    )

    def execute(self, policy):
        with tempfile.TemporaryDirectory() as tmp:
            config = RunConfig(
                method="cap_method_2",
                num_blocks=2,
                seed=0,
                save_video=False,
                output_root=tmp,
                prompt_version="typed_v5",
                post_settling_steps=50,
                final_release_check=True,
            )
            with contextlib.redirect_stdout(io.StringIO()):
                return run(
                    config,
                    ModelConfig(max_output_tokens=16384),
                    backend=ScriptedModel([policy]),
                    capture_stdout=False,
                )

    def test_holding_the_last_block_fails_and_releasing_it_passes(self):
        held = self.execute(self.PICK)
        # The gap: the shared settled check accepts the block held over b0.
        self.assertTrue(held["task_success"]["success"])
        self.assertEqual(held["status"], "unreleased_block", held.get("reason"))
        self.assertFalse(held["final_release"]["released"])
        self.assertTrue(held["final_release"]["fingers_closed"])
        self.assertEqual(held["final_release"]["unreleased_blocks"], [1])
        self.assertFalse(held["task_spec_success"])
        released = self.execute(self.PICK + "Release(b0, 0.15)\n")
        self.assertTrue(released["final_release"]["released"], released)
        self.assertIsNone(released["final_release"]["ledger_held"])
        self.assertTrue(released["task_spec_success"], released)


if __name__ == "__main__":
    unittest.main()
