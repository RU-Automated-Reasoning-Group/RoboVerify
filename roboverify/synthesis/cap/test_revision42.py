"""Independent prompt factors, trace-based primitive evidence, and batch ablations."""

import contextlib
import io
import itertools
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from synthesis.cfg.reset import restore

from .batch import create_batch
from .cap_google_baseline.compatibility import module_config
from .cap_method_1.generation import prompt
from .config import (
    ModelConfig,
    PrimitiveFeedbackConfig,
    PrimitivePromptConfig,
    RecoveryConfig,
    RunConfig,
)
from .contracts import api_description, task_api_names
from .llm import ScriptedModel
from .primitive_feedback import (
    build_diagnostics,
    pair_actions,
    phase_diagnostic,
    target_diagnostic,
)
from .primitive_prompts import (
    BLOCKS,
    OPTION_NAMES,
    controller_values,
    provenance,
    render,
)
from .recovery import collect_failure, select_relevant_evidence
from .runner import run
from .test_adapter import RobotFixture


def prompt_flags():
    return PrimitivePromptConfig(**dict.fromkeys(OPTION_NAMES, True))


def feedback_flags():
    return PrimitiveFeedbackConfig(
        target_derivation=True, controller_phases=True, action_context=True
    )


def synthetic_record():
    states = []
    for i in range(3):
        obs = np.zeros(43)
        obs[:3] = [1, 2, 3 + 0.01 * i]
        obs[3:5] = [0.02, 0.02] if i == 0 else [0.04, 0.04]
        obs[10:13] = [1.3 + i, 2.3 + i, 3.3 + i]
        obs[22:25] = [4.3 + i, 5.3 + i, 6.3 + i]
        states.append(obs)
    a = {"object_handle": "a"}
    b = {"object_handle": "b"}

    def event(kind, ident, state, **extra):
        return dict(
            kind=kind,
            event_id=ident,
            run_id="attempt",
            state_index=state,
            source_line=7,
            **extra,
        )

    cfg = {
        "position_tolerance": 0.002,
        "gripper_threshold": 0.052,
        "gripper_tolerance": 0.001,
    }
    events = [
        event("primitive_start", 1, 0, primitive="Pick", arguments=[b]),
        event(
            "primitive_end",
            2,
            0,
            primitive="Pick",
            parent_event_id=1,
            converged=True,
            steps=0,
            position_error=None,
        ),
        event(
            "primitive_start", 3, 0, primitive="Move", arguments=[a, a, a, [0, 0, 0.1]]
        ),
        event(
            "primitive_end",
            4,
            1,
            primitive="Move",
            parent_event_id=3,
            converged=True,
            steps=1,
            position_error=0.001,
        ),
        event(
            "primitive_start",
            5,
            1,
            primitive="Move",
            arguments=[a, b, a, [0.1, 0.2, 0.3]],
            controller_config=cfg,
            step_limit=50,
        ),
        event(
            "controller_phase",
            6,
            1,
            primitive="Move",
            primitive_event_id=5,
            phase="move",
            target=[2.4, 6.5, 4.6],
            steps=0,
        ),
        event(
            "primitive_end",
            7,
            2,
            primitive="Move",
            parent_event_id=5,
            converged=False,
            steps=50,
            position_error=3.0,
        ),
    ]
    return {
        "events": events,
        "trace": SimpleNamespace(states=states),
        "result": {"status": "primitive_failed"},
    }, {"a": 0, "b": 1}


class PrimitivePromptTests(unittest.TestCase):
    def test_all_16_combinations_are_independent_and_deterministic(self):
        base = api_description(task_api_names("stack"), prompt_version="typed_v3")
        for values in itertools.product((False, True), repeat=4):
            flags = dict(zip(OPTION_NAMES, values))
            supplement = render(flags)
            for name, enabled in flags.items():
                self.assertEqual(BLOCKS[name].splitlines()[0] in supplement, enabled)
            self.assertEqual(supplement, render(flags))
            api = api_description(
                task_api_names("stack"),
                prompt_version="typed_v3",
                primitive_prompt=flags,
            )
            if not any(values):
                self.assertEqual(api, base)
            else:
                self.assertEqual(api.replace(supplement, ""), base)
            self.assertEqual(
                provenance(flags)["primitive_prompt_enabled_blocks"],
                [k for k, v in flags.items() if v],
            )

    def test_all_off_matches_original_api_rendering_for_all_base_versions(self):
        from .prompt_package import package

        for version in ("legacy_v1", "typed_v2", "typed_v3"):
            for mode in ("dsl_only", "coordinate_predicates"):
                d = package(version)
                if version == "legacy_v1":
                    expected = "\n".join(
                        line
                        for line in d["api_description"].splitlines()
                        if not line.startswith("ON_star_zero")
                    )
                else:
                    expected = (
                        d["api_header"]
                        + "\n".join(
                            v.format(higher_tolerance="0.001")
                            for k, v in d["api_entries"].items()
                            if k in task_api_names("stack")
                        )
                        + d["api_footer"]
                        + d["examples"]
                    ).rstrip("\n")
                expected += "\n" + d["condition_rules"][mode].rstrip("\n")
                self.assertEqual(
                    api_description(
                        task_api_names("stack"),
                        mode,
                        prompt_version=version,
                        primitive_prompt=PrimitivePromptConfig(),
                    ),
                    expected,
                )

    def test_role_specific_prompts_get_same_general_blocks(self):
        flags = prompt_flags()
        supplement = render(flags)
        for role, fields in [
            ("policy", dict(instruction="Stack", context="b0 is input")),
            ("helper", dict(signature="lift(b)", role="body", unit="lift")),
            ("repair", dict(source="pass", diagnostics="bad", role="body")),
        ]:
            self.assertIn(
                supplement,
                prompt(
                    role, prompt_version="typed_v3", primitive_prompt=flags, **fields
                ),
            )
        for helper in (False, True):
            text = module_config(
                helper=helper, prompt_version="typed_v3", primitive_prompt=flags
            )["prompt_text"]
            self.assertIn(supplement, text)
            if helper:
                self.assertIn("Output only the requested function definition", text)

    def test_values_follow_controller_configuration_and_validate_flags(self):
        values = controller_values()
        self.assertEqual(values["step_limit"], 50)
        self.assertEqual(values["position_tolerance"], 0.002)
        with self.assertRaises(ValueError):
            PrimitivePromptConfig(explain_target_offsets=1)
        with self.assertRaises(ValueError):
            PrimitiveFeedbackConfig(action_context="yes")
        with self.assertRaises(ValueError):
            PrimitiveFeedbackConfig(context_lookback=1, context_actions=2)


class PrimitiveEvidenceTests(unittest.TestCase):
    def test_target_uses_entry_state_and_mixed_axis_references(self):
        record, mapping = synthetic_record()
        action = pair_actions(record["events"])[-1]
        result = target_diagnostic(record, action, mapping)
        np.testing.assert_allclose(
            [x["computed_target"] for x in result["axes"]], [2.4, 6.5, 4.6]
        )
        self.assertEqual(result["axes"][1]["reference_object"], 1)
        self.assertNotEqual(
            result["axes"][0]["reference_coordinate"], record["trace"].states[2][10]
        )

    def test_independent_diagnostic_groups_and_action_priority(self):
        record, mapping = synthetic_record()
        for enabled in itertools.product((False, True), repeat=3):
            flags = PrimitiveFeedbackConfig(
                **dict(
                    zip(
                        ("target_derivation", "controller_phases", "action_context"),
                        enabled,
                    )
                )
            )
            d, candidates = build_diagnostics(record, mapping, flags)
            if not any(enabled):
                self.assertIsNone(d)
                continue
            for name, on in zip(
                ("target_derivation", "controller_phases", "action_context"), enabled
            ):
                self.assertEqual(name in d, on)
            if flags.action_context:
                self.assertEqual([c["action_id"] for c in candidates], [3, 1])
                self.assertEqual(
                    candidates[0]["selection_reason"], "nearest_successful_predecessor"
                )
                self.assertFalse(
                    d["action_context"]["selected_object_is_grasp_evidence"]
                )

    def test_pairing_handles_zero_steps_incomplete_calls_and_attempt_identity(self):
        record, _ = synthetic_record()
        events = record["events"]
        actions = pair_actions(events[:-1])
        self.assertIsNone(actions[-1]["end"])
        self.assertEqual(
            actions[0]["start"]["state_index"], actions[0]["end"]["state_index"]
        )
        events = events + [dict(events[0], run_id="other")]
        with self.assertRaisesRegex(ValueError, "Mixed attempts"):
            pair_actions(events)

    def test_release_target_is_frozen_and_opening_is_distinct_from_retreat(self):
        record, mapping = synthetic_record()
        action = pair_actions(record["events"])[-1]
        action["start"].update(
            primitive="Release", state_index=0, arguments=[{"object_handle": "a"}, 0.1]
        )
        action["phases"] = [
            dict(phase="open", state_index=0, steps=0, target=None),
            dict(phase="retreat", state_index=1, steps=1, target=[1, 2, 3.4]),
        ]
        action["end"].update(
            primitive="Release", state_index=2, steps=50, position_error=0.5
        )
        target = target_diagnostic(record, action, mapping)
        np.testing.assert_allclose(target["axes"][0]["computed_target"], 3.4)
        self.assertEqual(target["frozen_xy_before_open"], [1, 2])
        phases = phase_diagnostic(record, action)["phases"]
        self.assertTrue(phases[0]["completed"])
        self.assertEqual(phases[0]["phase_steps"], 1)
        self.assertEqual(phases[1]["phase_steps"], 49)
        self.assertFalse(phases[1]["completed"])
        self.assertIsNone(phases[0]["position_error"])
        self.assertEqual(phases[0]["comparison"], ">")

    def test_legacy_criteria_unavailable_not_invented(self):
        record, _ = synthetic_record()
        action = pair_actions(record["events"])[-1]
        del action["start"]["controller_config"]
        del action["start"]["step_limit"]
        d = phase_diagnostic(record, action)
        self.assertIsNone(d["step_limit_shared"])
        self.assertIsNone(d["phases"][0]["threshold"])

    def test_release_boundary_does_not_imply_a_current_grasp(self):
        record, mapping = synthetic_record()
        record["events"][2]["primitive"] = "Release"
        record["events"][3]["primitive"] = "Release"
        d, candidates = build_diagnostics(record, mapping, feedback_flags())
        self.assertEqual(candidates[0]["primitive"], "Release")
        self.assertNotIn(
            "recent_selected_object_not_verified_grasp",
            [c["selection_reason"] for c in candidates],
        )

    def test_all_methods_deliver_prompt_and_bounded_feedback_and_restore(self):
        for method in ("cap_google_baseline", "cap_method_1", "cap_method_2"):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as tmp:
                envs = []

                def factory(config, snapshot):
                    env = RobotFixture(1, success=True, stuck=not envs)
                    if snapshot:
                        restore(env, snapshot)
                    envs.append(env)
                    return env

                cfg = RunConfig(
                    method=method,
                    num_blocks=1,
                    prompt_version="typed_v3",
                    save_video=False,
                    output_root=tmp,
                    recovery=RecoveryConfig(enabled=True),
                    primitive_prompt=prompt_flags(),
                    primitive_feedback=feedback_flags(),
                )
                backend = ScriptedModel(["Pick(b0)", "pass"])
                with contextlib.redirect_stdout(io.StringIO()):
                    result = run(
                        cfg,
                        ModelConfig(max_output_tokens=16384),
                        backend=backend,
                        environment_factory=factory,
                        capture_stdout=False,
                    )
                self.assertTrue(result["success"])
                self.assertEqual(result["model_calls"], 2)
                self.assertTrue(
                    all(render(prompt_flags()) in p for p in backend.prompts)
                )
                root = Path(result["attempts"][1]["run_dir"]) / "artifacts/trial_001"
                payload = json.loads((root / "recovery_feedback.json").read_text())
                self.assertLessEqual(payload["feedback_token_bound"], 8192)
                self.assertIn("primitive_diagnostics", payload["evidence"])
                phase = payload["evidence"]["primitive_diagnostics"][
                    "controller_phases"
                ]
                self.assertEqual(phase["step_limit_shared"], 50)
                meta = json.loads(
                    (root / "lmp/call_0001/prompt_metadata.json").read_text()
                )
                self.assertEqual(meta["primitive_prompt"], asdict(prompt_flags()))
                record = collect_failure(result["attempts"][0])
                record["duplicate_of"] = None
                record["changed_from_parent"] = False
                self.assertEqual(
                    select_relevant_evidence(record, [], 8192),
                    select_relevant_evidence(
                        record, [], 8192, PrimitiveFeedbackConfig()
                    ),
                )
                repeated = select_relevant_evidence(record, [], 8192, feedback_flags())
                self.assertEqual(repeated, payload)

    def test_phase_metadata_is_not_leaked_into_phase_ablation(self):
        for enabled in (False, True):
            with tempfile.TemporaryDirectory() as tmp:
                config = RunConfig(
                    method="cap_method_1",
                    num_blocks=1,
                    prompt_version="typed_v3",
                    save_video=False,
                    output_root=tmp,
                    primitive_feedback=PrimitiveFeedbackConfig(
                        target_derivation=True, controller_phases=enabled
                    ),
                )

                def factory(config, snapshot):
                    return RobotFixture(1, success=True, stuck=True)

                with contextlib.redirect_stdout(io.StringIO()):
                    result = run(
                        config,
                        ModelConfig(),
                        backend=ScriptedModel(["Pick(b0)"]),
                        environment_factory=factory,
                        capture_stdout=False,
                    )
                events = [
                    json.loads(line)
                    for line in (
                        Path(result["run_dir"]) / "artifacts/trial_001/timeline.jsonl"
                    )
                    .read_text()
                    .splitlines()
                ]
                start = next(e for e in events if e["kind"] == "primitive_start")
                self.assertEqual("controller_config" in start, enabled)
                self.assertEqual("step_limit" in start, enabled)

    def test_220_job_manifest_balances_variants_without_duplicate_ids(self):
        variants = [
            {
                "name": "full",
                "run_config": {
                    "primitive_prompt": asdict(prompt_flags()),
                    "primitive_feedback": asdict(feedback_flags()),
                },
            }
        ]
        for name, key in [
            ("no_prompt", None),
            ("no_target", "target_derivation"),
            ("no_phases", "controller_phases"),
            ("no_context", "action_context"),
        ]:
            p = asdict(prompt_flags())
            f = asdict(feedback_flags())
            if key:
                f[key] = False
            else:
                p = asdict(PrimitivePromptConfig())
            variants.append(
                {
                    "name": name,
                    "methods": ["cap_method_1", "cap_method_2"],
                    "run_config": {"primitive_prompt": p, "primitive_feedback": f},
                }
            )
        with tempfile.TemporaryDirectory() as tmp:
            manifest = create_batch(
                Path(tmp) / "batch",
                RunConfig(
                    prompt_version="typed_v3", recovery=RecoveryConfig(enabled=True)
                ),
                ModelConfig(),
                trials=20,
                modes=["dsl_only"],
                experiment_variants=variants,
            )
            jobs = manifest["jobs"]
            self.assertEqual(len(jobs), 220)
            self.assertEqual(len({j["id"] for j in jobs}), 220)
            from collections import Counter

            counts = Counter((j["run_config"]["method"], j["variant"]) for j in jobs)
            self.assertEqual(len(counts), 11)
            self.assertEqual(set(counts.values()), {20})
            self.assertEqual(
                len(
                    [
                        j
                        for j in jobs
                        if j["run_config"]["method"] == "cap_google_baseline"
                    ]
                ),
                20,
            )


if __name__ == "__main__":
    unittest.main()
