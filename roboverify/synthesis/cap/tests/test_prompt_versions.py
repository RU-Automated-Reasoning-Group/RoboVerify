"""Prompt compatibility, request roles, provenance, and unchanged execution."""

import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ..budget import Budget
from ..cap_google_baseline import GoogleBaseline
from ..cap_google_baseline.compatibility import module_config
from ..cap_method_1 import Method1
from ..cap_method_1.generation import prompt
from ..checkpoints import Checkpoints
from ..config import ModelConfig, RunConfig
from ..contracts import (
    API_DESCRIPTION,
    CONDITION_RULES,
    Bundle,
    ConstraintFailure,
    api_description,
    task_api_names,
)
from ..llm import LoggedModel, ScriptedModel, TransportError
from ..logger import CapLogger
from ..prompt_package import package, package_bytes, prompt_provenance
from ..runner import run
from ..validation import validate
from .test_adapter import RobotFixture
from .test_methods import FakeAdapter


class PromptVersionTests(unittest.TestCase):
    def test_legacy_matches_prechange_hashes_for_both_modes_tasks_and_roles(self):
        cases = json.loads(
            (
                Path(__file__).with_name("testdata") / "legacy_prompt_hashes.json"
            ).read_text()
        )
        for case in cases:
            with self.subTest(case=case):
                options = dict(
                    api_names=task_api_names(case["task"]), condition_mode=case["mode"]
                )
                if case["method"] == "cap_method_1":
                    text = prompt(case["role"], **options, **case["fields"])
                else:
                    text = (
                        module_config(helper=case["role"] == "helper", **options)[
                            "prompt_text"
                        ]
                        + case["tail"]
                    )
                self.assertEqual(
                    hashlib.sha256(text.encode()).hexdigest(), case["sha256"]
                )
        legacy = package("legacy_v1")
        self.assertEqual(legacy["api_description"], API_DESCRIPTION)
        self.assertEqual(legacy["condition_rules"], CONDITION_RULES)
        for role, text in legacy["method1_templates"].items():
            self.assertEqual(
                text,
                (
                    Path(__file__).parents[1] / f"cap_method_1/prompts/{role}.txt"
                ).read_text(),
            )

    def test_task_filter_tolerance_and_role_specific_instructions(self):
        for mode in CONDITION_RULES:
            for task in ("stack", "reverse"):
                options = dict(
                    prompt_version="typed_v2",
                    condition_mode=mode,
                    api_names=task_api_names(task),
                    higher_tolerance=0.0025,
                )
                api = api_description(
                    options["api_names"],
                    mode,
                    prompt_version="typed_v2",
                    higher_tolerance=0.0025,
                )
                self.assertEqual("ON_star_zero" in api, task == "reverse")
                self.assertEqual(
                    "frozen trial-entry geometry" in api, task == "reverse"
                )
                self.assertIn("0.0025 metres (the resolved Higher tolerance)", api)
                self.assertNotIn("Emit top-level statements", api)
                policy = prompt(
                    "policy",
                    instruction="Manipulate",
                    context="b0 is the input block.",
                    **options,
                )
                helper = prompt(
                    "helper",
                    signature="ready(b0)",
                    role="condition",
                    unit="ready",
                    **options,
                )
                helper_repair = prompt(
                    "repair",
                    source="def ready(b):\n    return True",
                    role="condition",
                    diagnostics="example",
                    unit="ready",
                    **options,
                )
                action_helper_repair = prompt(
                    "repair",
                    source="def lift(b):\n    Move(b,b,b,[0,0,0.1])",
                    role="body",
                    diagnostics="example",
                    unit="lift",
                    **options,
                )
                policy_repair = prompt(
                    "repair",
                    source="pass",
                    role="body",
                    diagnostics="example",
                    **options,
                )
                for text in (policy, policy_repair):
                    self.assertIn(
                        "If you define a policy entry function, invoke it at the top level.",
                        text,
                    )
                for text in (
                    helper,
                    helper_repair,
                    action_helper_repair,
                    module_config(helper=True, **options)["prompt_text"],
                ):
                    self.assertIn(
                        "The caller is responsible for invoking this helper.", text
                    )
                    self.assertNotIn("invoke it at the top level", text)
                    self.assertIn(api, text)

    def test_unknown_versions_and_invalid_resolved_tolerances_fail(self):
        with self.assertRaises(ValueError):
            RunConfig(prompt_version="typo")
        with self.assertRaises(ValueError):
            module_config(prompt_version="typo")
        for value in (-0.001, 0.025, float("nan")):
            with self.assertRaises(ValueError):
                api_description(
                    task_api_names("stack"),
                    prompt_version="typed_v2",
                    higher_tolerance=value,
                )
        self.assertEqual(RunConfig().prompt_version, "legacy_v1")

    def test_documented_examples_and_condition_modes_keep_existing_contract(self):
        source = "u=b0\nv=b0\nw=b0\np=get_block_position(u)\nMove(u,v,w,[0.01,-0.02,0.08])\nif exists(lambda x: eq(x,b0)):\n    selected=get(lambda x:eq(x,b0))\nMove(u,v,w,[0,0,p[2]-get_gripper_position()[2]])"
        for mode in CONDITION_RULES:
            validate(Bundle(source), condition_mode=mode)
            for bad in (
                "Move(u,v,w,0,0,0.08)",
                "Move(p,p,p,[0,0,0.08])",
                "Move(p[0],p[1],p[2],[0,0,0.08])",
                "Higher(get_block_position(u),get_block_position(v))",
            ):
                with self.assertRaises(ConstraintFailure):
                    validate(
                        Bundle("u=b0\nv=b0\nw=b0\np=get_block_position(u)\n" + bad),
                        condition_mode=mode,
                    )
        mixed = Bundle(
            "count=0\nwhile Higher(b0,b0) and count < 2:\n    count=count+1\nif get_block_position(b0)[2]>0:\n    pass"
        )
        validate(mixed, condition_mode="coordinate_predicates")
        with self.assertRaises(ConstraintFailure):
            validate(mixed, condition_mode="dsl_only")

    def test_recursive_helper_and_both_repair_roles_keep_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            logger = CapLogger(tmp, console=False)
            try:
                budget = Budget(timeout=20)
                cp = Checkpoints(
                    logger, method="cap_method_1", prompt_version="typed_v2"
                )
                backend = ScriptedModel(
                    [
                        "if Pick(b0):\n    pass",
                        "if ready(b0):\n    perform(b0)",
                        "def ready(b):\n    return 1<2",
                        "def ready(b):\n    return Higher(b,b)",
                        "def perform(b):\n    grab(b)",
                        "def grab(b):\n    Pick(b)",
                    ]
                )
                model = LoggedModel(backend, logger, budget)
                adapter = FakeAdapter()
                method = Method1(
                    model,
                    adapter,
                    logger,
                    cp,
                    budget,
                    prompt_version="typed_v2",
                    higher_tolerance=0.0025,
                )
                method("Manipulate")
                self.assertEqual([a[0] for a in adapter.actions], ["Pick"])
                self.assertEqual(len(backend.prompts), 6)
                for text in backend.prompts:
                    self.assertIn("0.0025 metres (the resolved Higher tolerance)", text)
                    self.assertNotIn("Emit top-level statements", text)
                self.assertIn("invoke it at the top level", backend.prompts[1])
                self.assertNotIn("invoke it at the top level", backend.prompts[3])
                self.assertIn("The caller is responsible", backend.prompts[3])
                for manifest_path in logger.root.glob("checkpoints/*/manifest.json"):
                    manifest = json.loads(manifest_path.read_text())
                    self.assertEqual(manifest["prompt_version"], "typed_v2")
                    for src in manifest["sources"].values():
                        if (Path(tmp) / src["path"]).read_text().strip():
                            self.assertTrue(src["generation_matches"])
                            for match in src["generation_matches"]:
                                metadata = json.loads(
                                    (Path(tmp) / match["metadata_path"]).read_text()
                                )
                                self.assertEqual(
                                    metadata["prompt_package_sha256"],
                                    manifest["prompt_package_sha256"],
                                )
                with self.assertRaises(ValueError):
                    Method1(
                        model, adapter, logger, cp, budget, prompt_version="legacy_v1"
                    )
            finally:
                logger.close()

    def test_baseline_recursive_helpers_and_policy_assembly(self):
        for version in ("legacy_v1", "typed_v2"):
            with self.subTest(version=version), tempfile.TemporaryDirectory() as tmp:
                logger = CapLogger(tmp, console=False)
                try:
                    budget = Budget(timeout=20)
                    cp = Checkpoints(
                        logger, method="cap_google_baseline", prompt_version=version
                    )
                    backend = ScriptedModel(
                        [
                            "perform(b0)",
                            "def perform(b):\n    grab(b)",
                            "def grab(b):\n    Pick(b)",
                        ]
                    )
                    model = LoggedModel(backend, logger, budget)
                    adapter = FakeAdapter()
                    method = GoogleBaseline(
                        model,
                        adapter,
                        logger,
                        cp,
                        budget,
                        prompt_version=version,
                        higher_tolerance=0.0025,
                    )
                    assembled, _ = method.policy.build_prompt(
                        "Manipulate", context="b0 is the input block."
                    )
                    self.assertEqual(
                        assembled,
                        prompt(
                            "policy",
                            instruction="Manipulate",
                            context="b0 is the input block.",
                            api_names=adapter.api_names,
                            prompt_version=version,
                            higher_tolerance=0.0025,
                        ),
                    )
                    method("Manipulate")
                    self.assertEqual([a[0] for a in adapter.actions], ["Pick"])
                    self.assertEqual(len(backend.prompts), 3)
                    if version == "typed_v2":
                        self.assertTrue(
                            all(
                                "0.0025 metres (the resolved Higher tolerance)" in p
                                for p in backend.prompts
                            )
                        )
                        self.assertTrue(
                            all(
                                "The caller is responsible" in p
                                for p in backend.prompts[1:]
                            )
                        )
                    for path in logger.root.glob("checkpoints/*/manifest.json"):
                        manifest = json.loads(path.read_text())
                        self.assertEqual(manifest["validation"], "not_run")
                        self.assertTrue(
                            all(
                                s["generation_matches"]
                                for s in manifest["sources"].values()
                            )
                        )
                finally:
                    logger.close()

    def test_retry_requests_have_exact_hashes_and_same_version(self):
        class RetryModel(ScriptedModel):
            failed = False

            def generate(self, prompt, timeout):
                if not self.failed:
                    self.failed = True
                    raise TransportError("test retry", True)
                return super().generate(prompt, timeout)

        with tempfile.TemporaryDirectory() as tmp:
            logger = CapLogger(tmp, console=False)
            try:
                Checkpoints(logger, method="cap_method_1", prompt_version="typed_v2")
                model = LoggedModel(
                    RetryModel(["pass"]), logger, Budget(timeout=20), retries=1
                )
                with patch("synthesis.cap.llm.time.sleep"):
                    model.complete("exact prompt\n", purpose="repair")
                metas = []
                for path in sorted(logger.root.glob("lmp/*/prompt_metadata.json")):
                    meta = json.loads(path.read_text())
                    metas.append(meta)
                    self.assertEqual(
                        meta["prompt_sha256"],
                        hashlib.sha256(
                            (path.parent / "prompt.txt").read_bytes()
                        ).hexdigest(),
                    )
                    request = json.loads((path.parent / "request.json").read_text())
                    self.assertEqual(
                        meta["request_sha256"],
                        hashlib.sha256(
                            json.dumps(
                                request,
                                sort_keys=True,
                                separators=(",", ":"),
                                ensure_ascii=True,
                            ).encode()
                        ).hexdigest(),
                    )
                    self.assertEqual(meta["role"], "repair")
                    self.assertEqual(meta["method"], "cap_method_1")
                    self.assertEqual(meta["condition_mode"], "dsl_only")
                    self.assertEqual(meta["prompt_version"], "typed_v2")
                self.assertEqual([m["retry"] for m in metas], [0, 1])
                self.assertEqual(metas[0]["prompt_sha256"], metas[1]["prompt_sha256"])
            finally:
                logger.close()

    def test_runner_versions_modes_share_actions_and_record_package(self):
        import numpy as np

        from synthesis.experiment.report import RunSummary

        actions = []
        with tempfile.TemporaryDirectory() as tmp:
            for version in ("legacy_v1", "typed_v2"):
                for method in ("cap_google_baseline", "cap_method_1"):
                    for mode in CONDITION_RULES:
                        env = RobotFixture(1, success=True)
                        backend = ScriptedModel(
                            [
                                "p=get_block_position(b0)\nPick(b0)\nMove(b0,b0,b0,[0,0,0.1])"
                            ]
                        )
                        with contextlib.redirect_stdout(io.StringIO()):
                            result = run(
                                RunConfig(
                                    method=method,
                                    condition_mode=mode,
                                    prompt_version=version,
                                    num_blocks=1,
                                    save_video=False,
                                    output_root=tmp,
                                    higher_tolerance=0.0025,
                                ),
                                ModelConfig(),
                                backend=backend,
                                environment_factory=lambda config, snapshot: env,
                                capture_stdout=False,
                            )
                        self.assertEqual(
                            result["status"], "completed", result.get("reason")
                        )
                        actions.append(np.array(env.actions))
                        directory = Path(result["run_dir"])
                        root = directory / "artifacts/trial_001"
                        config = json.loads((directory / "config.json").read_text())
                        self.assertEqual(config["config"]["prompt_version"], version)
                        self.assertEqual(
                            result["prompt_package_sha256"],
                            prompt_provenance(version)["prompt_package_sha256"],
                        )
                        self.assertEqual(
                            hashlib.sha256(
                                (root / "prompt_package.json").read_bytes()
                            ).hexdigest(),
                            hashlib.sha256(package_bytes(version)).hexdigest(),
                        )
                        self.assertIn(
                            f"prompt_version: {version}", RunSummary(directory).render()
                        )
                        self.assertEqual(
                            "0.0025 metres (the resolved Higher tolerance)"
                            in backend.prompts[0],
                            version == "typed_v2",
                        )
        for action in actions[1:]:
            np.testing.assert_array_equal(actions[0], action)

    def test_cli_propagates_prompt_version(self):
        from synthesis.entry.run_cap import main

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "responses.json"
            path.write_text('["pass"]')
            result = dict(status="generated", success=None, run_dir=tmp)
            with (
                patch("synthesis.entry.run_cap.run", return_value=result) as mocked,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(
                    main(
                        [
                            "--method",
                            "cap_method_1",
                            "--responses-file",
                            str(path),
                            "--prompt-version",
                            "typed_v2",
                            "--generate-only",
                        ]
                    ),
                    0,
                )
            self.assertEqual(mocked.call_args.args[0].prompt_version, "typed_v2")


if __name__ == "__main__":
    unittest.main()
