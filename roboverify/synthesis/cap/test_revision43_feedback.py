"""Revision 4.3 feedback-ablation launcher: seeds, profiles and audit (offline)."""

import json
import random
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from .batch import create_batch, variant_config
from .config import FEEDBACK_ITEMS, ModelConfig, RunConfig
from .request_control import dispatch_reservation, request_allowance
from .revision42_experiment import PRIOR
from .revision43_feedback_experiment import (
    PROFILES,
    QUOTAS,
    audit,
    classify,
    create_stage_a,
    load_seed_manifest,
    mcnemar_exact,
    profile_settings,
    prompt_flags,
    sample,
    scaled_quotas,
    select_seeds,
    stage_b_variants,
)

FLAGS = prompt_flags("1111")


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value if isinstance(value, str) else json.dumps(value))
    return path


def fake_batch(root, attempts, model="qwen3.5-9b", flags=FLAGS):
    """A read-only-shaped batch whose jobs each hold one round-0 attempt."""
    jobs = []
    for index, (method, result, rolled) in enumerate(attempts):
        output = root / "trials" / f"job{index}"
        attempt = root / "attempts" / f"a{index}"
        write(attempt / "artifacts/trial_001/initial_state.npz", "state")
        write(
            attempt / "artifacts/trial_001/timeline.jsonl",
            '{"kind": "policy_rollout_start"}\n' if rolled else '{"kind": "x"}\n',
        )
        parent = output / "cap/run"
        write(parent / "config.json", {"config": {}})
        write(
            parent / "artifacts/trial_001/rounds/000/result.json",
            result | {"run_dir": str(attempt)},
        )
        jobs.append(
            {
                "id": f"job{index}",
                "variant": "cell",
                "model_config": {"model": model},
                "run_config": {
                    "method": method,
                    "output_root": str(output),
                    "primitive_prompt": flags,
                },
            }
        )
    write(root / "manifest.json", {"jobs": jobs})
    return root


def primitive(name):
    return {
        "status": "primitive_failed",
        "reason": f"PrimitiveFailure: {name} did not converge within its shared primitive budget",
    }


class ClassificationTests(unittest.TestCase):
    def test_round_zero_failure_classes(self):
        cases = [
            (primitive("Move"), True, "move_nonconvergence"),
            (primitive("Pick"), True, "pick_nonconvergence"),
            (primitive("Release"), True, "release_nonconvergence"),
            (
                {
                    "status": "failed",
                    "reason": "CapError: No object satisfies get condition",
                },
                True,
                "no_witness",
            ),
            (
                {
                    "status": "budget_exhausted",
                    "budget_exhausted_by": "execution_ticks",
                },
                True,
                "execution_budget",
            ),
            (
                {"status": "budget_exhausted", "budget_exhausted_by": "tokens"},
                True,
                "other_runtime",
            ),
            ({"status": "goal_failed", "reason": ""}, True, "goal_failure"),
            (
                {"status": "failed", "reason": "NameError: name 'top'"},
                True,
                "other_runtime",
            ),
            (
                {"status": "failed", "reason": "SyntaxError: invalid"},
                False,
                "preparation",
            ),
            ({"status": "completed", "task_success": True}, True, None),
        ]
        for result, rolled, expected in cases:
            self.assertEqual(classify(result, rolled), expected, result)


class SamplingTests(unittest.TestCase):
    def pool(self, counts):
        return [
            {"class": c, "seed_result": f"{c}-{i:03d}"}
            for c, n in counts.items()
            for i in range(n)
        ]

    def test_quota_sampling_is_deterministic_and_complete(self):
        pool = self.pool(dict.fromkeys(QUOTAS, 30))
        first, report = sample(pool, QUOTAS, random.Random("x"))
        second, _ = sample(list(reversed(pool)), QUOTAS, random.Random("x"))
        self.assertEqual(
            [s["seed_result"] for s in first], [s["seed_result"] for s in second]
        )
        self.assertEqual(len(first), 50)
        self.assertTrue(
            all(r["shortfall"] == 0 and r["refill"] == 0 for r in report.values())
        )

    def test_shortfall_is_refilled_proportionally_from_surplus(self):
        counts = dict.fromkeys(QUOTAS, 0) | {
            "move_nonconvergence": 40,
            "pick_nonconvergence": 8,
            "release_nonconvergence": 18,
            "no_witness": 1,
        }
        chosen, report = sample(self.pool(counts), QUOTAS, random.Random(1))
        self.assertEqual(len(chosen), 50)
        self.assertEqual(report["no_witness"]["shortfall"], 4)
        # 50 - (15 + 8 + 8 + 1) = 18 refilled from surpluses 25 (Move) and 10 (Release).
        self.assertEqual(report["move_nonconvergence"]["refill"], 13)
        self.assertEqual(report["release_nonconvergence"]["refill"], 5)
        self.assertEqual(report["pick_nonconvergence"]["refill"], 0)
        self.assertEqual(len({s["seed_result"] for s in chosen}), 50)

    def test_insufficient_pool_reports_shortfall(self):
        chosen, report = sample(
            self.pool({"goal_failure": 2}), QUOTAS, random.Random(1)
        )
        self.assertEqual(len(chosen), 2)
        self.assertEqual(report["move_nonconvergence"]["shortfall"], 15)


class SeedManifestTests(unittest.TestCase):
    def test_manifest_is_immutable_and_filters_model_prompt_and_preparation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            batch = fake_batch(
                root / "batch",
                [
                    ("cap_method_1", primitive("Move"), True),
                    (
                        "cap_method_1",
                        {"status": "failed", "reason": "SyntaxError"},
                        False,
                    ),
                    (
                        "cap_method_2",
                        {"status": "completed", "task_success": True},
                        True,
                    ),
                    ("cap_method_2", {"status": "goal_failed"}, True),
                ],
            )
            other = fake_batch(
                root / "other",
                [("cap_method_1", primitive("Pick"), True)],
                flags=prompt_flags("0000"),
            )
            out = root / "seeds/seeds.json"
            manifest = select_seeds(
                out, [f"{batch}:cell", str(other)], model="qwen3.5-9b", flags=FLAGS
            )
            self.assertEqual(
                [(s["method"], s["class"]) for s in manifest["seeds"]],
                [
                    ("cap_method_1", "move_nonconvergence"),
                    ("cap_method_2", "goal_failure"),
                ],
            )
            self.assertEqual(
                manifest["strata"]["cap_method_1"]["preparation_failures"], 1
            )
            self.assertEqual(manifest["strata"]["cap_method_2"]["successes"], 1)
            with self.assertRaises(FileExistsError):
                select_seeds(out, [str(batch)], model="qwen3.5-9b", flags=FLAGS)
            self.assertEqual(len(load_seed_manifest(out)["seeds"]), 2)
            Path(manifest["seeds"][0]["seed_result"]).write_text("{}")
            with self.assertRaises(ValueError):
                load_seed_manifest(out)


class ProfileTests(unittest.TestCase):
    def test_fifteen_valid_profiles_with_fixed_items(self):
        base = asdict(RunConfig(recovery={"enabled": True}))
        self.assertEqual(len(PROFILES), 15)
        for name in PROFILES:
            settings = profile_settings(name)
            config = RunConfig(**variant_config(base, settings))
            self.assertEqual(config.recovery.feedback.evidence_layout, "fixed_items")
            self.assertEqual(config.recovery.feedback.supporting_event_window, 40)
            self.assertEqual(config.recovery.feedback.evidence_bytes, 32768)
            self.assertTrue(config.recovery.feedback.dsl_diagnostics)

    def test_profile_flag_mapping(self):
        full = profile_settings("full")
        self.assertTrue(all(full["recovery"]["feedback"][i] for i in FEEDBACK_ITEMS))
        self.assertTrue(all(full["primitive_feedback"].values()))
        self.assertEqual(full["recovery"]["history"]["mode"], "executed_programs")
        events = profile_settings("no_failure_events")["recovery"]["feedback"]
        self.assertFalse(events["failure_events"] or events["query_candidates"])
        self.assertTrue(events["keyframes"])
        self.assertFalse(
            profile_settings("no_context")["primitive_feedback"]["action_context"]
        )
        source = profile_settings("no_current_source")["recovery"]
        self.assertFalse(source["feedback"]["current_source"])
        self.assertEqual(source["history"]["mode"], "summaries")
        minimal = profile_settings("minimal")
        enabled = {k for k, v in minimal["recovery"]["feedback"].items() if v is True}
        self.assertEqual(enabled, {"current_source", "dsl_diagnostics"})
        self.assertEqual(minimal["recovery"]["history"]["mode"], "none")
        legacy = profile_settings("legacy_v1")
        self.assertFalse(any(legacy["primitive_feedback"].values()))
        self.assertEqual(legacy["recovery"]["history"]["mode"], "summaries")

    def test_stage_b_variants_and_seeded_batch_jobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            seed = write(Path(tmp) / "seed/result.json", "{}")
            manifest = {
                "prompt_flags": FLAGS,
                "seeds": [
                    {
                        "seed_id": "cap_method_1-s000",
                        "method": "cap_method_1",
                        "seed_result": str(seed),
                    },
                    {
                        "seed_id": "cap_method_2-s000",
                        "method": "cap_method_2",
                        "seed_result": str(seed),
                    },
                ],
            }
            variants = stage_b_variants(manifest, profiles=("full", "minimal"))
            self.assertEqual(
                [v["name"] for v in variants],
                ["m1_s000_full", "m1_s000_minimal", "m2_s000_full", "m2_s000_minimal"],
            )
            config = RunConfig(method="cap_method_1", prompt_version="typed_v3")
            config.recovery.enabled = True
            batch = create_batch(
                Path(tmp) / "batch",
                config,
                ModelConfig(model="scripted"),
                methods=["cap_method_1", "cap_method_2"],
                modes=["dsl_only"],
                trials=1,
                recovery_variants=[True],
                experiment_variants=variants,
            )
            self.assertEqual(len(batch["jobs"]), 4)
            self.assertTrue(
                all(j["seed_attempt"] == str(seed.resolve()) for j in batch["jobs"])
            )
            self.assertEqual(
                {(j["run_config"]["method"], j["variant"][:2]) for j in batch["jobs"]},
                {("cap_method_1", "m1"), ("cap_method_2", "m2")},
            )
            with self.assertRaises(ValueError):
                create_batch(
                    Path(tmp) / "off",
                    config,
                    ModelConfig(model="scripted"),
                    methods=["cap_method_1"],
                    modes=["dsl_only"],
                    trials=1,
                    recovery_variants=[False],
                    experiment_variants=variants[:1],
                )


class AuditTests(unittest.TestCase):
    def branch(self, root, name, variant, seed_hash, delivery):
        output = root / "trials" / name
        run = output / "cap/run"
        settings = variant["run_config"]
        write(
            run / "config.json",
            {
                "config": {
                    "recovery": settings["recovery"],
                    "primitive_feedback": settings["primitive_feedback"],
                    "primitive_prompt": settings["primitive_prompt"],
                }
            },
        )
        write(
            run / "artifacts/trial_001/seed.json",
            {
                "seed_result": variant["seed_attempt"],
                "seed_result_sha256": seed_hash,
                "initial_state_sha256": "state",
                "seed_run_dir": "attempt",
            },
        )
        write(
            run / "artifacts/trial_001/rounds/001/feedback.json",
            {
                "ready": True,
                "history_mode": settings["recovery"]["history"]["mode"],
                "item_delivery": delivery,
            },
        )
        return {
            "id": name,
            "variant": variant["name"],
            "seed_attempt": variant["seed_attempt"],
            "run_config": {"output_root": str(output)},
        }

    def test_audit_detects_seed_and_delivery_problems(self):
        manifest = {
            "prompt_flags": FLAGS,
            "seeds": [
                {
                    "seed_id": "cap_method_1-s000",
                    "method": "cap_method_1",
                    "seed_result": "/seed.json",
                }
            ],
        }
        variants = stage_b_variants(manifest, profiles=("full", "no_keyframes"))
        full, no_keyframes = variants
        complete = dict.fromkeys(FEEDBACK_ITEMS, "complete") | {
            "target_derivation": "complete",
            "controller_phases": "not_applicable",
            "action_context": "complete",
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            jobs = [
                self.branch(root, "a", full, "h", complete),
                self.branch(
                    root, "b", no_keyframes, "h", complete | {"keyframes": "disabled"}
                ),
            ]
            write(
                root / "manifest.json", {"experiment_variants": variants, "jobs": jobs}
            )
            self.assertEqual(audit(root), [])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            jobs = [
                self.branch(
                    root,
                    "a",
                    full,
                    "h",
                    complete | {"supporting_events": {"partial": 3, "available": 40}},
                ),
                self.branch(root, "b", no_keyframes, "other", complete),
            ]
            write(
                root / "manifest.json", {"experiment_variants": variants, "jobs": jobs}
            )
            issues = audit(root)
            self.assertTrue(any("supporting_events delivered" in i for i in issues))
            self.assertTrue(any("disabled keyframes" in i for i in issues))
            self.assertTrue(any("do not share one round-0" in i for i in issues))

    def test_exact_mcnemar(self):
        self.assertEqual(mcnemar_exact(0, 0), 1.0)
        self.assertAlmostEqual(mcnemar_exact(0, 6), 2 / 64)
        self.assertEqual(mcnemar_exact(3, 3), 1.0)


class ScaledQuotaTest(unittest.TestCase):
    def test_scaled_quotas_keep_total_and_proportions(self):
        self.assertEqual(scaled_quotas(50), QUOTAS)
        self.assertEqual(sum(scaled_quotas(70).values()), 70)
        self.assertEqual(scaled_quotas(70)["move_nonconvergence"], 21)
        self.assertEqual(
            scaled_quotas(3),
            {
                k: int(
                    k
                    in {
                        "move_nonconvergence",
                        "pick_nonconvergence",
                        "release_nonconvergence",
                    }
                )
                for k in QUOTAS
            },
        )


class StageARequestLimitTest(unittest.TestCase):
    @unittest.skipUnless(PRIOR.exists(), "needs the archived Revision 4.2 batch")
    def test_request_limit_admits_every_trial_concurrently(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "stage_a"
            manifest = create_stage_a(
                directory,
                model_name="qwen3.8-27b-fp8",
                flags=prompt_flags("0001"),
                trials=2,
                max_workers=6,
            )
            jobs = manifest["jobs"]
            limit = jobs[0]["transport_control"]["request_limit"]
            for job in jobs:
                others = [j for j in jobs if j is not job]
                decision = dispatch_reservation(
                    job, others, {}, request_limit=limit, free_remaining=limit
                )
                self.assertTrue(decision["allowed"])
            self.assertEqual(limit, len(jobs) * request_allowance(jobs[0]))


if __name__ == "__main__":
    unittest.main()
