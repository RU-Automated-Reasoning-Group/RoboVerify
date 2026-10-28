"""Revision 7.5: provider prompt caching without changing any prompt text."""

import contextlib
import io
import json
import tempfile
import unittest
import urllib.error
from dataclasses import asdict, replace
from pathlib import Path

from synthesis.cfg.reset import restore

from ..batch import create_batch, execute_batch
from ..config import ModelConfig, RecoveryConfig, RunConfig
from ..configs import config_final_v1 as v1
from ..configs import config_final_v2 as v2
from ..demonstrations import DemonstrationError, count_tokens, measure_once, prepare
from ..llm import (
    RATE_LIMIT_BACKOFF_SECONDS,
    HTTPModel,
    ModelError,
    TransportError,
    cache_parts,
)
from ..runner import run
from .test_adapter import RobotFixture
from .test_demonstrations import Fixture
from .test_final_release import install_contacts

MARK = {"type": "ephemeral"}


class RecordingHTTP(HTTPModel):
    """HTTPModel whose transport returns scripted chat replies with cache usage."""

    def __init__(self, config, responses):
        super().__init__(config)
        self.responses, self.bodies = iter(responses), []

    def post(self, route, body, timeout, headers=None):
        self.bodies.append(body)
        usage = {
            "prompt_tokens": 100,
            "completion_tokens": 10,
            "total_tokens": 110,
            "cost": 0.5,
            "prompt_tokens_details": {"cached_tokens": 40, "cache_write_tokens": 60},
        }
        return {
            "choices": [
                {"message": {"content": next(self.responses)}, "finish_reason": "stop"}
            ],
            "usage": usage,
            "provider": "Anthropic",
        }


def text(content):
    return content if isinstance(content, str) else "".join(p["text"] for p in content)


class PartsTests(unittest.TestCase):
    def test_parts_concatenate_to_the_prompt(self):
        prompt = "demo block\nmethod prompt\n\nRECOVERY feedback"
        parts = cache_parts(prompt, (11, 25), "5m")
        self.assertEqual(text(parts), prompt)
        self.assertEqual([p.get("cache_control") for p in parts], [MARK, MARK, None])
        self.assertEqual(parts[0]["text"], "demo block\n")
        whole = cache_parts(prompt, (len(prompt), 0, 99), "1h")
        self.assertEqual(
            whole,
            [{"type": "text", "text": prompt, "cache_control": MARK | {"ttl": "1h"}}],
        )
        with self.assertRaises(ModelError):
            cache_parts(prompt, (1, 2, 3, 4, 5), "5m")

    def test_payload_uses_parts_only_with_prompt_cache(self):
        plain = HTTPModel(ModelConfig()).payload("abc", (1,))
        self.assertEqual(plain["messages"][0]["content"], "abc")
        cached = HTTPModel(ModelConfig(prompt_cache="5m")).payload("abc", (1,))
        self.assertEqual(text(cached["messages"][0]["content"]), "abc")
        self.assertEqual(
            HTTPModel(ModelConfig(prompt_cache="5m")).payload("abc"),
            HTTPModel(ModelConfig()).payload("abc"),
        )
        with self.assertRaises(ValueError):
            ModelConfig(prompt_cache="10m")
        with self.assertRaises(ValueError):
            ModelConfig(prompt_cache="5m", api_style="responses")


class RateLimitTests(unittest.TestCase):
    def raise_429(self, headers):
        def opener(request, timeout):
            raise urllib.error.HTTPError(
                request.full_url, 429, "busy", headers, io.BytesIO(b"{}")
            )

        return HTTPModel(ModelConfig(api_key_env=""), opener)

    def test_429_waits_for_retry_after_or_the_default(self):
        for headers, expected in (
            ({}, RATE_LIMIT_BACKOFF_SECONDS),
            ({"Retry-After": "7"}, 7.0),
        ):
            with self.assertRaises(TransportError) as caught:
                self.raise_429(headers).generate("x", 10)
            self.assertTrue(caught.exception.retryable)
            self.assertEqual(caught.exception.retry_after, expected)


class FinalV2Tests(unittest.TestCase):
    def test_only_request_and_batch_settings_change(self):
        for task in v2.TASKS:
            self.assertEqual(
                asdict(v2.run_config(task, seed=3)), asdict(v1.run_config(task, seed=3))
            )
            self.assertEqual(v2.instruction(task), v1.instruction(task))
        old, new = asdict(v1.model_config()), asdict(v2.model_config())
        changed = {k for k in new if old[k] != new[k]}
        self.assertEqual(changed, {"prompt_cache", "extra_body"})
        self.assertEqual(new["prompt_cache"], "5m")
        self.assertEqual(
            new["extra_body"], old["extra_body"] | {"provider": v2.PROVIDER_ROUTING}
        )
        self.assertTrue(v2.DEMONSTRATION_TOKENS_ONCE)
        self.assertEqual(v2.BATCH, {"cache_warmup": True})


class MeasuredTokenTests(Fixture):
    def test_a_measured_count_is_reused_only_for_its_block(self):
        config = self.config()
        measured = measure_once(config, ModelConfig())
        self.assertEqual(measured["measurement"], {"method": "utf8_bytes_upper_bound"})
        reused = replace(
            config,
            demonstrations=replace(
                config.demonstrations,
                token_counter="provider_usage",
                measured_tokens=measured,
            ),
        )
        # provider_usage would need a network request; the stored count is used.
        context = prepare(reused, ModelConfig(api_key_env=""))
        self.assertEqual(context.tokens, measured["tokens"])
        self.assertEqual(context.sha256, measured["sha256"])
        self.assertTrue(context.manifest["token_measurement"]["measured_once"])
        wrong = replace(
            reused.demonstrations, measured_tokens=measured | {"sha256": "0" * 64}
        )
        with self.assertRaises(DemonstrationError):
            count_tokens(context.text, wrong, ModelConfig())
        with self.assertRaises(ValueError):
            replace(config.demonstrations, measured_tokens={"sha256": "x", "tokens": 0})


class RequestTests(Fixture):
    """Round 0 with one DSL repair, then a recovery round, with demonstrations."""

    RESPONSES = ["if Pick(b0):\n    pass\n", "pass", "pass"]

    def factory(self, config, snapshot):
        env = RobotFixture(config.num_blocks, success=True)
        if snapshot:
            restore(env, snapshot)
        held = [("robot0:l_gripper_finger_link", "object0")] if self.first else []
        self.first = False
        install_contacts(env, held)
        return env

    def execute(self, prompt_cache, **demonstrations):
        self.first = True
        config = replace(
            self.config(**demonstrations),
            method="cap_method_2",
            post_settling_steps=50,
            recovery=RecoveryConfig(
                enabled=True,
                max_rounds=1,
                success_field="task_spec_success",
                feedback={"evidence_layout": "fixed_items"},
            ),
        )
        backend = RecordingHTTP(
            ModelConfig(prompt_cache=prompt_cache, max_output_tokens=16384),
            self.RESPONSES,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            result = run(
                config,
                backend.config,
                backend=backend,
                environment_factory=self.factory,
                capture_stdout=False,
            )
        return result, backend.bodies

    def test_cached_requests_split_the_same_text_at_shared_prefixes(self):
        result, bodies = self.execute("5m")
        self.assertEqual(result["attempts"][0]["status"], "unreleased_block")
        self.assertEqual(result["recovery_rounds"], 1)
        plain_result, plain = self.execute(None)
        self.assertEqual(len(bodies), 3)
        contents = [b["messages"][0]["content"] for b in bodies]
        # Byte-identical text, with and without caching. Recovery feedback
        # names random object handles, so only its shared prefix is compared.
        plain = [b["messages"][0]["content"] for b in plain]
        self.assertEqual([text(c) for c in contents[:2]], plain[:2])
        self.assertTrue(plain[2].startswith(text(contents[0]) + "\n\nRECOVERY"))
        policy, repair, recovery = contents
        demo = policy[0]["text"]
        self.assertEqual(demo, prepare(self.config(), ModelConfig()).text)
        self.assertEqual([p.get("cache_control") for p in policy], [MARK, MARK])
        self.assertEqual([p.get("cache_control") for p in repair], [MARK, None])
        self.assertEqual(repair[0]["text"], demo)
        self.assertEqual([p.get("cache_control") for p in recovery], [MARK, MARK, None])
        self.assertEqual(recovery[0]["text"] + recovery[1]["text"], text(policy))
        self.assertTrue(recovery[2]["text"].startswith("\n\nRECOVERY REQUEST"))
        cache = result["prefix_cache"]
        self.assertEqual(cache["cached_prompt_tokens"], 120)
        self.assertEqual(cache["cache_write_tokens"], 180)
        self.assertEqual(cache["provider_cost_usd"], 1.5)
        self.assertEqual(cache["served_by"], {"Anthropic": 3})
        root = Path(result["attempts"][0]["run_dir"]) / "artifacts/trial_001/lmp"
        metadata = json.loads((root / "call_0001/prompt_metadata.json").read_text())
        self.assertEqual(metadata["cache_breakpoints"], [len(demo), len(text(policy))])
        self.assertEqual((root / "call_0001/prompt.txt").read_text(), text(policy))

    def test_the_cache_summary_is_reported_without_demonstrations(self):
        result, bodies = self.execute("5m", enabled=False)
        self.assertNotIn("demonstration_tokens", result)
        # No demonstration breakpoint: round 0 is one cached part.
        self.assertEqual(
            [p.get("cache_control") for p in bodies[0]["messages"][0]["content"]],
            [MARK],
        )
        self.assertEqual(result["prefix_cache"]["provider_cost_usd"], 1.5)
        attempt = result["attempts"][0]
        self.assertEqual(attempt["prefix_cache"]["cached_prompt_tokens"], 80)


class WarmupTests(unittest.TestCase):
    def test_other_workers_wait_for_the_first_response(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "batch"
            manifest = create_batch(
                root,
                RunConfig(num_blocks=1, save_video=False, prompt_version="typed_v3"),
                ModelConfig(),
                methods=("cap_method_2",),
                modes=("dsl_only",),
                trials=3,
                max_workers=3,
                responses=["Pick(b0)"],
                fixture=True,
                cache_warmup=True,
            )
            self.assertTrue(manifest["cache_warmup"])
            with contextlib.redirect_stdout(io.StringIO()):
                summary = execute_batch(root)
            self.assertEqual(summary["finished"], 3)
            session = json.loads((root / "execution_metadata.json").read_text())[
                "sessions"
            ][-1]
            released = session["cache_warmup"]["released_epoch"]
            schedule = json.loads((root / "schedule.json").read_text())
            first = session["cache_warmup"]["job"]
            for ident, state in schedule.items():
                if ident != first:
                    self.assertGreaterEqual(state["dispatched_epoch"], released)


if __name__ == "__main__":
    unittest.main()
