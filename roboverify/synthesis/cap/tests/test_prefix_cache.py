"""Prefix-cache routing, measurement and startup verification (Revision 6, §3.6)."""

import io
import json
import unittest

from ..batch import prefix_cache_model
from ..budget import Budget
from ..config import ModelConfig
from ..llm import HTTPModel
from ..recovery import RoundBudget
from ..serving.prefix_cache import (
    AFFINITY_HEADER,
    BACKEND_HEADER,
    PrefixCacheUnavailable,
    affinity_key,
    cached_prompt_tokens,
    choose_backend,
    rank_backends,
    verify_prefix_caching,
)

BACKENDS = ["http://127.0.0.1:8126", "http://127.0.0.1:8127"]


class Handle(io.BytesIO):
    def __init__(self, payload, headers=None):
        super().__init__(json.dumps(payload).encode())
        self.headers = headers or {}


def completion(cached=None, backend=None):
    usage = {"prompt_tokens": 100, "total_tokens": 101}
    if cached is not None:
        usage["prompt_tokens_details"] = {"cached_tokens": cached}
    payload = {
        "choices": [{"message": {"content": "pass"}, "finish_reason": "stop"}],
        "usage": usage,
    }
    return Handle(payload, {BACKEND_HEADER: backend} if backend else {})


def vllm(**overrides):
    return ModelConfig(
        **{
            "provider": "vllm",
            "model": "m",
            "base_url": "http://127.0.0.1:8106/v1",
            "api_key_env": "",
        }
        | overrides
    )


class RoutingTests(unittest.TestCase):
    def test_requests_without_a_key_go_to_the_least_loaded_instance(self):
        self.assertEqual(choose_backend(BACKENDS, {BACKENDS[0]: 3}), BACKENDS[1])
        self.assertEqual(choose_backend(BACKENDS, {}), BACKENDS[0])

    def test_a_key_stays_on_one_instance_within_the_slack(self):
        key = affinity_key("batch", "job-1")
        preferred = rank_backends(BACKENDS, key)[0]
        other = next(b for b in BACKENDS if b != preferred)
        for load in ({}, {preferred: 2}, {preferred: 3, other: 1}):
            with self.subTest(load=load):
                self.assertEqual(choose_backend(BACKENDS, load, key), preferred)
        self.assertEqual(choose_backend(BACKENDS, {preferred: 3}, key), other)

    def test_keys_spread_over_instances_and_survive_removing_another(self):
        keys = [affinity_key("batch", f"job-{i}") for i in range(40)]
        chosen = [choose_backend(BACKENDS, {}, k) for k in keys]
        self.assertEqual(set(chosen), set(BACKENDS))
        three = BACKENDS + ["http://127.0.0.1:8128"]
        for key in keys:
            top = rank_backends(three, key)[0]
            if top != three[2]:
                self.assertEqual(rank_backends(BACKENDS, key)[0], top)

    def test_affinity_keys_are_stable_and_opaque(self):
        key = affinity_key("/runs/batch", "job-1")
        self.assertEqual(key, affinity_key("/runs/batch", "job-1"))
        self.assertNotIn("runs", key)
        self.assertEqual(len(key), 32)


class ClientTests(unittest.TestCase):
    def test_vllm_requests_carry_the_key_and_report_the_instance(self):
        requests = []

        def opener(request, timeout):
            requests.append(request)
            return completion(cached=64, backend=BACKENDS[1])

        response = HTTPModel(vllm(), opener, affinity_key="k1").generate("p", 5)
        self.assertEqual(requests[0].get_header(AFFINITY_HEADER.capitalize()), "k1")
        self.assertEqual(response.backend, BACKENDS[1])
        self.assertEqual(cached_prompt_tokens(response.usage), 64)

    def test_other_providers_never_receive_the_key(self):
        requests = []

        def opener(request, timeout):
            requests.append(request)
            return completion()

        config = ModelConfig(base_url="http://localhost/v1", api_key_env="")
        response = HTTPModel(config, opener, affinity_key="k1").generate("p", 5)
        self.assertIsNone(requests[0].get_header(AFFINITY_HEADER.capitalize()))
        self.assertIsNone(response.backend)

    def test_cached_tokens_are_none_when_the_server_does_not_report_them(self):
        self.assertIsNone(cached_prompt_tokens({"total_tokens": 3}))
        self.assertIsNone(cached_prompt_tokens(None))
        self.assertEqual(
            cached_prompt_tokens({"prompt_tokens_details": {"cached_tokens": 0}}), 0
        )

    def test_round_budgets_record_cache_use_on_the_trial_ledger(self):
        ledger = Budget()
        round_budget = RoundBudget()
        round_budget.parent = ledger
        round_budget.record_prefix_cache(80, BACKENDS[0])
        round_budget.record_prefix_cache(None, BACKENDS[0])
        for budget in (ledger, round_budget):
            self.assertEqual(
                budget.prefix_cache_summary(),
                {
                    "cached_prompt_tokens": 80,
                    "cache_unreported_calls": 1,
                    "served_by": {BACKENDS[0]: 2},
                },
            )
        self.assertEqual(ledger.tokens, 0)


class VerificationTests(unittest.TestCase):
    def server(self, cached_second):
        calls = []

        def opener(request, timeout):
            calls.append(request)
            if request.full_url.endswith("/balancer/status"):
                return Handle({"backends": BACKENDS})
            body = json.loads(request.data)
            self.assertEqual(body["max_tokens"], 1)
            previous = [
                c
                for c in calls[:-1]
                if c.full_url == request.full_url and c.data == request.data
            ]
            return completion(cached=cached_second if previous else 0)

        return opener, calls

    def test_every_instance_behind_the_balancer_is_probed_twice(self):
        opener, calls = self.server(cached_second=1024)
        record = verify_prefix_caching(vllm(), opener=opener)
        self.assertEqual(
            [p["backend"] for p in record["probes"]], [b + "/v1" for b in BACKENDS]
        )
        self.assertTrue(all(p["verified"] for p in record["probes"]))
        self.assertEqual(len(calls), 1 + 2 * len(BACKENDS))

    def test_no_reuse_or_no_report_refuses_the_batch(self):
        for cached in (0, None):
            with self.subTest(cached=cached):
                opener, _ = self.server(cached_second=cached)
                with self.assertRaisesRegex(PrefixCacheUnavailable, "8126"):
                    verify_prefix_caching(vllm(), opener=opener)

    def test_a_plain_endpoint_is_probed_as_one_instance(self):
        calls = []

        def opener(request, timeout):
            calls.append(request.full_url)
            if request.full_url.endswith("/balancer/status"):
                raise OSError("not a balancer")
            return completion(cached=512 if len(calls) > 2 else 0)

        record = verify_prefix_caching(vllm(), opener=opener)
        self.assertEqual(record["probes"][0]["backend"], "http://127.0.0.1:8106/v1")

    def test_only_live_vllm_demonstration_jobs_require_the_probe(self):
        on = {"demonstrations": {"enabled": True}}
        model = vllm().__dict__
        job = {"run_config": on, "model_config": model, "responses": None}
        self.assertEqual(prefix_cache_model([job]).model, "m")
        for changed in (
            {"run_config": {"demonstrations": {"enabled": False}}},
            {"run_config": {}},
            {"model_config": ModelConfig().__dict__},
            {"responses": ["pass"]},
            {"fixture": True},
        ):
            with self.subTest(changed=changed):
                self.assertIsNone(prefix_cache_model([job | changed]))
        with self.assertRaisesRegex(ValueError, "one model"):
            prefix_cache_model([job, job | {"model_config": vllm(model="n").__dict__}])


if __name__ == "__main__":
    unittest.main()
