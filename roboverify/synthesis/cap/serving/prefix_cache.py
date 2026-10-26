"""Prefix-cache routing and verification for local vLLM serving (Revision 6, §3.6).

The demonstration block is an identical prefix of every request of a trial, so
vLLM's prefix cache can reuse its prefill. Three pieces keep that reuse
effective and measured without changing any request:

- an affinity key per trial, sent as ``AFFINITY_HEADER``; the balancer
  (``serving/balancer.py``) keeps a key on one instance via ``choose_backend``;
- ``cached_prompt_tokens`` reads the server-reported cached prompt tokens
  (``--enable-prompt-tokens-details``);
- ``verify_prefix_caching`` probes every instance before a batch dispatches.

Nothing here runs inside vLLM, and none of it alters prompts or sampling.
"""

import hashlib
import json
import os
import secrets
import time
import urllib.error
import urllib.request

AFFINITY_HEADER = "X-CaP-Affinity"
BACKEND_HEADER = "X-CaP-Backend"
VERSION = "cap-prefix-cache-v1"
# Least-loaded fallback once the preferred instance has this many more requests
# in flight than the least-loaded one.
DEFAULT_SLACK = 2
# Required server flags (``vllm serve``); recorded with every probe.
SERVE_FLAGS = ("--enable-prefix-caching", "--enable-prompt-tokens-details")
PROBE_FILLER_WORDS = 1500


class PrefixCacheUnavailable(RuntimeError):
    pass


def affinity_key(*parts):
    """A stable, opaque key for one trial (no paths or names leave the client)."""
    return hashlib.sha256("\0".join(map(str, parts)).encode()).hexdigest()[:32]


def rank_backends(backends, key):
    """Rendezvous (highest-random-weight) order of ``backends`` for ``key``."""
    return sorted(
        backends,
        key=lambda b: hashlib.sha256(f"{key}\0{b}".encode()).digest(),
        reverse=True,
    )


def choose_backend(backends, inflight, key=None, slack=DEFAULT_SLACK):
    """The instance for one request.

    Without a key: least in flight (ties in list order). With a key: the
    key's top-ranked instance, unless it has more than ``slack`` requests in
    flight above the least-loaded instance.
    """
    if not backends:
        raise ValueError("No backend available")
    least = min(backends, key=lambda b: inflight.get(b, 0))
    if key is None:
        return least
    preferred = rank_backends(backends, key)[0]
    if inflight.get(preferred, 0) > inflight.get(least, 0) + slack:
        return least
    return preferred


def cached_prompt_tokens(usage):
    """Server-reported cached prompt tokens, or None when not reported."""
    details = (usage or {}).get("prompt_tokens_details") or {}
    value = details.get("cached_tokens")
    return int(value) if value is not None else None


def _request(url, key, body=None, timeout=60, opener=None):
    opener = opener or urllib.request.urlopen
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode() if body is not None else None,
        headers=headers,
        method="POST" if body is not None else "GET",
    )
    with opener(request, timeout=timeout) as handle:
        return json.load(handle)


def endpoint_backends(model_config, opener=None, timeout=10):
    """Instances behind ``model_config.base_url``.

    A CaP balancer lists them at ``/balancer/status``; any other endpoint is
    treated as a single instance at ``base_url``.
    """
    base = model_config.base_url.rstrip("/")
    root = base[: -len("/v1")] if base.endswith("/v1") else base
    try:
        status = _request(
            root + "/balancer/status", None, timeout=timeout, opener=opener
        )
        listed = [b.rstrip("/") + "/v1" for b in status.get("backends", [])]
    except (urllib.error.URLError, TimeoutError, OSError, ValueError):
        listed = []
    return listed or [base]


def probe_backend(url, model_config, opener=None, timeout=120):
    """Send one nonce-prefixed prompt twice; report the second's cached tokens."""
    key = (
        os.environ.get(model_config.api_key_env, "") if model_config.api_key_env else ""
    )
    nonce = secrets.token_hex(8)
    words = " ".join(f"w{i % 97}" for i in range(PROBE_FILLER_WORDS))
    body = {
        "model": model_config.model,
        "messages": [
            {"role": "user", "content": f"Probe {nonce}. {words}\nReply with OK."}
        ],
        "max_tokens": 1,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    usages = []
    for _ in range(2):
        start = time.monotonic()
        raw = _request(url + "/chat/completions", key, body, timeout, opener)
        usages.append(
            {
                "prompt_tokens": (raw.get("usage") or {}).get("prompt_tokens"),
                "cached_tokens": cached_prompt_tokens(raw.get("usage")),
                "seconds": round(time.monotonic() - start, 3),
            }
        )
    cached = usages[1]["cached_tokens"]
    return {
        "backend": url,
        "requests": usages,
        "cached_tokens_reported": cached is not None,
        "verified": bool(cached),
    }


def verify_prefix_caching(model_config, opener=None):
    """Probe every instance; raise unless each one reuses and reports a prefix."""
    try:
        backends = endpoint_backends(model_config, opener=opener)
        probes = [probe_backend(b, model_config, opener=opener) for b in backends]
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        raise PrefixCacheUnavailable(
            f"Prefix-cache probe failed: {type(exc).__name__}"
        ) from exc
    record = {
        "version": VERSION,
        "checked_epoch": time.time(),
        "required_serve_flags": list(SERVE_FLAGS),
        "probes": probes,
    }
    failed = [p["backend"] for p in probes if not p["verified"]]
    if failed:
        raise PrefixCacheUnavailable(
            "Prefix caching not verified on "
            + ", ".join(failed)
            + f"; serve with {' '.join(SERVE_FLAGS)}"
        )
    return record
