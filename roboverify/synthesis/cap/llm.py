"""Text-only model transports shared by both methods; no SDK coupling."""

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

from .contracts import CapError
from .timing import timed


class ModelError(CapError):
    pass


class TransportError(ModelError):
    def __init__(self, message, retryable=False):
        self.retryable = retryable
        super().__init__(message)


@dataclass
class Response:
    text: str
    usage: dict = field(default_factory=dict)
    finish_reason: str | None = None
    raw: dict = field(default_factory=dict)


class HTTPModel:
    """OpenRouter/OpenAI/vLLM chat API, plus a separate Responses wire adapter."""

    def __init__(self, config, opener=None):
        self.config = config
        self.opener = opener or urllib.request.urlopen

    def payload(self, prompt):
        c = self.config
        body = {"model": c.model, "stream": False}
        if c.api_style == "responses":
            body.update(
                input=[{"role": "user", "content": prompt}],
                max_output_tokens=c.max_output_tokens,
            )
        else:
            body.update(messages=[{"role": "user", "content": prompt}])
            body[c.token_field] = c.max_output_tokens
        if c.temperature is not None:
            body["temperature"] = c.temperature
        if c.seed is not None:
            body["seed"] = c.seed
        if set(body) & set(c.extra_body):
            raise ModelError(
                "Provider options cannot override normalized request fields"
            )
        if any(k in c.extra_body for k in ("tools", "messages", "input", "modalities")):
            raise ModelError("This client supports text code generation only")
        return body | c.extra_body

    def generate(self, prompt, timeout):
        c = self.config
        key = os.environ.get(c.api_key_env, "") if c.api_key_env else ""
        if c.api_key_env and not key:
            raise ModelError(
                f"Missing credential environment variable: {c.api_key_env}"
            )
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        route = "responses" if c.api_style == "responses" else "chat/completions"
        request = urllib.request.Request(
            c.base_url.rstrip("/") + "/" + route,
            data=json.dumps(self.payload(prompt)).encode(),
            headers=headers,
            method="POST",
        )
        try:
            with self.opener(
                request, timeout=min(timeout, c.request_timeout)
            ) as handle:
                raw = json.load(handle)
        except urllib.error.HTTPError as exc:
            detail = exc.read(8192).decode("utf-8", errors="replace")
            if key:
                detail = detail.replace(key, "[REDACTED]")
            for name in getattr(self.opener, "credential_envs", ()):
                credential = os.environ.get(name)
                if credential:
                    detail = detail.replace(credential, "[REDACTED]")
            raise TransportError(
                f"Model HTTP status {exc.code}: {detail[:1000]}",
                exc.code == 429 or exc.code >= 500,
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise TransportError(
                f"Model transport failed: {type(exc).__name__}", True
            ) from exc
        if c.api_style == "responses":
            text = "".join(
                part.get("text", "")
                for item in raw.get("output", [])
                if item.get("type") == "message"
                for part in item.get("content", [])
                if part.get("type") == "output_text"
            )
            reason = raw.get("status")
        else:
            choice = raw.get("choices", [{}])[0]
            text = choice.get("message", {}).get("content") or ""
            reason = choice.get("finish_reason")
        return Response(text, raw.get("usage") or {}, reason, raw)


LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def local_model_config(
    model,
    base_url,
    *,
    api_key_env="VLLM_API_KEY",
    thinking=True,
    max_output_tokens=16384,
    request_timeout=300.0,
    transport_retries=1,
):
    """Model settings for a local OpenAI-compatible vLLM server.

    Qwen-style chat templates take a thinking switch instead of OpenRouter's
    reasoning-effort levels. The key is read from ``api_key_env`` at request time.
    """
    from urllib.parse import urlparse

    from .config import ModelConfig

    if urlparse(base_url).hostname not in LOOPBACK_HOSTS:
        raise ValueError("Local model mode requires a loopback base_url")
    return ModelConfig(
        provider="vllm",
        model=model,
        base_url=base_url,
        api_key_env=api_key_env,
        max_output_tokens=max_output_tokens,
        request_timeout=request_timeout,
        transport_retries=transport_retries,
        extra_body={"chat_template_kwargs": {"enable_thinking": thinking}},
    )


def server_provenance(config, opener=None, timeout=10):
    """Served model identity and vLLM version, recorded without credentials."""
    opener = opener or urllib.request.urlopen
    key = os.environ.get(config.api_key_env, "") if config.api_key_env else ""
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    base = config.base_url.rstrip("/")

    def fetch(url):
        request = urllib.request.Request(url, headers=headers, method="GET")
        try:
            with opener(request, timeout=timeout) as handle:
                return json.load(handle)
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            raise ModelError(
                f"Local server provenance unavailable: {type(exc).__name__}"
            ) from exc

    models = fetch(base + "/models")
    root = base[: -len("/v1")] if base.endswith("/v1") else base
    try:
        version = fetch(root + "/version").get("version")
    except ModelError:
        version = None
    return {
        "base_url": config.base_url,
        "requested_model": config.model,
        "server_version": version,
        "served_models": [
            {k: m.get(k) for k in ("id", "root", "max_model_len", "owned_by")}
            for m in models.get("data", [])
        ],
        "extra_body": config.extra_body,
    }


class ScriptedModel:
    """Deterministic responses for tests and explicit offline runs."""

    def __init__(self, responses):
        self.responses = iter(responses)
        self.prompts = []

    def payload(self, prompt):
        return {"model": "scripted", "prompt": prompt}

    def generate(self, prompt, timeout):
        self.prompts.append(prompt)
        try:
            text = next(self.responses)
        except StopIteration as exc:
            raise ModelError("Scripted responses exhausted") from exc
        return Response(text, {}, "stop", {"text": text, "model": "scripted"})


class LoggedModel:
    def __init__(self, backend, logger, budget, retries=0, recovery_feedback=None):
        self.backend, self.logger, self.budget = backend, logger, budget
        self.retries = retries
        self.recovery_feedback = recovery_feedback

    @timed("generation_control")
    def complete(self, prompt, *, purpose, **context):
        is_policy = purpose == "policy" or (
            purpose == "google_generation" and not self.logger.context.get("helper")
        )
        if self.recovery_feedback is not None and is_policy:
            from .recovery import recovery_prompt

            prompt = recovery_prompt(prompt, self.recovery_feedback)
        for retry in range(self.retries + 1):
            if hasattr(self.budget, "reserve_request"):
                reservation = self.budget.reserve_request(prompt)
                self.logger.emit("request_reservation", **reservation)

            self.budget.model_call()
            call = f"call_{self.budget.calls:04d}"
            path = f"lmp/{call}"
            request = self.backend.payload(prompt)
            role = purpose
            if purpose == "google_generation":
                role = "helper" if self.logger.context.get("helper") else "policy"
            if self.recovery_feedback is not None and is_policy:
                role = "recovery_policy"
            provenance = {
                **self.logger.prompt_provenance,
                "call_id": call,
                "role": role,
                "retry": retry,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "request_sha256": hashlib.sha256(
                    json.dumps(
                        request,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=True,
                    ).encode()
                ).hexdigest(),
                "request_hash_encoding": "json-sort-keys-compact-ascii-utf8",
            }
            self.logger.artifact(f"{path}/prompt.txt", prompt)
            self.logger.artifact(f"{path}/request.json", request)
            metadata_path = self.logger.artifact(
                f"{path}/prompt_metadata.json", provenance
            )
            event = self.logger.emit(
                "generation_start",
                purpose=purpose,
                call_id=call,
                retry=retry,
                **context,
            )
            start = time.monotonic()
            try:
                response = self.budget.timings.call(
                    "model_" + role,
                    self.backend.generate,
                    prompt,
                    self.budget.remaining(),
                )
                request_elapsed = time.monotonic() - start
            except TransportError as exc:
                if hasattr(self.budget, "abandon_request"):
                    self.budget.abandon_request()
                self.logger.emit(
                    "generation_error",
                    call_id=call,
                    error=str(exc),
                    elapsed=time.monotonic() - start,
                    parent_event_id=event,
                )
                if not exc.retryable or retry == self.retries:
                    raise
                self.budget.timings.call(
                    "transport_backoff",
                    time.sleep,
                    min(2**retry, self.budget.remaining()),
                )
                continue
            except Exception as exc:
                self.logger.emit(
                    "generation_error",
                    call_id=call,
                    error=str(exc),
                    elapsed=time.monotonic() - start,
                    parent_event_id=event,
                )
                raise
            self.logger.artifact(f"{path}/response.json", response.raw)
            self.logger.artifact(f"{path}/response.txt", response.text)
            self.logger.emit(
                "generation_end",
                **(
                    context
                    | dict(
                        call_id=call,
                        purpose=purpose,
                        parent_event_id=event,
                        elapsed=request_elapsed,
                        usage=response.usage,
                        finish_reason=response.finish_reason,
                    )
                ),
            )
            self.budget.account_tokens(response.usage.get("total_tokens"))
            if hasattr(self.budget, "reserve_request"):
                self.logger.emit(
                    "request_reconciliation",
                    reservation=reservation,
                    actual_tokens=response.usage.get("total_tokens"),
                    charged_tokens=self.budget.charged_tokens,
                    cumulative_charged_tokens=self.budget.parent.charged_tokens,
                )

            self.budget.remaining()
            if (
                response.finish_reason in {"length", "incomplete", "content_filter"}
                or not response.text.strip()
            ):
                raise ModelError(
                    f"Incomplete/empty model response: {response.finish_reason}"
                )
            source = extract_source(response.text)
            source_key = hashlib.sha256(source.strip().encode()).hexdigest()
            self.logger.source_generations.setdefault(source_key, []).append(
                {
                    **provenance,
                    "metadata_path": self.logger.relative(metadata_path),
                    "returned_source_sha256": hashlib.sha256(
                        source.encode()
                    ).hexdigest(),
                    "match": "outer_whitespace_ignored",
                }
            )
            return source


def extract_source(text):
    """Remove one surrounding Markdown fence, without silently rewriting code."""
    text = text.strip()
    lines = text.splitlines()
    if (
        len(lines) >= 2
        and lines[0].strip() in {"```python", "```py", "```"}
        and lines[-1].strip() == "```"
    ):
        return "\n".join(lines[1:-1]) + "\n"
    return text + "\n"
