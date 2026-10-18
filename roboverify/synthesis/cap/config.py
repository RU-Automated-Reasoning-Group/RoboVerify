"""Explicit provider and experiment configuration; no credentials in artifacts."""

import re
from dataclasses import dataclass, field

from .contracts import validate_condition_mode


@dataclass
class ModelConfig:
    provider: str = "openrouter"
    model: str = "deepseek/deepseek-v4.1-flash"
    base_url: str = "https://openrouter.ai/api/v1"
    api_key_env: str = "OPENROUTER_API_KEY"
    api_style: str = "chat_completions"
    max_output_tokens: int = 4096
    temperature: float | None = None
    seed: int | None = None
    token_field: str = "max_tokens"
    request_timeout: float = 45.0
    transport_retries: int = 2
    extra_body: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.api_key_env and not re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]*", self.api_key_env
        ):
            raise ValueError(
                "api_key_env must be an environment-variable name, not a credential value"
            )
        if self.api_style not in {"chat_completions", "responses"}:
            raise ValueError("Unknown API style")
        if self.token_field not in {"max_tokens", "max_completion_tokens"}:
            raise ValueError("Unsupported output-token parameter")
        if (
            not self.model
            or self.max_output_tokens <= 0
            or self.request_timeout <= 0
            or self.transport_retries < 0
        ):
            raise ValueError("Invalid model configuration")


@dataclass
class RunConfig:
    method: str = "cap_method_1"
    task: str = "stack"
    num_blocks: int = 3
    seed: int = 0
    timeout: float = 600.0
    max_attempts: int = 3
    max_model_calls: int = 30
    max_steps: int = 10000
    max_depth: int = 12
    max_tokens: int = 100000
    higher_tolerance: float = 0.001
    save_video: bool = True
    output_root: str = "runs"
    condition_mode: str = "dsl_only"
    max_sim_steps: int = 10000

    def __post_init__(self):
        validate_condition_mode(self.condition_mode)
