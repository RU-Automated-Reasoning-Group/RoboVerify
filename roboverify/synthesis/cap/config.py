"""Explicit provider and experiment configuration; no credentials in artifacts."""

import math
import re
from dataclasses import dataclass, field

from .contracts import validate_condition_mode
from .prompt_package import validate_prompt_version


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
class RecoveryConfig:
    enabled: bool = False
    max_rounds: int = 4
    on_dsl_noncompliance: bool = False
    feedback_tokens: int = 8192
    round_tokens: int = 100000
    round_calls: int = 16
    round_timeout: float = 600.0
    total_tokens: int = 450000
    total_calls: int = 76
    total_steps: int = 50000
    total_sim_steps: int = 50000
    total_timeout: float = 3000.0
    context_tokens: int = 200000

    def __post_init__(self):
        if not 0 <= self.max_rounds <= 4:
            raise ValueError("Recovery permits zero to four additional rounds")
        for name, value in vars(self).items():
            if name not in {"enabled", "on_dsl_noncompliance", "max_rounds"}:
                if not math.isfinite(value) or value <= 0:
                    raise ValueError(f"Recovery {name} must be positive")


@dataclass
class PrimitivePromptConfig:
    explain_target_offsets: bool = False
    explain_convergence_criteria: bool = False
    explain_partial_execution: bool = False
    explain_motion_planning_limits: bool = False

    def __post_init__(self):
        if any(type(v) is not bool for v in vars(self).values()):
            raise ValueError("Primitive prompt options must be Boolean")


@dataclass
class PrimitiveFeedbackConfig:
    target_derivation: bool = False
    controller_phases: bool = False
    action_context: bool = False
    context_lookback: int = 16
    context_actions: int = 4

    def __post_init__(self):
        for name in ("target_derivation", "controller_phases", "action_context"):
            if type(getattr(self, name)) is not bool:
                raise ValueError("Primitive feedback options must be Boolean")
        for name in ("context_lookback", "context_actions"):
            value = getattr(self, name)
            if type(value) is not int or value < 1 or value > 128:
                raise ValueError("Primitive context limits must be integers in [1,128]")
        if self.context_actions > self.context_lookback:
            raise ValueError("Context action count exceeds its lookback")

    @property
    def enabled(self):
        return self.target_derivation or self.controller_phases or self.action_context


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
    prompt_version: str = "legacy_v1"
    recovery: RecoveryConfig = field(default_factory=RecoveryConfig)
    primitive_prompt: PrimitivePromptConfig = field(
        default_factory=PrimitivePromptConfig
    )
    primitive_feedback: PrimitiveFeedbackConfig = field(
        default_factory=PrimitiveFeedbackConfig
    )
    baseline_audit: bool = True
    audit_timeout: float = 10.0

    def __post_init__(self):
        if isinstance(self.recovery, dict):
            self.recovery = RecoveryConfig(**self.recovery)
        if isinstance(self.primitive_prompt, dict):
            self.primitive_prompt = PrimitivePromptConfig(**self.primitive_prompt)
        if isinstance(self.primitive_feedback, dict):
            self.primitive_feedback = PrimitiveFeedbackConfig(**self.primitive_feedback)
        validate_condition_mode(self.condition_mode)
        validate_prompt_version(self.prompt_version)
        if not math.isfinite(self.audit_timeout) or self.audit_timeout <= 0:
            raise ValueError("Audit timeout must be positive")


@dataclass
class GPUConfig:
    """Optional batch resource policy; UUIDs confer permission, load does not."""

    gpu_allowlist: list[str]
    reservation_directory: str
    gpu_policy: str = "auto"
    max_workers_per_gpu: int = 1
    min_free_memory_mb: float = 0
    worker_memory_mb: float = 0
    gpu_poll_interval_seconds: float = 2
    gpu_sample_window_seconds: float = 10
    admission_timeout_seconds: float = 60

    def __post_init__(self):
        if self.gpu_policy not in {"fixed", "auto"}:
            raise ValueError("gpu_policy must be fixed or auto")
        if (
            not self.gpu_allowlist
            or len(set(self.gpu_allowlist)) != len(self.gpu_allowlist)
            or any(
                not re.fullmatch(
                    r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", u
                )
                for u in self.gpu_allowlist
            )
        ):
            raise ValueError("gpu_allowlist must contain unique physical GPU UUIDs")
        if self.gpu_policy == "fixed" and len(self.gpu_allowlist) != 1:
            raise ValueError("Fixed mode requires exactly one GPU UUID")
        if not self.reservation_directory:
            raise ValueError("A shared reservation_directory is required")
        if type(self.max_workers_per_gpu) is not int or self.max_workers_per_gpu < 1:
            raise ValueError("max_workers_per_gpu must be a positive integer")
        for name in (
            "min_free_memory_mb",
            "worker_memory_mb",
            "gpu_poll_interval_seconds",
            "gpu_sample_window_seconds",
            "admission_timeout_seconds",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid GPU setting: {name}")
        if self.gpu_poll_interval_seconds <= 0 or self.admission_timeout_seconds <= 0:
            raise ValueError("GPU polling and admission timeout must be positive")
        if self.gpu_policy == "auto" and (
            self.min_free_memory_mb <= 0 or self.worker_memory_mb <= 0
        ):
            raise ValueError(
                "Auto mode requires measured memory headroom and worker reservation"
            )
