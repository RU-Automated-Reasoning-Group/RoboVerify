"""Explicit provider and experiment configuration; no credentials in artifacts."""

import math
import re
from dataclasses import dataclass, field

from .contracts import validate_condition_mode
from .prompt_package import package, validate_prompt_version


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
        if self.provider == "vllm" and "reasoning" in self.extra_body:
            raise ValueError(
                "vLLM ignores OpenRouter reasoning options; use chat_template_kwargs"
            )
        if (
            not self.model
            or self.max_output_tokens <= 0
            or self.request_timeout <= 0
            or self.transport_retries < 0
        ):
            raise ValueError("Invalid model configuration")


FEEDBACK_ITEMS = (
    "current_source",
    "failure_location",
    "failure_events",
    "query_candidates",
    "keyframes",
    "goal_clauses",
    "dsl_diagnostics",
    "supporting_events",
    "event_counts",
    "selection_metadata",
)


def _nested(cls, value, label):
    if isinstance(value, cls):
        return value
    if not isinstance(value, dict):
        raise ValueError(f"Recovery {label} must be an object")
    unknown = set(value) - set(cls.__dataclass_fields__)
    if unknown:
        raise ValueError(f"Unknown recovery {label} settings: {sorted(unknown)}")
    return cls(**value)


@dataclass
class RecoveryFeedbackConfig:
    """Which recorded evidence items a recovery request delivers (Revision 4.3)."""

    current_source: bool = True
    failure_location: bool = True
    failure_events: bool = True
    query_candidates: bool = True
    keyframes: bool = True
    goal_clauses: bool = True
    dsl_diagnostics: bool = True
    supporting_events: bool = True
    supporting_event_window: int = 80
    event_counts: bool = True
    selection_metadata: bool = True
    evidence_layout: str = "legacy_fill"
    # None keeps RecoveryConfig.feedback_tokens as the evidence cap.
    evidence_bytes: int | None = None

    def __post_init__(self):
        for name in FEEDBACK_ITEMS:
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"Recovery feedback {name} must be Boolean")
        window = self.supporting_event_window
        if type(window) is not int or not 0 <= window <= 512:
            raise ValueError("supporting_event_window must be an integer in [0,512]")
        if self.evidence_layout not in {"legacy_fill", "fixed_items"}:
            raise ValueError("evidence_layout must be legacy_fill or fixed_items")
        if self.evidence_bytes is not None and (
            type(self.evidence_bytes) is not int or self.evidence_bytes <= 0
        ):
            raise ValueError("evidence_bytes must be a positive integer or null")

    def disabled(self):
        return [name for name in FEEDBACK_ITEMS if not getattr(self, name)]


@dataclass
class RecoveryHistoryConfig:
    """How earlier failed attempts of the same trial enter a recovery request."""

    mode: str = "summaries"
    edit_ledger: bool = True
    source_diff: bool = False
    source_bytes: int = 16384

    def __post_init__(self):
        if self.mode not in {"none", "summaries", "executed_programs"}:
            raise ValueError(
                "History mode must be none, summaries or executed_programs"
            )
        if type(self.edit_ledger) is not bool or type(self.source_diff) is not bool:
            raise ValueError("History ledger/diff options must be Boolean")
        if self.source_diff and self.mode != "executed_programs":
            raise ValueError("source_diff requires executed_programs history")
        if type(self.source_bytes) is not int or self.source_bytes <= 0:
            raise ValueError("History source_bytes must be a positive integer")


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
    # Attempt field that ends recovery. "task_success" preserves cap-recovery
    # behavior; "task_spec_success" uses RoboVerify's criterion (Revision 5).
    success_field: str = "task_success"
    feedback: RecoveryFeedbackConfig = field(default_factory=RecoveryFeedbackConfig)
    history: RecoveryHistoryConfig = field(default_factory=RecoveryHistoryConfig)

    def __post_init__(self):
        self.feedback = _nested(RecoveryFeedbackConfig, self.feedback, "feedback")
        self.history = _nested(RecoveryHistoryConfig, self.history, "history")
        if not 0 <= self.max_rounds <= 4:
            raise ValueError("Recovery permits zero to four additional rounds")
        for name, value in vars(self).items():
            if name not in {
                "enabled",
                "on_dsl_noncompliance",
                "max_rounds",
                "success_field",
                "feedback",
                "history",
            }:
                if not math.isfinite(value) or value <= 0:
                    raise ValueError(f"Recovery {name} must be positive")
        if self.success_field not in {"task_success", "task_spec_success"}:
            raise ValueError(
                "Recovery success_field must be task_success or task_spec_success"
            )
        if (
            self.history.mode == "executed_programs"
            and not self.feedback.current_source
        ):
            raise ValueError("executed_programs history requires current_source")

    @property
    def evidence_cap(self):
        return self.feedback.evidence_bytes or self.feedback_tokens

    @property
    def default_profile(self):
        """True when delivered feedback must reproduce cap-recovery-v1 exactly."""
        return (
            self.feedback == RecoveryFeedbackConfig()
            and self.history.mode == "summaries"
        )


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


DEMONSTRATION_TOKEN_COUNTERS = ("server_tokenize", "utf8_bytes")


@dataclass
class DemonstrationConfig:
    """Demonstration input (Revision 6); off keeps every prompt unchanged.

    ``archive`` is a ``demonstrations.npz`` written by synthesis.cfg.recordings.
    ``trace_ids`` selects and orders its trajectories (``"d0"``, ``"d1"``, ...;
    null selects all in archive order). ``archive_sha256`` optionally pins the
    archive bytes. ``waypoint_tolerance`` is the RDP tolerance in metres for
    gripper-path waypoints. ``max_tokens`` limits the rendered block; exceeding
    it fails the trial. ``token_counter`` measures the block once per trial:
    ``server_tokenize`` uses the vLLM ``/tokenize`` endpoint (``tokenizer_url``,
    default derived from the model base URL); ``utf8_bytes`` is a conservative
    byte bound for offline diagnostics.
    """

    enabled: bool = False
    archive: str | None = None
    trace_ids: list[str] | None = None
    archive_sha256: str | None = None
    waypoint_tolerance: float = 0.01
    max_tokens: int = 131072
    token_counter: str = "server_tokenize"
    tokenizer_url: str | None = None

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError("demonstrations.enabled must be Boolean")
        tolerance = self.waypoint_tolerance
        if (
            isinstance(tolerance, bool)
            or not isinstance(tolerance, (int, float))
            or not math.isfinite(tolerance)
            or tolerance <= 0
        ):
            raise ValueError("demonstrations.waypoint_tolerance must be finite and > 0")
        self.waypoint_tolerance = float(tolerance)
        if type(self.max_tokens) is not int or self.max_tokens <= 0:
            raise ValueError("demonstrations.max_tokens must be a positive integer")
        if self.token_counter not in DEMONSTRATION_TOKEN_COUNTERS:
            raise ValueError(
                "demonstrations.token_counter must be server_tokenize or utf8_bytes"
            )
        if self.archive is not None and (
            not isinstance(self.archive, str) or not self.archive.endswith(".npz")
        ):
            raise ValueError("demonstrations.archive must be an .npz path")
        if self.trace_ids is not None:
            ids = list(self.trace_ids)
            if (
                not ids
                or len(set(ids)) != len(ids)
                or not all(
                    isinstance(i, str) and re.fullmatch(r"d(0|[1-9][0-9]*)", i)
                    for i in ids
                )
            ):
                raise ValueError(
                    "demonstrations.trace_ids must be unique ids d0, d1, ... or null"
                )
            self.trace_ids = ids
        if self.archive_sha256 is not None and not re.fullmatch(
            r"[0-9a-f]{64}", str(self.archive_sha256)
        ):
            raise ValueError("demonstrations.archive_sha256 must be a SHA-256 digest")
        if self.tokenizer_url is not None and not re.match(
            r"https?://", str(self.tokenizer_url)
        ):
            raise ValueError("demonstrations.tokenizer_url must be an HTTP(S) URL")
        if self.enabled and not self.archive:
            raise ValueError("Enabled demonstrations require an archive")


TASK_OPTIONS = {
    "initial_tower_blocks": {"restack"},
    "grid_rows": {"grid"},
    "grid_cols": {"grid"},
    "bottom_row_blocks": {"pyramid"},
}


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
    save_video: bool | None = None
    video_mode: str | None = None
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
    demonstrations: DemonstrationConfig = field(default_factory=DemonstrationConfig)
    baseline_audit: bool = True
    audit_timeout: float = 10.0
    # Task layout options, resolved by the upstream resolvers (Revision 5).
    initial_tower_blocks: int | None = None
    grid_rows: int | None = None
    grid_cols: int | None = None
    bottom_row_blocks: int | None = None

    def __post_init__(self):
        for name, tasks in TASK_OPTIONS.items():
            if getattr(self, name) is not None and self.task not in tasks:
                raise ValueError(f"{name} applies only to {'/'.join(sorted(tasks))}")
        if isinstance(self.recovery, dict):
            self.recovery = _nested(RecoveryConfig, self.recovery, "configuration")
        if isinstance(self.primitive_prompt, dict):
            self.primitive_prompt = PrimitivePromptConfig(**self.primitive_prompt)
        if isinstance(self.primitive_feedback, dict):
            self.primitive_feedback = PrimitiveFeedbackConfig(**self.primitive_feedback)
        if isinstance(self.demonstrations, dict):
            unknown = set(self.demonstrations) - set(
                DemonstrationConfig.__dataclass_fields__
            )
            if unknown:
                raise ValueError(f"Unknown demonstration settings: {sorted(unknown)}")
            self.demonstrations = DemonstrationConfig(**self.demonstrations)
        if not isinstance(self.demonstrations, DemonstrationConfig):
            raise ValueError("demonstrations must be an object")
        if self.save_video is not None and type(self.save_video) is not bool:
            raise ValueError("save_video must be Boolean or omitted")
        if self.video_mode is None:
            self.video_mode = (
                "deferred"
                if self.save_video is None
                else "inline" if self.save_video else "off"
            )
        if self.video_mode not in {"inline", "deferred", "off"}:
            raise ValueError("video_mode must be inline, deferred, or off")
        if self.save_video is not None and self.save_video != (
            self.video_mode != "off"
        ):
            raise ValueError("Conflicting save_video and video_mode settings")
        self.save_video = self.video_mode != "off"
        validate_condition_mode(self.condition_mode)
        validate_prompt_version(self.prompt_version)
        if self.demonstrations.enabled and "demonstrations" not in package(
            self.prompt_version
        ):
            raise ValueError(
                f"Prompt package {self.prompt_version} has no demonstration"
                " templates; use typed_v5"
            )
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
