"""Shared single-trial runner; all outcomes and artifacts use RunLogger."""

import hashlib
import math
import signal
import time
import traceback
import uuid
from contextlib import contextmanager, nullcontext
from dataclasses import asdict
from pathlib import Path

from .adapter import GenerationOnlyAdapter, RoboVerifyAdapter
from .audit import audit_run
from .budget import Budget
from .checker import AnalysisFailure, DSLContract
from .checkpoints import Checkpoints
from .contracts import (
    CHECKER_VERSION,
    OBSERVATION_API_VERSION,
    AdapterGap,
    BudgetExceeded,
    ConstraintFailure,
    EnvironmentUnavailable,
    PrimitiveFailure,
    validate_condition_mode,
)
from .llm import HTTPModel, LoggedModel, ModelError
from .logger import CapLogger
from .prompt_package import package, prompt_provenance
from .tasks import (
    TASK_INSTRUCTIONS,
    check_task,
    evaluate_terminal,
    prepare_environment,
    validate_start,
)


@contextmanager
def deadline(budget):
    """POSIX wall-clock deadline also interrupts original Python infinite loops."""
    previous = signal.getsignal(signal.SIGALRM)
    timer = signal.getitimer(signal.ITIMER_REAL)
    started = time.monotonic()

    def expired(signum, frame):
        budget.exhaust("wall_clock", "Wall-clock trial deadline exhausted")

    remaining = budget.remaining()
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(
        signal.ITIMER_REAL, min(remaining, timer[0]) if timer[0] else remaining
    )
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        if timer[0]:
            signal.setitimer(
                signal.ITIMER_REAL,
                max(0.000001, timer[0] - (time.monotonic() - started)),
                timer[1],
            )


def make_method(
    name,
    model,
    adapter,
    logger,
    checkpoints,
    budget,
    attempts,
    condition_mode="dsl_only",
    prompt_version="legacy_v1",
    higher_tolerance=0.001,
):
    if name in {"cap_method_1", "cap_method_2"}:
        from .cap_method_1 import Method1
        from .cap_method_2 import Method2

        method_class = Method1 if name == "cap_method_1" else Method2
        return method_class(
            model,
            adapter,
            logger,
            checkpoints,
            budget,
            attempts,
            condition_mode=condition_mode,
            prompt_version=prompt_version,
            higher_tolerance=higher_tolerance,
        )
    if name == "cap_google_baseline":
        from .cap_google_baseline import GoogleBaseline

        return GoogleBaseline(
            model,
            adapter,
            logger,
            checkpoints,
            budget,
            condition_mode=condition_mode,
            prompt_version=prompt_version,
            higher_tolerance=higher_tolerance,
        )
    raise ValueError(f"Unknown method: {name}")


def failure_status(error):
    for kind, status in (
        (AnalysisFailure, "analysis_failed"),
        (AdapterGap, "adapter_gap"),
        (EnvironmentUnavailable, "environment_unavailable"),
        (BudgetExceeded, "budget_exhausted"),
        (ConstraintFailure, "constraint_failed"),
        (PrimitiveFailure, "primitive_failed"),
        (ModelError, "generation_failed"),
    ):
        if isinstance(error, kind):
            return status
    return "failed"


def run_once(
    config,
    model_config,
    *,
    instruction=None,
    backend=None,
    initial_state=None,
    trace_index=0,
    generate_only=False,
    environment_factory=None,
    capture_stdout=True,
    queue_wait_seconds=0.0,
    batch_context=None,
    shared_budget=None,
    recovery_feedback=None,
):
    from synthesis.experiment.run_logger import RunLogger

    if not math.isfinite(queue_wait_seconds) or queue_wait_seconds < 0:
        raise ValueError("Queue wait must be finite and nonnegative")
    validate_condition_mode(config.condition_mode)
    provenance = prompt_provenance(config.prompt_version, config.primitive_prompt)
    if config.method not in {"cap_method_1", "cap_method_2", "cap_google_baseline"}:
        raise ValueError("Unknown CaP method")
    if config.timeout <= 0 or config.max_attempts < 1:
        raise ValueError("Invalid trial budget")
    timeout = min(config.timeout, 60) if config.task == "unstack" else config.timeout
    budget_class = Budget
    if shared_budget is not None:
        from .recovery import RoundBudget

        budget_class = RoundBudget
    budget = budget_class(
        timeout,
        config.max_model_calls,
        config.max_steps,
        config.max_depth,
        config.max_tokens,
        max_sim_steps=config.max_sim_steps,
    )
    if shared_budget is not None:
        budget.parent = shared_budget
        budget.context_tokens = config.recovery.context_tokens
        budget.output_tokens = model_config.max_output_tokens
    instruction = instruction or TASK_INSTRUCTIONS.get(config.task)
    resolved = asdict(config) | {
        **provenance,
        "timeout": timeout,
        "budget_policy": "resource-limits-with-wall-clock-safety-cap-v1",
        "queue_wait_seconds": queue_wait_seconds,
        "batch_context": dict(batch_context or {}),
        "model": asdict(model_config),
        "generate_only": generate_only,
        "instruction": instruction,
        "initial_state": str(initial_state) if initial_state else None,
        "trace_index": trace_index,
        "validation_enabled": config.method != "cap_google_baseline",
        "refinement_enabled": config.method != "cap_google_baseline",
        "observation_api_version": OBSERVATION_API_VERSION,
        "checker_version": (
            CHECKER_VERSION if config.method != "cap_google_baseline" else None
        ),
        "pre_execution_dsl_check": config.method != "cap_google_baseline",
        "dsl_contract": DSLContract.for_task(
            config.task, config.condition_mode
        ).identity(),
        "posthoc_dsl_check": config.method == "cap_google_baseline"
        and config.baseline_audit,
        "auxiliary_lmp_modules": [],
        "model_backend": type(backend).__name__ if backend else "HTTPModel",
    }
    run_log = RunLogger(
        config.output_root,
        "cap",
        resolved,
        slug=f"{config.method}-{config.task}-{uuid.uuid4().hex[:8]}",
        capture_stdout=capture_stdout,
    )
    logger = CapLogger(run_log.run_dir, run=run_log, timings=budget.timings)
    logger.primitive_phase_feedback_enabled = (
        config.primitive_feedback.controller_phases
    )
    logger.context.update(recovery_round=(batch_context or {}).get("recovery_round", 0))
    if recovery_feedback is not None:
        logger.artifact("recovery_feedback.json", recovery_feedback)
    checkpoints = Checkpoints(
        logger,
        method=config.method,
        condition_mode=config.condition_mode,
        prompt_version=config.prompt_version,
        primitive_prompt=config.primitive_prompt,
    )
    from .primitive_prompts import BLOCKS
    from .primitive_prompts import VERSION as PRIMITIVE_PROMPT_VERSION

    if (
        any(asdict(config.primitive_prompt).values())
        or config.primitive_feedback.enabled
    ):
        logger.artifact(
            "primitive_prompt_package.json",
            {
                "version": PRIMITIVE_PROMPT_VERSION,
                "blocks": BLOCKS,
                "flags": asdict(config.primitive_prompt),
            },
        )
    logger.artifact("prompt_package.json", package(config.prompt_version))
    model = LoggedModel(
        backend or HTTPModel(model_config),
        logger,
        budget,
        model_config.transport_retries,
        recovery_feedback=recovery_feedback,
    )
    env, recorder, entry = None, None, None
    outcome = {"success": False, "environment_success": None, "dsl_postcondition": None}
    status, reason = "failed", ""
    try:
        with deadline(budget):
            check_task(config.task)
            instruction = instruction or TASK_INSTRUCTIONS[config.task]
            if generate_only:
                adapter = GenerationOnlyAdapter(config.task)
                method = make_method(
                    config.method,
                    model,
                    adapter,
                    logger,
                    checkpoints,
                    budget,
                    config.max_attempts,
                    condition_mode=config.condition_mode,
                    prompt_version=config.prompt_version,
                    higher_tolerance=config.higher_tolerance,
                )
                if config.method != "cap_google_baseline":
                    budget.timings.call(
                        "generation_control", method.generator.generate, instruction
                    )
                else:
                    method.policy._cfg["debug_mode"] = True
                    budget.timings.call("generation_control", method, instruction)
                status = "generated"
                outcome["success"] = None
            else:
                from synthesis.cfg.recordings import load_traces
                from synthesis.cfg.reset import inner_env
                from synthesis.util.on import using_higher_tolerance

                from .recording import Recorder

                snapshot = None
                if initial_state:
                    traces = load_traces(initial_state)
                    if not 0 <= trace_index < len(traces):
                        raise ValueError("Initial-state trace index is out of range")
                    trace = traces[trace_index]
                    aliases = {"restack": "partial_stack"}
                    if trace.num_blocks != config.num_blocks or aliases.get(
                        trace.task, trace.task
                    ) != aliases.get(config.task, config.task):
                        raise ValueError("Initial-state task/object count mismatch")
                    snapshot = trace.snapshots[0]
                    logger.emit(
                        "snapshot_input",
                        path=str(Path(initial_state).resolve()),
                        sha256=hashlib.sha256(
                            Path(initial_state).read_bytes()
                        ).hexdigest(),
                        trace_index=trace_index,
                    )
                with using_higher_tolerance(config.higher_tolerance):
                    with budget.timings.measure("setup"):
                        env = (
                            environment_factory(config, snapshot)
                            if environment_factory
                            else prepare_environment(config, snapshot)
                        )
                        adapter = RoboVerifyAdapter(
                            env, logger, budget, task=config.task
                        )
                        entry = adapter.entry_observation
                        if snapshot is not None:
                            from .recovery import verify_restoration

                            verify_restoration(env, snapshot, logger)

                        from synthesis.api.control import (
                            DEFAULT_CONTROL,
                            DEFAULT_PICK_CONTROL,
                        )

                        logger.artifact(
                            "adapter_config.json",
                            {
                                "controller": asdict(DEFAULT_CONTROL),
                                "pick_controller": asdict(DEFAULT_PICK_CONTROL),
                                "primitive_step_limit": 50,
                                "higher_tolerance": config.higher_tolerance,
                                "predicate_domain": "physical_blocks",
                                "entry_geometry": "frozen_trial_start",
                                "api_names": sorted(adapter.api_names),
                                "object_handles": {
                                    handle.token: physical_id
                                    for handle, physical_id in adapter.ids.items()
                                },
                                "initialization": (
                                    "snapshot" if snapshot else "fresh_scene"
                                ),
                                "settling_steps": (
                                    0 if snapshot or config.task != "stack" else 50
                                ),
                            },
                        )
                        validate_start(env, config.task, entry)
                        recorder = Recorder(
                            env,
                            logger,
                            budget,
                            video=config.save_video,
                            video_mode=config.video_mode,
                            queue_root=(batch_context or {}).get("batch_dir")
                            or (batch_context or {}).get("render_queue_root"),
                            seed=config.seed,
                            task=config.task,
                        )
                    method = make_method(
                        config.method,
                        model,
                        adapter,
                        logger,
                        checkpoints,
                        budget,
                        config.max_attempts,
                        condition_mode=config.condition_mode,
                        prompt_version=config.prompt_version,
                        higher_tolerance=config.higher_tolerance,
                    )
                    with recorder:
                        budget.timings.call("generation_control", method, instruction)
                    outcome.update(
                        budget.timings.call(
                            "evaluation", evaluate_terminal, env, config.task, entry
                        )
                    )
                    # Physical success is the primary metric; DSL agreement is
                    # retained independently rather than silently changing it.
                    outcome["success"] = outcome["environment_success"]
                    status = "completed" if outcome["success"] else "goal_failed"
    except Exception as exc:
        status = failure_status(exc)
        if isinstance(exc, ImportError) and env is None and not generate_only:
            status = "environment_unavailable"
        if generate_only and status == "failed":
            status = "generation_failed"
        reason = f"{type(exc).__name__}: {exc}"
        logger.artifact("execution/traceback.txt", traceback.format_exc())
        logger.emit("trial_failure", status=status, reason=reason)
        if env is not None and entry is not None:
            try:
                with using_higher_tolerance(config.higher_tolerance):
                    outcome.update(
                        budget.timings.call(
                            "evaluation", evaluate_terminal, env, config.task, entry
                        )
                    )
            except Exception as evaluation_error:
                logger.emit("evaluation_failure", error=str(evaluation_error))
        outcome["success"] = (
            None
            if generate_only or status in {"adapter_gap", "environment_unavailable"}
            else False
        )
    finally:
        if env is not None:
            try:
                budget.timings.call("cleanup", env.close)
            except Exception as exc:
                logger.emit("close_failure", error=str(exc))
        outcome.update(
            **provenance,
            method=config.method,
            condition_mode=config.condition_mode,
            observation_api_version=OBSERVATION_API_VERSION,
            task=config.task,
            model_calls=budget.calls,
            known_tokens=budget.tokens,
            unknown_usage_calls=budget.unknown_usage_calls,
            execution_ticks=budget.steps,
            simulation_steps=budget.simulation_steps,
            max_observed_depth=budget.max_observed_depth,
            budget_exhausted_by=budget.exhausted_by,
            queue_wait_seconds=queue_wait_seconds,
            checkpoint_id=checkpoints.previous,
            active_checkpoint_id=checkpoints.active,
            artifacts_path=logger.relative(logger.root),
        )
        if config.method != "cap_google_baseline":
            outcome["dsl_compliance"] = (
                "pass"
                if checkpoints.active
                else ("fail" if status == "constraint_failed" else "incomplete")
            )
        outcome["video"] = (
            recorder.video_metadata if recorder else {"status": "not_started"}
        )
        outcome.update(budget.timings.snapshot())
        logger.artifact(
            "timings.json",
            {
                key: outcome[key]
                for key in (
                    "active_wall_seconds",
                    "queue_wait_seconds",
                    "phase_seconds",
                    "phase_accounting",
                    "budget_exhausted_by",
                )
            },
        )
        logger.emit("trial_result", status=status, **outcome)
        logger.close()
        run_log.log_metrics(
            0,
            phase="cap",
            success_rate=(
                int(outcome["success"]) if outcome["success"] is not None else None
            ),
            model_calls=budget.calls,
            known_tokens=budget.tokens,
            status=status,
        )
        run_log.finish(status, exit_reason=reason, **outcome)
    result = {
        "status": status,
        "run_dir": str(run_log.run_dir),
        "reason": reason,
        **outcome,
    }
    if config.method == "cap_google_baseline" and config.baseline_audit:
        audit_timeout = config.audit_timeout
        if shared_budget is not None:
            try:
                audit_timeout = min(audit_timeout, budget.remaining())
            except BudgetExceeded:
                result["dsl_compliance"] = "not_available"
                result["audit_skipped_reason"] = "attempt_or_trial_time_exhausted"
                return result
        audit = audit_run(run_log.run_dir, timeout=audit_timeout)
        result["dsl_compliance"] = audit["status"]
        result["audit_seconds"] = audit["elapsed_seconds"]
    elif config.method != "cap_google_baseline":
        result["dsl_compliance"] = (
            "pass"
            if checkpoints.active
            else ("fail" if status == "constraint_failed" else "incomplete")
        )
    return result


def run(config, model_config, **kwargs):
    if config.recovery.enabled and not kwargs.get("generate_only", False):
        from .recovery import run_recovery

        return run_recovery(config, model_config, **kwargs)
    return run_once(config, model_config, **kwargs)
