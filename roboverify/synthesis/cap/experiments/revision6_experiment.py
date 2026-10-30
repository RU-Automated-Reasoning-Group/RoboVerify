"""Revision 6 demonstration experiments under ``config_frozen_v3`` (or ``v6``).

One batch per (model, task) at ``ROOT/<model>/<task>``, all batches in
parallel, then one combined report at ``ROOT/report.md``
(``reports/batch_set_report.py``)::

    python -m synthesis.cap.experiments.revision6_experiment --root ROOT \\
        --model local-qwen3.8-27b-fp8 --model claude-haiku-5.5 \\
        --task stack --task grid --trials 5 --seed-start 0 --workers 5

Run configs come from ``config_frozen_v3`` (demonstrations on unless
``--demonstrations off``); ``--frozen v6`` selects ``config_frozen_v6``
(Revision 7.1), whose demonstration ablation runs both arms on the same scene
seeds under separate roots, ``--frozen v7`` the hosted-model
``config_frozen_v7`` (Revision 7.3), and ``--frozen final_v1`` the final
configuration ``config_final_v1``: v7 with the final release check (Revision 7.4);
``--frozen final_v2`` adds provider prompt caching (Revision 7.5), and
``--frozen final_v2_wo_demo`` is final v2 without demonstrations (its default;
``--demonstrations on`` is rejected). The local model uses v3's model config unchanged.
Hosted OpenRouter models replace only the endpoint, the model name and the
reasoning option (OpenRouter's ``reasoning.effort`` in place of vLLM's chat
template switch), and measure the demonstration block with the provider's
prompt-token counts (``provider_usage``), since they have no local tokenizer.
Credentials are read from ``VLLM_API_KEY`` / ``OPENROUTER_API_KEY`` only.
``--model`` also accepts any OpenRouter model id (``provider/name``), configured
like the listed hosted models. ``--gpu UUID`` (repeatable) replaces this host's
GPU allowlist; ``--no-gpu-scheduler`` runs workers without GPU reservations.
``--max-analysis-steps N`` sets the static DSL analysis work budget
(``RunConfig.max_analysis_steps``; default: the checker's 100,000) and records it
as an override. ``--report-only`` rewrites the combined report from existing batches.
"""

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

from ..batch import atomic_json, create_batch, execute_batch
from ..config import GPUConfig
from ..configs import (
    config_final_v1,
    config_final_v2,
    config_final_v2_wo_demo,
    config_frozen_v3,
    config_frozen_v6,
    config_frozen_v7,
)
from ..demonstrations import measure_once
from ..reports import batch_set_report
from .revision42_experiment import GPU_UUIDS

OPENROUTER = {
    "provider": "openrouter",
    "base_url": "https://openrouter.ai/api/v1",
    "api_key_env": "OPENROUTER_API_KEY",
    "extra_body": {"reasoning": {"effort": "medium"}},
}
MODELS = {
    "local-qwen3.8-27b-fp8": None,  # config_frozen_v3.MODEL (local vLLM)
    "qwen3.8-27b": OPENROUTER | {"model": "qwen/qwen3.8-27b"},
    "claude-haiku-5.5": OPENROUTER | {"model": "anthropic/claude-haiku-5.5"},
    "claude-opus-5.5": OPENROUTER | {"model": "anthropic/claude-opus-5.5"},
}
FROZEN = {
    "v3": config_frozen_v3,
    "v6": config_frozen_v6,
    "v7": config_frozen_v7,
    "final_v1": config_final_v1,
    "final_v2": config_final_v2,
    "final_v2_wo_demo": config_final_v2_wo_demo,
}
EXPERIMENTS = {
    "v3": "revision6_frozen_v3_demonstrations",
    "v6": "revision7_1_frozen_v6_demonstrations",
    "v7": "revision7_3_frozen_v7",
    "final_v1": "final_v1",
    "final_v2": "final_v2",
    "final_v2_wo_demo": "final_v2_wo_demo",
}
TASKS = tuple(config_frozen_v3.TASKS)


def model_settings(label):
    """Model settings for a listed label or an OpenRouter id (``provider/name``)."""
    if label in MODELS:
        return MODELS[label]
    if "/" in label:
        return OPENROUTER | {"model": label}
    raise ValueError(
        f"Unknown model {label!r}: use one of {sorted(MODELS)} or an OpenRouter id"
    )


def configuration(
    task,
    label,
    *,
    demonstrations=True,
    seed=0,
    frozen="v3",
    max_analysis_steps=None,
):
    """(RunConfig, ModelConfig, instruction, overrides) for one batch."""
    frozen = FROZEN[frozen]
    config, model, instruction = frozen.configuration(task, demonstrations, seed=seed)
    overrides = {}
    if max_analysis_steps is not None:
        config = replace(config, max_analysis_steps=max_analysis_steps)
        overrides["max_analysis_steps"] = max_analysis_steps
    if model_settings(label) is not None:
        # Record only what differs from the frozen version (v7 is hosted). A
        # version that lists the model keeps its own routing and cache settings.
        settings = getattr(frozen, "MODELS", {}).get(label, model_settings(label))
        hosted = frozen.model_config(**settings)
        if hosted != model:
            model = hosted
            overrides["model_config"] = settings
        if demonstrations and config.demonstrations.token_counter != "provider_usage":
            config.demonstrations = replace(
                config.demonstrations, token_counter="provider_usage"
            )
            overrides["demonstrations.token_counter"] = "provider_usage"
    return config, model, instruction, overrides


def serving_record(model, frozen):
    if model.provider == "vllm":
        return frozen.SERVING | {
            "endpoint": model.base_url,
            "instances": "GPU 6 (8126), GPU 7 (8127)",
            "prefix_cache": "cap-prefix-cache-v1",
        }
    return {
        "endpoint": model.base_url,
        "note": "hosted model; no think-close guard, no prefix-cache affinity",
    } | (
        {
            "prompt_cache": model.prompt_cache,
            "provider_routing": model.extra_body.get("provider"),
        }
        if model.prompt_cache
        else {}
    )


def create(
    directory,
    task,
    label,
    *,
    trials,
    seed_start,
    workers,
    demonstrations,
    frozen,
    max_analysis_steps=None,
    gpus=GPU_UUIDS,
):
    config, model, instruction, overrides = configuration(
        task,
        label,
        demonstrations=demonstrations,
        seed=seed_start,
        frozen=frozen,
        max_analysis_steps=max_analysis_steps,
    )
    version = frozen
    frozen = FROZEN[version]
    if (
        getattr(frozen, "DEMONSTRATION_TOKENS_ONCE", False)
        and config.demonstrations.enabled
    ):
        # Revision 7.5: one measurement of the block for every trial of the batch.
        measured = measure_once(config, model)
        config.demonstrations = replace(config.demonstrations, measured_tokens=measured)
        overrides["demonstrations.measured_tokens"] = measured
    request_limit = trials * config.recovery.total_calls
    create_batch(
        directory,
        config,
        model,
        instruction=instruction,
        methods=[config.method],
        modes=[config.condition_mode],
        trials=trials,
        max_workers=workers,
        schedule_seed=6000 + seed_start,
        gpu_config=(
            GPUConfig(
                list(gpus),
                "/tmp/roboverify-cap-gpu-reservations",
                max_workers_per_gpu=48,
                min_free_memory_mb=2048,
                worker_memory_mb=128,
                gpu_poll_interval_seconds=2,
                gpu_sample_window_seconds=10,
                admission_timeout_seconds=60,
            )
            if gpus
            else None
        ),
        transport_control={
            "state_path": str(directory / "request_gate.json"),
            "interval_seconds": 0.0,
            "request_limit": request_limit,
            "credential_limits": {model.api_key_env: request_limit},
        },
        seed_per_trial=True,
        cache_warmup=getattr(frozen, "BATCH", {}).get("cache_warmup", False),
    )
    atomic_json(
        directory / "comparison_plan.json",
        {
            "experiment": EXPERIMENTS[version],
            "frozen_config": frozen.VERSION,
            "frozen_config_source": f"synthesis/cap/configs/{Path(frozen.__file__).name}",
            "instruction_version": getattr(
                frozen, "INSTRUCTION_VERSION", "task-instructions-v1"
            ),
            "driver": "synthesis/cap/experiments/revision6_experiment.py",
            "overrides": overrides,
            "task": task,
            "model_label": label,
            "demonstrations": demonstrations,
            "trials": trials,
            "scene_seeds": [seed_start, seed_start + trials - 1],
            "task_settings": frozen.TASKS[task],
            "serving": serving_record(model, frozen),
            "worker_gpus": list(gpus or []),
            "request_limit": request_limit,
            "git_sha": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "git_dirty": bool(
                subprocess.check_output(
                    ["git", "status", "--porcelain", "synthesis"], text=True
                ).strip()
            ),
            "run_config": asdict(config),
            "model_config": asdict(model),
            "no_failed_trial_replacements": True,
        },
    )


def run_batch(directory, task, label, **settings):
    """Create (unless present) and execute one batch; resumes a stopped batch."""
    directory = Path(directory).resolve()
    if not directory.exists():
        create(directory, task, label, **settings)
    return execute_batch(directory, resume=(directory / "summary.json").exists())


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--model",
        action="append",
        help=f"one of {sorted(MODELS)} or an OpenRouter model id (provider/name)",
    )
    parser.add_argument("--task", action="append", choices=TASKS)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--workers", type=int, default=5, help="per batch")
    parser.add_argument(
        "--demonstrations",
        choices=("on", "off"),
        help="default: the frozen version's setting (on unless it disables them)",
    )
    parser.add_argument("--frozen", choices=sorted(FROZEN), default="v3")
    parser.add_argument(
        "--max-analysis-steps",
        type=int,
        help="DSL analysis work budget (default: the checker's 100,000)",
    )
    gpu = parser.add_mutually_exclusive_group()
    gpu.add_argument(
        "--gpu",
        action="append",
        metavar="UUID",
        help="allowed GPU UUID, repeatable (default: this host's allowlist)",
    )
    gpu.add_argument(
        "--no-gpu-scheduler",
        action="store_true",
        help="run workers without GPU reservations",
    )
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--title", default="Revision 6 demonstrations")
    parser.add_argument("--note", action="append", default=[])
    parser.add_argument(
        "--batch", nargs=2, metavar=("MODEL", "TASK"), help=argparse.SUPPRESS
    )
    args = parser.parse_args()
    root = args.root.resolve()
    if args.demonstrations is None:
        enabled = getattr(FROZEN[args.frozen], "DEMONSTRATIONS_ENABLED", True)
        args.demonstrations = "on" if enabled else "off"
    settings = dict(
        trials=args.trials,
        seed_start=args.seed_start,
        workers=args.workers,
        demonstrations=args.demonstrations == "on",
        frozen=args.frozen,
        max_analysis_steps=args.max_analysis_steps,
        gpus=[] if args.no_gpu_scheduler else args.gpu or GPU_UUIDS,
    )
    for label in args.model or []:
        try:
            model_settings(label)
        except ValueError as error:
            parser.error(str(error))
    if args.batch:
        label, task = args.batch
        summary = run_batch(root / label / task, task, label, **settings)
        print(
            json.dumps(
                {
                    "model": label,
                    "task": task,
                    "finished": summary["finished"],
                    "planned": summary["planned"],
                }
            ),
            flush=True,
        )
        return
    if not args.report_only:
        models, tasks = args.model or ["local-qwen3.8-27b-fp8"], args.task or list(
            TASKS
        )
        for label in models:
            env = (
                "VLLM_API_KEY"
                if model_settings(label) is None
                else OPENROUTER["api_key_env"]
            )
            if not os.environ.get(env):
                parser.error(f"{label} needs the {env} environment variable")
        root.mkdir(parents=True, exist_ok=True)
        processes = []
        for label in models:
            for task in tasks:
                log = (root / f"launch_{label}_{task}.log").open("a")
                command = [
                    sys.executable,
                    "-m",
                    __spec__.name,
                    "--root",
                    str(root),
                    "--batch",
                    label,
                    task,
                    "--trials",
                    str(args.trials),
                    "--seed-start",
                    str(args.seed_start),
                    "--workers",
                    str(args.workers),
                    "--demonstrations",
                    args.demonstrations,
                    "--frozen",
                    args.frozen,
                ]
                if args.max_analysis_steps is not None:
                    command += ["--max-analysis-steps", str(args.max_analysis_steps)]
                if args.no_gpu_scheduler:
                    command += ["--no-gpu-scheduler"]
                for uuid in args.gpu or []:
                    command += ["--gpu", uuid]
                processes.append(
                    (
                        label,
                        task,
                        subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT),
                    )
                )
        for label, task, process in processes:
            code = process.wait()
            print(
                json.dumps({"model": label, "task": task, "exit_code": code}),
                flush=True,
            )
    print(batch_set_report.write(root, title=args.title, notes=args.note), flush=True)


if __name__ == "__main__":
    main()
