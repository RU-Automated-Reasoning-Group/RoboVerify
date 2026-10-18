"""Run the Google CaP baseline or constrained method 1 as a module."""

import argparse
import json
from pathlib import Path

from synthesis.cap.config import ModelConfig, RunConfig
from synthesis.cap.contracts import CONDITION_MODES
from synthesis.cap.llm import ScriptedModel
from synthesis.cap.prompt_package import PROMPT_VERSIONS
from synthesis.cap.runner import run
from synthesis.cap.tasks import TASK_INSTRUCTIONS


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        choices=["cap_google_baseline", "cap_method_1", "cap_method_2"],
        required=True,
    )
    parser.add_argument("--condition-mode", choices=CONDITION_MODES, default="dsl_only")
    parser.add_argument(
        "--prompt-version", choices=PROMPT_VERSIONS, default="legacy_v1"
    )
    parser.add_argument(
        "--baseline-audit", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--audit-timeout", type=float, default=10.0)
    parser.add_argument("--task", choices=TASK_INSTRUCTIONS, default="stack")
    parser.add_argument("--num-blocks", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--instruction")
    parser.add_argument("--initial-state", type=Path)
    parser.add_argument("--trace-index", type=int, default=0)
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument(
        "--responses-file",
        type=Path,
        help="Offline JSON list of model response strings",
    )
    parser.add_argument(
        "--provider", choices=["openrouter", "openai", "vllm"], default="openrouter"
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url")
    parser.add_argument(
        "--api-key-env",
        help="Credential variable name; empty disables authentication for a local endpoint",
    )
    parser.add_argument(
        "--api-style",
        choices=["chat_completions", "responses"],
        default="chat_completions",
    )
    parser.add_argument(
        "--token-field",
        choices=["max_tokens", "max_completion_tokens"],
        default="max_tokens",
    )
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--model-seed", type=int)
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument(
        "--extra-body", type=Path, help="Provider-specific JSON request options"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=600,
        help="Wall-clock safety cap in seconds; Unstack remains capped at 60",
    )
    parser.add_argument("--request-timeout", type=float, default=45)
    parser.add_argument("--transport-retries", type=int, default=2)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--max-model-calls", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--max-sim-steps", type=int, default=10000)
    parser.add_argument("--max-depth", type=int, default=12)
    parser.add_argument("--max-tokens", type=int, default=100000)
    parser.add_argument("--higher-tolerance", type=float, default=0.001)
    parser.add_argument(
        "--save-video", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--output-dir", default="runs")
    args = parser.parse_args(argv)
    defaults = {
        "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
        "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
        "vllm": ("http://localhost:8000/v1", ""),
    }
    if args.provider != "openrouter" and not args.model and not args.responses_file:
        parser.error("--model is required for OpenAI and vLLM")
    model = ModelConfig(
        provider=args.provider,
        model=args.model or "deepseek/deepseek-v4.1-flash",
        base_url=args.base_url or defaults[args.provider][0],
        api_key_env=(
            args.api_key_env
            if args.api_key_env is not None
            else defaults[args.provider][1]
        ),
        api_style=args.api_style,
        temperature=args.temperature,
        seed=args.model_seed,
        max_output_tokens=args.max_output_tokens,
        token_field=args.token_field,
        request_timeout=args.request_timeout,
        transport_retries=args.transport_retries,
        extra_body=json.loads(args.extra_body.read_text()) if args.extra_body else {},
    )
    config = RunConfig(
        method=args.method,
        baseline_audit=args.baseline_audit,
        audit_timeout=args.audit_timeout,
        condition_mode=args.condition_mode,
        prompt_version=args.prompt_version,
        task=args.task,
        num_blocks=args.num_blocks,
        seed=args.seed,
        timeout=args.timeout,
        max_attempts=args.max_attempts,
        max_model_calls=args.max_model_calls,
        max_steps=args.max_steps,
        max_sim_steps=args.max_sim_steps,
        max_depth=args.max_depth,
        max_tokens=args.max_tokens,
        higher_tolerance=args.higher_tolerance,
        save_video=args.save_video,
        output_root=args.output_dir,
    )
    backend = None
    if args.responses_file:
        responses = json.loads(args.responses_file.read_text())
        if not isinstance(responses, list) or not all(
            isinstance(item, str) for item in responses
        ):
            parser.error("--responses-file must contain a JSON list of strings")
        backend = ScriptedModel(responses)
    result = run(
        config,
        model,
        instruction=args.instruction,
        backend=backend,
        initial_state=args.initial_state,
        trace_index=args.trace_index,
        generate_only=args.generate_only,
    )
    output = json.dumps(result, indent=2)
    with (Path(result["run_dir"]) / "stdout.log").open("a") as handle:
        handle.write(output + "\n")
    print(output)
    return 0 if result["status"] == "generated" or result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
