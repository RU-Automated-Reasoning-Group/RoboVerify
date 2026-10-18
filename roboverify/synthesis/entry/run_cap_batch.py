"""Run or resume a manifest-defined, process-isolated CaP batch."""

import argparse
import json
from pathlib import Path

from synthesis.cap.batch import METHODS, MODES, create_batch, execute_batch
from synthesis.cap.config import ModelConfig, RunConfig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--config",
        type=Path,
        help="JSON containing run_config and model_config, without credentials",
    )
    parser.add_argument("--initial-state", type=Path)
    parser.add_argument("--instruction-file", type=Path)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--condition-modes", nargs="+", choices=MODES, default=MODES)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--schedule-seed", type=int, default=0)
    parser.add_argument("--responses-file", type=Path)
    args = parser.parse_args(argv)
    if not args.resume:
        if not args.config:
            parser.error("--config is required for a new batch")
        spec = json.loads(args.config.read_text())
        create_batch(
            args.output_dir,
            RunConfig(**spec["run_config"]),
            ModelConfig(**spec["model_config"]),
            initial_state=args.initial_state,
            instruction=(
                args.instruction_file.read_text() if args.instruction_file else None
            ),
            methods=args.methods,
            modes=args.condition_modes,
            trials=args.trials,
            max_workers=args.max_workers,
            schedule_seed=args.schedule_seed,
            responses=(
                json.loads(args.responses_file.read_text())
                if args.responses_file
                else None
            ),
        )
    result = execute_batch(args.output_dir, resume=args.resume)
    print(
        json.dumps(
            {
                "finished": result["finished"],
                "planned": result["planned"],
                "report": str(args.output_dir / "report.md"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
