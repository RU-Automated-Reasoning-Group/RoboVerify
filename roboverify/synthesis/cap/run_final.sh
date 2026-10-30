#!/usr/bin/env bash
# Run the final CaP experiment in one command.
#
#   bash synthesis/cap/run_final.sh --demo on --task stack --model claude-opus-5.5
#
# Demonstrations on runs config_final_v2 (docs/final_v1_report.md); off runs
# config_final_v2_wo_demo with its 1,000,000-step analysis budget
# (docs/final_v2_wo_demo_report.md). Every other setting comes from those frozen
# configurations. The combined report is written to ROOT/report.md.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash synthesis/cap/run_final.sh [options]

  --demo on|off      demonstrations in every request (default: on)
  --task TASK        stack, unstack, reverse, restack, grid or pyramid;
                     repeatable (default: all six)
  --model MODEL      claude-opus-5.5 (default), claude-haiku-5.5, or any
                     OpenRouter model id such as openai/gpt-6.1-sol; repeatable
  --trials N         trials (scenes) per task (default: 5; the reports use 50)
  --seed-start S     first scene seed (default: 0)
  --workers W        parallel trials per task (default: 2; keep <= 3 per key)
  --root DIR         output directory (default: runs/cap-final/<time>-demo-<on|off>)
  --gpu UUID         reserve this GPU for workers, repeatable (default: none)

Reads OPENROUTER_API_KEY from the environment, or from .runtime/cap-openrouter.env.
EOF
}

demo=on
trials=5
seed_start=0
workers=2
root=""
tasks=()
models=()
gpus=()
while [ $# -gt 0 ]; do
    case "$1" in
        --demo) demo="$2"; shift 2 ;;
        --task) tasks+=(--task "$2"); shift 2 ;;
        --model) models+=(--model "$2"); shift 2 ;;
        --trials) trials="$2"; shift 2 ;;
        --seed-start) seed_start="$2"; shift 2 ;;
        --workers) workers="$2"; shift 2 ;;
        --root) root="$2"; shift 2 ;;
        --gpu) gpus+=(--gpu "$2"); shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

case "$demo" in
    on) frozen=(--frozen final_v2) ;;
    off) frozen=(--frozen final_v2_wo_demo --max-analysis-steps 1000000) ;;
    *) echo "--demo must be on or off" >&2; exit 2 ;;
esac
[ ${#models[@]} -gt 0 ] || models=(--model claude-opus-5.5)
[ ${#gpus[@]} -gt 0 ] || gpus=(--no-gpu-scheduler)

# Simulator environment (AGENTS.md, "Environment"), from roboverify/.
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
unset LD_PRELOAD
if [ -d "$PWD/.runtime/mujoco210" ]; then
    export MUJOCO_PY_MUJOCO_PATH="$PWD/.runtime/mujoco210"
else
    export MUJOCO_PY_MUJOCO_PATH="$HOME/.mujoco/mujoco210"
fi
export LD_LIBRARY_PATH="$MUJOCO_PY_MUJOCO_PATH/bin:/usr/lib/nvidia${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

if [ -z "${OPENROUTER_API_KEY:-}" ] && [ -f .runtime/cap-openrouter.env ]; then
    set -a
    # shellcheck disable=SC1091
    . .runtime/cap-openrouter.env
    set +a
fi
if [ -z "${OPENROUTER_API_KEY:-}" ]; then
    echo "Set OPENROUTER_API_KEY (or write it to roboverify/.runtime/cap-openrouter.env)" >&2
    exit 2
fi

if [ -x .venv/bin/python ]; then
    python=(.venv/bin/python)
else
    python=(uv run python)
fi

if [ "$demo" = on ]; then
    # The pinned demonstration archives are not in git; fail before any request.
    "${python[@]}" - "${tasks[@]}" <<'PY'
import sys
from pathlib import Path

from synthesis.cap.configs import config_final_v2 as frozen

tasks = sys.argv[2::2] or list(frozen.TASKS)
missing = [frozen.ARCHIVES[t][0] for t in tasks if not Path(frozen.ARCHIVES[t][0]).is_file()]
if missing:
    sys.exit(
        "Missing demonstration archives:\n  "
        + "\n  ".join(missing)
        + "\nRun `bash synthesis/cap/collect_demos.sh` first, or use --demo off."
    )
PY
fi

[ -n "$root" ] || root="runs/cap-final/$(date -u +%Y%m%d-%H%M%S)-demo-$demo"
"${python[@]}" -m synthesis.cap.experiments.revision6_experiment \
    --root "$root" "${models[@]}" "${tasks[@]}" "${frozen[@]}" "${gpus[@]}" \
    --trials "$trials" --seed-start "$seed_start" --workers "$workers" \
    --title "Final CaP experiment (demonstrations $demo)"
echo "Results: roboverify/$root/report.md"
