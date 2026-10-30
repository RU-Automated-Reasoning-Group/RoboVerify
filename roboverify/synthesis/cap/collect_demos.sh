#!/usr/bin/env bash
# Collect the demonstration archives that the final CaP experiment pins.
#
#   bash synthesis/cap/collect_demos.sh
#
# Runs RoboVerify's groundtruth programs (synthesis.entry.collect_demos) on
# seeds 1000-1014 at the paper's block counts, then checks each archive
# against its pin in config_final_v2 before installing it at the pinned path.
# Collection is deterministic; a re-collected archive differs from the original
# only in recorded wall times and matches the pin's content digest
# (demonstrations.CONTENT_PINS). Run it once before run_final.sh --demo on.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash synthesis/cap/collect_demos.sh [--task TASK]...

  --task TASK   stack, unstack, reverse, restack, grid or pyramid; repeatable
                (default: all six)

An archive already present and matching its pin is kept.
EOF
}

tasks=()
while [ $# -gt 0 ]; do
    case "$1" in
        --task) tasks+=("$2"); shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done
[ ${#tasks[@]} -gt 0 ] || tasks=(stack unstack reverse restack grid pyramid)

# Collection settings of each pinned archive (docs/revision_6_add_demonstration.md,
# Section 6.6; Unstack and Reverse were re-collected from settled starts in v4).
options() {
    case "$1" in
        stack) echo "--num-blocks 4" ;;
        unstack|reverse) echo "--num-blocks 4 --post-settling-steps 50" ;;
        restack) echo "--num-blocks 5 --initial-tower-blocks 3" ;;
        grid) echo "--grid-rows 3 --grid-cols 2" ;;
        pyramid) echo "--bottom-row-blocks 3" ;;
        *) echo "Unknown task: $1" >&2; return 2 ;;
    esac
}

# Simulator environment (AGENTS.md, "Environment"), from roboverify/.
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
unset LD_PRELOAD
if [ -d "$PWD/.runtime/mujoco210" ]; then
    export MUJOCO_PY_MUJOCO_PATH="$PWD/.runtime/mujoco210"
else
    export MUJOCO_PY_MUJOCO_PATH="$HOME/.mujoco/mujoco210"
fi
export LD_LIBRARY_PATH="$MUJOCO_PY_MUJOCO_PATH/bin:/usr/lib/nvidia${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

if [ -x .venv/bin/python ]; then
    python=(.venv/bin/python)
else
    python=(uv run python)
fi

# Pinned path and digest of a task's archive; "check" verifies a file.
pin() {
    "${python[@]}" - "$@" <<'PY'
import sys

from synthesis.cap.configs import config_final_v2 as frozen
from synthesis.cap.demonstrations import DemonstrationError, check_archive_pin

command, task = sys.argv[1:3]
path, sha256 = frozen.ARCHIVES[task]
if command == "path":
    print(path)
else:
    try:
        print(check_archive_pin(sys.argv[3], sha256))
    except DemonstrationError as error:
        sys.exit(f"{task}: {error}")
PY
}

work="demos/.collect-$$"
mkdir -p "$work"
trap 'rm -rf "$work"' EXIT
pids=()
todo=()
for task in "${tasks[@]}"; do
    opts=$(options "$task")
    target=$(pin path "$task")
    if [ -f "$target" ] && pin check "$task" "$target" >/dev/null 2>&1; then
        echo "$task: $target already matches its pin"
        continue
    elif [ -e "$(dirname "$target")" ]; then
        echo "$task: $(dirname "$target") exists but does not match its pin; move it away first" >&2
        exit 1
    fi
    echo "$task: collecting 15 demonstrations (log: $work/$task.log)"
    # shellcheck disable=SC2086
    "${python[@]}" -m synthesis.entry.collect_demos --task "$task" $opts \
        --seeds $(seq 1000 1014) --output-dir "$work/$task" >"$work/$task.log" 2>&1 &
    pids+=($!)
    todo+=("$task")
done

status=0
for i in "${!todo[@]}"; do
    task=${todo[$i]}
    target=$(pin path "$task")
    if ! wait "${pids[$i]}"; then
        echo "$task: collection failed; last log lines:" >&2
        tail -n 20 "$work/$task.log" >&2
        status=1
    elif match=$(pin check "$task" "$work/$task/demonstrations.npz"); then
        mkdir -p "$(dirname "$(dirname "$target")")"
        mv "$work/$task" "$(dirname "$target")"
        cp "$work/$task.log" "$(dirname "$target").log"
        echo "$task: installed $target (pin match: $match)"
    else
        status=1
    fi
done
exit $status
