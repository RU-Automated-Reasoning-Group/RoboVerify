#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
unset LD_PRELOAD
mujoco_path="${MUJOCO_PY_MUJOCO_PATH:-$HOME/.mujoco/mujoco210}"
export LD_LIBRARY_PATH="$mujoco_path/bin:/usr/lib/nvidia${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

# The Fetch runtime uses Gym 0.21. Build it with the project's locked
# packaging tools, which accept the legacy release's dependency metadata.
uv run --locked python -m pip --disable-pip-version-check install \
  --no-deps --no-build-isolation --require-hashes -r /dev/stdin <<'GYM_REQUIREMENT'
gym==0.21.0 --hash=sha256:0fd1ce165c754b4017e37a617b097c032b8c3feb8a0394ccc8777c7c50dddff3
GYM_REQUIREMENT

inputs=demos/stack/contract-reproduction
mkdir -p runs/stack-reproduction
output_dir=$(mktemp -d runs/stack-reproduction/run.XXXXXX)
echo "Stack reproduction results: $PWD/$output_dir"

uv run --locked python -m synthesis.entry.synthesize_cfg \
  --task stack --num-blocks 4 --mode synthesize \
  --demos "$inputs/demonstrations.npz" \
  --synthesis-approach id-first --search-strategy contract --contract-refinement --quotient \
  --slots 6 --iterations 7 --refinements 20 \
  --cem-iterations 0 --cem-samples 1 --cem-elites 1 \
  --distance mmd --theta .1 --epsilon .1 \
  --inner-samples 64 --final-samples 256 --reset-mode reset --seed 0 \
  --loop-body-distance arc --loop-body-theta .03 --loop-body-epsilon .003 \
  --contract-beam-width 12 --contract-pool-limit 8 --contract-seconds 60 \
  --contract-max-expansions 100000 --predicate-seconds 5 \
  --validation-starts "$inputs/development-starts.npz" \
  --candidate-validation-limit 8 --candidate-validation-seconds 900 \
  --candidate-selection frame-margin \
  --candidate-parameter-trials 40 --candidate-parameter-step .001 \
  --candidate-max-total-actions 300000 --wall-timeout-seconds 1200 \
  --output-dir "$output_dir" --run-name stack-synthesis

shopt -s nullglob
programs=("$output_dir"/cfg/*-stack-synthesis/artifacts/program.json)
if (( ${#programs[@]} != 1 )); then
  echo "Expected one synthesized Stack program under $output_dir" >&2
  exit 1
fi

uv run --locked python -m synthesis.entry.synthesize_cfg \
  --task stack --num-blocks 4 --mode verify --candidate-program \
  --program "${programs[0]}" --demos "$inputs/demonstrations.npz" \
  --verification-level symbolic --synthesis-approach relational --reset-mode reset \
  --invariant-relations ON_star Higher Scattered equality --invariant-minimizer pyeda \
  --symbolic-iterations 0 --motion-iterations 0 --verification-timeout-ms 10000 \
  --higher-tolerance .001 --wp-system old --no-minimize-unsat-core \
  --max-loop-iterations 100 --trajectory-timeout-seconds 20 --wall-timeout-seconds 110 \
  --output-dir "$output_dir" --run-name stack-symbolic

echo "Synthesized Stack program: $PWD/${programs[0]}"
