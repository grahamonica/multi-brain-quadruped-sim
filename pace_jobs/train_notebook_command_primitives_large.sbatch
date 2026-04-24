#!/bin/bash
#SBATCH --job-name=quad-commands-large
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/pace_commands_large_%j.out
#SBATCH --error=logs/pace_commands_large_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs checkpoints
source venv/bin/activate

CONFIG=${CONFIG:-configs/default.yaml}
MODEL_TYPE=${MODEL_TYPE:-notebook_command_brain_v1}
LOG_ID=${LOG_ID:-command_large_${SLURM_JOB_ID:-$(date +%s)}}
OUT_DIR=${OUT_DIR:-checkpoints}
GENERATIONS=${GENERATIONS:-1800}
SEED=${SEED:-11}
POPULATION_SIZE=${POPULATION_SIZE:-320}
EPISODE_SECONDS=${EPISODE_SECONDS:-60}
COMMAND_VOCAB=${COMMAND_VOCAB:-stand,trot,turn_left,turn_right,back_up}
COMMAND_SPEED=${COMMAND_SPEED:-0.45}
EXTRA_ARGS=${EXTRA_ARGS:-}

export MODEL_TYPE
python3 - <<'PY'
import os
import sys
from brains.models import get_model_definition

model_type = os.environ["MODEL_TYPE"]
try:
    get_model_definition(model_type)
except Exception as exc:
    print(
        f"[ERROR] Model type {model_type!r} is not registered. "
        "Run notebooks/brain_command_primitives_playground.ipynb registration first.",
        file=sys.stderr,
    )
    raise
PY

TRAIN_ARGS=(
  --config "${CONFIG}"
  --model-type "${MODEL_TYPE}"
  --log-id "${LOG_ID}"
  --out-dir "${OUT_DIR}"
  --generations "${GENERATIONS}"
  --seed "${SEED}"
  --population-size "${POPULATION_SIZE}"
  --episode-seconds "${EPISODE_SECONDS}"
  --control-mode command_primitives
  --command-vocabulary "${COMMAND_VOCAB}"
  --command-speed "${COMMAND_SPEED}"
)

if [[ -n "${EXTRA_ARGS}" ]]; then
  TRAIN_ARGS+=(${EXTRA_ARGS})
fi

python train_headless.py "${TRAIN_ARGS[@]}"

