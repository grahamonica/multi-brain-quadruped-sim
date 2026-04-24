#!/bin/bash
#SBATCH --job-name=quad-shared-large
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/pace_shared_large_%j.out
#SBATCH --error=logs/pace_shared_large_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs checkpoints
source venv/bin/activate

CONFIG=${CONFIG:-configs/default.yaml}
MODEL_TYPE=${MODEL_TYPE:-notebook_shared_trunk_v1}
LOG_ID=${LOG_ID:-shared_large_${SLURM_JOB_ID:-$(date +%s)}}
OUT_DIR=${OUT_DIR:-checkpoints}
GENERATIONS=${GENERATIONS:-1500}
SEED=${SEED:-11}
POPULATION_SIZE=${POPULATION_SIZE:-256}
EPISODE_SECONDS=${EPISODE_SECONDS:-45}
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
        "Run the corresponding notebook registration flow first.",
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
)

if [[ -n "${EXTRA_ARGS}" ]]; then
  TRAIN_ARGS+=(${EXTRA_ARGS})
fi

python train_headless.py "${TRAIN_ARGS[@]}"

