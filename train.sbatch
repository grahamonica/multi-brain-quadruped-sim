#!/bin/bash
#SBATCH --job-name=quadruped-train
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

set -euo pipefail

OUT_DIR=${OUT_DIR:-checkpoints}

mkdir -p logs "${OUT_DIR}"

# Activate the virtual environment prepared on the cluster login node.
source venv/bin/activate

CONFIG=${CONFIG:-configs/default.yaml}
MODEL_TYPE=${MODEL_TYPE:-shared_trunk_es}
LOG_ID=${LOG_ID:-${SLURM_JOB_ID:-local_$(date +%s)}}
GENERATIONS=${GENERATIONS:-100}
SEED=${SEED:-42}
RESUME=${RESUME:-}
EXTRA_ARGS=${EXTRA_ARGS:-}

# Run headless training. Reusing LOG_ID resumes the same model weight path:
# ${OUT_DIR}/${MODEL_TYPE}_${LOG_ID}/latest.npz
TRAIN_ARGS=(
  --config "${CONFIG}"
  --model-type "${MODEL_TYPE}"
  --log-id "${LOG_ID}"
  --generations "${GENERATIONS}"
  --seed "${SEED}"
  --out-dir "${OUT_DIR}"
)

if [[ -n "${RESUME}" ]]; then
  TRAIN_ARGS+=(--resume "${RESUME}")
fi

if [[ -n "${EXTRA_ARGS}" ]]; then
  # Intentional word splitting lets callers pass flags such as:
  # EXTRA_ARGS="--population-size 64 --episode-seconds 20"
  TRAIN_ARGS+=(${EXTRA_ARGS})
fi

python train_headless.py "${TRAIN_ARGS[@]}"
