#!/bin/bash
#SBATCH --job-name=quad-vla-harness
#SBATCH --gres=gpu:h100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/pace_vla_harness_%j.out
#SBATCH --error=logs/pace_vla_harness_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs
source venv/bin/activate

CONFIG=${CONFIG:-configs/default.yaml}
MODEL_ID=${MODEL_ID:-}
STEPS=${STEPS:-1600}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-24}
CAMERA_WIDTH=${CAMERA_WIDTH:-256}
CAMERA_HEIGHT=${CAMERA_HEIGHT:-256}
CAMERA_FOVY=${CAMERA_FOVY:-70}
INSTRUCTION=${INSTRUCTION:-walk forward then turn left and stabilize}

if [[ -z "${MODEL_ID}" ]]; then
  echo "[ERROR] MODEL_ID must be provided, e.g. MODEL_ID=Qwen/Qwen2.5-VL-3B-Instruct" >&2
  exit 2
fi

python run_vla_harness.py \
  --config "${CONFIG}" \
  --model-id "${MODEL_ID}" \
  --instruction "${INSTRUCTION}" \
  --steps "${STEPS}" \
  --camera-width "${CAMERA_WIDTH}" \
  --camera-height "${CAMERA_HEIGHT}" \
  --camera-fovy "${CAMERA_FOVY}" \
  --max-new-tokens "${MAX_NEW_TOKENS}"

