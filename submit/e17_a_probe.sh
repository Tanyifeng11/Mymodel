#!/bin/bash
#SBATCH -J E17_A_probe
#SBATCH -p gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -o /share/home/u2515283058/Mymodel/e17/e17_a_%j.log
#SBATCH -e /share/home/u2515283058/Mymodel/e17/e17_a_%j.err
set -eo pipefail
if [[ "${DRY_RUN:-0}" != 1 ]]; then
  source /share/apps/anaconda3/etc/profile.d/conda.sh
  conda activate Mymodel
fi
set -u
PROJECT_ROOT="${PROJECT_ROOT:-/share/home/u2515283058/Mymodel}"
E17_ROOT="${E17_ROOT:-${PROJECT_ROOT}/e17}"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=8
mkdir -p "${E17_ROOT}"
for PAIR in baseline:output/texture_adapter_bf_e20/checkpoint-final/pytorch_model.bin c_pattern64:output/e16_c_pattern64/checkpoint-final/pytorch_model.bin; do
  TAG="${PAIR%%:*}"
  CKPT="${PROJECT_ROOT}/${PAIR#*:}"
  if [[ ! -f "${CKPT}" && "${DRY_RUN:-0}" != 1 ]]; then
    echo "[skip] ${CKPT} absent"
    continue
  fi
  CMD=(python -m tools.e17_probe
    --manifest "${PROJECT_ROOT}/data/processed/bf_full_audit_v1/validation_clean.json"
    --data-root "${DATA_ROOT:-/share/home/u2515283058/datasets/BF}"
    --checkpoint "${CKPT}"
    --base-model "${PROJECT_ROOT}/models/stable-diffusion-v1-5"
    --clip-model "${PROJECT_ROOT}/models/clip"
    --labels "${PROJECT_ROOT}/eval_outputs/e14_candidates/candidates/labels.csv"
    --count "${COUNT:-256}" --orientation-count "${ORIENTATION_COUNT:-120}"
    --device cuda:0 --output "${E17_ROOT}/a_${TAG}")
  printf '%q ' "${CMD[@]}"; printf '\n'
  if [[ "${DRY_RUN:-0}" != 1 ]]; then "${CMD[@]}"; fi
done
