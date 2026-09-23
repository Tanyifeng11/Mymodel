#!/bin/bash
#SBATCH -J E14_matched
#SBATCH -p gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH -o log_e14_matched_%j.log
#SBATCH -e log_e14_matched_%j.err
set -eo pipefail
if [[ "${DRY_RUN:-0}" != 1 ]]; then
 source /share/apps/anaconda3/etc/profile.d/conda.sh
 conda activate Mymodel
fi
set -u
PROJECT_ROOT="${PROJECT_ROOT:-/share/home/u2515283058/Mymodel}"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4
CMD=(python -m tools.e14_matched_denoising
 --manifest "${TRAIN_JSON:-${PROJECT_ROOT}/data/train_bf_texture.json}"
 --data-root "${DATA_ROOT:-/share/home/u2515283058/datasets/BF/training}"
 --checkpoint "${CHECKPOINT:-${PROJECT_ROOT}/output/texture_adapter_bf_e20/checkpoint-final/pytorch_model.bin}"
 --base-model "${PROJECT_ROOT}/models/stable-diffusion-v1-5"
 --clip-model "${PROJECT_ROOT}/models/clip" --count "${COUNT:-32}"
 --output "${EVAL_ROOT:-${PROJECT_ROOT}/eval_outputs/e14_matched_denoising/${SLURM_JOB_ID:-local}}")
printf '%q ' "${CMD[@]}"; printf '\n'
if [[ "${DRY_RUN:-0}" != 1 ]]; then "${CMD[@]}"; fi
