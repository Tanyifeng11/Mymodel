#!/bin/bash
#SBATCH -J E14_train_inputs
#SBATCH -p gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -o log_e14_train_inputs_%j.log
#SBATCH -e log_e14_train_inputs_%j.err
set -eo pipefail
if [[ "${DRY_RUN:-0}" != 1 ]]; then
  source /share/apps/anaconda3/etc/profile.d/conda.sh
  conda activate Mymodel
fi
set -u
PROJECT_ROOT="${PROJECT_ROOT:-/share/home/u2515283058/Mymodel}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
cd "${PROJECT_ROOT}"
CMD=(python -m tools.e14_checkpoint_history run
  --suite texture_endpoints --preprocess-protocol texture_train
  --checkpoint-root "${CHECKPOINT_ROOT:-${PROJECT_ROOT}/output}"
  --inputs "${INPUTS_ROOT:-${PROJECT_ROOT}/eval_outputs/e14_real_orientation_inputs}"
  --output "${EVAL_ROOT:-${PROJECT_ROOT}/eval_outputs/e14_train_inputs/${SLURM_JOB_ID:-local}}"
  --base-model "${SD_MODEL:-${PROJECT_ROOT}/models/stable-diffusion-v1-5}"
  --clip-model "${CLIP_MODEL:-${PROJECT_ROOT}/models/clip}")
printf '%q ' "${CMD[@]}"; printf '\n'
if [[ "${DRY_RUN:-0}" != 1 ]]; then "${CMD[@]}"; fi
