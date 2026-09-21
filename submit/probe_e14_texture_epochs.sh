#!/bin/bash
#SBATCH -J E14_texture_epochs
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -o log_e14_texture_epochs_%j.log
#SBATCH -e log_e14_texture_epochs_%j.err
set -eo pipefail
if [[ "${DRY_RUN:-0}" != 1 ]]; then
  source /share/apps/anaconda3/etc/profile.d/conda.sh
  conda activate Mymodel
fi
set -u
PROJECT_ROOT="${PROJECT_ROOT:-/share/home/u2515283058/Mymodel}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${PROJECT_ROOT}/output}"
INPUTS_ROOT="${INPUTS_ROOT:-${PROJECT_ROOT}/eval_outputs/e14_real_orientation_inputs}"
EVAL_ROOT="${EVAL_ROOT:-${PROJECT_ROOT}/eval_outputs/e14_texture_epochs/${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
cd "${PROJECT_ROOT}"
CMD=(python -m tools.e14_checkpoint_history run --suite texture_epochs
  --checkpoint-root "${CHECKPOINT_ROOT}" --inputs "${INPUTS_ROOT}" --output "${EVAL_ROOT}"
  --base-model "${SD_MODEL:-${PROJECT_ROOT}/models/stable-diffusion-v1-5}"
  --clip-model "${CLIP_MODEL:-${PROJECT_ROOT}/models/clip}")
if [[ "${AUDIT_ONLY:-0}" == 1 ]]; then CMD+=(--audit-only); fi
printf '%q ' "${CMD[@]}"; printf '\n'
if [[ "${DRY_RUN:-0}" != 1 ]]; then "${CMD[@]}"; fi
echo "输出目录：${EVAL_ROOT}"
echo "下载后本地运行：python -m tools.e14_checkpoint_history evaluate --root <结果目录>"
