#!/bin/bash
#SBATCH -J E14_supervision
#SBATCH -p gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH -o log_e14_supervision_%j.log
#SBATCH -e log_e14_supervision_%j.err
set -eo pipefail
if [[ "${DRY_RUN:-0}" != 1 ]]; then
  source /share/apps/anaconda3/etc/profile.d/conda.sh
  conda activate Mymodel
fi
set -u
PROJECT_ROOT="${PROJECT_ROOT:-/share/home/u2515283058/Mymodel}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4
cd "${PROJECT_ROOT}"
CMD=(python -m tools.e14_training_supervision_audit
  --manifest "${TRAIN_JSON:-${PROJECT_ROOT}/data/train_bf_texture.json}"
  --data-root "${DATA_ROOT:-/share/home/u2515283058/datasets/BF/training}"
  --output "${EVAL_ROOT:-${PROJECT_ROOT}/eval_outputs/e14_supervision/${SLURM_JOB_ID:-local}}"
  --mode "${TEXTURE_PREPROCESS_MODE:-plain_resize}" --width "${WIDTH:-384}" --height "${HEIGHT:-512}"
  --count "${COUNT:-40}" --device cuda:0)
# 默认检查数据与预处理；已有VGG缓存时用VGG=1追加真实Gram损失敏感性。
if [[ "${VGG:-0}" == 1 ]]; then CMD+=(--vgg); fi
printf '%q ' "${CMD[@]}"; printf '\n'
if [[ "${DRY_RUN:-0}" != 1 ]]; then "${CMD[@]}"; fi
