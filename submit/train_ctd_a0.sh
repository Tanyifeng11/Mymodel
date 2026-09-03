#!/bin/bash
# A0 续训控制组。提交前以环境变量复现 CTD 全量 run 的实际训练配置。
# 例：A0_MAX_TRAIN_SAMPLES=... A0_NUM_TRAIN_EPOCHS=... sbatch submit/train_ctd_a0.sh

#SBATCH -J ctd_a0
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --gres=gpu:1
#SBATCH -o /share/home/u2515283058/Mymodel/log_ctd_a0_%j.log
#SBATCH -e /share/home/u2515283058/Mymodel/log_ctd_a0_%j.err

set -eo pipefail
source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate Mymodel
set -u

export HTTP_PROXY="${HTTP_PROXY:-http://211.67.63.75:3128}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://211.67.63.75:3128}"
export http_proxy="${http_proxy:-${HTTP_PROXY}}"
export https_proxy="${https_proxy:-${HTTPS_PROXY}}"

PROJECT_ROOT="${PROJECT_ROOT:-/share/home/u2515283058/Mymodel}"
OUTPUT_BASE="${OUTPUT_BASE:-${PROJECT_ROOT}/output}"

# 这些默认值复现 submit/train_ctd_stage_a.sh；若 ctd_full 使用过覆盖值，必须同步覆盖。
A0_OUTPUT_DIR="${A0_OUTPUT_DIR:-${OUTPUT_BASE}/phase1_ctd_stage_a_gamut_p030_full_a0}"
A0_RESUME_CKPT="${A0_RESUME_CKPT:-${OUTPUT_BASE}/phase1_e5_tcpm_lite_e3/checkpoint-final/joint_model.pt}"

cd "${PROJECT_ROOT}"

NUM_GPUS="${NUM_GPUS:-1}" \
A0_OUTPUT_DIR="${A0_OUTPUT_DIR}" \
A0_RESUME_CKPT="${A0_RESUME_CKPT}" \
A0_TARGET_STRATEGY="${A0_TARGET_STRATEGY:-gamut_aware}" \
MAX_TRAIN_SAMPLES="${A0_MAX_TRAIN_SAMPLES:-${MAX_TRAIN_SAMPLES:-2000}}" \
CTD_NUM_TRAIN_EPOCHS="${A0_NUM_TRAIN_EPOCHS:-${CTD_NUM_TRAIN_EPOCHS:-3}}" \
CTD_MAX_TRAIN_STEPS="${A0_MAX_TRAIN_STEPS:-${CTD_MAX_TRAIN_STEPS:-300}}" \
WANDB_MODE="${WANDB_MODE:-offline}" \
DRY_RUN="${DRY_RUN:-0}" \
bash scripts/train_phase1_ctd_a0.sh
