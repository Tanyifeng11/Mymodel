#!/bin/bash

# CTD Stage A 小规模训练：默认先跑 300 步 D0/D3 诊断，避免在登录节点直接启动训练。
# 提交：sbatch submit/train_ctd_stage_a.sh
# 覆盖：CTD_PROB=0.3 CTD_OUTPUT_DIR=/path sbatch submit/train_ctd_stage_a.sh

#SBATCH -J ctd_a
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --gres=gpu:1
#SBATCH -o /share/home/u2515283058/Mymodel/log_ctd_a_%j.log
#SBATCH -e /share/home/u2515283058/Mymodel/log_ctd_a_%j.err

set -eo pipefail
source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate Mymodel
set -u

export HTTP_PROXY="${HTTP_PROXY:-http://211.67.63.75:3128}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://211.67.63.75:3128}"
export http_proxy="${http_proxy:-${HTTP_PROXY}}"
export https_proxy="${https_proxy:-${HTTPS_PROXY}}"

PROJECT_ROOT="${PROJECT_ROOT:-/share/home/u2515283058/Mymodel}"
CTD_OUTPUT_DIR="${CTD_OUTPUT_DIR:-${PROJECT_ROOT}/output/phase1_ctd_stage_a_gamut_p030}"
CTD_MAX_TRAIN_STEPS="${CTD_MAX_TRAIN_STEPS:-300}"

cd "${PROJECT_ROOT}"

NUM_GPUS="${NUM_GPUS:-1}" \
CTD_TARGET_STRATEGY="${CTD_TARGET_STRATEGY:-gamut_aware}" \
CTD_PROB="${CTD_PROB:-0.3}" \
CTD_OUTPUT_DIR="${CTD_OUTPUT_DIR}" \
CTD_MAX_TRAIN_STEPS="${CTD_MAX_TRAIN_STEPS}" \
WANDB_MODE="${WANDB_MODE:-offline}" \
DRY_RUN="${DRY_RUN:-0}" \
bash scripts/train_phase1_ctd_stage_a.sh
