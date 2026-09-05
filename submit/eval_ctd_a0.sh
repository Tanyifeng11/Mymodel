#!/bin/bash
# A0 vs CTD 的筛选评测：默认 S1/S2/S3 各 1 seed；确认方向后再覆盖为 full 跑 3 seed。
# A0 和 CTD 必须来自同一 E5 起点、同一训练 schedule；两者唯一差异是 ctd_prob。

#SBATCH -J ctd_a0_eval
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --gres=gpu:1
#SBATCH -o /share/home/u2515283058/Mymodel/log_ctd_a0_eval_%j.log
#SBATCH -e /share/home/u2515283058/Mymodel/log_ctd_a0_eval_%j.err

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
A0_CKPT="${A0_CKPT:-${OUTPUT_BASE}/phase1_ctd_stage_a_gamut_p030_full_a0/checkpoint-final/joint_model.pt}"
CTD_CKPT="${CTD_CKPT:-${OUTPUT_BASE}/phase1_ctd_stage_a_gamut_p030_full/checkpoint-final/joint_model.pt}"
A0_EVAL_ROOT="${A0_EVAL_ROOT:-${PROJECT_ROOT}/output_eval/ctd_stage_a_gamut_p030_full_a0_eval}"

cd "${PROJECT_ROOT}"

CONTROL_NAME=a0 \
CONTROL_CKPT="${A0_CKPT}" \
CTD_CKPT="${CTD_CKPT}" \
CTD_EVAL_ROOT="${A0_EVAL_ROOT}" \
CTD_EVAL_REPORT_ROOT="${A0_EVAL_ROOT}/report" \
CTD_EVAL_MODE="${CTD_EVAL_MODE:-screen_all}" \
GENERATION_SEEDS="${GENERATION_SEEDS:-42}" \
SINGLE_SEED_REFERENCE_REPORT="${SINGLE_SEED_REFERENCE_REPORT:-1}" \
bash scripts/eval_ctd_stage_a.sh
