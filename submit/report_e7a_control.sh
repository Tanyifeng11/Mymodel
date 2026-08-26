#!/bin/bash

# 仅聚合已有逐样本 CSV，不重新生成图像或运行模型。
# 可在提交时通过 E7A_CONTROL_EVAL_ROOT、E7A_CONTROL_AUTO_ROOT、
# E7A_CONTROL_REPORT_ROOT、GENERATION_SEEDS 覆盖默认路径与 seed。
#SBATCH -J E7a_report
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -o /share/home/u2515283058/Mymodel/log_e7a_report_%j.log
#SBATCH -e /share/home/u2515283058/Mymodel/log_e7a_report_%j.err

set -eo pipefail
source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate Mymodel
set -u

PROJECT_ROOT="${PROJECT_ROOT:-/share/home/u2515283058/Mymodel}"
EVAL_ROOT="${E7A_CONTROL_EVAL_ROOT:-${PROJECT_ROOT}/eval_outputs/e7a_control/sketch_only}"
AUTO_ROOT="${E7A_CONTROL_AUTO_ROOT:-${PROJECT_ROOT}/eval_outputs/e7a_control/auto}"
REPORT_ROOT="${E7A_CONTROL_REPORT_ROOT:-${PROJECT_ROOT}/eval_outputs/e7a_control/report}"
GENERATION_SEEDS="${GENERATION_SEEDS:-42,123,2026}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-2000}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

python tools/report_e7a_control.py \
  --eval_root "${EVAL_ROOT}" \
  --output_dir "${REPORT_ROOT}/sketch_only" \
  --generation_seeds "${GENERATION_SEEDS}" \
  --bootstrap_samples "${BOOTSTRAP_SAMPLES}"

python tools/report_e7a_control.py \
  --eval_root "${AUTO_ROOT}" \
  --output_dir "${REPORT_ROOT}/auto" \
  --generation_seeds "${GENERATION_SEEDS}" \
  --bootstrap_samples "${BOOTSTRAP_SAMPLES}"

echo "[done] reports saved to ${REPORT_ROOT}"
