#!/bin/bash

# 仅重跑 caption 颜色扫描的评分阶段：不重新生成 576 张图，不申请 GPU。
# 使用前先将 tools/sweep_caption_color.py 的评分修复同步到服务器。
# 提交：sbatch submit/score_caption_color_sweep.sh
# 可覆盖：PROJECT_ROOT=/path OUT_DIR=/path sbatch submit/score_caption_color_sweep.sh

#SBATCH -J color_score
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -o /share/home/u2515283058/Mymodel/log_color_score_%j.log
#SBATCH -e /share/home/u2515283058/Mymodel/log_color_score_%j.err

set -eo pipefail
source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate Mymodel
set -u

PROJECT_ROOT="${PROJECT_ROOT:-/share/home/u2515283058/Mymodel}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/output_eval/caption_color_sweep/e5}"
MASK_POLICY="${MASK_POLICY:-sketch_only}"
REQUIRE_MASK_BACKEND="${REQUIRE_MASK_BACKEND:-opencv}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
cd "${PROJECT_ROOT}"

echo "[score] out_dir=${OUT_DIR}"
python tools/sweep_caption_color.py \
  --stage score \
  --out_dir "${OUT_DIR}" \
  --mask_policy "${MASK_POLICY}" \
  --require_mask_backend "${REQUIRE_MASK_BACKEND}"

echo "[done] ${OUT_DIR}/sweep_summary.json"
