#!/usr/bin/env bash
# A0：与 CTD-A 完全同配方的续训控制组，唯一差异是关闭条件端色度扰动。
# 提交前必须把 A0_* 覆盖为 CTD 全量 run 实际使用的样本数、步数与其他训练参数。
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/share/home/u2515283058/Mymodel}"
OUTPUT_BASE="${OUTPUT_BASE:-${PROJECT_ROOT}/output}"

A0_OUTPUT_DIR="${A0_OUTPUT_DIR:-${OUTPUT_BASE}/phase1_ctd_stage_a_gamut_p030_full_a0}"
A0_RESUME_CKPT="${A0_RESUME_CKPT:-${OUTPUT_BASE}/phase1_e5_tcpm_lite_e3/checkpoint-final/joint_model.pt}"

export CTD_OUTPUT_DIR="${A0_OUTPUT_DIR}"
export CTD_RESUME_CKPT="${A0_RESUME_CKPT}"
export CTD_PROB=0
export CTD_ALL_SAMPLES=0
export CTD_TARGET_STRATEGY="${A0_TARGET_STRATEGY:-gamut_aware}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-ctd_stage_a_a0}"

echo "[A0] 仅关闭 ctd_prob；其余 CTD_* / GAM_* 环境变量会原样传入 CTD 训练脚本。"
echo "[A0] resume=${CTD_RESUME_CKPT}"
echo "[A0] output=${CTD_OUTPUT_DIR}"

exec bash "${PROJECT_ROOT}/scripts/train_phase1_ctd_stage_a.sh"
