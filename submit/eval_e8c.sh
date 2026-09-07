#!/bin/bash
# 已有 E8c 权重时单独验证：sbatch submit/eval_e8c.sh
# 指定中间权重：TEXT_CKPT=/完整路径/checkpoint-250/joint_model.pt sbatch submit/eval_e8c.sh

#SBATCH -J E8c_eval
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -o /share/home/u2515283058/Mymodel/log_e8c_eval_%j.log
#SBATCH -e /share/home/u2515283058/Mymodel/log_e8c_eval_%j.err

set -eo pipefail
if [[ "${DRY_RUN:-0}" != "1" ]]; then
  source /share/apps/anaconda3/etc/profile.d/conda.sh
  conda activate Mymodel
fi
set -u

export PROJECT_ROOT="${PROJECT_ROOT:-/share/home/u2515283058/Mymodel}"
export DATASETS_ROOT="${DATASETS_ROOT:-/share/home/u2515283058/datasets}"
export OUTPUT_BASE="${OUTPUT_BASE:-${PROJECT_ROOT}/output}"
export TEXT_CKPT="${TEXT_CKPT:-${OUTPUT_BASE}/phase1_resampler_text_only/checkpoint-final/joint_model.pt}"
export BF_SPLIT="${BF_SPLIT:-validation}"
export NUM_SAMPLES="${NUM_SAMPLES:-100}"
export GENERATION_SEEDS="${GENERATION_SEEDS:-42}"
export EVAL_RUN_ID="${EVAL_RUN_ID:-${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}"
export EVAL_ROOT="${EVAL_ROOT:-${PROJECT_ROOT}/eval_outputs/resampler_text_only_${BF_SPLIT}_${NUM_SAMPLES}/${EVAL_RUN_ID}}"
export EXPERIMENTS="${EXPERIMENTS:-e5,text_only_off,text_only_on}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
cd "${PROJECT_ROOT}"

echo "[E8c] 单独验证，划分=${BF_SPLIT}，每组=${NUM_SAMPLES} 张，权重=${TEXT_CKPT}"
DEVICE=cuda:0 bash scripts/eval_phase1_text_guided_resampler.sh
echo "[E8c] 评测目录：${EVAL_ROOT}（DRY_RUN=${DRY_RUN:-0}）"
