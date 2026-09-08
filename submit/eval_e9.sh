#!/bin/bash
# 已有 E9 权重时单独验证：sbatch submit/eval_e9.sh
# 指定中间权重：E9_B_CKPT=/完整路径/checkpoint-250/joint_model.pt sbatch submit/eval_e9.sh
# 图案定性对照：EVAL_SCOPE=pattern NUM_SAMPLES=48 sbatch submit/eval_e9.sh

#SBATCH -J E9_eval
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -o /share/home/u2515283058/Mymodel/log_e9_eval_%j.log
#SBATCH -e /share/home/u2515283058/Mymodel/log_e9_eval_%j.err

set -eo pipefail
if [[ "${DRY_RUN:-0}" != "1" ]]; then
  source /share/apps/anaconda3/etc/profile.d/conda.sh
  conda activate Mymodel
fi
set -u

export PROJECT_ROOT="${PROJECT_ROOT:-/share/home/u2515283058/Mymodel}"
export DATASETS_ROOT="${DATASETS_ROOT:-/share/home/u2515283058/datasets}"
export OUTPUT_BASE="${OUTPUT_BASE:-${PROJECT_ROOT}/output}"
export E5_CKPT="${E5_CKPT:-${OUTPUT_BASE}/phase1_e5_tcpm_lite_e3/checkpoint-final/joint_model.pt}"
export E9_A_CKPT="${E9_A_CKPT:-${OUTPUT_BASE}/phase1_local_detail_resampled/checkpoint-final/joint_model.pt}"
export E9_B_CKPT="${E9_B_CKPT:-${OUTPUT_BASE}/phase1_local_detail_local/checkpoint-final/joint_model.pt}"
export TEXTURE_CKPT="${TEXTURE_CKPT:-${OUTPUT_BASE}/texture_adapter_bf_e20/checkpoint-final/texture_adapter.bin}"
export BF_SPLIT="${BF_SPLIT:-validation}"
export NUM_SAMPLES="${NUM_SAMPLES:-100}"
export GENERATION_SEEDS="${GENERATION_SEEDS:-42}"
export EVAL_SCOPE="${EVAL_SCOPE:-metrics}"
export EVAL_RUN_ID="${EVAL_RUN_ID:-${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}"
export EVAL_ROOT="${EVAL_ROOT:-${PROJECT_ROOT}/eval_outputs/local_detail_${EVAL_SCOPE}_${BF_SPLIT}_${NUM_SAMPLES}/${EVAL_RUN_ID}}"
export EXPERIMENTS="${EXPERIMENTS:-e5,e9_a,e9_b}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
cd "${PROJECT_ROOT}"

echo "[E9] 单独验证，scope=${EVAL_SCOPE}，划分=${BF_SPLIT}，每组=${NUM_SAMPLES} 张"
echo "[E9] A=${E9_A_CKPT}"
echo "[E9] B=${E9_B_CKPT}"
DATA_ROOT_PATH="${DATASETS_ROOT}/BF/${BF_SPLIT}" DEVICE=cuda:0 \
bash scripts/eval_phase1_local_detail.sh
echo "[E9] 评测目录：${EVAL_ROOT}（DRY_RUN=${DRY_RUN:-0}）"
