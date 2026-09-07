#!/bin/bash
# 提交：sbatch submit/run_e8c.sh
# 从 E5 新训 E8c：完全冻结原模型，只训练新增文本模块，再自动验证三组。

#SBATCH -J E8c
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -o /share/home/u2515283058/Mymodel/log_e8c_%j.log
#SBATCH -e /share/home/u2515283058/Mymodel/log_e8c_%j.err

set -eo pipefail
if [[ "${DRY_RUN:-0}" != "1" ]]; then
  source /share/apps/anaconda3/etc/profile.d/conda.sh
  conda activate Mymodel
fi
set -u

export PROJECT_ROOT="${PROJECT_ROOT:-/share/home/u2515283058/Mymodel}"
export DATASETS_ROOT="${DATASETS_ROOT:-/share/home/u2515283058/datasets}"
export OUTPUT_BASE="${OUTPUT_BASE:-${PROJECT_ROOT}/output}"
export BASE_CKPT="${BASE_CKPT:-${OUTPUT_BASE}/phase1_e5_tcpm_lite_e3/checkpoint-final/joint_model.pt}"
export TEXTURE_ADAPTER_CKPT="${TEXTURE_ADAPTER_CKPT:-${OUTPUT_BASE}/texture_adapter_bf_e20/checkpoint-final/texture_adapter.bin}"
export OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_BASE}/phase1_resampler_text_only}"
export BF_SPLIT="${BF_SPLIT:-validation}"
export NUM_SAMPLES="${NUM_SAMPLES:-100}"
export GENERATION_SEEDS="${GENERATION_SEEDS:-42}"
export EVAL_RUN_ID="${EVAL_RUN_ID:-${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}"
export EVAL_ROOT="${EVAL_ROOT:-${PROJECT_ROOT}/eval_outputs/resampler_text_only_${BF_SPLIT}_${NUM_SAMPLES}/${EVAL_RUN_ID}}"
export NUM_GPUS=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
cd "${PROJECT_ROOT}"

for checkpoint in "${OUTPUT_DIR}"/checkpoint-*/joint_model.pt; do
  if [[ -f "${checkpoint}" ]]; then
    echo "已有训练权重：${checkpoint}；请用 submit/eval_e8c.sh 验证，或指定新的 OUTPUT_DIR。" >&2
    exit 1
  fi
done
echo "[E8c] 从 E5 初始化，仅训练新增文本模块，步数=${MAX_TRAIN_STEPS:-1000}"
# 禁止沿用 E8a/E8b 的恢复状态；数据始终使用官方 training 划分。
RESAMPLER_MODE=text_only TGR_RESUME_CKPT="" START_GLOBAL_STEP=0 \
DATA_ROOT_PATH="${DATASETS_ROOT}/BF/training" \
bash scripts/train_phase1_text_guided_resampler.sh

# 训练成功后再评测；off/on 严格使用同一个训练后 checkpoint。
echo "[E8c] 开始验证 E5 / 关闭文本模块 / 开启文本模块，每组=${NUM_SAMPLES} 张"
E5_CKPT="${BASE_CKPT}" \
TEXT_CKPT="${OUTPUT_DIR}/checkpoint-final/joint_model.pt" \
TEXTURE_CKPT="${TEXTURE_ADAPTER_CKPT}" \
DATA_ROOT_PATH="${DATASETS_ROOT}/BF/${BF_SPLIT}" \
EXPERIMENTS=e5,text_only_off,text_only_on DEVICE=cuda:0 \
bash scripts/eval_phase1_text_guided_resampler.sh

echo "[E8c] 流程结束，评测目录：${EVAL_ROOT}（DRY_RUN=${DRY_RUN:-0}）"
