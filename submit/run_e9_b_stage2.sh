#!/bin/bash
# 提交：sbatch submit/run_e9_b_stage2.sh
# E9-B 第二阶段：从已验证的 local(B 组) checkpoint 再训练到总步数 MAX_TRAIN_STEPS，
# 然后用多 generation seed 分别完成指标集和图案集评测。

#SBATCH -J E9B_s2
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -o /share/home/u2515283058/Mymodel/log_e9_b_s2_%j.log
#SBATCH -e /share/home/u2515283058/Mymodel/log_e9_b_s2_%j.err

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
export E9_B_RESUME_CKPT="${E9_B_RESUME_CKPT:-${OUTPUT_BASE}/phase1_local_detail_local/checkpoint-final/joint_model.pt}"

# MAX_TRAIN_STEPS 是总步数：默认从 checkpoint-1000 续训 1000 步至 2000 步。
export MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-2000}"
export B_STAGE2_OUTPUT_DIR="${B_STAGE2_OUTPUT_DIR:-${OUTPUT_BASE}/phase1_local_detail_local_stage2_${MAX_TRAIN_STEPS}}"
export CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-250}"
export NUM_WARMUP_STEPS="${NUM_WARMUP_STEPS:-50}"
export LOCAL_DETAIL_GRID="${LOCAL_DETAIL_GRID:-16}"
export LOCAL_DETAIL_DIM="${LOCAL_DETAIL_DIM:-128}"
export LOCAL_DETAIL_HEADS="${LOCAL_DETAIL_HEADS:-4}"
export LOCAL_DETAIL_LR="${LOCAL_DETAIL_LR:-5e-5}"
export TRAIN_SEED="${TRAIN_SEED:-42}"

export BF_SPLIT="${BF_SPLIT:-validation}"
export EVAL_NUM_SAMPLES="${EVAL_NUM_SAMPLES:-100}"
export PATTERN_NUM_SAMPLES="${PATTERN_NUM_SAMPLES:-64}"
export GENERATION_SEEDS="${GENERATION_SEEDS:-42,52,62}"
export RUN_PATTERN_EVAL="${RUN_PATTERN_EVAL:-1}"
export EVAL_RUN_ID="${EVAL_RUN_ID:-${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}"
export METRICS_EVAL_ROOT="${METRICS_EVAL_ROOT:-${PROJECT_ROOT}/eval_outputs/local_detail_b_stage2_metrics_${BF_SPLIT}_${EVAL_NUM_SAMPLES}/${EVAL_RUN_ID}}"
export PATTERN_EVAL_ROOT="${PATTERN_EVAL_ROOT:-${PROJECT_ROOT}/eval_outputs/local_detail_b_stage2_pattern_${BF_SPLIT}_${PATTERN_NUM_SAMPLES}/${EVAL_RUN_ID}}"
export NUM_GPUS=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
cd "${PROJECT_ROOT}"

if [[ "${DRY_RUN:-0}" != "1" ]]; then
  for path in "${BASE_CKPT}" "${TEXTURE_ADAPTER_CKPT}" "${E9_B_RESUME_CKPT}"; do
    [[ -f "${path}" ]] || { echo "缺少 checkpoint：${path}" >&2; exit 1; }
  done
  if [[ -e "${B_STAGE2_OUTPUT_DIR}/checkpoint-final/joint_model.pt" ]]; then
    echo "目标目录已有最终权重：${B_STAGE2_OUTPUT_DIR}；请验证该结果或指定新的 B_STAGE2_OUTPUT_DIR。" >&2
    exit 1
  fi
fi

echo "[E9-B stage2] 起点=${E9_B_RESUME_CKPT}，总步数=${MAX_TRAIN_STEPS}，输出=${B_STAGE2_OUTPUT_DIR}"
# 先确认起点本身只包含合法的 B 组局部旁路，避免在错误权重上继续训练。
if [[ "${DRY_RUN:-0}" != "1" ]]; then
  python tools/check_e9_frozen.py --e5-ckpt "${BASE_CKPT}" --e9-b-ckpt "${E9_B_RESUME_CKPT}" \
    --output-dir "${B_STAGE2_OUTPUT_DIR}/preflight_frozen_check"
fi

LOCAL_DETAIL_SOURCE=local LOCAL_DETAIL_RESUME_CKPT="${E9_B_RESUME_CKPT}" \
BASE_CKPT="${BASE_CKPT}" TEXTURE_ADAPTER_CKPT="${TEXTURE_ADAPTER_CKPT}" \
OUTPUT_DIR="${B_STAGE2_OUTPUT_DIR}" DATA_ROOT_PATH="${DATASETS_ROOT}/BF/training" \
bash scripts/train_phase1_local_detail.sh

STAGE2_CKPT="${B_STAGE2_OUTPUT_DIR}/checkpoint-final/joint_model.pt"
if [[ "${DRY_RUN:-0}" != "1" ]]; then
  python tools/check_e9_frozen.py --e5-ckpt "${BASE_CKPT}" --e9-b-ckpt "${STAGE2_CKPT}" \
    --output-dir "${B_STAGE2_OUTPUT_DIR}/postflight_frozen_check"
fi

echo "[E9-B stage2] 多 seed 指标评测：${GENERATION_SEEDS}"
E5_CKPT="${BASE_CKPT}" E9_B_CKPT="${STAGE2_CKPT}" TEXTURE_CKPT="${TEXTURE_ADAPTER_CKPT}" \
DATA_ROOT_PATH="${DATASETS_ROOT}/BF/${BF_SPLIT}" EXPERIMENTS=e5,e9_b \
NUM_SAMPLES="${EVAL_NUM_SAMPLES}" EVAL_SCOPE=metrics EVAL_ROOT="${METRICS_EVAL_ROOT}" \
GENERATION_SEEDS="${GENERATION_SEEDS}" DEVICE=cuda:0 \
bash scripts/eval_phase1_local_detail.sh

if [[ "${RUN_PATTERN_EVAL}" == "1" ]]; then
  echo "[E9-B stage2] 多 seed 图案评测：${GENERATION_SEEDS}"
  E5_CKPT="${BASE_CKPT}" E9_B_CKPT="${STAGE2_CKPT}" TEXTURE_CKPT="${TEXTURE_ADAPTER_CKPT}" \
  DATA_ROOT_PATH="${DATASETS_ROOT}/BF/${BF_SPLIT}" EXPERIMENTS=e5,e9_b \
  NUM_SAMPLES="${PATTERN_NUM_SAMPLES}" EVAL_SCOPE=pattern EVAL_ROOT="${PATTERN_EVAL_ROOT}" \
  GENERATION_SEEDS="${GENERATION_SEEDS}" DEVICE=cuda:0 \
  bash scripts/eval_phase1_local_detail.sh
fi

echo "[E9-B stage2] 完成：训练=${B_STAGE2_OUTPUT_DIR}"
echo "[E9-B stage2] 指标评测=${METRICS_EVAL_ROOT}"
if [[ "${RUN_PATTERN_EVAL}" == "1" ]]; then
  echo "[E9-B stage2] 图案评测=${PATTERN_EVAL_ROOT}"
fi
