#!/bin/bash
# 提交：sbatch submit/run_e9_c_highpass_gate.sh
# E9-C 快速门槛：从原始 E5 新训高通约束的 local(B 路)旁路，并评测 seed 42 / 100 张指标集。

#SBATCH -J E9C_gate
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -o /share/home/u2515283058/Mymodel/log_e9_c_gate_%j.log
#SBATCH -e /share/home/u2515283058/Mymodel/log_e9_c_gate_%j.err

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

# E9-C 与 E9-B 相同的训练预算；只改变 local adapter 的输出约束。
export MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-1000}"
export C_OUTPUT_DIR="${C_OUTPUT_DIR:-${OUTPUT_BASE}/phase1_local_detail_c_highpass}"
export CHECKPOINTING_STEPS="${CHECKPOINTING_STEPS:-250}"
export NUM_WARMUP_STEPS="${NUM_WARMUP_STEPS:-50}"
export LOCAL_DETAIL_GRID="${LOCAL_DETAIL_GRID:-16}"
export LOCAL_DETAIL_DIM="${LOCAL_DETAIL_DIM:-128}"
export LOCAL_DETAIL_HEADS="${LOCAL_DETAIL_HEADS:-4}"
export LOCAL_DETAIL_LR="${LOCAL_DETAIL_LR:-5e-5}"
export LOCAL_DETAIL_OUTPUT_CONSTRAINT=highpass
export LOCAL_DETAIL_HIGHPASS_KERNEL="${LOCAL_DETAIL_HIGHPASS_KERNEL:-3}"
export TRAIN_SEED="${TRAIN_SEED:-42}"

# 快速门槛只跑一组 seed；报告会明确标记为方向筛选，不能替代多 seed 结论。
export BF_SPLIT=validation
export EVAL_NUM_SAMPLES=100
export GENERATION_SEEDS=42
export EVAL_RUN_ID="${EVAL_RUN_ID:-${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}"
export METRICS_EVAL_ROOT="${METRICS_EVAL_ROOT:-${PROJECT_ROOT}/eval_outputs/local_detail_c_highpass_gate_metrics_${BF_SPLIT}_${EVAL_NUM_SAMPLES}/${EVAL_RUN_ID}}"
export NUM_GPUS=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
cd "${PROJECT_ROOT}"

if [[ "${DRY_RUN:-0}" != "1" ]]; then
  for path in "${BASE_CKPT}" "${TEXTURE_ADAPTER_CKPT}"; do
    [[ -f "${path}" ]] || { echo "缺少 checkpoint：${path}" >&2; exit 1; }
  done
  if [[ -e "${C_OUTPUT_DIR}/checkpoint-final/joint_model.pt" ]]; then
    echo "目标目录已有最终权重：${C_OUTPUT_DIR}；请评估已有 E9-C 或指定新的 C_OUTPUT_DIR。" >&2
    exit 1
  fi
fi

echo "[E9-C] 从 E5 新训：高通核=${LOCAL_DETAIL_HIGHPASS_KERNEL}，总步数=${MAX_TRAIN_STEPS}"
LOCAL_DETAIL_SOURCE=local \
BASE_CKPT="${BASE_CKPT}" TEXTURE_ADAPTER_CKPT="${TEXTURE_ADAPTER_CKPT}" \
OUTPUT_DIR="${C_OUTPUT_DIR}" DATA_ROOT_PATH="${DATASETS_ROOT}/BF/training" \
bash scripts/train_phase1_local_detail.sh

C_CKPT="${C_OUTPUT_DIR}/checkpoint-final/joint_model.pt"
if [[ "${DRY_RUN:-0}" != "1" ]]; then
  python tools/check_e9_frozen.py --e5-ckpt "${BASE_CKPT}" --e9-c-ckpt "${C_CKPT}" \
    --output-dir "${C_OUTPUT_DIR}/frozen_check"
fi

echo "[E9-C] 快速 metrics 评测：seed=${GENERATION_SEEDS}，样本=${EVAL_NUM_SAMPLES}"
E5_CKPT="${BASE_CKPT}" E9_C_CKPT="${C_CKPT}" TEXTURE_CKPT="${TEXTURE_ADAPTER_CKPT}" \
DATA_ROOT_PATH="${DATASETS_ROOT}/BF/${BF_SPLIT}" EXPERIMENTS=e5,e9_c \
NUM_SAMPLES="${EVAL_NUM_SAMPLES}" EVAL_SCOPE=metrics EVAL_ROOT="${METRICS_EVAL_ROOT}" \
GENERATION_SEEDS="${GENERATION_SEEDS}" DEVICE=cuda:0 \
bash scripts/eval_phase1_local_detail.sh

echo "[E9-C] 完成：训练=${C_OUTPUT_DIR}"
echo "[E9-C] 快速评测=${METRICS_EVAL_ROOT}"
