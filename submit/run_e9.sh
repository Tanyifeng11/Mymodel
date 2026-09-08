#!/bin/bash
# 提交：sbatch submit/run_e9.sh
# E9 首轮：从 E5 分别新训 A(旁路读原 16 token) 与 B(旁路读压缩前局部 token) 两组，
# 完全冻结原模型，只训练新增局部旁路，训练完成后自动验证 E5 / A / B 三组。

#SBATCH -J E9
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -o /share/home/u2515283058/Mymodel/log_e9_%j.log
#SBATCH -e /share/home/u2515283058/Mymodel/log_e9_%j.err

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
export A_OUTPUT_DIR="${A_OUTPUT_DIR:-${OUTPUT_BASE}/phase1_local_detail_resampled}"
export B_OUTPUT_DIR="${B_OUTPUT_DIR:-${OUTPUT_BASE}/phase1_local_detail_local}"
export BF_SPLIT="${BF_SPLIT:-validation}"
export NUM_SAMPLES="${NUM_SAMPLES:-100}"
export GENERATION_SEEDS="${GENERATION_SEEDS:-42}"
export EVAL_SCOPE="${EVAL_SCOPE:-metrics}"
export EVAL_RUN_ID="${EVAL_RUN_ID:-${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}"
export EVAL_ROOT="${EVAL_ROOT:-${PROJECT_ROOT}/eval_outputs/local_detail_${EVAL_SCOPE}_${BF_SPLIT}_${NUM_SAMPLES}/${EVAL_RUN_ID}}"
# A/B 必须同训练量、同数据、同新增参数配置，唯一差别是旁路读取的特征来源。
export MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-1000}"
export LOCAL_DETAIL_GRID="${LOCAL_DETAIL_GRID:-16}"
export LOCAL_DETAIL_DIM="${LOCAL_DETAIL_DIM:-128}"
export LOCAL_DETAIL_HEADS="${LOCAL_DETAIL_HEADS:-4}"
export LOCAL_DETAIL_LR="${LOCAL_DETAIL_LR:-5e-5}"
export TRAIN_SEED="${TRAIN_SEED:-42}"
export NUM_GPUS=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
cd "${PROJECT_ROOT}"

for output_dir in "${A_OUTPUT_DIR}" "${B_OUTPUT_DIR}"; do
  for checkpoint in "${output_dir}"/checkpoint-*/joint_model.pt; do
    if [[ -f "${checkpoint}" ]]; then
      echo "已有训练权重：${checkpoint}；请用 submit/eval_e9.sh 验证，或指定新的输出目录。" >&2
      exit 1
    fi
  done
done

# 两组顺序训练；A 组失败即退出，避免只拿到半组结果做比较。
for variant in resampled:${A_OUTPUT_DIR} local:${B_OUTPUT_DIR}; do
  source_name="${variant%%:*}"
  output_dir="${variant#*:}"
  echo "[E9] 从 E5 新训局部旁路：source=${source_name}, 步数=${MAX_TRAIN_STEPS}, 输出=${output_dir}"
  LOCAL_DETAIL_SOURCE="${source_name}" OUTPUT_DIR="${output_dir}" \
  DATA_ROOT_PATH="${DATASETS_ROOT}/BF/training" \
  bash scripts/train_phase1_local_detail.sh
done

# 训练成功后再评测；三组严格使用同一划分与同一生成 seed。
echo "[E9] 开始验证 E5 / E9-A / E9-B，scope=${EVAL_SCOPE}，每组=${NUM_SAMPLES} 张"
E5_CKPT="${BASE_CKPT}" \
E9_A_CKPT="${A_OUTPUT_DIR}/checkpoint-final/joint_model.pt" \
E9_B_CKPT="${B_OUTPUT_DIR}/checkpoint-final/joint_model.pt" \
TEXTURE_CKPT="${TEXTURE_ADAPTER_CKPT}" \
DATA_ROOT_PATH="${DATASETS_ROOT}/BF/${BF_SPLIT}" \
EXPERIMENTS=e5,e9_a,e9_b DEVICE=cuda:0 \
bash scripts/eval_phase1_local_detail.sh

echo "[E9] 流程结束，评测目录：${EVAL_ROOT}（DRY_RUN=${DRY_RUN:-0}）"
echo "[E9] 判读顺序：先看 frozen_check 是否 passed，再看 image_check 确认旁路已生效，"
echo "[E9] 最后比较 B 相对 A 的增益；仅颜色变化不算达到本轮目标。"
