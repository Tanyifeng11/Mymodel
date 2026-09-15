#!/bin/bash
# 提交：sbatch submit/run_e12.sh
# 同一作业内训练 E12，再测试 E5 / OFF / Correct / Shuffled。
# DRY_RUN=1 bash submit/run_e12.sh 查看命令；BF_SPLIT=test 切换到测试集。
#SBATCH -J E12
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -o /share/home/u2515283058/Mymodel/log_e12_%j.log
#SBATCH -e /share/home/u2515283058/Mymodel/log_e12_%j.err

set -eo pipefail
if [[ "${DRY_RUN:-0}" != "1" ]]; then
  source /share/apps/anaconda3/etc/profile.d/conda.sh
  conda activate Mymodel
fi
set -u
export PROJECT_ROOT="${PROJECT_ROOT:-/share/home/u2515283058/Mymodel}"
export DATASETS_ROOT="${DATASETS_ROOT:-/share/home/u2515283058/datasets}"
export DATA_ROOT_PATH="${DATA_ROOT_PATH:-${DATASETS_ROOT}/BF}"
export OUTPUT_BASE="${OUTPUT_BASE:-${PROJECT_ROOT}/output}"
export OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_BASE}/phase1_e12_nexus}"
export BASE_CKPT="${BASE_CKPT:-${OUTPUT_BASE}/phase1_e5_tcpm_lite_e3/checkpoint-final/joint_model.pt}"
export TEXTURE_ADAPTER_CKPT="${TEXTURE_ADAPTER_CKPT:-${OUTPUT_BASE}/texture_adapter_bf_e20/checkpoint-final/texture_adapter.bin}"
# 使用 E11 已有的正确文本训练清单，不重新采样或构造冲突数据。
export TRAIN_JSON="${TRAIN_JSON:-${PROJECT_ROOT}/data/processed/bf_film_v1/training_current.json}"
export TRAIN_SEED="${TRAIN_SEED:-42}"
export BF_SPLIT="${BF_SPLIT:-validation}"
case "${BF_SPLIT}" in validation|test) ;; *) echo "BF_SPLIT 必须为 validation 或 test" >&2; exit 1 ;; esac
export NUM_SAMPLES="${NUM_SAMPLES:-200}"
export DATASET_JSON="${DATASET_JSON:-${PROJECT_ROOT}/data/processed/bf_full_audit_v1/${BF_SPLIT}_clean.json}"
export SPLIT_PATH="${SPLIT_PATH:-${PROJECT_ROOT}/eval/benchmarks/bf_${BF_SPLIT}_e12_${NUM_SAMPLES}.json}"
export EVAL_RUN_ID="${EVAL_RUN_ID:-${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}"
export EVAL_ROOT="${EVAL_ROOT:-${PROJECT_ROOT}/eval_outputs/e12_${BF_SPLIT}_${NUM_SAMPLES}/${EVAL_RUN_ID}}"
export NUM_GPUS=1 PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
cd "${PROJECT_ROOT}"
run() {
  printf '%q ' "$@"
  printf '\n'
  if [[ "${DRY_RUN:-0}" != "1" ]]; then "$@"; fi
}
if [[ "${DRY_RUN:-0}" != "1" ]]; then
  for path in "${TRAIN_JSON}" "${DATASET_JSON}"; do
    [[ -f "${path}" ]] || { echo "缺少已有清单：${path}；请同步或设置对应路径，图像路径需相对 BF 根目录。" >&2; exit 1; }
  done
fi
echo "[E12 1/3] 准备固定评测清单"
run python tools/sync_bf_eval_captions.py --bf-root "${DATA_ROOT_PATH}" --apply
run python -c 'import sys; from eval.benchmark_utils import create_or_load_fixed_split; create_or_load_fixed_split(sys.argv[1], sys.argv[2], num_samples=int(sys.argv[3]), seed=42)' \
  "${DATASET_JSON}" "${SPLIT_PATH}" "${NUM_SAMPLES}"

echo "[E12 2/3] 训练正确文本 Adapter，步数=${MAX_TRAIN_STEPS:-500}"
bash scripts/train_e12.sh
export E12_CKPT="${OUTPUT_DIR}/checkpoint-final/joint_model.pt"
if [[ "${DRY_RUN:-0}" != "1" && ! -f "${E12_CKPT}" ]]; then
  echo "训练未生成最终权重：${E12_CKPT}" >&2
  exit 1
fi
echo "[E12 3/3] 测试 E5 / OFF / Correct / Shuffled，划分=${BF_SPLIT}，每组=${NUM_SAMPLES} 张"
E5_CKPT="${BASE_CKPT}" TEXTURE_CKPT="${TEXTURE_ADAPTER_CKPT}" DEVICE=cuda:0 \
  bash scripts/eval_e12.sh
echo "[E12] 完成，评测目录：${EVAL_ROOT}（DRY_RUN=${DRY_RUN:-0}）"
