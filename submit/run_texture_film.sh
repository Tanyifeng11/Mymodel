#!/bin/bash
# 提交：sbatch submit/run_texture_film.sh
# 同一作业内准备清单、训练 FiLM，再评测 E5 / FiLM 关闭 / FiLM 开启。
# 默认验证集 200 张；最终测试可用 BF_SPLIT=test 指定，清单路径均相对 BF 根目录。

#SBATCH -J FiLM
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH -o /share/home/u2515283058/Mymodel/log_film_%j.log
#SBATCH -e /share/home/u2515283058/Mymodel/log_film_%j.err

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
export FILM_TEXT_MODE="${FILM_TEXT_MODE:-matched}"
case "${FILM_TEXT_MODE}" in matched|shuffled) ;; *) echo "FILM_TEXT_MODE 必须为 matched 或 shuffled" >&2; exit 1 ;; esac
export OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_BASE}/phase1_film_${FILM_TEXT_MODE}}"
export BASE_CKPT="${BASE_CKPT:-${OUTPUT_BASE}/phase1_e5_tcpm_lite_e3/checkpoint-final/joint_model.pt}"
export TEXTURE_ADAPTER_CKPT="${TEXTURE_ADAPTER_CKPT:-${OUTPUT_BASE}/texture_adapter_bf_e20/checkpoint-final/texture_adapter.bin}"
export TRAIN_JSON="${TRAIN_JSON:-${PROJECT_ROOT}/data/processed/bf_film_v1/training_current.json}"
TRAIN_SOURCE_JSON="${TRAIN_SOURCE_JSON:-${PROJECT_ROOT}/data/processed/bf_full_audit_v1/training_clean.json}"
export TRAIN_SEED="${TRAIN_SEED:-42}"
export BF_SPLIT="${BF_SPLIT:-validation}"
case "${BF_SPLIT}" in validation|test) ;; *) echo "BF_SPLIT 必须为 validation 或 test" >&2; exit 1 ;; esac
export NUM_SAMPLES="${NUM_SAMPLES:-200}"
export DATASET_JSON="${DATASET_JSON:-${PROJECT_ROOT}/data/processed/bf_full_audit_v1/${BF_SPLIT}_clean.json}"
export SPLIT_PATH="${SPLIT_PATH:-${PROJECT_ROOT}/eval/benchmarks/bf_${BF_SPLIT}_film_${NUM_SAMPLES}.json}"
export EVAL_RUN_ID="${EVAL_RUN_ID:-${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}"
export EVAL_ROOT="${EVAL_ROOT:-${PROJECT_ROOT}/eval_outputs/texture_film_${FILM_TEXT_MODE}_${BF_SPLIT}_${NUM_SAMPLES}/${EVAL_RUN_ID}}"
export NUM_GPUS=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
cd "${PROJECT_ROOT}"
run() {
  printf '%q ' "$@"
  printf '\n'
  if [[ "${DRY_RUN:-0}" != "1" ]]; then "$@"; fi
}

# data/ 不随 Git 分发，先检查源清单，避免训练结束才发现无法评测。
if [[ "${DRY_RUN:-0}" != "1" ]]; then
  [[ -f "${DATASET_JSON}" ]] || { echo "缺少评测源清单：${DATASET_JSON}；请先从本地同步该 JSON。" >&2; exit 1; }
  if [[ ! -f "${TRAIN_JSON}" && ! -f "${TRAIN_SOURCE_JSON}" ]]; then
    echo "缺少训练源清单：${TRAIN_SOURCE_JSON}；请先从本地同步该 JSON，或设置已有 TRAIN_JSON。" >&2
    exit 1
  fi
fi

echo "[FiLM 1/3] 同步验证/测试文本并准备清单"
run python tools/sync_bf_eval_captions.py --bf-root "${DATA_ROOT_PATH}" --apply
if [[ ! -f "${TRAIN_JSON}" ]]; then
  run python tools/prepare_bf_film_manifest.py --bf-root "${DATA_ROOT_PATH}" \
    --source-manifest "${TRAIN_SOURCE_JSON}" --output "${TRAIN_JSON}" \
    --max-samples "${TRAIN_MANIFEST_SAMPLES:-5000}" --seed "${TRAIN_SEED}"
fi
# 沿用 benchmark 的固定采样逻辑，已有样本及顺序保持不变。
run python -c 'import sys; from eval.benchmark_utils import create_or_load_fixed_split; create_or_load_fixed_split(sys.argv[1], sys.argv[2], num_samples=int(sys.argv[3]), seed=42)' \
  "${DATASET_JSON}" "${SPLIT_PATH}" "${NUM_SAMPLES}"

echo "[FiLM 2/3] 开始训练，模式=${FILM_TEXT_MODE}，步数=${MAX_TRAIN_STEPS:-1000}"
bash scripts/train_phase1_texture_film.sh

# 只评测本次输出的最终权重，训练失败时 set -e 会立即终止作业。
export FILM_CKPT="${OUTPUT_DIR}/checkpoint-final/joint_model.pt"
if [[ "${DRY_RUN:-0}" != "1" && ! -f "${FILM_CKPT}" ]]; then
  echo "训练未生成最终权重：${FILM_CKPT}" >&2
  exit 1
fi
echo "[FiLM 3/3] 开始评测 E5 / FiLM 关闭 / FiLM 开启，划分=${BF_SPLIT}，每组=${NUM_SAMPLES} 张"
E5_CKPT="${BASE_CKPT}" TEXTURE_CKPT="${TEXTURE_ADAPTER_CKPT}" DEVICE=cuda:0 \
  bash scripts/eval_phase1_texture_film.sh
echo "[FiLM] 流程结束，评测目录：${EVAL_ROOT}（DRY_RUN=${DRY_RUN:-0}）"
