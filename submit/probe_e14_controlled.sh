#!/bin/bash
# sbatch submit/probe_e14_controlled.sh
# DRY_RUN=1 bash submit/probe_e14_controlled.sh
# 消除预处理灰度混杂：INPUT_COLOR_CONTROL=rank_binary sbatch submit/probe_e14_controlled.sh
#SBATCH -J E14_controlled
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -o log_e14_controlled_%j.log
#SBATCH -e log_e14_controlled_%j.err
set -eo pipefail
if [[ "${DRY_RUN:-0}" != "1" ]]; then
  source /share/apps/anaconda3/etc/profile.d/conda.sh
  conda activate Mymodel
fi
set -u
PROJECT_ROOT="${PROJECT_ROOT:-/share/home/u2515283058/Mymodel}"
EVAL_ROOT="${EVAL_ROOT:-${PROJECT_ROOT}/eval_outputs/e14_controlled/${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}"
E5_CKPT="${E5_CKPT:-${PROJECT_ROOT}/output/phase1_e5_tcpm_lite_e3/checkpoint-final/joint_model.pt}"
CLIP_MODEL="${CLIP_MODEL:-${PROJECT_ROOT}/models/clip}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
cd "${PROJECT_ROOT}"
run() {
  printf '%q ' "$@"
  printf '\n'
  if [[ "${DRY_RUN:-0}" != "1" ]]; then "$@"; fi
}
extra=()
[[ -z "${BASE_MODEL:-}" ]] || extra+=(--base-model "${BASE_MODEL}")
run python -m tools.e14_controlled_patterns --output "${EVAL_ROOT}/inputs"
run python -m tools.e14_pattern_probe extract --labels "${EVAL_ROOT}/inputs/labels.csv" \
  --data-root "${EVAL_ROOT}/inputs" --checkpoint "${E5_CKPT}" --clip-model "${CLIP_MODEL}" \
  --output "${EVAL_ROOT}/representations" --input-color-control "${INPUT_COLOR_CONTROL:-original}" "${extra[@]}"
eval_extra=()
# LINEAR=1 需要环境已有 scikit-learn；默认先做无训练读出。
[[ "${LINEAR:-0}" != "1" ]] || eval_extra+=(--linear)
run python -m tools.e14_pattern_probe evaluate --features "${EVAL_ROOT}/representations" "${eval_extra[@]}"
