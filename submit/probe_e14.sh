#!/bin/bash
# STAGE=prepare sbatch submit/probe_e14.sh
# STAGE=extract LABELS=/path/to/labels.csv sbatch submit/probe_e14.sh
# DRY_RUN=1 STAGE=prepare bash submit/probe_e14.sh
#SBATCH -J E14_pattern
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -o log_e14_%j.log
#SBATCH -e log_e14_%j.err

set -eo pipefail
if [[ "${DRY_RUN:-0}" != "1" ]]; then
  source /share/apps/anaconda3/etc/profile.d/conda.sh
  conda activate Mymodel
fi
set -u
PROJECT_ROOT="${PROJECT_ROOT:-/share/home/u2515283058/Mymodel}"
DATA_ROOT_PATH="${DATA_ROOT_PATH:-/share/home/u2515283058/datasets/BF}"
STAGE="${STAGE:-prepare}"
EVAL_ROOT="${EVAL_ROOT:-${PROJECT_ROOT}/eval_outputs/e14/${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}"
LABELS="${LABELS:-}"
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
case "${STAGE}" in
  prepare)
    run python -m tools.e14_pattern_probe prepare \
      --manifest "${MANIFEST:-data/processed/bf_full_audit_v1/validation_clean.json}" \
      --data-root "${DATA_ROOT_PATH}" --output "${EVAL_ROOT}/candidates" --per-class "${PER_CLASS:-40}"
    ;;
  extract)
    [[ -n "${LABELS}" ]] || { echo '请设置人工确认后的 LABELS 路径' >&2; exit 1; }
    extra=()
    [[ -z "${BASE_MODEL:-}" ]] || extra+=(--base-model "${BASE_MODEL}")
    run python -m tools.e14_pattern_probe extract --labels "${LABELS}" \
      --data-root "${DATA_ROOT_PATH}" --checkpoint "${E5_CKPT}" --clip-model "${CLIP_MODEL}" \
      --output "${EVAL_ROOT}/representations" "${extra[@]}"
    eval_extra=()
    [[ "${LINEAR:-0}" != "1" ]] || eval_extra+=(--linear)
    run python -m tools.e14_pattern_probe evaluate --features "${EVAL_ROOT}/representations" "${eval_extra[@]}"
    ;;
  evaluate)
    eval_extra=()
    [[ "${LINEAR:-0}" != "1" ]] || eval_extra+=(--linear)
    run python -m tools.e14_pattern_probe evaluate --features "${EVAL_ROOT}/representations" "${eval_extra[@]}"
    ;;
  *) echo "未知 STAGE=${STAGE}" >&2; exit 1 ;;
esac
