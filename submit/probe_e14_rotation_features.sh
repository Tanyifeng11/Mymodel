#!/bin/bash
# sbatch submit/probe_e14_rotation_features.sh
#SBATCH -J E14_rotation_features
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -o log_e14_rotation_features_%j.log
#SBATCH -e log_e14_rotation_features_%j.err
set -eo pipefail
if [[ "${DRY_RUN:-0}" != "1" ]]; then
  source /share/apps/anaconda3/etc/profile.d/conda.sh
  conda activate Mymodel
fi
set -u
PROJECT_ROOT="${PROJECT_ROOT:-/share/home/u2515283058/Mymodel}"
INPUTS_ROOT="${INPUTS_ROOT:-${PROJECT_ROOT}/eval_outputs/e14_real_rotation/113141/inputs}"
EVAL_ROOT="${EVAL_ROOT:-${PROJECT_ROOT}/eval_outputs/e14_rotation_features/${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
cd "${PROJECT_ROOT}"
run() { printf '%q ' "$@"; printf '\n'; if [[ "${DRY_RUN:-0}" != "1" ]]; then "$@"; fi; }
if [[ "${DRY_RUN:-0}" != "1" ]]; then
  [[ -f "${INPUTS_ROOT}/inputs.json" ]] || { echo "缺少 ${INPUTS_ROOT}/inputs.json；请设置 INPUTS_ROOT" >&2; exit 1; }
  [[ ! -e "${EVAL_ROOT}" ]] || { echo '请使用新的 EVAL_ROOT' >&2; exit 1; }
fi
run mkdir -p "${EVAL_ROOT}"
run cp -r "${INPUTS_ROOT}" "${EVAL_ROOT}/inputs"
run python -m tools.e14_rotation_features prepare --root "${EVAL_ROOT}/inputs"
run python -m tools.e14_pattern_probe extract --labels "${EVAL_ROOT}/inputs/feature_labels.csv" \
  --data-root "${EVAL_ROOT}/inputs" --allow-unmatched-extraction \
  --checkpoint "${E5_CKPT:-${PROJECT_ROOT}/output/phase1_e5_tcpm_lite_e3/checkpoint-final/joint_model.pt}" \
  --clip-model "${CLIP_MODEL:-${PROJECT_ROOT}/models/clip}" --output "${EVAL_ROOT}/representations"
run python -m tools.e14_rotation_features report --root "${EVAL_ROOT}/representations"
