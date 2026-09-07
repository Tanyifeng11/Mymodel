#!/bin/bash
# 提交：sbatch submit/diagnose_e8.sh
# 仅权重检查：DIAG_WEIGHTS_ONLY=1 sbatch submit/diagnose_e8.sh

#SBATCH -J E8_diagnose
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -o /share/home/u2515283058/Mymodel/log_e8_diagnose_%j.log
#SBATCH -e /share/home/u2515283058/Mymodel/log_e8_diagnose_%j.err

set -eo pipefail
if [[ "${DRY_RUN:-0}" != "1" ]]; then
  source /share/apps/anaconda3/etc/profile.d/conda.sh
  conda activate Mymodel
fi
set -u

export PROJECT_ROOT="${PROJECT_ROOT:-/share/home/u2515283058/Mymodel}"
DATASETS_ROOT="${DATASETS_ROOT:-/share/home/u2515283058/datasets}"
OUTPUT_BASE="${OUTPUT_BASE:-${PROJECT_ROOT}/output}"
EVAL_RUN_ROOT="${EVAL_RUN_ROOT:-${PROJECT_ROOT}/eval_outputs/resampler_validation_100/20260906_233924}"
DIAG_RUN_ID="${DIAG_RUN_ID:-${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}"
DIAG_OUTPUT_DIR="${DIAG_OUTPUT_DIR:-${PROJECT_ROOT}/eval_outputs/e8_diagnosis/${DIAG_RUN_ID}}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
cd "${PROJECT_ROOT}"

cmd=(
  python tools/diagnose_e8.py
  --e5-ckpt "${E5_CKPT:-${OUTPUT_BASE}/phase1_e5_tcpm_lite_e3/checkpoint-final/joint_model.pt}"
  --e8a-ckpt "${VISUAL_CKPT:-${OUTPUT_BASE}/phase1_resampler_visual/checkpoint-final/joint_model.pt}"
  --e8b-ckpt "${TEXT_CKPT:-${OUTPUT_BASE}/phase1_resampler_text/checkpoint-final/joint_model.pt}"
  --texture-ckpt "${TEXTURE_CKPT:-${OUTPUT_BASE}/texture_adapter_bf_e20/checkpoint-final/texture_adapter.bin}"
  --samples-json "${SAMPLES_JSON:-${EVAL_RUN_ROOT}/seed_42/e5/benchmark_samples.json}"
  --data-root "${DATA_ROOT_PATH:-${DATASETS_ROOT}/BF/validation}"
  --num-samples "${NUM_SAMPLES:-100}"
  --base-model-path "${SD_MODEL:-auto}" --image-encoder-path "${IMAGE_ENCODER_PATH:-auto}"
  --device cuda:0 --dtype "${DIAG_DTYPE:-fp16}"
  --width "${WIDTH:-384}" --height "${HEIGHT:-512}"
  --num-threads "${SLURM_CPUS_PER_TASK:-4}" --output-dir "${DIAG_OUTPUT_DIR}"
)
if [[ "${DIAG_WEIGHTS_ONLY:-0}" == "1" ]]; then cmd+=(--weights-only); fi

echo "[E8 诊断] 结果目录：${DIAG_OUTPUT_DIR}"
printf '%q ' "${cmd[@]}"
printf '\n'
if [[ "${DRY_RUN:-0}" != "1" ]]; then
  "${cmd[@]}"
fi
