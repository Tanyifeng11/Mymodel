#!/bin/bash
# 提交：sbatch submit/diagnose_e9_b.sh
# 默认 32 张固定验证样本、seed 42、13 个干预条件；全量快速门槛：NUM_SAMPLES=100 sbatch submit/diagnose_e9_b.sh

#SBATCH -J E9B_diag
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -o /share/home/u2515283058/Mymodel/log_e9_b_diag_%j.log
#SBATCH -e /share/home/u2515283058/Mymodel/log_e9_b_diag_%j.err

set -euo pipefail
if [[ "${DRY_RUN:-0}" != "1" ]]; then
  source /share/apps/anaconda3/etc/profile.d/conda.sh
  conda activate Mymodel
fi

export PROJECT_ROOT="${PROJECT_ROOT:-/share/home/u2515283058/Mymodel}"
export DATASETS_ROOT="${DATASETS_ROOT:-/share/home/u2515283058/datasets}"
export OUTPUT_BASE="${OUTPUT_BASE:-${PROJECT_ROOT}/output}"
export E9_B_CKPT="${E9_B_CKPT:-${OUTPUT_BASE}/phase1_local_detail_local_stage2_2000/checkpoint-final/joint_model.pt}"
export E5_CKPT="${E5_CKPT:-${OUTPUT_BASE}/phase1_e5_tcpm_lite_e3/checkpoint-final/joint_model.pt}"
export TEXTURE_CKPT="${TEXTURE_CKPT:-${OUTPUT_BASE}/texture_adapter_bf_e20/checkpoint-final/texture_adapter.bin}"
export DATA_ROOT_PATH="${DATA_ROOT_PATH:-${DATASETS_ROOT}/BF/validation}"
export DATASET_JSON="${DATASET_JSON:-${PROJECT_ROOT}/data/bf_validation_resampler.json}"
export SPLIT_PATH="${SPLIT_PATH:-${PROJECT_ROOT}/eval/benchmarks/bf_validation_resampler_100.json}"
export NUM_SAMPLES="${NUM_SAMPLES:-32}"
export DIAG_RUN_ID="${DIAG_RUN_ID:-${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}"
export DIAG_OUTPUT_DIR="${DIAG_OUTPUT_DIR:-${PROJECT_ROOT}/eval_outputs/e9_b_diagnosis_validation_${NUM_SAMPLES}/${DIAG_RUN_ID}}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
cd "${PROJECT_ROOT}"

if [[ "${DRY_RUN:-0}" != "1" ]]; then
  for path in "${E5_CKPT}" "${E9_B_CKPT}" "${TEXTURE_CKPT}" "${DATASET_JSON}" "${SPLIT_PATH}"; do
    [[ -f "${path}" ]] || { echo "缺少必需文件：${path}" >&2; exit 1; }
  done
  python tools/check_e9_frozen.py --e5-ckpt "${E5_CKPT}" --e9-b-ckpt "${E9_B_CKPT}" \
    --output-dir "${DIAG_OUTPUT_DIR}/frozen_check"
fi

cmd=(
  python tools/run_e9_b_diagnostics.py
  --dataset-json "${DATASET_JSON}" --data-root "${DATA_ROOT_PATH}" --split-path "${SPLIT_PATH}"
  --gam-ckpt "${E9_B_CKPT}" --texture-ckpt "${TEXTURE_CKPT}"
  --clip-model-path "${CLIP_MODEL:-${PROJECT_ROOT}/models/clip}"
  --output-dir "${DIAG_OUTPUT_DIR}" --num-samples "${NUM_SAMPLES}"
  --seed 42 --generation-seed 42 --device "${DEVICE:-cuda:0}" --compute-fid "${COMPUTE_FID:-0}"
)
echo "[E9-B 诊断] 输出目录：${DIAG_OUTPUT_DIR}"
printf '%q ' "${cmd[@]}"
printf '\n'
if [[ "${DRY_RUN:-0}" != "1" ]]; then
  "${cmd[@]}"
fi
