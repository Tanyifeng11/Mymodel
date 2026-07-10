#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/share/home/u2515283058/Mymodel}"
DATASETS_ROOT="${DATASETS_ROOT:-/share/home/u2515283058/datasets}"
BF_ROOT="${BF_ROOT:-${DATASETS_ROOT}/BF}"
DATA_ROOT_PATH="${DATA_ROOT_PATH:-${BF_ROOT}/training}"
DATASET_JSON="${DATASET_JSON:-${PROJECT_ROOT}/data/train_bf_texture.json}"
SPLIT_PATH="${SPLIT_PATH:-${PROJECT_ROOT}/eval/benchmarks/phase1_bf_val_split.json}"

OUTPUT_BASE="${OUTPUT_BASE:-${PROJECT_ROOT}/output}"
TEXTURE_ADAPTER_CKPT="${TEXTURE_ADAPTER_CKPT:-${OUTPUT_BASE}/texture_adapter_bf_e20/checkpoint-final/texture_adapter.bin}"
E5_CKPT="${E5_CKPT:-${OUTPUT_BASE}/phase1_e5_tcpm_lite_e12/checkpoint-final/joint_model.pt}"

EVAL_BASE="${EVAL_BASE:-${PROJECT_ROOT}/eval_outputs/phase1_full_500}"
REPORT_DIR="${REPORT_DIR:-${EVAL_BASE}/report_e5_tcpm_lite_e12}"
NUM_SAMPLES="${NUM_SAMPLES:-500}"
EVAL_SEED="${EVAL_SEED:-42}"
GENERATION_SEED="${GENERATION_SEED:-42}"
EVAL_DEVICE="${EVAL_DEVICE:-cuda:0}"
REPORT_DEVICE="${REPORT_DEVICE:-cuda:0}"
CLIP_MODEL="${CLIP_MODEL:-${PROJECT_ROOT}/models/clip}"
E5_RUN_NAME="${E5_RUN_NAME:-e5_tcpm_lite_e12}"
EXPERIMENT_NAMES="${EXPERIMENT_NAMES:-e2b_color_safe_gate,e5_tcpm_lite,e5_tcpm_lite_e12}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

cd "${PROJECT_ROOT}"

for required_path in "${DATASET_JSON}" "${DATA_ROOT_PATH}" "${TEXTURE_ADAPTER_CKPT}" "${E5_CKPT}" "${CLIP_MODEL}"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "[ERROR] Required path does not exist: ${required_path}" >&2
    exit 1
  fi
done

mkdir -p "${EVAL_BASE}" "${REPORT_DIR}"

echo "============================================"
echo "Phase 1 E5 TCPM-lite 500-sample evaluation"
echo "E5_CKPT=${E5_CKPT}"
echo "EVAL_BASE=${EVAL_BASE}"
echo "E5_RUN_NAME=${E5_RUN_NAME}"
echo "GENERATION_SEED=${GENERATION_SEED}"
echo "EXPERIMENT_NAMES=${EXPERIMENT_NAMES}"
echo "============================================"

python tools/run_fixed_benchmark.py \
  --dataset_json "${DATASET_JSON}" \
  --data_root "${DATA_ROOT_PATH}" \
  --split_path "${SPLIT_PATH}" \
  --num_samples "${NUM_SAMPLES}" \
  --seed "${EVAL_SEED}" \
  --generation_seed "${GENERATION_SEED}" \
  --sample_id_start 0 \
  --sample_id_end "${NUM_SAMPLES}" \
  --resume_generation 1 \
  --skip_existing 1 \
  --overwrite 0 \
  --gam_ckpt "${E5_CKPT}" \
  --texture_ckpt "${TEXTURE_ADAPTER_CKPT}" \
  --device "${EVAL_DEVICE}" \
  --modes token \
  --texture_preprocess_mode plain_resize \
  --clip_model_path "${CLIP_MODEL}" \
  --write_text_sidecars 1 \
  --output_dir "${EVAL_BASE}" \
  --run_name "${E5_RUN_NAME}" \
  --evaluation_protocol original_image_size \
  --use_tcpm_lite 1

python tools/validate_benchmark_outputs.py \
  --experiments_dir "${EVAL_BASE}" \
  --experiment_names "${E5_RUN_NAME}" \
  --expected_count "${NUM_SAMPLES}"

python -m eval.ablation_report \
  --experiments_dir "${EVAL_BASE}" \
  --output_dir "${REPORT_DIR}" \
  --experiment_names "${EXPERIMENT_NAMES}" \
  --device "${REPORT_DEVICE}" \
  --evaluation_protocol original_image_size \
  --num_samples "${NUM_SAMPLES}" \
  --resume_generation 1 \
  --existing_samples_skipped 1

echo "============================================"
echo "E5 TCPM-lite evaluation done."
echo "Generated images and metrics: ${EVAL_BASE}/${E5_RUN_NAME}"
echo "Report: ${REPORT_DIR}"
echo "============================================"
