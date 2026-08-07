#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/share/home/u2515283058/Mymodel}"
DATASETS_ROOT="${DATASETS_ROOT:-/share/home/u2515283058/datasets}"
BF_TRAIN_ROOT="${BF_TRAIN_ROOT:-${DATASETS_ROOT}/BF/training}"
DATASET_JSON="${DATASET_JSON:-${PROJECT_ROOT}/data/train_bf_texture.json}"
SPLIT_PATH="${SPLIT_PATH:-${PROJECT_ROOT}/eval/benchmarks/phase1_bf_val_split.json}"

OUTPUT_BASE="${OUTPUT_BASE:-${PROJECT_ROOT}/output}"
TEXTURE_ADAPTER_CKPT="${TEXTURE_ADAPTER_CKPT:-${OUTPUT_BASE}/texture_adapter_bf_e20/checkpoint-final/texture_adapter.bin}"
E7A_CKPT="${E7A_CKPT:-${OUTPUT_BASE}/phase1_e7a/checkpoint-final/joint_model.pt}"
CLIP_MODEL="${CLIP_MODEL:-${PROJECT_ROOT}/models/clip}"

EVAL_BASE="${EVAL_BASE:-${PROJECT_ROOT}/eval_outputs/phase1_e7a_500}"
REPORT_DIR="${REPORT_DIR:-${EVAL_BASE}/report_e7a}"
NUM_SAMPLES="${NUM_SAMPLES:-500}"
EVAL_SEED="${EVAL_SEED:-42}"
GENERATION_SEED="${GENERATION_SEED:-42}"
EVAL_DEVICE="${EVAL_DEVICE:-cuda:0}"
REPORT_DEVICE="${REPORT_DEVICE:-cuda:0}"
RUN_NAME="${E7A_RUN_NAME:-e7a}"
EXPERIMENT_NAMES="${EXPERIMENT_NAMES:-${RUN_NAME}}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

cd "${PROJECT_ROOT}"

for required_path in "${DATASET_JSON}" "${BF_TRAIN_ROOT}" "${TEXTURE_ADAPTER_CKPT}" "${E7A_CKPT}" "${CLIP_MODEL}"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "[ERROR] Required path does not exist: ${required_path}" >&2
    exit 1
  fi
done

mkdir -p "${EVAL_BASE}" "${REPORT_DIR}"

python tools/run_fixed_benchmark.py \
  --dataset_json "${DATASET_JSON}" \
  --data_root "${BF_TRAIN_ROOT}" \
  --split_path "${SPLIT_PATH}" \
  --num_samples "${NUM_SAMPLES}" \
  --seed "${EVAL_SEED}" \
  --generation_seed "${GENERATION_SEED}" \
  --resume_generation 1 \
  --skip_existing 1 \
  --overwrite 0 \
  --gam_ckpt "${E7A_CKPT}" \
  --texture_ckpt "${TEXTURE_ADAPTER_CKPT}" \
  --device "${EVAL_DEVICE}" \
  --modes token \
  --texture_preprocess_mode plain_resize \
  --clip_model_path "${CLIP_MODEL}" \
  --write_text_sidecars 1 \
  --output_dir "${EVAL_BASE}" \
  --run_name "${RUN_NAME}" \
  --evaluation_protocol original_image_size \
  --use_texture_gate 1 \
  --use_tcpm_lite 1 \
  --use_aa_tcr_fuse 1

python tools/validate_benchmark_outputs.py \
  --experiments_dir "${EVAL_BASE}" \
  --experiment_names "${RUN_NAME}" \
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
