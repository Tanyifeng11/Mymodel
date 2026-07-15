#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/share/home/u2515283058/Mymodel}"
BF_TEST_ROOT="${BF_TEST_ROOT:-/share/home/u2515283058/datasets/BF/test}"
DATASET_JSON="${DATASET_JSON:-${PROJECT_ROOT}/data/bf_test_no_bag.json}"
SPLIT_PATH="${SPLIT_PATH:-${PROJECT_ROOT}/eval/benchmarks/bf_test_no_bag_full_split.json}"
OUTPUT_BASE="${OUTPUT_BASE:-${PROJECT_ROOT}/output}"
E5_CKPT="${E5_CKPT:-${OUTPUT_BASE}/phase1_e5_tcpm_lite_e3/checkpoint-final/joint_model.pt}"
TEXTURE_CKPT="${TEXTURE_CKPT:-${OUTPUT_BASE}/texture_adapter_bf_e20/checkpoint-final/texture_adapter.bin}"
EVAL_BASE="${EVAL_BASE:-${PROJECT_ROOT}/eval_outputs/e5_bf_full_test}"
RUN_NAME="${RUN_NAME:-e5_tcpm_lite_e3_bf_test}"
CLIP_MODEL="${CLIP_MODEL:-${PROJECT_ROOT}/models/clip}"
DEVICE="${DEVICE:-cuda}"
GENERATION_SEED="${GENERATION_SEED:-42}"

for path in "${BF_TEST_ROOT}" "${DATASET_JSON}" "${SPLIT_PATH}" "${E5_CKPT}" "${TEXTURE_CKPT}" "${CLIP_MODEL}"; do
  if [[ ! -e "${path}" ]]; then
    echo "[ERROR] Required path does not exist: ${path}" >&2
    exit 1
  fi
done

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

NUM_SAMPLES="$(python -c "import json; print(len(json.load(open('${SPLIT_PATH}', encoding='utf-8'))))")"
echo "[finalize] full samples=${NUM_SAMPLES}, device=${DEVICE}"

GENERATED_DIR="${EVAL_BASE}/${RUN_NAME}/token"
if [[ ! -d "${GENERATED_DIR}" ]]; then
  echo "[ERROR] Generated directory does not exist: ${GENERATED_DIR}" >&2
  exit 1
fi
GENERATED_COUNT="$(find "${GENERATED_DIR}" -type f -name 'generated_*.png' | wc -l)"
GENERATED_COUNT="${GENERATED_COUNT//[[:space:]]/}"
if [[ "${GENERATED_COUNT}" -ne "${NUM_SAMPLES}" ]]; then
  echo "[ERROR] Generated images incomplete: ${GENERATED_COUNT}/${NUM_SAMPLES}" >&2
  exit 1
fi

python tools/run_fixed_benchmark.py \
  --dataset_json "${DATASET_JSON}" \
  --data_root "${BF_TEST_ROOT}" \
  --split_path "${SPLIT_PATH}" \
  --num_samples "${NUM_SAMPLES}" \
  --sample_id_start 0 \
  --sample_id_end "${NUM_SAMPLES}" \
  --generation_seed "${GENERATION_SEED}" \
  --gam_ckpt "${E5_CKPT}" \
  --texture_ckpt "${TEXTURE_CKPT}" \
  --device "${DEVICE}" \
  --modes token \
  --texture_preprocess_mode plain_resize \
  --clip_model_path "${CLIP_MODEL}" \
  --clip_batch_size 16 \
  --fid_batch_size 16 \
  --progress_write_interval 500 \
  --write_text_sidecars 0 \
  --output_dir "${EVAL_BASE}" \
  --run_name "${RUN_NAME}" \
  --evaluation_protocol original_image_size \
  --layer_group_enabled 1 \
  --use_texture_gate 1 \
  --use_palette_tokens 0 \
  --use_tcpm_lite 1 \
  --metrics_only

python tools/validate_benchmark_outputs.py \
  --experiments_dir "${EVAL_BASE}" \
  --experiment_names "${RUN_NAME}" \
  --expected_count "${NUM_SAMPLES}"

echo "[finalize] metrics: ${EVAL_BASE}/${RUN_NAME}"
