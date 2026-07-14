#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/share/home/u2515283058/Mymodel}"
DATASETS_ROOT="${DATASETS_ROOT:-/share/home/u2515283058/datasets}"
DATA_ROOT_PATH="${DATA_ROOT_PATH:-${DATASETS_ROOT}/BF/training}"
DATASET_JSON="${DATASET_JSON:-${PROJECT_ROOT}/data/train_bf_texture.json}"
SPLIT_PATH="${SPLIT_PATH:-${PROJECT_ROOT}/eval/benchmarks/phase1_bf_val_split.json}"

OUTPUT_BASE="${OUTPUT_BASE:-${PROJECT_ROOT}/output}"
E5_CKPT="${E5_CKPT:-${OUTPUT_BASE}/phase1_e5_tcpm_lite_e3/checkpoint-final/joint_model.pt}"
E6_CKPT="${E6_CKPT:-${OUTPUT_BASE}/phase1_e6b_unet_late_dist_e1/checkpoint-final/joint_model.pt}"
TEXTURE_ADAPTER_CKPT="${TEXTURE_ADAPTER_CKPT:-${OUTPUT_BASE}/texture_adapter_bf_e20/checkpoint-final/texture_adapter.bin}"
INTERPOLATION_OUTPUT_BASE="${INTERPOLATION_OUTPUT_BASE:-${OUTPUT_BASE}/phase1_e5_e6_interpolation}"
INTERPOLATED_CKPT="${INTERPOLATED_CKPT:-${INTERPOLATION_OUTPUT_BASE}/e5_e6_interp_a075/checkpoint-final/joint_model.pt}"

EVAL_BASE="${EVAL_BASE:-${PROJECT_ROOT}/eval_outputs/phase1_e5_e6_interpolation_a075_500}"
REUSE_BASE="${REUSE_BASE:-${PROJECT_ROOT}/eval_outputs/phase1_e5_e6_interpolation_100}"
REPORT_DIR="${REPORT_DIR:-${EVAL_BASE}/report}"
NUM_SAMPLES="${NUM_SAMPLES:-500}"
EVAL_SEED="${EVAL_SEED:-42}"
GENERATION_SEED="${GENERATION_SEED:-42}"
DEVICE="${DEVICE:-cuda}"
CLIP_MODEL="${CLIP_MODEL:-${PROJECT_ROOT}/models/clip}"

UNET_LATE_BLOCKS="${UNET_LATE_BLOCKS:-2,3}"
INCLUDE_OUTPUT_LAYER="${INCLUDE_OUTPUT_LAYER:-1}"
INTERPOLATION_OVERWRITE="${INTERPOLATION_OVERWRITE:-0}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

cd "${PROJECT_ROOT}"

for required_path in \
  "${DATASET_JSON}" \
  "${DATA_ROOT_PATH}" \
  "${SPLIT_PATH}" \
  "${E5_CKPT}" \
  "${E6_CKPT}" \
  "${TEXTURE_ADAPTER_CKPT}" \
  "${CLIP_MODEL}"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "[ERROR] Required path does not exist: ${required_path}" >&2
    exit 1
  fi
done

mkdir -p "$(dirname "${INTERPOLATED_CKPT}")" "${EVAL_BASE}" "${REPORT_DIR}"

if [[ ! -f "${INTERPOLATED_CKPT}" || "${INTERPOLATION_OVERWRITE}" == "1" ]]; then
  interpolation_args=(
    python tools/interpolate_e5_e6_checkpoints.py
    --e5_ckpt "${E5_CKPT}"
    --e6_ckpt "${E6_CKPT}"
    --output_path "${INTERPOLATED_CKPT}"
    --alpha 0.75
    --unet_late_blocks "${UNET_LATE_BLOCKS}"
    --include_output_layer "${INCLUDE_OUTPUT_LAYER}"
  )
  if [[ "${INTERPOLATION_OVERWRITE}" == "1" ]]; then
    interpolation_args+=(--overwrite)
  fi
  "${interpolation_args[@]}"
fi

experiment_names=("e5_e6_interp_a075" "e6_interp_endpoint")
checkpoint_paths=("${INTERPOLATED_CKPT}" "${E6_CKPT}")

echo "============================================"
echo "E5/E6 alpha=0.75 vs E6, ${NUM_SAMPLES}-sample evaluation"
echo "DEVICE=${DEVICE}"
echo "EVAL_BASE=${EVAL_BASE}"
echo "REUSE_BASE=${REUSE_BASE}"
echo "============================================"

for index in "${!experiment_names[@]}"; do
  run_name="${experiment_names[$index]}"
  checkpoint_path="${checkpoint_paths[$index]}"
  reuse_args=()
  if [[ -d "${REUSE_BASE}/${run_name}" ]]; then
    reuse_args+=(--reuse_from_dir "${REUSE_BASE}/${run_name}")
  fi

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
    "${reuse_args[@]}" \
    --gam_ckpt "${checkpoint_path}" \
    --texture_ckpt "${TEXTURE_ADAPTER_CKPT}" \
    --device "${DEVICE}" \
    --modes token \
    --texture_preprocess_mode plain_resize \
    --clip_model_path "${CLIP_MODEL}" \
    --write_text_sidecars 1 \
    --output_dir "${EVAL_BASE}" \
    --run_name "${run_name}" \
    --evaluation_protocol original_image_size \
    --layer_group_enabled 1 \
    --use_texture_gate 0 \
    --use_palette_tokens 0 \
    --use_tcpm_lite 1
done

experiment_names_csv="$(IFS=','; echo "${experiment_names[*]}")"

python tools/validate_benchmark_outputs.py \
  --experiments_dir "${EVAL_BASE}" \
  --experiment_names "${experiment_names_csv}" \
  --expected_count "${NUM_SAMPLES}"

python -m eval.ablation_report \
  --experiments_dir "${EVAL_BASE}" \
  --output_dir "${REPORT_DIR}" \
  --experiment_names "${experiment_names_csv}" \
  --device "${DEVICE}" \
  --evaluation_protocol original_image_size \
  --num_samples "${NUM_SAMPLES}" \
  --resume_generation 1 \
  --existing_samples_skipped 1

echo "============================================"
echo "500-sample evaluation done."
echo "Results: ${EVAL_BASE}"
echo "Report: ${REPORT_DIR}"
echo "============================================"
