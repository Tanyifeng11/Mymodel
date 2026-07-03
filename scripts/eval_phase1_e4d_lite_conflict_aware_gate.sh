#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/share/home/u2515283058/Mymodel}"
DATASETS_ROOT="${DATASETS_ROOT:-/share/home/u2515283058/datasets}"
BF_ROOT="${BF_ROOT:-${DATASETS_ROOT}/BF}"
BF_TRAIN_ROOT="${BF_TRAIN_ROOT:-${BF_ROOT}/training}"

DATA_JSON_DIR="${DATA_JSON_DIR:-${PROJECT_ROOT}/data}"
VAL_JSON="${VAL_JSON:-${DATA_JSON_DIR}/train_bf_texture.json}"
VAL_ROOT_PATH="${VAL_ROOT_PATH:-${BF_TRAIN_ROOT}}"

OUTPUT_BASE="${OUTPUT_BASE:-${PROJECT_ROOT}/output}"
TEXTURE_ADAPTER_CKPT="${TEXTURE_ADAPTER_CKPT:-${OUTPUT_BASE}/texture_adapter_bf_e20/checkpoint-final/texture_adapter.bin}"
E4D_CKPT="${E4D_CKPT:-${OUTPUT_BASE}/phase1_e4d_lite_conflict_aware_gate_v2_e1/checkpoint-final/joint_model.pt}"

EVAL_BASE="${EVAL_BASE:-${PROJECT_ROOT}/eval_outputs/phase1_full_500_e4d_lite_conflict_aware_gate_v2}"
REPORT_DIR="${REPORT_DIR:-${EVAL_BASE}/report_normal}"
SPLIT_PATH="${SPLIT_PATH:-${PROJECT_ROOT}/eval/benchmarks/phase1_bf_val_split.json}"
EVAL_NUM_SAMPLES="${EVAL_NUM_SAMPLES:-500}"
EVAL_SEED="${EVAL_SEED:-42}"
EVAL_DEVICE="${EVAL_DEVICE:-cuda:0}"
EVAL_MODES="${EVAL_MODES:-token}"
TEXTURE_PREPROCESS_MODE="${TEXTURE_PREPROCESS_MODE:-plain_resize}"
REAL_IMG_DIR="${REAL_IMG_DIR:-${VAL_ROOT_PATH}/cloth}"
CLIP_MODEL_PATH="${CLIP_MODEL_PATH:-${PROJECT_ROOT}/models/clip}"
REPORT_DEVICE="${REPORT_DEVICE:-cuda}"

COMPUTE_FID="${COMPUTE_FID:-1}"
COMPUTE_CLIP_I="${COMPUTE_CLIP_I:-1}"
COMPUTE_LEAKAGE="${COMPUTE_LEAKAGE:-1}"
COMPUTE_STRUCTURE="${COMPUTE_STRUCTURE:-1}"
DEBUG_SAVE_MASKS="${DEBUG_SAVE_MASKS:-20}"
FAIL_ON_EMPTY_MASKS="${FAIL_ON_EMPTY_MASKS:-0}"
MIN_VALID_PIXELS="${MIN_VALID_PIXELS:-50}"
WRITE_TEXT_SIDECARS="${WRITE_TEXT_SIDECARS:-1}"
SAVE_BALANCED_GATE_TRACE="${SAVE_BALANCED_GATE_TRACE:-0}"

CONFLICT_TEXTURE_SUPPRESS_STRENGTH="${CONFLICT_TEXTURE_SUPPRESS_STRENGTH:-0.1}"
CONFLICT_PALETTE_SUPPRESS_STRENGTH="${CONFLICT_PALETTE_SUPPRESS_STRENGTH:-0.4}"
CONFLICT_DELTAE_NORM="${CONFLICT_DELTAE_NORM:-50.0}"
CONFLICT_THRESHOLD="${CONFLICT_THRESHOLD:-0.70}"

TEXTURE_IMAGES_DIR="${TEXTURE_IMAGES_DIR:-}"
SKETCH_IMAGES_DIR="${SKETCH_IMAGES_DIR:-}"
MASK_DIR="${MASK_DIR:-}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

cd "${PROJECT_ROOT}"

if [[ ! -f "${VAL_JSON}" ]]; then
  echo "[ERROR] VAL_JSON does not exist: ${VAL_JSON}" >&2
  exit 1
fi
if [[ ! -d "${VAL_ROOT_PATH}" ]]; then
  echo "[ERROR] VAL_ROOT_PATH does not exist: ${VAL_ROOT_PATH}" >&2
  exit 1
fi
if [[ ! -d "${REAL_IMG_DIR}" ]]; then
  REAL_IMG_DIR="${VAL_ROOT_PATH}"
fi
if [[ "${METRICS_ONLY:-0}" != "1" && ! -f "${TEXTURE_ADAPTER_CKPT}" ]]; then
  echo "[ERROR] TEXTURE_ADAPTER_CKPT does not exist: ${TEXTURE_ADAPTER_CKPT}" >&2
  exit 1
fi
if [[ "${METRICS_ONLY:-0}" != "1" && ! -f "${E4D_CKPT}" ]]; then
  echo "[ERROR] E4D_CKPT does not exist: ${E4D_CKPT}" >&2
  echo "[ERROR] Train it first with: bash scripts/train_phase1_e4d_lite_conflict_aware_gate.sh" >&2
  exit 1
fi

mkdir -p "${EVAL_BASE}" "${REPORT_DIR}"

benchmark_cmd=(
  python tools/run_fixed_benchmark.py
  --dataset_json "${VAL_JSON}"
  --data_root "${VAL_ROOT_PATH}"
  --split_path "${SPLIT_PATH}"
  --gam_ckpt "${E4D_CKPT}"
  --texture_ckpt "${TEXTURE_ADAPTER_CKPT}"
  --modes "${EVAL_MODES}"
  --num_samples "${EVAL_NUM_SAMPLES}"
  --seed "${EVAL_SEED}"
  --resume_generation 1
  --skip_existing 1
  --overwrite 0
  --device "${EVAL_DEVICE}"
  --texture_preprocess_mode "${TEXTURE_PREPROCESS_MODE}"
  --real_images_dir "${REAL_IMG_DIR}"
  --clip_model_path "${CLIP_MODEL_PATH}"
  --compute_fid "${COMPUTE_FID}"
  --compute_clip_i "${COMPUTE_CLIP_I}"
  --compute_leakage "${COMPUTE_LEAKAGE}"
  --compute_structure "${COMPUTE_STRUCTURE}"
  --debug_save_masks "${DEBUG_SAVE_MASKS}"
  --fail_on_empty_masks "${FAIL_ON_EMPTY_MASKS}"
  --min_valid_pixels "${MIN_VALID_PIXELS}"
  --write_text_sidecars "${WRITE_TEXT_SIDECARS}"
  --save_balanced_gate_trace "${SAVE_BALANCED_GATE_TRACE}"
  --use_conflict_aware_gate 1
  --conflict_texture_suppress_strength "${CONFLICT_TEXTURE_SUPPRESS_STRENGTH}"
  --conflict_palette_suppress_strength "${CONFLICT_PALETTE_SUPPRESS_STRENGTH}"
  --conflict_deltae_norm "${CONFLICT_DELTAE_NORM}"
  --conflict_threshold "${CONFLICT_THRESHOLD}"
  --output_dir "${EVAL_BASE}"
  --run_name E4d_lite_conflict_aware_gate_v2
)

if [[ -n "${TEXTURE_IMAGES_DIR}" ]]; then
  benchmark_cmd+=(--texture_images_dir "${TEXTURE_IMAGES_DIR}")
fi
if [[ -n "${SKETCH_IMAGES_DIR}" ]]; then
  benchmark_cmd+=(--sketch_images_dir "${SKETCH_IMAGES_DIR}")
fi
if [[ -n "${MASK_DIR}" ]]; then
  benchmark_cmd+=(--mask_dir "${MASK_DIR}")
fi
if [[ "${METRICS_ONLY:-0}" == "1" ]]; then
  benchmark_cmd+=(--metrics_only)
fi

echo "============================================"
echo "Phase E4D-lite conflict-aware evaluation"
echo "E4D_CKPT=${E4D_CKPT}"
echo "TEXTURE_ADAPTER_CKPT=${TEXTURE_ADAPTER_CKPT}"
echo "EVAL_BASE=${EVAL_BASE}"
echo "CONFLICT_TEXTURE_SUPPRESS_STRENGTH=${CONFLICT_TEXTURE_SUPPRESS_STRENGTH}"
echo "CONFLICT_PALETTE_SUPPRESS_STRENGTH=${CONFLICT_PALETTE_SUPPRESS_STRENGTH}"
echo "CONFLICT_DELTAE_NORM=${CONFLICT_DELTAE_NORM}, CONFLICT_THRESHOLD=${CONFLICT_THRESHOLD}"
echo "============================================"
printf '%q ' "${benchmark_cmd[@]}"
echo

if [[ "${DRY_RUN:-0}" != "1" ]]; then
  "${benchmark_cmd[@]}"
fi

report_cmd=(
  python -m eval.ablation_report
  --experiments_dir "${EVAL_BASE}"
  --real_images_dir "${REAL_IMG_DIR}"
  --output_dir "${REPORT_DIR}"
  --experiment_names E4d_lite_conflict_aware_gate_v2
  --device "${REPORT_DEVICE}"
  --clip_model_path "${CLIP_MODEL_PATH}"
)

echo "=== Building E4D-lite report ==="
printf '%q ' "${report_cmd[@]}"
echo

if [[ "${DRY_RUN:-0}" != "1" && "${RUN_REPORT:-1}" == "1" ]]; then
  "${report_cmd[@]}"
fi

echo "============================================"
echo "E4D-lite evaluation done."
echo "Eval output:   ${EVAL_BASE}/E4d_lite_conflict_aware_gate_v2"
echo "Bucket output: ${EVAL_BASE}/E4d_lite_conflict_aware_gate_v2/conflict_bucket_metrics.csv"
echo "Report output: ${REPORT_DIR}"
echo "============================================"
