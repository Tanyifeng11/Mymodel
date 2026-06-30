#!/usr/bin/env bash
set -euo pipefail

# 阶段 1/2A：默认只评估 E2A，并复用已有 E0/E1 结果生成对比报告。

PROJECT_ROOT="${PROJECT_ROOT:-/share/home/u2515283058/Mymodel}"
DATASETS_ROOT="${DATASETS_ROOT:-/share/home/u2515283058/datasets}"

BF_ROOT="${BF_ROOT:-${DATASETS_ROOT}/BF}"
BF_TRAIN_ROOT="${BF_TRAIN_ROOT:-${BF_ROOT}/training}"
BF_VAL_ROOT="${BF_VAL_ROOT:-${BF_ROOT}/validation}"

# 服务器上的其他数据集路径。默认 BF 评估暂时不会用到这些数据集。
MMDGARMENT_ROOT="${MMDGARMENT_ROOT:-${DATASETS_ROOT}/MMDGarment}"
MMDGARMENT_TRAIN_ROOT="${MMDGARMENT_TRAIN_ROOT:-${MMDGARMENT_ROOT}/train}"
MMDGARMENT_TEST_ROOT="${MMDGARMENT_TEST_ROOT:-${MMDGARMENT_ROOT}/test}"
VITONHD_ROOT="${VITONHD_ROOT:-${DATASETS_ROOT}/vitonhd}"
VITONHD_TRAIN_ROOT="${VITONHD_TRAIN_ROOT:-${VITONHD_ROOT}/train}"
VITONHD_TEST_ROOT="${VITONHD_TEST_ROOT:-${VITONHD_ROOT}/test}"
SF_DIFFUSION_ROOT="${SF_DIFFUSION_ROOT:-${DATASETS_ROOT}/SF_Diffusion}"
SF_DIFFUSION_CLOTH_ROOT="${SF_DIFFUSION_CLOTH_ROOT:-${SF_DIFFUSION_ROOT}/cloth}"
SF_DIFFUSION_SKETCH_ROOT="${SF_DIFFUSION_SKETCH_ROOT:-${SF_DIFFUSION_ROOT}/sketch}"
SF_DIFFUSION_TEXTURE_ROOT="${SF_DIFFUSION_TEXTURE_ROOT:-${SF_DIFFUSION_ROOT}/texture}"

DATA_JSON_DIR="${DATA_JSON_DIR:-${PROJECT_ROOT}/data}"
TRAIN_JSON="${TRAIN_JSON:-${DATA_JSON_DIR}/train_bf_texture.json}"
VAL_JSON="${VAL_JSON:-${TRAIN_JSON}}"
DATA_ROOT_PATH="${DATA_ROOT_PATH:-${BF_TRAIN_ROOT}}"
VAL_ROOT_PATH="${VAL_ROOT_PATH:-${DATA_ROOT_PATH}}"

OUTPUT_BASE="${OUTPUT_BASE:-${PROJECT_ROOT}/output}"
TEXTURE_OUTPUT_DIR="${TEXTURE_OUTPUT_DIR:-${OUTPUT_BASE}/texture_adapter_bf_e20}"
E0_OUTPUT_DIR="${E0_OUTPUT_DIR:-${OUTPUT_BASE}/phase1_e0_baseline_e5}"
E1_OUTPUT_DIR="${E1_OUTPUT_DIR:-${OUTPUT_BASE}/phase1_e1_grouped_e5}"
E2A_OUTPUT_DIR="${E2A_OUTPUT_DIR:-${OUTPUT_BASE}/phase1_e2a_region_e5}"
E3A_OUTPUT_DIR="${E3A_OUTPUT_DIR:-${OUTPUT_BASE}/phase1_e3a_palette_token_e3}"
E4A_OUTPUT_DIR="${E4A_OUTPUT_DIR:-${OUTPUT_BASE}/phase1_e4a_balanced_gate_e3}"
E2B_OUTPUT_DIR="${E2B_OUTPUT_DIR:-${OUTPUT_BASE}/phase1_e2b_layer_gate_e3}"
E2B_SAFE_OUTPUT_DIR="${E2B_SAFE_OUTPUT_DIR:-${OUTPUT_BASE}/phase1_e2b_safe_gate_e3}"
E2B_COLOR_SAFE_OUTPUT_DIR="${E2B_COLOR_SAFE_OUTPUT_DIR:-${OUTPUT_BASE}/phase1_e2b_color_safe_gate_e3}"

TEXTURE_ADAPTER_CKPT="${TEXTURE_ADAPTER_CKPT:-}"
E0_CKPT="${E0_CKPT:-}"
E1_CKPT="${E1_CKPT:-}"
E2A_CKPT="${E2A_CKPT:-}"
E3A_CKPT="${E3A_CKPT:-${E3A_OUTPUT_DIR}/checkpoint-final/joint_model.pt}"
E4A_CKPT="${E4A_CKPT:-${E4A_OUTPUT_DIR}/checkpoint-final/joint_model.pt}"
E2B_CKPT="${E2B_CKPT:-${E2B_OUTPUT_DIR}/checkpoint-final/joint_model.pt}"
E2B_SAFE_CKPT="${E2B_SAFE_CKPT:-${E2B_SAFE_OUTPUT_DIR}/checkpoint-final/joint_model.pt}"
E2B_COLOR_SAFE_CKPT="${E2B_COLOR_SAFE_CKPT:-${E2B_COLOR_SAFE_OUTPUT_DIR}/checkpoint-final/joint_model.pt}"

EVAL_BASE="${EVAL_BASE:-${PROJECT_ROOT}/eval_outputs/phase1_full}"
LEGACY_EVAL_BASE="${LEGACY_EVAL_BASE:-${PROJECT_ROOT}/eval_outputs/phase1_100_with_e2b}"
REPORT_DIR="${REPORT_DIR:-${EVAL_BASE}/report_normal}"
SPLIT_PATH="${SPLIT_PATH:-${PROJECT_ROOT}/eval/benchmarks/phase1_bf_val_split.json}"
EVAL_NUM_SAMPLES="${EVAL_NUM_SAMPLES:-100}"
EVAL_SEED="${EVAL_SEED:-42}"
EVAL_DEVICE="${EVAL_DEVICE:-cuda:0}"
EVAL_PARALLEL="${EVAL_PARALLEL:-0}"
E0_DEVICE="${E0_DEVICE:-${EVAL_DEVICE}}"
E1_DEVICE="${E1_DEVICE:-${EVAL_DEVICE}}"
E2A_DEVICE="${E2A_DEVICE:-${EVAL_DEVICE}}"
E3A_DEVICE="${E3A_DEVICE:-${E2A_DEVICE}}"
E4A_DEVICE="${E4A_DEVICE:-${E3A_DEVICE}}"
E2B_DEVICE="${E2B_DEVICE:-${EVAL_DEVICE}}"
E2B_SAFE_DEVICE="${E2B_SAFE_DEVICE:-${E2B_DEVICE}}"
E2B_COLOR_SAFE_DEVICE="${E2B_COLOR_SAFE_DEVICE:-${E2B_DEVICE}}"
# Default to generating only missing samples.
EVAL_METRICS_ONLY="${EVAL_METRICS_ONLY:-0}"
REPORT_DEVICE="${REPORT_DEVICE:-cuda}"
EVAL_MODES="${EVAL_MODES:-token}"
TEXTURE_PREPROCESS_MODE="${TEXTURE_PREPROCESS_MODE:-plain_resize}"
DEFAULT_RUN_E0=1
DEFAULT_RUN_E1=1
RUN_E0="${RUN_E0:-${DEFAULT_RUN_E0}}"
RUN_E1="${RUN_E1:-${DEFAULT_RUN_E1}}"
RUN_E2A="${RUN_E2A:-1}"
RUN_E3A="${RUN_E3A:-0}"
if [[ "${RUN_E3:-0}" == "1" ]]; then
  RUN_E3A=1
fi
RUN_E4A="${RUN_E4A:-0}"
RUN_E2B="${RUN_E2B:-1}"
RUN_E2B_SAFE="${RUN_E2B_SAFE:-0}"
RUN_E2B_COLOR_SAFE="${RUN_E2B_COLOR_SAFE:-0}"
if [[ -z "${REPORT_EXPERIMENTS:-}" ]]; then
  REPORT_EXPERIMENTS=""
  if [[ "${RUN_E0}" == "1" ]]; then
    REPORT_EXPERIMENTS+="${REPORT_EXPERIMENTS:+,}e0_baseline"
  fi
  if [[ "${RUN_E1}" == "1" ]]; then
    REPORT_EXPERIMENTS+="${REPORT_EXPERIMENTS:+,}e1_grouped"
  fi
  if [[ "${RUN_E2A}" == "1" ]]; then
    REPORT_EXPERIMENTS+="${REPORT_EXPERIMENTS:+,}e2a_region"
  fi
  if [[ "${RUN_E3A}" == "1" ]]; then
    REPORT_EXPERIMENTS+="${REPORT_EXPERIMENTS:+,}e3a_palette_token"
  fi
  if [[ "${RUN_E4A}" == "1" ]]; then
    REPORT_EXPERIMENTS+="${REPORT_EXPERIMENTS:+,}e4a_balanced_gate"
  fi
  if [[ "${RUN_E2B}" == "1" ]]; then
    REPORT_EXPERIMENTS+="${REPORT_EXPERIMENTS:+,}e2b_gate"
  fi
  if [[ "${RUN_E2B_SAFE}" == "1" ]]; then
    REPORT_EXPERIMENTS+="${REPORT_EXPERIMENTS:+,}e2b_safe_gate"
  fi
  if [[ "${RUN_E2B_COLOR_SAFE}" == "1" ]]; then
    REPORT_EXPERIMENTS+="${REPORT_EXPERIMENTS:+,}e2b_color_safe_gate"
  fi
fi
CLIP_MODEL_PATH="${CLIP_MODEL_PATH:-${PROJECT_ROOT}/models/clip}"
COMPUTE_FID="${COMPUTE_FID:-1}"
COMPUTE_CLIP_I="${COMPUTE_CLIP_I:-1}"
COMPUTE_LEAKAGE="${COMPUTE_LEAKAGE:-1}"
COMPUTE_STRUCTURE="${COMPUTE_STRUCTURE:-1}"
DEBUG_SAVE_MASKS="${DEBUG_SAVE_MASKS:-20}"
FAIL_ON_EMPTY_MASKS="${FAIL_ON_EMPTY_MASKS:-0}"
MIN_VALID_PIXELS="${MIN_VALID_PIXELS:-50}"
WRITE_TEXT_SIDECARS="${WRITE_TEXT_SIDECARS:-1}"
TEXTURE_IMAGES_DIR="${TEXTURE_IMAGES_DIR:-}"
SKETCH_IMAGES_DIR="${SKETCH_IMAGES_DIR:-}"
MASK_DIR="${MASK_DIR:-}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

find_texture_ckpt() {
  local base_dir="$1"
  local ckpt=""

  if [[ -f "${base_dir}/checkpoint-final/texture_adapter.bin" ]]; then
    ckpt="${base_dir}/checkpoint-final/texture_adapter.bin"
  else
    ckpt="$(find "${base_dir}" -maxdepth 2 -path '*/texture_adapter.bin' -type f 2>/dev/null | sort -V | tail -1 || true)"
  fi

  echo "${ckpt}"
}

find_latest_gam_ckpt() {
  local base_dir="$1"
  if [[ -f "${base_dir}/checkpoint-final/joint_model.pt" ]]; then
    echo "${base_dir}/checkpoint-final/joint_model.pt"
    return
  fi
  find "${base_dir}" -maxdepth 2 -path '*/joint_model.pt' -type f 2>/dev/null | sort -V | tail -1 || true
}

count_generated_images() {
  local dir="$1"
  if [[ ! -d "${dir}" ]]; then
    echo 0
    return
  fi
  find "${dir}" -path '*/generated_*.png' -type f 2>/dev/null | wc -l
}

count_generated_text_sidecars() {
  local dir="$1"
  if [[ ! -d "${dir}" ]]; then
    echo 0
    return
  fi
  find "${dir}" -path '*/generated_*.txt' -type f 2>/dev/null | wc -l
}

count_evaluation_images() {
  local dir="$1"
  if [[ ! -d "${dir}" ]]; then
    echo 0
    return
  fi
  find "${dir}" -maxdepth 1 -name '*.png' -type f 2>/dev/null | wc -l
}

run_benchmark_if_needed() {
  local run_name="$1"
  local gam_ckpt="$2"
  local eval_device="${3:-${EVAL_DEVICE}}"
  local force_metrics_only="${4:-0}"
  local run_dir="${EVAL_BASE}/${run_name}"
  local metrics_only="${EVAL_METRICS_ONLY}"
  if [[ "${force_metrics_only}" == "1" ]]; then
    metrics_only=1
  fi
  local existing_count=0
  existing_count="$(count_generated_images "${run_dir}")"
  local generated_text_count=0
  local text_sidecars_ready=1
  generated_text_count="$(count_generated_text_sidecars "${run_dir}")"
  if [[ "${WRITE_TEXT_SIDECARS}" == "1" && "${generated_text_count}" -lt "${EVAL_NUM_SAMPLES}" ]]; then
    text_sidecars_ready=0
  fi
  local generated_compare_count=0
  local real_compare_count=0
  generated_compare_count="$(count_evaluation_images "${run_dir}/generated")"
  real_compare_count="$(count_evaluation_images "${run_dir}/real")"
  if [[ "${metrics_only}" == "1" && "${existing_count}" -lt "${EVAL_NUM_SAMPLES}" ]]; then
    echo "[WARN] ${run_name}: metrics-only requested but only ${existing_count}/${EVAL_NUM_SAMPLES} generated images found; generating missing samples."
    metrics_only=0
  fi

  if [[ "${FORCE_EVAL:-0}" != "1" && "${force_metrics_only}" != "1" && "${metrics_only}" != "1" && "${existing_count}" -ge "${EVAL_NUM_SAMPLES}" && "${generated_compare_count}" -ge "${EVAL_NUM_SAMPLES}" && "${real_compare_count}" -ge "${EVAL_NUM_SAMPLES}" && "${text_sidecars_ready}" == "1" && -f "${run_dir}/metrics_summary.json" && -f "${run_dir}/diagnostics.json" ]]; then
    echo "[SKIP] Evaluation exists: ${run_dir}/metrics_summary.json"
    return
  fi

  local cmd=(
    python tools/run_fixed_benchmark.py
    --dataset_json "${VAL_JSON}"
    --data_root "${VAL_ROOT_PATH}"
    --split_path "${SPLIT_PATH}"
    --gam_ckpt "${gam_ckpt}"
    --texture_ckpt "${TEXTURE_ADAPTER_CKPT}"
    --modes "${EVAL_MODES}"
    --num_samples "${EVAL_NUM_SAMPLES}"
    --seed "${EVAL_SEED}"
    --resume_generation 1
    --skip_existing 1
    --overwrite 0
    --device "${eval_device}"
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
    --output_dir "${EVAL_BASE}"
    --run_name "${run_name}"
  )
  if [[ -n "${TEXTURE_IMAGES_DIR}" ]]; then
    cmd+=(--texture_images_dir "${TEXTURE_IMAGES_DIR}")
  fi
  if [[ -n "${SKETCH_IMAGES_DIR}" ]]; then
    cmd+=(--sketch_images_dir "${SKETCH_IMAGES_DIR}")
  fi
  if [[ -n "${MASK_DIR}" ]]; then
    cmd+=(--mask_dir "${MASK_DIR}")
  fi
  if [[ "${metrics_only}" == "1" ]]; then
    cmd+=(--metrics_only)
  fi

  printf '%q ' "${cmd[@]}"
  echo
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    return
  fi

  "${cmd[@]}"
}

migrate_legacy_e2b_results() {
  local source_dir="${LEGACY_EVAL_BASE}/e2b_gate"
  local target_dir="${EVAL_BASE}/e2b_gate"

  E2B_RECOMPUTE_METRICS=0
  if [[ ! -d "${source_dir}" ]]; then
    return
  fi

  if [[ -e "${target_dir}" ]]; then
    echo "[WARN] E2B target already exists; legacy directory was not moved: ${source_dir}"
    echo "[WARN] Remove or merge it manually if the legacy directory still contains additional samples."
    return
  fi

  echo "[MIGRATE] Moving E2B results:"
  echo "          ${source_dir}"
  echo "       -> ${target_dir}"
  mv "${source_dir}" "${target_dir}"
  E2B_RECOMPUTE_METRICS=1
}

if [[ ! -d "${PROJECT_ROOT}" ]]; then
  echo "[ERROR] PROJECT_ROOT does not exist: ${PROJECT_ROOT}" >&2
  exit 1
fi
cd "${PROJECT_ROOT}"

if [[ ! -f "${VAL_JSON}" ]]; then
  echo "[WARN] VAL_JSON not found: ${VAL_JSON}"
  echo "[WARN] Falling back to TRAIN_JSON for fixed benchmark."
  VAL_JSON="${TRAIN_JSON}"
  VAL_ROOT_PATH="${DATA_ROOT_PATH}"
fi
if [[ ! -f "${VAL_JSON}" ]]; then
  echo "[ERROR] VAL_JSON does not exist: ${VAL_JSON}" >&2
  exit 1
fi
if [[ ! -d "${VAL_ROOT_PATH}" ]]; then
  echo "[WARN] VAL_ROOT_PATH not found: ${VAL_ROOT_PATH}"
  echo "[WARN] Falling back to DATA_ROOT_PATH."
  VAL_ROOT_PATH="${DATA_ROOT_PATH}"
fi
if [[ ! -d "${VAL_ROOT_PATH}" ]]; then
  echo "[ERROR] VAL_ROOT_PATH does not exist: ${VAL_ROOT_PATH}" >&2
  exit 1
fi

REAL_IMG_DIR="${REAL_IMG_DIR:-${VAL_ROOT_PATH}/cloth}"
if [[ ! -d "${REAL_IMG_DIR}" ]]; then
  REAL_IMG_DIR="${VAL_ROOT_PATH}"
fi

if [[ -z "${TEXTURE_ADAPTER_CKPT}" ]]; then
  TEXTURE_ADAPTER_CKPT="$(find_texture_ckpt "${TEXTURE_OUTPUT_DIR}")"
fi
if [[ -z "${E0_CKPT}" ]]; then
  E0_CKPT="$(find_latest_gam_ckpt "${E0_OUTPUT_DIR}")"
fi
if [[ -z "${E1_CKPT}" ]]; then
  E1_CKPT="$(find_latest_gam_ckpt "${E1_OUTPUT_DIR}")"
fi
if [[ -z "${E2A_CKPT}" ]]; then
  E2A_CKPT="$(find_latest_gam_ckpt "${E2A_OUTPUT_DIR}")"
fi
if [[ -z "${E3A_CKPT}" || ! -f "${E3A_CKPT}" ]]; then
  E3A_CKPT="$(find_latest_gam_ckpt "${E3A_OUTPUT_DIR}")"
fi
if [[ -z "${E4A_CKPT}" || ! -f "${E4A_CKPT}" ]]; then
  E4A_CKPT="$(find_latest_gam_ckpt "${E4A_OUTPUT_DIR}")"
fi
if [[ -z "${E2B_CKPT}" ]]; then
  E2B_CKPT="$(find_latest_gam_ckpt "${E2B_OUTPUT_DIR}")"
fi
if [[ -z "${E2B_SAFE_CKPT}" || ! -f "${E2B_SAFE_CKPT}" ]]; then
  E2B_SAFE_CKPT="$(find_latest_gam_ckpt "${E2B_SAFE_OUTPUT_DIR}")"
fi
if [[ -z "${E2B_COLOR_SAFE_CKPT}" || ! -f "${E2B_COLOR_SAFE_CKPT}" ]]; then
  E2B_COLOR_SAFE_CKPT="$(find_latest_gam_ckpt "${E2B_COLOR_SAFE_OUTPUT_DIR}")"
fi

echo "============================================"
echo "Phase 1 E0/E1/E2A/E2B evaluation"
echo "VAL_JSON=${VAL_JSON}"
echo "VAL_ROOT_PATH=${VAL_ROOT_PATH}"
echo "TEXTURE_ADAPTER_CKPT=${TEXTURE_ADAPTER_CKPT}"
echo "E0_CKPT=${E0_CKPT}"
echo "E1_CKPT=${E1_CKPT}"
echo "E2A_CKPT=${E2A_CKPT}"
echo "E3A_CKPT=${E3A_CKPT}"
echo "E4A_CKPT=${E4A_CKPT}"
echo "E2B_CKPT=${E2B_CKPT}"
echo "E2B_SAFE_CKPT=${E2B_SAFE_CKPT}"
echo "E2B_COLOR_SAFE_CKPT=${E2B_COLOR_SAFE_CKPT}"
echo "EVAL_BASE=${EVAL_BASE}"
echo "LEGACY_EVAL_BASE=${LEGACY_EVAL_BASE}"
echo "REPORT_DIR=${REPORT_DIR}"
echo "EVAL_PARALLEL=${EVAL_PARALLEL}"
echo "EVAL_METRICS_ONLY=${EVAL_METRICS_ONLY}"
echo "RUN_E0=${RUN_E0}, RUN_E1=${RUN_E1}, RUN_E2A=${RUN_E2A}, RUN_E3A=${RUN_E3A}, RUN_E4A=${RUN_E4A}, RUN_E2B=${RUN_E2B}, RUN_E2B_SAFE=${RUN_E2B_SAFE}, RUN_E2B_COLOR_SAFE=${RUN_E2B_COLOR_SAFE}"
echo "E0_DEVICE=${E0_DEVICE}, E1_DEVICE=${E1_DEVICE}, E2A_DEVICE=${E2A_DEVICE}, E3A_DEVICE=${E3A_DEVICE}, E4A_DEVICE=${E4A_DEVICE}, E2B_DEVICE=${E2B_DEVICE}, E2B_SAFE_DEVICE=${E2B_SAFE_DEVICE}, E2B_COLOR_SAFE_DEVICE=${E2B_COLOR_SAFE_DEVICE}"
echo "COMPUTE_FID=${COMPUTE_FID}, COMPUTE_CLIP_I=${COMPUTE_CLIP_I}, COMPUTE_LEAKAGE=${COMPUTE_LEAKAGE}, COMPUTE_STRUCTURE=${COMPUTE_STRUCTURE}"
echo "============================================"

mkdir -p "${EVAL_BASE}" "${REPORT_DIR}"
migrate_legacy_e2b_results

if [[ "${RUN_EVAL:-1}" == "1" ]]; then
  if [[ "${EVAL_METRICS_ONLY}" != "1" && ! -f "${TEXTURE_ADAPTER_CKPT}" ]]; then
    echo "[ERROR] TEXTURE_ADAPTER_CKPT does not exist: ${TEXTURE_ADAPTER_CKPT}" >&2
    exit 1
  fi
  if [[ "${RUN_E0}" == "1" && "${EVAL_METRICS_ONLY}" != "1" && ! -f "${E0_CKPT}" ]]; then
    echo "[ERROR] E0_CKPT does not exist: ${E0_CKPT}" >&2
    exit 1
  fi
  if [[ "${RUN_E1}" == "1" && "${EVAL_METRICS_ONLY}" != "1" && ! -f "${E1_CKPT}" ]]; then
    echo "[ERROR] E1_CKPT does not exist: ${E1_CKPT}" >&2
    exit 1
  fi
  if [[ "${RUN_E2A}" == "1" && "${EVAL_METRICS_ONLY}" != "1" && ! -f "${E2A_CKPT}" ]]; then
    echo "[ERROR] E2A_CKPT does not exist: ${E2A_CKPT}" >&2
    exit 1
  fi
  if [[ "${RUN_E3A}" == "1" && "${EVAL_METRICS_ONLY}" != "1" && ! -f "${E3A_CKPT}" ]]; then
    echo "[ERROR] E3A_CKPT does not exist: ${E3A_CKPT}" >&2
    echo "[ERROR] Train it first with: bash scripts/train_phase1_e3a.sh" >&2
    exit 1
  fi
  if [[ "${RUN_E4A}" == "1" && "${EVAL_METRICS_ONLY}" != "1" && ! -f "${E4A_CKPT}" ]]; then
    echo "[ERROR] E4A_CKPT does not exist: ${E4A_CKPT}" >&2
    echo "[ERROR] Train it first with: bash scripts/train_phase1_e4a.sh" >&2
    exit 1
  fi
  if [[ "${RUN_E2B}" == "1" && "${EVAL_METRICS_ONLY}" != "1" && ! -f "${E2B_CKPT}" ]]; then
    echo "[ERROR] E2B_CKPT does not exist: ${E2B_CKPT}" >&2
    exit 1
  fi
  if [[ "${RUN_E2B_SAFE}" == "1" && "${EVAL_METRICS_ONLY}" != "1" && ! -f "${E2B_SAFE_CKPT}" ]]; then
    echo "[ERROR] E2B_SAFE_CKPT does not exist: ${E2B_SAFE_CKPT}" >&2
    exit 1
  fi
  if [[ "${RUN_E2B_COLOR_SAFE}" == "1" && "${EVAL_METRICS_ONLY}" != "1" && ! -f "${E2B_COLOR_SAFE_CKPT}" ]]; then
    echo "[ERROR] E2B_COLOR_SAFE_CKPT does not exist: ${E2B_COLOR_SAFE_CKPT}" >&2
    exit 1
  fi

  if [[ "${EVAL_PARALLEL}" == "1" ]]; then
    pids=()
    names=()
    if [[ "${RUN_E0}" == "1" ]]; then
      run_benchmark_if_needed "e0_baseline" "${E0_CKPT}" "${E0_DEVICE}" &
      pids+=("$!")
      names+=("E0")
    fi
    if [[ "${RUN_E1}" == "1" ]]; then
      run_benchmark_if_needed "e1_grouped" "${E1_CKPT}" "${E1_DEVICE}" &
      pids+=("$!")
      names+=("E1")
    fi
    if [[ "${RUN_E2A}" == "1" ]]; then
      run_benchmark_if_needed "e2a_region" "${E2A_CKPT}" "${E2A_DEVICE}" &
      pids+=("$!")
      names+=("E2A")
    fi
    if [[ "${RUN_E3A}" == "1" ]]; then
      run_benchmark_if_needed "e3a_palette_token" "${E3A_CKPT}" "${E3A_DEVICE}" &
      pids+=("$!")
      names+=("E3A")
    fi
    if [[ "${RUN_E4A}" == "1" ]]; then
      run_benchmark_if_needed "e4a_balanced_gate" "${E4A_CKPT}" "${E4A_DEVICE}" &
      pids+=("$!")
      names+=("E4A")
    fi
    if [[ "${RUN_E2B}" == "1" ]]; then
      run_benchmark_if_needed "e2b_gate" "${E2B_CKPT}" "${E2B_DEVICE}" "${E2B_RECOMPUTE_METRICS}" &
      pids+=("$!")
      names+=("E2B")
    fi
    if [[ "${RUN_E2B_SAFE}" == "1" ]]; then
      run_benchmark_if_needed "e2b_safe_gate" "${E2B_SAFE_CKPT}" "${E2B_SAFE_DEVICE}" &
      pids+=("$!")
      names+=("E2B-safe")
    fi
    if [[ "${RUN_E2B_COLOR_SAFE}" == "1" ]]; then
      run_benchmark_if_needed "e2b_color_safe_gate" "${E2B_COLOR_SAFE_CKPT}" "${E2B_COLOR_SAFE_DEVICE}" &
      pids+=("$!")
      names+=("E2B-color-safe")
    fi

    for i in "${!pids[@]}"; do
      status=0
      wait "${pids[$i]}" || status=$?
      if [[ "${status}" != "0" ]]; then
        echo "[ERROR] ${names[$i]} evaluation failed. status=${status}" >&2
        exit "${status}"
      fi
    done
  else
    if [[ "${RUN_E0}" == "1" ]]; then
      echo "=== Fixed benchmark: E0 ==="
      run_benchmark_if_needed "e0_baseline" "${E0_CKPT}" "${E0_DEVICE}"
    fi
    if [[ "${RUN_E1}" == "1" ]]; then
      echo "=== Fixed benchmark: E1 ==="
      run_benchmark_if_needed "e1_grouped" "${E1_CKPT}" "${E1_DEVICE}"
    fi
    if [[ "${RUN_E2A}" == "1" ]]; then
      echo "=== Fixed benchmark: E2A ==="
      run_benchmark_if_needed "e2a_region" "${E2A_CKPT}" "${E2A_DEVICE}"
    fi
    if [[ "${RUN_E3A}" == "1" ]]; then
      echo "=== Fixed benchmark: E3A palette token ==="
      run_benchmark_if_needed "e3a_palette_token" "${E3A_CKPT}" "${E3A_DEVICE}"
    fi
    if [[ "${RUN_E4A}" == "1" ]]; then
      echo "=== Fixed benchmark: E4A balanced gate ==="
      run_benchmark_if_needed "e4a_balanced_gate" "${E4A_CKPT}" "${E4A_DEVICE}"
    fi
    if [[ "${RUN_E2B}" == "1" ]]; then
      echo "=== Fixed benchmark: E2B ==="
      run_benchmark_if_needed "e2b_gate" "${E2B_CKPT}" "${E2B_DEVICE}" "${E2B_RECOMPUTE_METRICS}"
    fi
    if [[ "${RUN_E2B_SAFE}" == "1" ]]; then
      echo "=== Fixed benchmark: E2B-safe ==="
      run_benchmark_if_needed "e2b_safe_gate" "${E2B_SAFE_CKPT}" "${E2B_SAFE_DEVICE}"
    fi
    if [[ "${RUN_E2B_COLOR_SAFE}" == "1" ]]; then
      echo "=== Fixed benchmark: E2B-color-safe ==="
      run_benchmark_if_needed "e2b_color_safe_gate" "${E2B_COLOR_SAFE_CKPT}" "${E2B_COLOR_SAFE_DEVICE}"
    fi
  fi
else
  echo "[SKIP] RUN_EVAL=0"
fi

if [[ "${RUN_REPORT:-1}" == "1" ]]; then
  IFS=',' read -r -a report_experiments <<< "${REPORT_EXPERIMENTS}"
  for experiment_name in "${report_experiments[@]}"; do
    experiment_name="${experiment_name//[[:space:]]/}"
    if [[ ! -f "${EVAL_BASE}/${experiment_name}/metrics_summary.json" ]]; then
      echo "[ERROR] Missing new-format metrics: ${EVAL_BASE}/${experiment_name}/metrics_summary.json" >&2
      echo "[ERROR] Recompute all requested experiments with EVAL_METRICS_ONLY=1 FORCE_EVAL=1." >&2
      exit 1
    fi
    if [[ ! -f "${EVAL_BASE}/${experiment_name}/diagnostics.json" ]]; then
      echo "[ERROR] Missing diagnostics: ${EVAL_BASE}/${experiment_name}/diagnostics.json" >&2
      echo "[ERROR] Recompute all requested experiments with EVAL_METRICS_ONLY=1 FORCE_EVAL=1." >&2
      exit 1
    fi
  done

  report_cmd=(
    python -m eval.ablation_report
    --experiments_dir "${EVAL_BASE}"
    --real_images_dir "${REAL_IMG_DIR}"
    --output_dir "${REPORT_DIR}"
    --experiment_names "${REPORT_EXPERIMENTS}"
    --device "${REPORT_DEVICE}"
    --clip_model_path "${CLIP_MODEL_PATH}"
  )

  echo "=== Building report and radar chart ==="
  printf '%q ' "${report_cmd[@]}"
  echo
  if [[ "${DRY_RUN:-0}" != "1" ]]; then
    "${report_cmd[@]}"
  fi
else
  echo "[SKIP] RUN_REPORT=0"
fi

echo "============================================"
echo "Evaluation done."
echo "Eval output:   ${EVAL_BASE}"
echo "Report output: ${REPORT_DIR}"
echo "Radar chart:   ${REPORT_DIR}/radar_chart.html"
echo "============================================"
