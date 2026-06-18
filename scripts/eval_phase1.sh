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

TEXTURE_ADAPTER_CKPT="${TEXTURE_ADAPTER_CKPT:-}"
E0_CKPT="${E0_CKPT:-}"
E1_CKPT="${E1_CKPT:-}"
E2A_CKPT="${E2A_CKPT:-}"

EVAL_BASE="${EVAL_BASE:-${PROJECT_ROOT}/eval_outputs/phase1_full}"
REPORT_DIR="${REPORT_DIR:-${EVAL_BASE}/report}"
SPLIT_PATH="${SPLIT_PATH:-${PROJECT_ROOT}/eval/benchmarks/phase1_bf_val_split.json}"
EVAL_NUM_SAMPLES="${EVAL_NUM_SAMPLES:-100}"
EVAL_SEED="${EVAL_SEED:-42}"
EVAL_DEVICE="${EVAL_DEVICE:-cuda:0}"
EVAL_PARALLEL="${EVAL_PARALLEL:-0}"
E0_DEVICE="${E0_DEVICE:-cuda:0}"
E1_DEVICE="${E1_DEVICE:-cuda:1}"
E2A_DEVICE="${E2A_DEVICE:-${EVAL_DEVICE}}"
EVAL_METRICS_ONLY="${EVAL_METRICS_ONLY:-0}"
REPORT_DEVICE="${REPORT_DEVICE:-cuda}"
EVAL_MODES="${EVAL_MODES:-token}"
TEXTURE_PREPROCESS_MODE="${TEXTURE_PREPROCESS_MODE:-plain_resize}"
RUN_E0="${RUN_E0:-0}"
RUN_E1="${RUN_E1:-0}"
RUN_E2A="${RUN_E2A:-1}"
REPORT_EXPERIMENTS="${REPORT_EXPERIMENTS:-e0_baseline,e1_grouped,e2a_region}"

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

run_benchmark_if_needed() {
  local run_name="$1"
  local gam_ckpt="$2"
  local eval_device="${3:-${EVAL_DEVICE}}"
  local run_dir="${EVAL_BASE}/${run_name}"

  if [[ "${FORCE_EVAL:-0}" != "1" && "${EVAL_METRICS_ONLY}" != "1" && -f "${run_dir}/summary_metrics.json" ]]; then
    echo "[SKIP] Evaluation exists: ${run_dir}/summary_metrics.json"
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
    --device "${eval_device}"
    --texture_preprocess_mode "${TEXTURE_PREPROCESS_MODE}"
    --output_dir "${EVAL_BASE}"
    --run_name "${run_name}"
  )
  if [[ "${EVAL_METRICS_ONLY}" == "1" ]]; then
    cmd+=(--metrics_only)
  fi

  printf '%q ' "${cmd[@]}"
  echo
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    return
  fi

  "${cmd[@]}"
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

echo "============================================"
echo "Phase 1/2A evaluation"
echo "VAL_JSON=${VAL_JSON}"
echo "VAL_ROOT_PATH=${VAL_ROOT_PATH}"
echo "TEXTURE_ADAPTER_CKPT=${TEXTURE_ADAPTER_CKPT}"
echo "E0_CKPT=${E0_CKPT}"
echo "E1_CKPT=${E1_CKPT}"
echo "E2A_CKPT=${E2A_CKPT}"
echo "EVAL_BASE=${EVAL_BASE}"
echo "REPORT_DIR=${REPORT_DIR}"
echo "EVAL_PARALLEL=${EVAL_PARALLEL}"
echo "EVAL_METRICS_ONLY=${EVAL_METRICS_ONLY}"
echo "RUN_E0=${RUN_E0}, RUN_E1=${RUN_E1}, RUN_E2A=${RUN_E2A}"
echo "E0_DEVICE=${E0_DEVICE}, E1_DEVICE=${E1_DEVICE}, E2A_DEVICE=${E2A_DEVICE}"
echo "============================================"

mkdir -p "${EVAL_BASE}" "${REPORT_DIR}"

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
  fi
else
  echo "[SKIP] RUN_EVAL=0"
fi

if [[ "${RUN_REPORT:-1}" == "1" ]]; then
  report_cmd=(
    python -m eval.ablation_report
    --experiments_dir "${EVAL_BASE}"
    --real_images_dir "${REAL_IMG_DIR}"
    --output_dir "${REPORT_DIR}"
    --experiment_names "${REPORT_EXPERIMENTS}"
    --device "${REPORT_DEVICE}"
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
