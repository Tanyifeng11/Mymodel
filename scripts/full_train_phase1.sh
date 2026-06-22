#!/usr/bin/env bash
set -euo pipefail

# 阶段 1 完整流程：训练 Texture Adapter、E0、E1，并调用评估脚本生成结果和图表。

PROJECT_ROOT="${PROJECT_ROOT:-/share/home/u2515283058/Mymodel}"
DATASETS_ROOT="${DATASETS_ROOT:-/share/home/u2515283058/datasets}"

BF_ROOT="${BF_ROOT:-${DATASETS_ROOT}/BF}"
BF_TRAIN_ROOT="${BF_TRAIN_ROOT:-${BF_ROOT}/training}"
BF_VAL_ROOT="${BF_VAL_ROOT:-${BF_ROOT}/validation}"

# 服务器上的其他数据集路径。默认 BF 训练暂时不会用到这些数据集。
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

SD_MODEL="${SD_MODEL:-${PROJECT_ROOT}/models/stable-diffusion-v1-5}"
VAE_MODEL="${VAE_MODEL:-${SD_MODEL}/vae}"
CLIP_MODEL="${CLIP_MODEL:-${PROJECT_ROOT}/models/clip}"

OUTPUT_BASE="${OUTPUT_BASE:-${PROJECT_ROOT}/output}"
TEXTURE_OUTPUT_DIR="${TEXTURE_OUTPUT_DIR:-${OUTPUT_BASE}/texture_adapter_bf_e20}"
E0_OUTPUT_DIR="${E0_OUTPUT_DIR:-${OUTPUT_BASE}/phase1_e0_baseline_e5}"
E1_OUTPUT_DIR="${E1_OUTPUT_DIR:-${OUTPUT_BASE}/phase1_e1_grouped_e5}"

EVAL_BASE="${EVAL_BASE:-${PROJECT_ROOT}/eval_outputs/phase1_full}"
REPORT_DIR="${REPORT_DIR:-${EVAL_BASE}/report}"
SPLIT_PATH="${SPLIT_PATH:-${PROJECT_ROOT}/eval/benchmarks/phase1_bf_val_split.json}"
EVAL_NUM_SAMPLES="${EVAL_NUM_SAMPLES:-100}"
EVAL_SEED="${EVAL_SEED:-42}"
EVAL_DEVICE="${EVAL_DEVICE:-cuda:0}"
REPORT_DEVICE="${REPORT_DEVICE:-cuda}"
EVAL_MODES="${EVAL_MODES:-token}"

NUM_GPUS="${NUM_GPUS:-4}"
E1_NUM_GPUS="${E1_NUM_GPUS:-1}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-0}"
MIXED_PRECISION="${MIXED_PRECISION:-fp16}"
WIDTH="${WIDTH:-384}"
HEIGHT="${HEIGHT:-512}"
TEXTURE_PREPROCESS_MODE="${TEXTURE_PREPROCESS_MODE:-plain_resize}"
REPORT_TO="${REPORT_TO:-wandb}"
WANDB_PROJECT="${WANDB_PROJECT:-Mymodel}"
WANDB_MODE="${WANDB_MODE:-online}"

export PROJECT_ROOT DATASETS_ROOT
export BF_ROOT BF_TRAIN_ROOT BF_VAL_ROOT
export MMDGARMENT_ROOT MMDGARMENT_TRAIN_ROOT MMDGARMENT_TEST_ROOT
export VITONHD_ROOT VITONHD_TRAIN_ROOT VITONHD_TEST_ROOT
export SF_DIFFUSION_ROOT SF_DIFFUSION_CLOTH_ROOT SF_DIFFUSION_SKETCH_ROOT SF_DIFFUSION_TEXTURE_ROOT
export DATA_JSON_DIR TRAIN_JSON VAL_JSON DATA_ROOT_PATH VAL_ROOT_PATH
export SD_MODEL VAE_MODEL CLIP_MODEL
export OUTPUT_BASE TEXTURE_OUTPUT_DIR E0_OUTPUT_DIR E1_OUTPUT_DIR
export EVAL_BASE REPORT_DIR SPLIT_PATH EVAL_NUM_SAMPLES EVAL_SEED EVAL_DEVICE REPORT_DEVICE EVAL_MODES
export NUM_GPUS MAIN_PROCESS_PORT MIXED_PRECISION WIDTH HEIGHT
export TEXTURE_PREPROCESS_MODE REPORT_TO WANDB_PROJECT WANDB_MODE
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

if [[ ! -d "${PROJECT_ROOT}" ]]; then
  echo "[ERROR] PROJECT_ROOT does not exist: ${PROJECT_ROOT}" >&2
  exit 1
fi
cd "${PROJECT_ROOT}"

if [[ ! -f "${TRAIN_JSON}" ]]; then
  echo "[ERROR] TRAIN_JSON does not exist: ${TRAIN_JSON}" >&2
  exit 1
fi
if [[ ! -d "${DATA_ROOT_PATH}" ]]; then
  echo "[ERROR] DATA_ROOT_PATH does not exist: ${DATA_ROOT_PATH}" >&2
  exit 1
fi

if [[ ! -f "${VAL_JSON}" ]]; then
  echo "[WARN] VAL_JSON not found: ${VAL_JSON}"
  echo "[WARN] Falling back to TRAIN_JSON for fixed benchmark."
  VAL_JSON="${TRAIN_JSON}"
  VAL_ROOT_PATH="${DATA_ROOT_PATH}"
fi
if [[ ! -d "${VAL_ROOT_PATH}" ]]; then
  echo "[WARN] VAL_ROOT_PATH not found: ${VAL_ROOT_PATH}"
  echo "[WARN] Falling back to DATA_ROOT_PATH."
  VAL_ROOT_PATH="${DATA_ROOT_PATH}"
fi

REAL_IMG_DIR="${REAL_IMG_DIR:-${VAL_ROOT_PATH}/cloth}"
if [[ ! -d "${REAL_IMG_DIR}" ]]; then
  REAL_IMG_DIR="${VAL_ROOT_PATH}"
fi

echo "============================================"
echo "Full Phase 1 pipeline"
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "TRAIN_JSON=${TRAIN_JSON}"
echo "VAL_JSON=${VAL_JSON}"
echo "DATA_ROOT_PATH=${DATA_ROOT_PATH}"
echo "VAL_ROOT_PATH=${VAL_ROOT_PATH}"
echo "TEXTURE_OUTPUT_DIR=${TEXTURE_OUTPUT_DIR}"
echo "E0_OUTPUT_DIR=${E0_OUTPUT_DIR}"
echo "E1_OUTPUT_DIR=${E1_OUTPUT_DIR}"
echo "EVAL_BASE=${EVAL_BASE}"
echo "REPORT_DIR=${REPORT_DIR}"
echo "NUM_GPUS=${NUM_GPUS}, E1_NUM_GPUS=${E1_NUM_GPUS}"
echo "============================================"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1, printing child commands only."
  bash scripts/train_phase1_texture.sh
  bash scripts/train_phase1_e0.sh
  NUM_GPUS="${E1_NUM_GPUS}" bash scripts/train_phase1_e1.sh
  exit 0
fi

bash scripts/train_phase1_texture.sh

TEXTURE_ADAPTER_CKPT="${TEXTURE_ADAPTER_CKPT:-$(find_texture_ckpt "${TEXTURE_OUTPUT_DIR}")}"
export TEXTURE_ADAPTER_CKPT
if [[ ! -f "${TEXTURE_ADAPTER_CKPT}" ]]; then
  echo "[ERROR] Cannot find Texture Adapter checkpoint under ${TEXTURE_OUTPUT_DIR}" >&2
  exit 1
fi

bash scripts/train_phase1_e0.sh
NUM_GPUS="${E1_NUM_GPUS}" bash scripts/train_phase1_e1.sh

E0_CKPT="$(find_latest_gam_ckpt "${E0_OUTPUT_DIR}")"
E1_CKPT="$(find_latest_gam_ckpt "${E1_OUTPUT_DIR}")"
if [[ ! -f "${E0_CKPT}" ]]; then
  echo "[ERROR] Cannot find E0 checkpoint under ${E0_OUTPUT_DIR}" >&2
  exit 1
fi
if [[ ! -f "${E1_CKPT}" ]]; then
  echo "[ERROR] Cannot find E1 checkpoint under ${E1_OUTPUT_DIR}" >&2
  exit 1
fi

mkdir -p "${EVAL_BASE}" "${REPORT_DIR}"

export TEXTURE_ADAPTER_CKPT E0_CKPT E1_CKPT REAL_IMG_DIR
bash scripts/eval_phase1.sh

echo "============================================"
echo "Done."
echo "Texture Adapter: ${TEXTURE_ADAPTER_CKPT}"
echo "E0 checkpoint:   ${E0_CKPT}"
echo "E1 checkpoint:   ${E1_CKPT}"
echo "Eval output:     ${EVAL_BASE}"
echo "Report output:   ${REPORT_DIR}"
echo "Radar chart:     ${REPORT_DIR}/radar_chart.html"
echo "============================================"
