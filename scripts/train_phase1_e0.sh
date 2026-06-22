#!/usr/bin/env bash
set -euo pipefail

# 阶段 1：训练 E0 基线版。使用纹理 token 条件，不启用层级分组。

# 显卡数量：直接改这里，或运行时用 NUM_GPUS=1 bash scripts/train_phase1_e0.sh 覆盖。
NUM_GPUS="${NUM_GPUS:-4}"

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
DATA_ROOT_PATH="${DATA_ROOT_PATH:-${BF_TRAIN_ROOT}}"

SD_MODEL="${SD_MODEL:-${PROJECT_ROOT}/models/stable-diffusion-v1-5}"
VAE_MODEL="${VAE_MODEL:-${SD_MODEL}/vae}"
CLIP_MODEL="${CLIP_MODEL:-${PROJECT_ROOT}/models/clip}"

OUTPUT_BASE="${OUTPUT_BASE:-${PROJECT_ROOT}/output}"
TEXTURE_ADAPTER_DIR="${TEXTURE_ADAPTER_DIR:-${OUTPUT_BASE}/texture_adapter_bf_e20}"
TEXTURE_ADAPTER_CKPT="${TEXTURE_ADAPTER_CKPT:-}"
OUTPUT_DIR="${E0_OUTPUT_DIR:-${OUTPUT_BASE}/phase1_e0_baseline_e5}"

MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-0}"
MIXED_PRECISION="${MIXED_PRECISION:-fp16}"
WIDTH="${WIDTH:-384}"
HEIGHT="${HEIGHT:-512}"
CUDA_DEBUG="${CUDA_DEBUG:-0}"

TRAIN_BATCH_SIZE="${GAM_TRAIN_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GAM_GRADIENT_ACCUMULATION_STEPS:-2}"
NUM_TRAIN_EPOCHS="${GAM_NUM_TRAIN_EPOCHS:-5}"
MAX_TRAIN_STEPS="${GAM_MAX_TRAIN_STEPS:--1}"
CHECKPOINTING_EPOCHS="${GAM_CHECKPOINTING_EPOCHS:-1}"
LEARNING_RATE="${E0_LEARNING_RATE:-${GAM_LEARNING_RATE:-5e-5}}"
NUM_WARMUP_STEPS="${GAM_NUM_WARMUP_STEPS:-500}"
MAX_GRAD_NORM="${GAM_MAX_GRAD_NORM:-1.0}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-1}"

BF_NUM_TOKENS="${BF_NUM_TOKENS:-16}"
TEXTURE_MODE="${TEXTURE_MODE:-patch_resampled}"
TEXTURE_CONDITION_MODE="${TEXTURE_CONDITION_MODE:-token}"
TEXTURE_PREPROCESS_MODE="${TEXTURE_PREPROCESS_MODE:-plain_resize}"
CLIP_HIDDEN_LAYER="${CLIP_HIDDEN_LAYER:--1}"

LAMBDA_STYLE="${LAMBDA_STYLE:-1.0}"
STYLE_LOSS_TYPE="${STYLE_LOSS_TYPE:-gram}"
LAMBDA_EDGE="${LAMBDA_EDGE:-0.05}"
LAMBDA_TEXTURE_COLOR="${LAMBDA_TEXTURE_COLOR:-0.1}"
JOINT_T_DROP_RATE="${JOINT_T_DROP_RATE:-0.2}"
JOINT_I_DROP_RATE="${JOINT_I_DROP_RATE:-0.05}"
JOINT_TI_DROP_RATE="${JOINT_TI_DROP_RATE:-0.05}"
VAL_VIS_STEPS="${VAL_VIS_STEPS:-0}"
VIS_EVERY_N_STEPS="${VIS_EVERY_N_STEPS:-0}"

REPORT_TO="${REPORT_TO:-wandb}"
WANDB_PROJECT="${WANDB_PROJECT:-Mymodel}"
WANDB_RUN_NAME="${E0_WANDB_RUN_NAME:-phase1_e0_baseline_e5}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
DEBUG_CHECKPOINT_LOAD="${DEBUG_CHECKPOINT_LOAD:-0}"
FORCE_BF_NUM_TOKENS_OVERRIDE="${FORCE_BF_NUM_TOKENS_OVERRIDE:-0}"
FORCE_RESOLUTION_OVERRIDE="${FORCE_RESOLUTION_OVERRIDE:-0}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
if [[ "${CUDA_DEBUG}" == "1" ]]; then
  export CUDA_LAUNCH_BLOCKING=1
  export TORCH_SHOW_CPP_STACKTRACES=1
fi

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

if [[ -z "${TEXTURE_ADAPTER_CKPT}" ]]; then
  TEXTURE_ADAPTER_CKPT="$(find_texture_ckpt "${TEXTURE_ADAPTER_DIR}")"
fi

if [[ "${FORCE_TRAIN:-0}" != "1" ]]; then
  EXISTING_CKPT="$(find_latest_gam_ckpt "${OUTPUT_DIR}")"
  if [[ -n "${EXISTING_CKPT}" ]]; then
    echo "[SKIP] E0 checkpoint already exists: ${EXISTING_CKPT}"
    exit 0
  fi
fi

if [[ ! -f "${TRAIN_JSON}" ]]; then
  echo "[ERROR] TRAIN_JSON does not exist: ${TRAIN_JSON}" >&2
  exit 1
fi
if [[ ! -d "${DATA_ROOT_PATH}" ]]; then
  echo "[ERROR] DATA_ROOT_PATH does not exist: ${DATA_ROOT_PATH}" >&2
  exit 1
fi
if [[ ! -f "${TEXTURE_ADAPTER_CKPT}" ]]; then
  echo "[ERROR] TEXTURE_ADAPTER_CKPT does not exist: ${TEXTURE_ADAPTER_CKPT}" >&2
  exit 1
fi
if [[ ! -d "${SD_MODEL}" || ! -d "${VAE_MODEL}" || ! -d "${CLIP_MODEL}" ]]; then
  echo "[ERROR] Model path does not exist. SD=${SD_MODEL}, VAE=${VAE_MODEL}, CLIP=${CLIP_MODEL}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

CMD=(
  accelerate launch
  --num_processes="${NUM_GPUS}"
  --main_process_port "${MAIN_PROCESS_PORT}"
  --mixed_precision="${MIXED_PRECISION}"
  train_GAM_texture_joint.py
  --pretrained_model_name_or_path "${SD_MODEL}"
  --pretrained_vae_model_path "${VAE_MODEL}"
  --image_encoder_path "${CLIP_MODEL}"
  --dataset_json_path "${TRAIN_JSON}"
  --data_root_path "${DATA_ROOT_PATH}"
  --texture_adapter_ckpt "${TEXTURE_ADAPTER_CKPT}"
  --output_dir "${OUTPUT_DIR}"
  --texture_condition_mode "${TEXTURE_CONDITION_MODE}"
  --layer_group_enabled 0
  --texture_mode "${TEXTURE_MODE}"
  --texture_preprocess_mode "${TEXTURE_PREPROCESS_MODE}"
  --clip_hidden_layer "${CLIP_HIDDEN_LAYER}"
  --bf_num_tokens "${BF_NUM_TOKENS}"
  --train_batch_size "${TRAIN_BATCH_SIZE}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
  --num_train_epochs "${NUM_TRAIN_EPOCHS}"
  --max_train_steps "${MAX_TRAIN_STEPS}"
  --checkpointing_epochs "${CHECKPOINTING_EPOCHS}"
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}"
  --mixed_precision "${MIXED_PRECISION}"
  --learning_rate "${LEARNING_RATE}"
  --num_warmup_steps "${NUM_WARMUP_STEPS}"
  --max_grad_norm "${MAX_GRAD_NORM}"
  --width "${WIDTH}"
  --height "${HEIGHT}"
  --lambda_style "${LAMBDA_STYLE}"
  --lambda_edge "${LAMBDA_EDGE}"
  --lambda_texture_color "${LAMBDA_TEXTURE_COLOR}"
  --style_loss_type "${STYLE_LOSS_TYPE}"
  --joint_t_drop_rate "${JOINT_T_DROP_RATE}"
  --joint_i_drop_rate "${JOINT_I_DROP_RATE}"
  --joint_ti_drop_rate "${JOINT_TI_DROP_RATE}"
  --val_vis_steps "${VAL_VIS_STEPS}"
  --vis_every_n_steps "${VIS_EVERY_N_STEPS}"
  --report_to "${REPORT_TO}"
  --wandb_project "${WANDB_PROJECT}"
  --wandb_run_name "${WANDB_RUN_NAME}"
  --wandb_mode "${WANDB_MODE}"
)

if [[ -n "${WANDB_ENTITY}" ]]; then
  CMD+=(--wandb_entity "${WANDB_ENTITY}")
fi
if [[ "${DEBUG_CHECKPOINT_LOAD}" == "1" ]]; then
  CMD+=(--debug_checkpoint_load)
fi
if [[ "${FORCE_BF_NUM_TOKENS_OVERRIDE}" == "1" ]]; then
  CMD+=(--force_bf_num_tokens_override)
fi
if [[ "${FORCE_RESOLUTION_OVERRIDE}" == "1" ]]; then
  CMD+=(--force_resolution_override)
fi

echo "============================================"
echo "Phase 1 E0 baseline training"
echo "TRAIN_JSON=${TRAIN_JSON}"
echo "DATA_ROOT_PATH=${DATA_ROOT_PATH}"
echo "TEXTURE_ADAPTER_CKPT=${TEXTURE_ADAPTER_CKPT}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "NUM_GPUS=${NUM_GPUS}, MIXED_PRECISION=${MIXED_PRECISION}"
echo "WIDTH=${WIDTH}, HEIGHT=${HEIGHT}, BF_NUM_TOKENS=${BF_NUM_TOKENS}"
echo "DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS}, MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS}"
echo "CUDA_DEBUG=${CUDA_DEBUG}"
echo "FORCE_BF_NUM_TOKENS_OVERRIDE=${FORCE_BF_NUM_TOKENS_OVERRIDE}, FORCE_RESOLUTION_OVERRIDE=${FORCE_RESOLUTION_OVERRIDE}"
echo "============================================"
printf '%q ' "${CMD[@]}"
echo

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1, command not executed."
  exit 0
fi

"${CMD[@]}"
