#!/usr/bin/env bash
set -euo pipefail

# 阶段 1：训练 Texture Adapter。默认使用服务器路径。

# 显卡数量：直接改这里，或运行时用 NUM_GPUS=1 bash scripts/train_phase1_texture.sh 覆盖。
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
CLIP_MODEL="${CLIP_MODEL:-${PROJECT_ROOT}/models/clip}"

OUTPUT_BASE="${OUTPUT_BASE:-${PROJECT_ROOT}/output}"
OUTPUT_DIR="${TEXTURE_OUTPUT_DIR:-${OUTPUT_BASE}/texture_adapter_bf_e20}"
LOGGING_DIR="${LOGGING_DIR:-logs}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
PRETRAINED_TEXTURE_ADAPTER_PATH="${PRETRAINED_TEXTURE_ADAPTER_PATH:-}"

MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-0}"
MIXED_PRECISION="${MIXED_PRECISION:-fp16}"
WIDTH="${WIDTH:-384}"
HEIGHT="${HEIGHT:-512}"
RESOLUTION="${RESOLUTION:-512}"

LEARNING_RATE="${TEXTURE_LEARNING_RATE:-1e-4}"
WEIGHT_DECAY="${TEXTURE_WEIGHT_DECAY:-1e-2}"
NUM_TRAIN_EPOCHS="${TEXTURE_NUM_TRAIN_EPOCHS:-20}"
TRAIN_BATCH_SIZE="${TEXTURE_TRAIN_BATCH_SIZE:-4}"
GRADIENT_ACCUMULATION_STEPS="${TEXTURE_GRADIENT_ACCUMULATION_STEPS:-1}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-1}"
SAVE_STEPS="${TEXTURE_SAVE_STEPS:-0}"
VALIDATION_STEPS="${TEXTURE_VALIDATION_STEPS:-2000}"
VALIDATION_NUM_TEXTURES="${TEXTURE_VALIDATION_NUM_TEXTURES:-4}"

BF_NUM_TOKENS="${BF_NUM_TOKENS:-16}"
BF_BASE_CHANNELS="${BF_BASE_CHANNELS:-32}"
TEXTURE_MODE="${TEXTURE_MODE:-patch_resampled}"
TEXTURE_PREPROCESS_MODE="${TEXTURE_PREPROCESS_MODE:-plain_resize}"
TEXTURE_CROP_SCALE_MIN="${TEXTURE_CROP_SCALE_MIN:-0.4}"
TEXTURE_CROP_SCALE_MAX="${TEXTURE_CROP_SCALE_MAX:-0.9}"
TEXTURE_LOSS_TARGET_MODE="${TEXTURE_LOSS_TARGET_MODE:-conditioned_texture}"
LAMBDA_TEXTURE_STYLE="${LAMBDA_TEXTURE_STYLE:-0.1}"
LAMBDA_TEXTURE_GLOBAL="${LAMBDA_TEXTURE_GLOBAL:-0.05}"
CLIP_HIDDEN_LAYER="${CLIP_HIDDEN_LAYER:--1}"
FIXED_SEED="${FIXED_SEED:-1234}"

I_DROP_RATE="${I_DROP_RATE:-0.05}"
T_DROP_RATE="${T_DROP_RATE:-0.2}"
TI_DROP_RATE="${TI_DROP_RATE:-0.05}"
UNFREEZE_UP_BLOCKS="${UNFREEZE_UP_BLOCKS:-2}"
LR_SCHEDULER="${TEXTURE_LR_SCHEDULER:-cosine}"
LR_WARMUP_STEPS="${TEXTURE_LR_WARMUP_STEPS:-500}"
LOSS_TYPE="${TEXTURE_LOSS_TYPE:-huber}"
HUBER_C="${TEXTURE_HUBER_C:-0.1}"
MAX_GRAD_NORM="${TEXTURE_MAX_GRAD_NORM:-1.0}"

REPORT_TO="${REPORT_TO:-wandb}"
WANDB_PROJECT="${WANDB_PROJECT:-Mymodel}"
WANDB_RUN_NAME="${TEXTURE_WANDB_RUN_NAME:-texture_adapter_bf_e20}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_ENTITY="${WANDB_ENTITY:-}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

if [[ ! -d "${PROJECT_ROOT}" ]]; then
  echo "[ERROR] PROJECT_ROOT does not exist: ${PROJECT_ROOT}" >&2
  exit 1
fi
cd "${PROJECT_ROOT}"

if [[ "${FORCE_TRAIN:-0}" != "1" && -f "${OUTPUT_DIR}/checkpoint-final/texture_adapter.bin" ]]; then
  echo "[SKIP] Texture Adapter checkpoint already exists: ${OUTPUT_DIR}/checkpoint-final/texture_adapter.bin"
  exit 0
fi

if [[ ! -f "${TRAIN_JSON}" ]]; then
  echo "[ERROR] TRAIN_JSON does not exist: ${TRAIN_JSON}" >&2
  exit 1
fi
if [[ ! -d "${DATA_ROOT_PATH}" ]]; then
  echo "[ERROR] DATA_ROOT_PATH does not exist: ${DATA_ROOT_PATH}" >&2
  exit 1
fi
if [[ ! -d "${SD_MODEL}" ]]; then
  echo "[ERROR] SD_MODEL does not exist: ${SD_MODEL}" >&2
  exit 1
fi
if [[ ! -d "${CLIP_MODEL}" ]]; then
  echo "[ERROR] CLIP_MODEL does not exist: ${CLIP_MODEL}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

CMD=(
  accelerate launch
  --num_processes="${NUM_GPUS}"
  --main_process_port "${MAIN_PROCESS_PORT}"
  --mixed_precision="${MIXED_PRECISION}"
  train_texture_adapter.py
  --pretrained_model_name_or_path "${SD_MODEL}"
  --data_json_file "${TRAIN_JSON}"
  --data_root_path "${DATA_ROOT_PATH}"
  --image_encoder_path "${CLIP_MODEL}"
  --output_dir "${OUTPUT_DIR}"
  --logging_dir "${LOGGING_DIR}"
  --resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}"
  --resolution "${RESOLUTION}"
  --width "${WIDTH}"
  --height "${HEIGHT}"
  --learning_rate "${LEARNING_RATE}"
  --weight_decay "${WEIGHT_DECAY}"
  --num_train_epochs "${NUM_TRAIN_EPOCHS}"
  --train_batch_size "${TRAIN_BATCH_SIZE}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}"
  --save_steps "${SAVE_STEPS}"
  --i_drop_rate "${I_DROP_RATE}"
  --t_drop_rate "${T_DROP_RATE}"
  --ti_drop_rate "${TI_DROP_RATE}"
  --bf_num_tokens "${BF_NUM_TOKENS}"
  --bf_base_channels "${BF_BASE_CHANNELS}"
  --texture_mode "${TEXTURE_MODE}"
  --texture_preprocess_mode "${TEXTURE_PREPROCESS_MODE}"
  --texture_crop_scale_min "${TEXTURE_CROP_SCALE_MIN}"
  --texture_crop_scale_max "${TEXTURE_CROP_SCALE_MAX}"
  --texture_loss_target_mode "${TEXTURE_LOSS_TARGET_MODE}"
  --lambda_texture_style "${LAMBDA_TEXTURE_STYLE}"
  --lambda_texture_global "${LAMBDA_TEXTURE_GLOBAL}"
  --clip_hidden_layer "${CLIP_HIDDEN_LAYER}"
  --fixed_seed "${FIXED_SEED}"
  --validation_steps "${VALIDATION_STEPS}"
  --validation_num_textures "${VALIDATION_NUM_TEXTURES}"
  --unfreeze_mid_block
  --unfreeze_up_blocks "${UNFREEZE_UP_BLOCKS}"
  --unfreeze_attention_only
  --mixed_precision "${MIXED_PRECISION}"
  --report_to "${REPORT_TO}"
  --wandb_project "${WANDB_PROJECT}"
  --wandb_run_name "${WANDB_RUN_NAME}"
  --wandb_mode "${WANDB_MODE}"
  --lr_scheduler "${LR_SCHEDULER}"
  --lr_warmup_steps "${LR_WARMUP_STEPS}"
  --loss_type "${LOSS_TYPE}"
  --huber_c "${HUBER_C}"
  --max_grad_norm "${MAX_GRAD_NORM}"
)

if [[ -n "${PRETRAINED_TEXTURE_ADAPTER_PATH}" ]]; then
  CMD+=(--pretrained_texture_adapter_path "${PRETRAINED_TEXTURE_ADAPTER_PATH}")
fi
if [[ -n "${WANDB_ENTITY}" ]]; then
  CMD+=(--wandb_entity "${WANDB_ENTITY}")
fi

echo "============================================"
echo "Phase 1 Texture Adapter training"
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "TRAIN_JSON=${TRAIN_JSON}"
echo "DATA_ROOT_PATH=${DATA_ROOT_PATH}"
echo "SD_MODEL=${SD_MODEL}"
echo "CLIP_MODEL=${CLIP_MODEL}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "NUM_GPUS=${NUM_GPUS}, MIXED_PRECISION=${MIXED_PRECISION}"
echo "============================================"
printf '%q ' "${CMD[@]}"
echo

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1, command not executed."
  exit 0
fi

"${CMD[@]}"
