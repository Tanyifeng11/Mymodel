#!/usr/bin/env bash
set -euo pipefail

# =========================
# Basic environment
# =========================
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false

# 不走代理，避免 HF 下载超时
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
export NO_PROXY=localhost,127.0.0.1,huggingface.co,cdn-lfs.huggingface.co,hf.co

# =========================
# Paths
# =========================

# SD1.5 base model
PRETRAINED_MODEL_NAME_OR_PATH="stable-diffusion-v1-5/stable-diffusion-v1-5"

# CLIP image encoder
IMAGE_ENCODER_PATH="openai/clip-vit-large-patch14"

# Training json
DATA_JSON_FILE="/mnt/d/tyf/fuxian/Mymodel/data/train_MMD_texture.json"

# Root folder for images referenced in json
DATA_ROOT_PATH="/mnt/d/tyf/fuxian/datasets/MMDGarment"

# Output directory
OUTPUT_DIR="/mnt/d/tyf/fuxian/Mymodel/output/texture_adapter_MMG"

# Logging subdir
LOGGING_DIR="logs"

# Resume from old adapter checkpoint
#PRETRAINED_TEXTURE_ADAPTER_PATH="./output/texture_adapter_bf_fashion/checkpoint-90000/texture_adapter.bin"
PRETRAINED_TEXTURE_ADAPTER_PATH=""
# =========================
# Training hyperparameters
# =========================
RESOLUTION=512
LEARNING_RATE=1e-4
WEIGHT_DECAY=1e-2
NUM_TRAIN_EPOCHS=25
TRAIN_BATCH_SIZE=8
DATALOADER_NUM_WORKERS=2
SAVE_STEPS=4380
MIXED_PRECISION="fp16"

REPORT_TO="wandb"
WANDB_PROJECT="Mymodel"
WANDB_RUN_NAME="texture-adapter-exp1"
WANDB_MODE="online"   # online/offline/disabled

# =========================
# Build command
# =========================
CMD=(
  accelerate launch
  --mixed_precision="${MIXED_PRECISION}"
  train_texture_adapter.py
  --pretrained_model_name_or_path "${PRETRAINED_MODEL_NAME_OR_PATH}"
  --data_json_file "${DATA_JSON_FILE}"
  --data_root_path "${DATA_ROOT_PATH}"
  --image_encoder_path "${IMAGE_ENCODER_PATH}"
  --output_dir "${OUTPUT_DIR}"
  --logging_dir "${LOGGING_DIR}"
  --resolution "${RESOLUTION}"
  --learning_rate "${LEARNING_RATE}"
  --weight_decay "${WEIGHT_DECAY}"
  --num_train_epochs "${NUM_TRAIN_EPOCHS}"
  --train_batch_size "${TRAIN_BATCH_SIZE}"
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}"
  --save_steps "${SAVE_STEPS}"
  --mixed_precision "${MIXED_PRECISION}"
  --report_to "${REPORT_TO}"
  --wandb_project "${WANDB_PROJECT}"
  --wandb_run_name "${WANDB_RUN_NAME}"
  --wandb_mode "${WANDB_MODE}"
)

if [ -n "${PRETRAINED_TEXTURE_ADAPTER_PATH}" ]; then
  CMD+=(--pretrained_texture_adapter_path "${PRETRAINED_TEXTURE_ADAPTER_PATH}")
fi

echo "Running command:"
printf '%q ' "${CMD[@]}"
echo

"${CMD[@]}"
