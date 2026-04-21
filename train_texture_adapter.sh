#!/usr/bin/env bash
set -euo pipefail

# =========================
# Basic environment
# =========================
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false

unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
export NO_PROXY=localhost,127.0.0.1,huggingface.co,cdn-lfs.huggingface.co,hf.co

# =========================
# Paths
# =========================
PRETRAINED_MODEL_NAME_OR_PATH="stable-diffusion-v1-5/stable-diffusion-v1-5"
IMAGE_ENCODER_PATH="openai/clip-vit-large-patch14"

DATA_JSON_FILE="/mnt/d/tyf/fuxian/Mymodel/data/train_MMD_texture.json"
DATA_ROOT_PATH="/mnt/d/tyf/fuxian/datasets/MMDGarment"

# 建议新目录，避免和旧实验混在一起
OUTPUT_DIR="/mnt/d/tyf/fuxian/Mymodel/output/texture_adapter_MMG_Bf_Texture"
LOGGING_DIR="logs"

# 置空("")表示从头训练；填写checkpoint路径表示继续训练（resume/finetune）
PRETRAINED_TEXTURE_ADAPTER_PATH="/mnt/d/tyf/fuxian/Mymodel/output/texture_adapter_MMG_Bf_Texture/checkpoint-43800/texture_adapter.bin"

# =========================
# Training hyperparameters
# =========================
RESOLUTION=512
WIDTH=512
HEIGHT=640

LEARNING_RATE=1e-5
WEIGHT_DECAY=1e-2
NUM_TRAIN_EPOCHS=15
TRAIN_BATCH_SIZE=4
DATALOADER_NUM_WORKERS=2
SAVE_STEPS=8760

I_DROP_RATE=0.02
T_DROP_RATE=0.02
TI_DROP_RATE=0.02

BF_NUM_TOKENS=4
BF_BASE_CHANNELS=32

MIXED_PRECISION="fp16"

REPORT_TO="wandb"
WANDB_PROJECT="Mymodel"
WANDB_RUN_NAME="texture-adapter-exp5"
WANDB_MODE="online"

ADAM_BETA1=0.9
ADAM_BETA2=0.999
ADAM_EPSILON=1e-8
LR_SCHEDULER="cosine"
LR_WARMUP_STEPS=300
LOSS_TYPE="huber"
HUBER_C=0.1
MAX_GRAD_NORM=1.0
GRADIENT_ACCUMULATION_STEPS=1

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
  --width "${WIDTH}"
  --height "${HEIGHT}"
  --learning_rate "${LEARNING_RATE}"
  --weight_decay "${WEIGHT_DECAY}"
  --num_train_epochs "${NUM_TRAIN_EPOCHS}"
  --train_batch_size "${TRAIN_BATCH_SIZE}"
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}"
  --save_steps "${SAVE_STEPS}"
  --i_drop_rate "${I_DROP_RATE}"
  --t_drop_rate "${T_DROP_RATE}"
  --ti_drop_rate "${TI_DROP_RATE}"
  --bf_num_tokens "${BF_NUM_TOKENS}"
  --bf_base_channels "${BF_BASE_CHANNELS}"
  --mixed_precision "${MIXED_PRECISION}"
  --report_to "${REPORT_TO}"
  --wandb_project "${WANDB_PROJECT}"
  --wandb_run_name "${WANDB_RUN_NAME}"
  --wandb_mode "${WANDB_MODE}"
  --adam_beta1 "${ADAM_BETA1}"
  --adam_beta2 "${ADAM_BETA2}"
  --adam_epsilon "${ADAM_EPSILON}"
  --lr_scheduler "${LR_SCHEDULER}"
  --lr_warmup_steps "${LR_WARMUP_STEPS}"
  --loss_type "${LOSS_TYPE}"
  --huber_c "${HUBER_C}"
  --max_grad_norm "${MAX_GRAD_NORM}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
)

if [ -n "${PRETRAINED_TEXTURE_ADAPTER_PATH}" ]; then
  CMD+=(--pretrained_texture_adapter_path "${PRETRAINED_TEXTURE_ADAPTER_PATH}")
fi

echo "Running command:"
printf '%q ' "${CMD[@]}"
echo

"${CMD[@]}"
