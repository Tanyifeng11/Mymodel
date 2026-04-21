#!/usr/bin/env bash
set -euo pipefail

# =========================
# Basic environment
# =========================
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false

# 减少显存碎片
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"

unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
export NO_PROXY=localhost,127.0.0.1,huggingface.co,cdn-lfs.huggingface.co,hf.co

# =========================
# Paths
# =========================
PRETRAINED_MODEL_NAME_OR_PATH="stable-diffusion-v1-5/stable-diffusion-v1-5"
PRETRAINED_VAE_MODEL_PATH="stabilityai/sd-vae-ft-mse"
IMAGE_ENCODER_PATH="openai/clip-vit-large-patch14"

DATA_ROOT_PATH="/mnt/d/tyf/fuxian/datasets/MMDGarment"
DATASET_JSON_PATH="/mnt/d/tyf/fuxian/Mymodel/data/train_MMD_texture.json"
TEXTURE_ADAPTER_CKPT="/mnt/d/tyf/fuxian/Mymodel/output/texture_adapter_MMG/checkpoint-52950/texture_adapter.bin"

RESUME_FROM_CHECKPOINT="/mnt/d/tyf/fuxian/Mymodel/output/gam/checkpoint-3000"

OUTPUT_DIR="/mnt/d/tyf/fuxian/Mymodel/output/gam"

# =========================
# Training hyperparameters
# =========================
TRAIN_BATCH_SIZE=1

# 注意：
# 这里通常表示“训练总步数上限”，不是“再训练多少步”
# 如果当前 resume 在 2000 步，且脚本按标准 resume 逻辑写的，
# MAX_TRAIN_STEPS=10000 一般表示继续训到总步数 10000，也就是再训 8000 步
MAX_TRAIN_STEPS=40000

LEARNING_RATE=5e-5
NUM_WARMUP_STEPS=1000

BF_NUM_TOKENS=16
BF_BASE_CHANNELS=32
TEXTURE_MODE="patch_resampled"
CLIP_HIDDEN_LAYER=-1

HEIGHT=576
WIDTH=448

DATALOADER_NUM_WORKERS=0
SAVE_STEPS=10000
MAX_GRAD_NORM=1.0

MAIN_PROCESS_PORT=29510

# =========================
# WandB
# =========================
REPORT_TO="wandb"
WANDB_PROJECT="Mymodel"
WANDB_RUN_NAME="gam-joint-exp2"
WANDB_MODE="online"
WANDB_ENTITY=""

# =========================
# Checks
# =========================
[ -d "${DATA_ROOT_PATH}" ] || { echo "DATA_ROOT_PATH 不存在: ${DATA_ROOT_PATH}"; exit 1; }
[ -f "${DATASET_JSON_PATH}" ] || { echo "DATASET_JSON_PATH 不存在: ${DATASET_JSON_PATH}"; exit 1; }
[ -f "${TEXTURE_ADAPTER_CKPT}" ] || { echo "TEXTURE_ADAPTER_CKPT 不存在: ${TEXTURE_ADAPTER_CKPT}"; exit 1; }

if [ -n "${RESUME_FROM_CHECKPOINT}" ]; then
  [ -d "${RESUME_FROM_CHECKPOINT}" ] || { echo "RESUME_FROM_CHECKPOINT 不存在: ${RESUME_FROM_CHECKPOINT}"; exit 1; }
else
  [ -f "${GAM_INIT_CKPT}" ] || { echo "GAM_INIT_CKPT 不存在: ${GAM_INIT_CKPT}"; exit 1; }
fi

mkdir -p "${OUTPUT_DIR}"

# =========================
# Build command
# =========================
CMD=(
  accelerate launch
  --num_processes 1
  --num_machines 1
  --mixed_precision fp16
  --dynamo_backend no
  --main_process_port "${MAIN_PROCESS_PORT}"
  train_GAM_texture_joint.py
  --pretrained_model_name_or_path "${PRETRAINED_MODEL_NAME_OR_PATH}"
  --pretrained_vae_model_path "${PRETRAINED_VAE_MODEL_PATH}"
  --image_encoder_path "${IMAGE_ENCODER_PATH}"
  --dataset_json_path "${DATASET_JSON_PATH}"
  --data_root_path "${DATA_ROOT_PATH}"
  --texture_adapter_ckpt "${TEXTURE_ADAPTER_CKPT}"
  --output_dir "${OUTPUT_DIR}"
  --train_batch_size "${TRAIN_BATCH_SIZE}"
  --max_train_steps "${MAX_TRAIN_STEPS}"
  --learning_rate "${LEARNING_RATE}"
  --num_warmup_steps "${NUM_WARMUP_STEPS}"
  --bf_num_tokens "${BF_NUM_TOKENS}"
  --bf_base_channels "${BF_BASE_CHANNELS}"
  --texture_mode "${TEXTURE_MODE}"
  --clip_hidden_layer "${CLIP_HIDDEN_LAYER}"
  --height "${HEIGHT}"
  --width "${WIDTH}"
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}"
  --save_steps "${SAVE_STEPS}"
  --max_grad_norm "${MAX_GRAD_NORM}"
  --report_to "${REPORT_TO}"
  --wandb_project "${WANDB_PROJECT}"
  --wandb_run_name "${WANDB_RUN_NAME}"
  --wandb_mode "${WANDB_MODE}"
)

if [ -n "${WANDB_ENTITY}" ]; then
  CMD+=(--wandb_entity "${WANDB_ENTITY}")
fi

# resume 优先；否则走旧 .pt 初始化
if [ -n "${RESUME_FROM_CHECKPOINT}" ]; then
  CMD+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
else
  CMD+=(--gam_init_ckpt "${GAM_INIT_CKPT}")
fi

echo "Running command:"
printf '%q ' "${CMD[@]}"
echo

"${CMD[@]}"