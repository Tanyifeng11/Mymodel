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
# Paths (modify these first)
# =========================
PRETRAINED_MODEL_NAME_OR_PATH="stable-diffusion-v1-5/stable-diffusion-v1-5"
PRETRAINED_VAE_MODEL_PATH="stabilityai/sd-vae-ft-mse"
IMAGE_ENCODER_PATH="openai/clip-vit-large-patch14"

DATASET_JSON_PATH="/path/to/train_joint_texture.json"
DATA_ROOT_PATH="/path/to/GarmentBench"
TEXTURE_ADAPTER_CKPT="/path/to/texture_adapter_checkpoint.pt"

OUTPUT_DIR="/path/to/output/joint_texture_output"

# =========================
# Training hyperparameters
# =========================
TRAIN_BATCH_SIZE=4
MAX_TRAIN_STEPS=20000
LEARNING_RATE=1e-4
NUM_WARMUP_STEPS=500

BF_NUM_TOKENS=16
TEXTURE_MODE="patch_resampled"  # choices: patch_resampled | legacy_pooled
CLIP_HIDDEN_LAYER=-1

MAIN_PROCESS_PORT=29510

# =========================
# Build command
# =========================
CMD=(
  accelerate launch
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
  --texture_mode "${TEXTURE_MODE}"
  --clip_hidden_layer "${CLIP_HIDDEN_LAYER}"
)

echo "Running command:"
printf '%q ' "${CMD[@]}"
echo

"${CMD[@]}"
