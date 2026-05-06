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
PRETRAINED_VAE_MODEL_PATH="stabilityai/sd-vae-ft-mse"
IMAGE_ENCODER_PATH="openai/clip-vit-large-patch14"

DATA_ROOT_PATH="/mnt/d/tyf/fuxian/datasets/MMDGarment"
DATASET_JSON_PATH="/mnt/d/tyf/fuxian/Mymodel/data/overfit_1sample.json"
TEXTURE_ADAPTER_CKPT="/mnt/d/tyf/fuxian/Mymodel/output/texture_adapter_MMG/checkpoint-52950/texture_adapter.bin"

OUTPUT_DIR="/mnt/d/tyf/fuxian/Mymodel/output/overfit_1sample"

# =========================
# Training hyperparameters
# =========================
TRAIN_BATCH_SIZE=1
MAX_TRAIN_STEPS=1000
LEARNING_RATE=1e-5
NUM_WARMUP_STEPS=50
MAX_GRAD_NORM=1.0

BF_NUM_TOKENS=16
TEXTURE_MODE="patch_resampled"
TEXTURE_CONDITION_MODE="hybrid"
FUSION_TYPE="minimal"
TEXTURE_PREPROCESS_MODE="crop_tile"
CLIP_HIDDEN_LAYER=-1

# 先关掉 style loss，避免把基线训歪
LAMBDA_STYLE=0.0
STYLE_LOSS_TYPE="gram"
LAMBDA_PATCH_STYLE=0.0

# 先关掉所有 dropout，先验证链路能不能记住这 1 条样本
JOINT_T_DROP_RATE=0.0
JOINT_I_DROP_RATE=0.0
JOINT_TI_DROP_RATE=0.0

ALPHA1=1.0
ALPHA2=1.0
ALPHA3=0.7
ALPHA4=0.5

WIDTH=384
HEIGHT=512

VAL_VIS_STEPS=100
VIS_EVERY_N_STEPS=100
NUM_VIS_SAMPLES=1

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
  --max_grad_norm "${MAX_GRAD_NORM}"
  --bf_num_tokens "${BF_NUM_TOKENS}"
  --texture_mode "${TEXTURE_MODE}"
  --texture_condition_mode "${TEXTURE_CONDITION_MODE}"
  --fusion_type "${FUSION_TYPE}"
  --texture_preprocess_mode "${TEXTURE_PREPROCESS_MODE}"
  --clip_hidden_layer "${CLIP_HIDDEN_LAYER}"
  --lambda_style "${LAMBDA_STYLE}"
  --style_loss_type "${STYLE_LOSS_TYPE}"
  --lambda_patch_style "${LAMBDA_PATCH_STYLE}"
  --joint_t_drop_rate "${JOINT_T_DROP_RATE}"
  --joint_i_drop_rate "${JOINT_I_DROP_RATE}"
  --joint_ti_drop_rate "${JOINT_TI_DROP_RATE}"
  --alpha1 "${ALPHA1}"
  --alpha2 "${ALPHA2}"
  --alpha3 "${ALPHA3}"
  --alpha4 "${ALPHA4}"
  --width "${WIDTH}"
  --height "${HEIGHT}"
  --val_vis_steps "${VAL_VIS_STEPS}"
  --vis_every_n_steps "${VIS_EVERY_N_STEPS}"
  --num_vis_samples "${NUM_VIS_SAMPLES}"
)

echo "Running command:"
printf '%q ' "${CMD[@]}"
echo

"${CMD[@]}"