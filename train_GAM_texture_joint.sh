#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false

unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
export NO_PROXY=localhost,127.0.0.1,huggingface.co,cdn-lfs.huggingface.co,hf.co

PRETRAINED_MODEL_NAME_OR_PATH="stable-diffusion-v1-5/stable-diffusion-v1-5"
PRETRAINED_VAE_MODEL_PATH="stabilityai/sd-vae-ft-mse"
IMAGE_ENCODER_PATH="openai/clip-vit-large-patch14"

DATA_ROOT_PATH="/mnt/d/tyf/fuxian/datasets/MMDGarment"
DATASET_JSON_PATH="/mnt/d/tyf/fuxian/Mymodel/data/train_MMD_texture.json"
TEXTURE_ADAPTER_CKPT="/mnt/d/tyf/fuxian/Mymodel/output/texture_adapter_MMG/checkpoint-52950/texture_adapter.bin"

OUTPUT_DIR="/mnt/d/tyf/fuxian/Mymodel/output/gam_smoke"

TRAIN_BATCH_SIZE=1
MAX_TRAIN_STEPS=300
LEARNING_RATE=1e-4
NUM_WARMUP_STEPS=300

BF_NUM_TOKENS=16
TEXTURE_MODE="patch_resampled"
TEXTURE_CONDITION_MODE="hybrid"
FUSION_TYPE="minimal"
TEXTURE_PREPROCESS_MODE="crop_tile"
CLIP_HIDDEN_LAYER=-1

LAMBDA_STYLE=1.0
STYLE_LOSS_TYPE="gram"
LAMBDA_PATCH_STYLE=0.0

JOINT_T_DROP_RATE=0.3
JOINT_I_DROP_RATE=0.05
JOINT_TI_DROP_RATE=0.05

ALPHA1=1.0
ALPHA2=1.0
ALPHA3=0.7
ALPHA4=0.5

MAIN_PROCESS_PORT=29510

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
  --width 384 \
  --height 512
)

echo "Running command:"
printf '%q ' "${CMD[@]}"
echo

"${CMD[@]}"