#!/usr/bin/env bash
set -euo pipefail

# Retrain GAM with the fixed spatial path.
# Default: initialize from the existing texture adapter, then train GAM attention
# processors plus spatial_texture_encoder/spatial_injection. This is the right
# choice when the current spatial-only result collapses into texture patches.

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,huggingface.co,cdn-lfs.huggingface.co,hf.co}"

PRETRAINED_MODEL_NAME_OR_PATH="${PRETRAINED_MODEL_NAME_OR_PATH:-stable-diffusion-v1-5/stable-diffusion-v1-5}"
PRETRAINED_VAE_MODEL_PATH="${PRETRAINED_VAE_MODEL_PATH:-stabilityai/sd-vae-ft-mse}"
IMAGE_ENCODER_PATH="${IMAGE_ENCODER_PATH:-openai/clip-vit-large-patch14}"

DATA_ROOT_PATH="${DATA_ROOT_PATH:-/mnt/d/tyf/fuxian/datasets/MMDGarment}"
DATASET_JSON_PATH="${DATASET_JSON_PATH:-/mnt/d/tyf/fuxian/Mymodel/data/train_MMD_texture.json}"
TEXTURE_ADAPTER_CKPT="${TEXTURE_ADAPTER_CKPT:-/mnt/d/tyf/fuxian/Mymodel/output/texture_adapter_MMG/checkpoint-52950/texture_adapter.bin}"

# Optional: set this to a good existing joint_model.pt that already generates
# garments. With a good GAM_INIT_CKPT, TRAIN_SPATIAL_ONLY=1 becomes reasonable.
GAM_INIT_CKPT="${GAM_INIT_CKPT:-}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"

OUTPUT_DIR="${OUTPUT_DIR:-/mnt/d/tyf/fuxian/Mymodel/output/gam_spatial_full_retrain}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-10000}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
NUM_WARMUP_STEPS="${NUM_WARMUP_STEPS:-300}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"

BF_NUM_TOKENS="${BF_NUM_TOKENS:-16}"
BF_BASE_CHANNELS="${BF_BASE_CHANNELS:-32}"
TEXTURE_MODE="${TEXTURE_MODE:-patch_resampled}"
TEXTURE_CONDITION_MODE="${TEXTURE_CONDITION_MODE:-spatial}"
TEXTURE_PREPROCESS_MODE="${TEXTURE_PREPROCESS_MODE:-plain_resize}"
CLIP_HIDDEN_LAYER="${CLIP_HIDDEN_LAYER:--1}"

LAMBDA_STYLE="${LAMBDA_STYLE:-0.5}"
STYLE_LOSS_TYPE="${STYLE_LOSS_TYPE:-gram}"
LAMBDA_PATCH_STYLE="${LAMBDA_PATCH_STYLE:-0.0}"
LAMBDA_EDGE="${LAMBDA_EDGE:-0.05}"

JOINT_T_DROP_RATE="${JOINT_T_DROP_RATE:-0.15}"
JOINT_I_DROP_RATE="${JOINT_I_DROP_RATE:-0.10}"
JOINT_TI_DROP_RATE="${JOINT_TI_DROP_RATE:-0.05}"
HYBRID_DROP_TOKEN_RATE="${HYBRID_DROP_TOKEN_RATE:-0.25}"
HYBRID_DROP_SPATIAL_RATE="${HYBRID_DROP_SPATIAL_RATE:-0.25}"

ALPHA1="${ALPHA1:-1.0}"
ALPHA2="${ALPHA2:-1.0}"
ALPHA3="${ALPHA3:-0.7}"
ALPHA4="${ALPHA4:-0.5}"

WIDTH="${WIDTH:-384}"
HEIGHT="${HEIGHT:-512}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29510}"

# Important:
# 0 = train GAM attention processors + spatial branch. Use this when starting
#     only from texture_adapter.bin.
# 1 = train only spatial branch. Use this only with a good GAM_INIT_CKPT.
TRAIN_SPATIAL_ONLY="${TRAIN_SPATIAL_ONLY:-0}"

VAL_VIS_STEPS="${VAL_VIS_STEPS:-0}"
VIS_EVERY_N_STEPS="${VIS_EVERY_N_STEPS:-500}"
NUM_VIS_SAMPLES="${NUM_VIS_SAMPLES:-2}"
FIXED_VIS_JSON="${FIXED_VIS_JSON:-}"

mkdir -p "${OUTPUT_DIR}"

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
  --bf_base_channels "${BF_BASE_CHANNELS}"
  --texture_mode "${TEXTURE_MODE}"
  --texture_condition_mode "${TEXTURE_CONDITION_MODE}"
  --texture_preprocess_mode "${TEXTURE_PREPROCESS_MODE}"
  --clip_hidden_layer "${CLIP_HIDDEN_LAYER}"
  --lambda_style "${LAMBDA_STYLE}"
  --style_loss_type "${STYLE_LOSS_TYPE}"
  --lambda_patch_style "${LAMBDA_PATCH_STYLE}"
  --lambda_edge "${LAMBDA_EDGE}"
  --joint_t_drop_rate "${JOINT_T_DROP_RATE}"
  --joint_i_drop_rate "${JOINT_I_DROP_RATE}"
  --joint_ti_drop_rate "${JOINT_TI_DROP_RATE}"
  --hybrid_drop_token_rate "${HYBRID_DROP_TOKEN_RATE}"
  --hybrid_drop_spatial_rate "${HYBRID_DROP_SPATIAL_RATE}"
  --alpha1 "${ALPHA1}"
  --alpha2 "${ALPHA2}"
  --alpha3 "${ALPHA3}"
  --alpha4 "${ALPHA4}"
  --val_vis_steps "${VAL_VIS_STEPS}"
  --vis_every_n_steps "${VIS_EVERY_N_STEPS}"
  --num_vis_samples "${NUM_VIS_SAMPLES}"
  --width "${WIDTH}"
  --height "${HEIGHT}"
)

if [[ -n "${GAM_INIT_CKPT}" ]]; then
  CMD+=(--gam_init_ckpt "${GAM_INIT_CKPT}")
fi

if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  CMD+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

if [[ -n "${FIXED_VIS_JSON}" ]]; then
  CMD+=(--fixed_vis_json "${FIXED_VIS_JSON}")
fi

if [[ "${TRAIN_SPATIAL_ONLY}" == "1" ]]; then
  CMD+=(--train_spatial_only)
fi

echo "Running command:"
printf '%q ' "${CMD[@]}"
echo

"${CMD[@]}"
