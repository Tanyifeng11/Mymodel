#!/usr/bin/env bash
set -euo pipefail

# E7a formal training: train only the AA-TCR Fuse on the full dataset.
NUM_GPUS="${NUM_GPUS:-1}"

PROJECT_ROOT="${PROJECT_ROOT:-/share/home/u2515283058/Mymodel}"
DATASETS_ROOT="${DATASETS_ROOT:-/share/home/u2515283058/datasets}"
BF_ROOT="${BF_ROOT:-${DATASETS_ROOT}/BF}"
BF_TRAIN_ROOT="${BF_TRAIN_ROOT:-${BF_ROOT}/training}"

DATA_JSON_DIR="${DATA_JSON_DIR:-${PROJECT_ROOT}/data}"
TRAIN_JSON="${TRAIN_JSON:-${DATA_JSON_DIR}/train_bf_texture.json}"
DATA_ROOT_PATH="${DATA_ROOT_PATH:-${BF_TRAIN_ROOT}}"

SD_MODEL="${SD_MODEL:-${PROJECT_ROOT}/models/stable-diffusion-v1-5}"
VAE_MODEL="${VAE_MODEL:-${SD_MODEL}/vae}"
CLIP_MODEL="${CLIP_MODEL:-${PROJECT_ROOT}/models/clip}"

OUTPUT_BASE="${OUTPUT_BASE:-${PROJECT_ROOT}/output}"
TEXTURE_ADAPTER_CKPT="${TEXTURE_ADAPTER_CKPT:-${OUTPUT_BASE}/texture_adapter_bf_e20/checkpoint-final/texture_adapter.bin}"
BASE_CKPT="${BASE_CKPT:-${OUTPUT_BASE}/phase1_e5_tcpm_lite_e3/checkpoint-final/joint_model.pt}"
RESUME_CKPT="${E7A_RESUME_CKPT:-}"
OUTPUT_DIR="${E7A_OUTPUT_DIR:-${OUTPUT_BASE}/phase1_e7a}"

MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-0}"
MIXED_PRECISION="${MIXED_PRECISION:-fp16}"
WIDTH="${WIDTH:-384}"
HEIGHT="${HEIGHT:-512}"

TRAIN_BATCH_SIZE="${E7A_TRAIN_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${E7A_GRADIENT_ACCUMULATION_STEPS:-2}"
NUM_TRAIN_EPOCHS="${E7A_NUM_TRAIN_EPOCHS:-3}"
MAX_TRAIN_STEPS="${E7A_MAX_TRAIN_STEPS:--1}"
CHECKPOINTING_EPOCHS="${E7A_CHECKPOINTING_EPOCHS:-1}"
DATALOADER_NUM_WORKERS="${E7A_DATALOADER_NUM_WORKERS:-1}"
NUM_WARMUP_STEPS="${E7A_NUM_WARMUP_STEPS:-500}"
MAX_GRAD_NORM="${E7A_MAX_GRAD_NORM:-1.0}"

BF_NUM_TOKENS="${BF_NUM_TOKENS:-16}"
TEXTURE_MODE="${TEXTURE_MODE:-patch_resampled}"
TEXTURE_CONDITION_MODE="${TEXTURE_CONDITION_MODE:-token}"
TEXTURE_PREPROCESS_MODE="${TEXTURE_PREPROCESS_MODE:-plain_resize}"
CLIP_HIDDEN_LAYER="${CLIP_HIDDEN_LAYER:--1}"

AA_TCR_LR="${AA_TCR_LR:-5e-5}"
AA_TCR_NUM_HEADS="${AA_TCR_NUM_HEADS:-4}"
AA_TCR_ALPHA_INIT="${AA_TCR_ALPHA_INIT:-0.0}"
AA_TCR_EMPTY_FALLBACK="${AA_TCR_EMPTY_FALLBACK:-1}"

LAMBDA_STYLE="${LAMBDA_STYLE:-1.0}"
STYLE_LOSS_TYPE="${STYLE_LOSS_TYPE:-gram}"
LAMBDA_EDGE="${LAMBDA_EDGE:-0.05}"
LAMBDA_TEXTURE_COLOR="${LAMBDA_TEXTURE_COLOR:-0.2}"
LAMBDA_REGION_TEXTURE="${LAMBDA_REGION_TEXTURE:-0.1}"
LAMBDA_REGION_COLOR_LAB="${LAMBDA_REGION_COLOR_LAB:-0.05}"
LAMBDA_BOUNDARY="${LAMBDA_BOUNDARY:-0.05}"
LAMBDA_LEAK="${LAMBDA_LEAK:-0.1}"
REGION_KERNEL_SIZE="${REGION_KERNEL_SIZE:-9}"
JOINT_T_DROP_RATE="${JOINT_T_DROP_RATE:-0.2}"
JOINT_I_DROP_RATE="${JOINT_I_DROP_RATE:-0.05}"
JOINT_TI_DROP_RATE="${JOINT_TI_DROP_RATE:-0.05}"
VAL_VIS_STEPS="${VAL_VIS_STEPS:-0}"
VIS_EVERY_N_STEPS="${VIS_EVERY_N_STEPS:-0}"

REPORT_TO="${REPORT_TO:-wandb}"
WANDB_PROJECT="${WANDB_PROJECT:-Mymodel}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-phase1_e7a}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_ENTITY="${WANDB_ENTITY:-}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

cd "${PROJECT_ROOT}"

for required_path in \
  "${TRAIN_JSON}" \
  "${DATA_ROOT_PATH}" \
  "${TEXTURE_ADAPTER_CKPT}" \
  "${BASE_CKPT}" \
  "${SD_MODEL}" \
  "${VAE_MODEL}" \
  "${CLIP_MODEL}"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "[ERROR] Required path does not exist: ${required_path}" >&2
    exit 1
  fi
done

if [[ -n "${RESUME_CKPT}" && ! -e "${RESUME_CKPT}" ]]; then
  echo "[ERROR] E7A_RESUME_CKPT does not exist: ${RESUME_CKPT}" >&2
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
  --gam_init_ckpt "${BASE_CKPT}"
  --output_dir "${OUTPUT_DIR}"
  --texture_condition_mode "${TEXTURE_CONDITION_MODE}"
  --layer_group_enabled 1
  --use_texture_gate 1
  --use_tcpm_lite 1
  --use_aa_tcr_fuse 1
  --freeze_all_but_aa_tcr 1
  --aa_tcr_lr "${AA_TCR_LR}"
  --aa_tcr_num_heads "${AA_TCR_NUM_HEADS}"
  --aa_tcr_alpha_init "${AA_TCR_ALPHA_INIT}"
  --aa_tcr_empty_fallback "${AA_TCR_EMPTY_FALLBACK}"
  --ddp_find_unused_parameters 1
  --disable_gradient_checkpointing 1
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
  --learning_rate "${AA_TCR_LR}"
  --num_warmup_steps "${NUM_WARMUP_STEPS}"
  --max_grad_norm "${MAX_GRAD_NORM}"
  --width "${WIDTH}"
  --height "${HEIGHT}"
  --lambda_style "${LAMBDA_STYLE}"
  --lambda_edge "${LAMBDA_EDGE}"
  --lambda_texture_color "${LAMBDA_TEXTURE_COLOR}"
  --lambda_region_texture "${LAMBDA_REGION_TEXTURE}"
  --lambda_region_color_lab "${LAMBDA_REGION_COLOR_LAB}"
  --lambda_boundary "${LAMBDA_BOUNDARY}"
  --lambda_leak "${LAMBDA_LEAK}"
  --region_kernel_size "${REGION_KERNEL_SIZE}"
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

if [[ -n "${RESUME_CKPT}" ]]; then
  CMD+=(--resume_from_checkpoint "${RESUME_CKPT}")
fi

if [[ -n "${WANDB_ENTITY}" ]]; then
  CMD+=(--wandb_entity "${WANDB_ENTITY}")
fi

echo "============================================"
echo "Phase 1 E7a AA-TCR Fuse formal training"
echo "BASE_CKPT=${BASE_CKPT}"
echo "RESUME_CKPT=${RESUME_CKPT}"
echo "TEXTURE_ADAPTER_CKPT=${TEXTURE_ADAPTER_CKPT}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "AA_TCR_LR=${AA_TCR_LR}, AA_TCR_NUM_HEADS=${AA_TCR_NUM_HEADS}"
echo "NUM_GPUS=${NUM_GPUS}, NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS}, MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS}"
echo "Trainable parameters: AA-TCR Fuse only"
echo "============================================"
printf '%q ' "${CMD[@]}"
echo

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1, command not executed."
  exit 0
fi

"${CMD[@]}"
