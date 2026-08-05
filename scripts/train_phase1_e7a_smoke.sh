#!/usr/bin/env bash
set -euo pipefail

# E7a Smoke Test: 验证 AA-TCR Fuse 模块前向/反向/保存/加载功能
# 方案第 270 行要求：数据 128~256 条，可训练参数仅 AA-TCR Fuse，步数 100~500

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
OUTPUT_DIR="${E7A_OUTPUT_DIR:-${OUTPUT_BASE}/phase1_e7a_smoke_test}"

MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-0}"
MIXED_PRECISION="${MIXED_PRECISION:-fp16}"
WIDTH="${WIDTH:-384}"
HEIGHT="${HEIGHT:-512}"

# E7a smoke test 特定配置
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-256}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-200}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-1}"
NUM_WARMUP_STEPS="${NUM_WARMUP_STEPS:-20}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"

# AA-TCR Fuse 配置
AA_TCR_LR="${AA_TCR_LR:-5e-5}"
AA_TCR_NUM_HEADS="${AA_TCR_NUM_HEADS:-4}"
AA_TCR_ALPHA_INIT="${AA_TCR_ALPHA_INIT:-0.0}"
AA_TCR_EMPTY_FALLBACK="${AA_TCR_EMPTY_FALLBACK:-1}"

# 其他超参（smoke test 简化，损失权重归零）
BF_NUM_TOKENS="${BF_NUM_TOKENS:-16}"
TEXTURE_MODE="${TEXTURE_MODE:-patch_resampled}"
TEXTURE_PREPROCESS_MODE="${TEXTURE_PREPROCESS_MODE:-plain_resize}"
CLIP_HIDDEN_LAYER="${CLIP_HIDDEN_LAYER:--1}"
LAMBDA_STYLE="${LAMBDA_STYLE:-0.0}"
LAMBDA_EDGE="${LAMBDA_EDGE:-0.0}"
LAMBDA_TEXTURE_COLOR="${LAMBDA_TEXTURE_COLOR:-0.0}"
LAMBDA_REGION_TEXTURE="${LAMBDA_REGION_TEXTURE:-0.0}"
LAMBDA_BOUNDARY="${LAMBDA_BOUNDARY:-0.0}"
LAMBDA_LEAK="${LAMBDA_LEAK:-0.0}"
LAMBDA_REGION_COLOR_LAB="${LAMBDA_REGION_COLOR_LAB:-0.0}"

REPORT_TO="${REPORT_TO:-wandb}"
WANDB_PROJECT="${WANDB_PROJECT:-Mymodel}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-phase1_e7a_smoke}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_ENTITY="${WANDB_ENTITY:-}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

cd "${PROJECT_ROOT}"

for path in "${TRAIN_JSON}" "${DATA_ROOT_PATH}" "${TEXTURE_ADAPTER_CKPT}" "${BASE_CKPT}" "${SD_MODEL}" "${VAE_MODEL}" "${CLIP_MODEL}"; do
  if [[ ! -e "${path}" ]]; then
    echo "[ERROR] Required path does not exist: ${path}" >&2
    exit 1
  fi
done

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
  --max_train_samples "${MAX_TRAIN_SAMPLES}"
  --texture_condition_mode token
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
  --max_train_steps "${MAX_TRAIN_STEPS}"
  --checkpointing_epochs 999
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
  --region_kernel_size 9
  --style_loss_type gram
  --joint_t_drop_rate 0.0
  --joint_i_drop_rate 0.0
  --joint_ti_drop_rate 0.0
  --val_vis_steps 0
  --vis_every_n_steps 0
  --report_to "${REPORT_TO}"
  --wandb_project "${WANDB_PROJECT}"
  --wandb_run_name "${WANDB_RUN_NAME}"
  --wandb_mode "${WANDB_MODE}"
)

if [[ -n "${WANDB_ENTITY}" ]]; then
  CMD+=(--wandb_entity "${WANDB_ENTITY}")
fi

echo "============================================"
echo "E7a Smoke Test - AA-TCR Fuse 功能验证"
echo "BASE_CKPT=${BASE_CKPT}"
echo "TEXTURE_ADAPTER_CKPT=${TEXTURE_ADAPTER_CKPT}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "MAX_TRAIN_SAMPLES=${MAX_TRAIN_SAMPLES}"
echo "MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS}"
echo "可训练参数: 仅 AA-TCR Fuse (7.09M)"
echo "AA_TCR_LR=${AA_TCR_LR}, ALPHA_INIT=${AA_TCR_ALPHA_INIT}"
echo "EMPTY_FALLBACK=${AA_TCR_EMPTY_FALLBACK}"
echo "NUM_GPUS=${NUM_GPUS}"
echo "============================================"
printf '%q ' "${CMD[@]}"
echo

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1, command not executed."
  exit 0
fi

"${CMD[@]}"
