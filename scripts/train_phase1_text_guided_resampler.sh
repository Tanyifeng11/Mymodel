#!/usr/bin/env bash
set -euo pipefail

# RESAMPLER_MODE=visual 为 B 组，text 为 C 组；两组从同一个 E5 初始化。
PROJECT_ROOT="${PROJECT_ROOT:-/share/home/u2515283058/Mymodel}"
DATASETS_ROOT="${DATASETS_ROOT:-/share/home/u2515283058/datasets}"
RESAMPLER_MODE="${RESAMPLER_MODE:-text}"
case "${RESAMPLER_MODE}" in visual|text) ;; *) echo "RESAMPLER_MODE 必须为 visual 或 text" >&2; exit 1 ;; esac
OUTPUT_BASE="${OUTPUT_BASE:-${PROJECT_ROOT}/output}"
BASE_CKPT="${BASE_CKPT:-${OUTPUT_BASE}/phase1_e5_tcpm_lite_e3/checkpoint-final/joint_model.pt}"
TEXTURE_ADAPTER_CKPT="${TEXTURE_ADAPTER_CKPT:-${OUTPUT_BASE}/texture_adapter_bf_e20/checkpoint-final/texture_adapter.bin}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_BASE}/phase1_resampler_${RESAMPLER_MODE}}"
TRAIN_JSON="${TRAIN_JSON:-${PROJECT_ROOT}/data/train_bf_texture.json}"
DATA_ROOT_PATH="${DATA_ROOT_PATH:-${DATASETS_ROOT}/BF/training}"
SD_MODEL="${SD_MODEL:-${PROJECT_ROOT}/models/stable-diffusion-v1-5}"
VAE_MODEL="${VAE_MODEL:-${SD_MODEL}/vae}"
CLIP_MODEL="${CLIP_MODEL:-${PROJECT_ROOT}/models/clip}"
NUM_GPUS="${NUM_GPUS:-1}"
MIXED_PRECISION="${MIXED_PRECISION:-fp16}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false
cd "${PROJECT_ROOT}"

if [[ "${DRY_RUN:-0}" != "1" ]]; then
  for path in "${BASE_CKPT}" "${TEXTURE_ADAPTER_CKPT}" "${TRAIN_JSON}" "${DATA_ROOT_PATH}" "${SD_MODEL}" "${VAE_MODEL}" "${CLIP_MODEL}"; do
    [[ -e "${path}" ]] || { echo "缺少路径：${path}" >&2; exit 1; }
  done
fi

start_step=0
resume_args=()
if [[ -n "${TGR_RESUME_CKPT:-}" ]]; then
  resume_args=(--resume_from_checkpoint "${TGR_RESUME_CKPT}")
  start_step=-1
fi
cmd=(
  accelerate launch --num_processes "${NUM_GPUS}" --main_process_port "${MAIN_PROCESS_PORT:-0}"
  --mixed_precision "${MIXED_PRECISION}" train_GAM_texture_joint.py
  --pretrained_model_name_or_path "${SD_MODEL}" --pretrained_vae_model_path "${VAE_MODEL}"
  --image_encoder_path "${CLIP_MODEL}" --dataset_json_path "${TRAIN_JSON}" --data_root_path "${DATA_ROOT_PATH}"
  --texture_adapter_ckpt "${TEXTURE_ADAPTER_CKPT}" --gam_init_ckpt "${BASE_CKPT}"
  --output_dir "${OUTPUT_DIR}"
  --start_global_step "${START_GLOBAL_STEP:-${start_step}}" "${resume_args[@]}"
  --resampler_training "${RESAMPLER_MODE}" --text_guidance_dim "${TGR_DIM:-256}"
  --text_guidance_heads "${TGR_HEADS:-4}" --text_guidance_max_ratio "${TGR_MAX_RATIO:-0.3}"
  --text_guidance_lr "${TGR_LR:-5e-5}" --resampler_lr "${RESAMPLER_LR:-1e-5}"
  --learning_rate "${RESAMPLER_LR:-1e-5}" --seed "${TRAIN_SEED:-42}"
  --texture_condition_mode token --texture_mode patch_resampled --texture_preprocess_mode plain_resize
  --bf_num_tokens 16 --clip_hidden_layer -1 --use_tcpm_lite 1 --use_texture_gate 1 --layer_group_enabled 1
  --freeze_for_tcpm_lite 0 --use_aa_tcr_fuse 0 --use_palette_tokens 0
  --width "${WIDTH:-384}" --height "${HEIGHT:-512}" --force_resolution_override
  --train_batch_size "${TRAIN_BATCH_SIZE:-1}" --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-8}"
  --max_train_steps "${MAX_TRAIN_STEPS:-1000}" --checkpointing_steps "${CHECKPOINTING_STEPS:-250}"
  --num_warmup_steps "${NUM_WARMUP_STEPS:-50}" --max_grad_norm 1.0
  --mixed_precision "${MIXED_PRECISION}" --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-1}"
  --ddp_find_unused_parameters 1 --disable_gradient_checkpointing 1 --debug_trainable_params
  --lambda_style "${LAMBDA_STYLE:-1.0}" --style_loss_type gram --lambda_edge "${LAMBDA_EDGE:-0.05}"
  --lambda_texture_color "${LAMBDA_TEXTURE_COLOR:-0.2}" --lambda_region_texture "${LAMBDA_REGION_TEXTURE:-0.1}"
  --lambda_region_color_lab "${LAMBDA_REGION_COLOR_LAB:-0.05}" --lambda_boundary "${LAMBDA_BOUNDARY:-0.05}"
  --lambda_leak "${LAMBDA_LEAK:-0.1}" --region_kernel_size 9 --tcpm_mask_inner_only 1
  --joint_t_drop_rate 0.2 --joint_i_drop_rate 0.05 --joint_ti_drop_rate 0.05
  --val_vis_steps 0 --vis_every_n_steps 0 --report_to "${REPORT_TO:-none}"
  --wandb_run_name "resampler_${RESAMPLER_MODE}"
)
printf '%q ' "${cmd[@]}"
printf '\n'
if [[ "${DRY_RUN:-0}" != "1" ]]; then
  "${cmd[@]}"
fi
