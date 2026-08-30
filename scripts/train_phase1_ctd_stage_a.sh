#!/usr/bin/env bash
# CTD Stage A 小规模训练 (docs/ctd_stage_a_spec.md §6, 第 3-5 天)
#
# 与 train_phase1_e5_tcpm_lite.sh 的唯一区别:
#   + --ctd_prob / --ctd_min_delta_e / --ctd_seed / --ctd_all_samples
#   + --max_train_samples 2000
# 其余逐 flag 相同 —— 冻结面完全沿用 E5 (--use_tcpm_lite 1 --freeze_for_tcpm_lite 1,
# 只有 TCPM-lite 可训练; texture_gate_delta 也是冻结的, Stage A 不要动它, 多一个
# 自由度就多一条归因不清的路径, 这是 E7a 的教训)。
#
# p_cf 取 0.3 而不是 0.5: 参考图 70% 可信时无条件忽略的代价大于收益, 损失面才会
# 推向"保留参考图色度、只在与 caption 严重矛盾时拉回"(spec §0.3)。
# A2 阶段扫 {0.15, 0.3, 0.5} 验证这个推理:
#     CTD_PROB=0.15 E5_OUTPUT_DIR=.../ctd_a_p015 bash scripts/train_phase1_ctd_stage_a.sh
#
# 消融 A1(全量扰动, 隔离管辖先验):  CTD_ALL_SAMPLES=1
# 等价性检查(应逐位等于 E5):        CTD_PROB=0
set -euo pipefail

NUM_GPUS="${NUM_GPUS:-4}"

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
BASE_CKPT="${BASE_CKPT:-${OUTPUT_BASE}/phase1_e2b_color_safe_gate_e3/checkpoint-final/joint_model.pt}"
# 基线: E5 TCPM-lite (spec 前言)
RESUME_CKPT="${CTD_RESUME_CKPT:-${OUTPUT_BASE}/phase1_e5_tcpm_lite_e3/checkpoint-final/joint_model.pt}"
OUTPUT_DIR="${CTD_OUTPUT_DIR:-${OUTPUT_BASE}/phase1_ctd_stage_a_p030}"

# ---- CTD-A 三个 flag + 小规模 ----
CTD_PROB="${CTD_PROB:-0.3}"
CTD_MIN_DELTA_E="${CTD_MIN_DELTA_E:-15.0}"
CTD_SEED="${CTD_SEED:-42}"
CTD_ALL_SAMPLES="${CTD_ALL_SAMPLES:-0}"
CTD_TARGET_STRATEGY="${CTD_TARGET_STRATEGY:-legacy}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-2000}"

MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-0}"
MIXED_PRECISION="${MIXED_PRECISION:-fp16}"
WIDTH="${WIDTH:-384}"
HEIGHT="${HEIGHT:-512}"
TRAIN_BATCH_SIZE="${GAM_TRAIN_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GAM_GRADIENT_ACCUMULATION_STEPS:-2}"
NUM_TRAIN_EPOCHS="${CTD_NUM_TRAIN_EPOCHS:-3}"
MAX_TRAIN_STEPS="${CTD_MAX_TRAIN_STEPS:--1}"
CHECKPOINTING_EPOCHS="${GAM_CHECKPOINTING_EPOCHS:-1}"
NUM_WARMUP_STEPS="${GAM_NUM_WARMUP_STEPS:-100}"
MAX_GRAD_NORM="${GAM_MAX_GRAD_NORM:-1.0}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-1}"

BF_NUM_TOKENS="${BF_NUM_TOKENS:-16}"
TEXTURE_MODE="${TEXTURE_MODE:-patch_resampled}"
TEXTURE_CONDITION_MODE="${TEXTURE_CONDITION_MODE:-token}"
TEXTURE_PREPROCESS_MODE="${TEXTURE_PREPROCESS_MODE:-plain_resize}"
CLIP_HIDDEN_LAYER="${CLIP_HIDDEN_LAYER:--1}"

TCPM_LR="${TCPM_LR:-5e-5}"
TCPM_SCALE_LR="${TCPM_SCALE_LR:-1e-5}"
TCPM_HIDDEN_RATIO="${TCPM_HIDDEN_RATIO:-0.25}"
TCPM_RESIDUAL_SCALE_INIT="${TCPM_RESIDUAL_SCALE_INIT:-0.0}"
TCPM_MASK_INNER_ONLY="${TCPM_MASK_INNER_ONLY:-1}"

LAMBDA_STYLE="${LAMBDA_STYLE:-1.0}"
STYLE_LOSS_TYPE="${STYLE_LOSS_TYPE:-gram}"
LAMBDA_EDGE="${LAMBDA_EDGE:-0.05}"
LAMBDA_TEXTURE_COLOR="${LAMBDA_TEXTURE_COLOR:-0.2}"
LAMBDA_REGION_TEXTURE="${LAMBDA_REGION_TEXTURE:-0.1}"
LAMBDA_BOUNDARY="${LAMBDA_BOUNDARY:-0.05}"
LAMBDA_LEAK="${LAMBDA_LEAK:-0.1}"
LAMBDA_REGION_COLOR_LAB="${LAMBDA_REGION_COLOR_LAB:-0.05}"
REGION_KERNEL_SIZE="${REGION_KERNEL_SIZE:-9}"
JOINT_T_DROP_RATE="${JOINT_T_DROP_RATE:-0.2}"
JOINT_I_DROP_RATE="${JOINT_I_DROP_RATE:-0.05}"
JOINT_TI_DROP_RATE="${JOINT_TI_DROP_RATE:-0.05}"
VAL_VIS_STEPS="${VAL_VIS_STEPS:-0}"
VIS_EVERY_N_STEPS="${VIS_EVERY_N_STEPS:-0}"

REPORT_TO="${REPORT_TO:-wandb}"
WANDB_PROJECT="${WANDB_PROJECT:-Mymodel}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-phase1_ctd_stage_a_p${CTD_PROB}}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_ENTITY="${WANDB_ENTITY:-}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

cd "${PROJECT_ROOT}"

for required_path in "${TRAIN_JSON}" "${DATA_ROOT_PATH}" "${TEXTURE_ADAPTER_CKPT}" "${BASE_CKPT}" "${RESUME_CKPT}" "${SD_MODEL}" "${VAE_MODEL}" "${CLIP_MODEL}"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "[ERROR] Required path does not exist: ${required_path}" >&2
    exit 1
  fi
done

mkdir -p "${OUTPUT_DIR}"

CMD=(accelerate launch)
if [[ "${NUM_GPUS}" -gt 1 ]]; then
  CMD+=(--multi_gpu)
fi
CMD+=(
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
  --resume_from_checkpoint "${RESUME_CKPT}"
  --output_dir "${OUTPUT_DIR}"
  --ctd_prob "${CTD_PROB}"
  --ctd_min_delta_e "${CTD_MIN_DELTA_E}"
  --ctd_seed "${CTD_SEED}"
  --ctd_all_samples "${CTD_ALL_SAMPLES}"
  --ctd_target_strategy "${CTD_TARGET_STRATEGY}"
  --max_train_samples "${MAX_TRAIN_SAMPLES}"
  --texture_condition_mode "${TEXTURE_CONDITION_MODE}"
  --layer_group_enabled 1
  --use_texture_gate 1
  --use_tcpm_lite 1
  --freeze_for_tcpm_lite 1
  --ddp_find_unused_parameters 1
  --disable_gradient_checkpointing 1
  --tcpm_lr "${TCPM_LR}"
  --tcpm_scale_lr "${TCPM_SCALE_LR}"
  --tcpm_hidden_ratio "${TCPM_HIDDEN_RATIO}"
  --tcpm_residual_scale_init "${TCPM_RESIDUAL_SCALE_INIT}"
  --tcpm_mask_inner_only "${TCPM_MASK_INNER_ONLY}"
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
  --learning_rate "${TCPM_LR}"
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

if [[ -n "${WANDB_ENTITY}" ]]; then
  CMD+=(--wandb_entity "${WANDB_ENTITY}")
fi

echo "============================================"
echo "CTD Stage A (条件端色度替换) small-scale training"
echo "RESUME_CKPT=${RESUME_CKPT}   <- E5 baseline"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "CTD_PROB=${CTD_PROB}  CTD_MIN_DELTA_E=${CTD_MIN_DELTA_E}  CTD_SEED=${CTD_SEED}"
echo "CTD_ALL_SAMPLES=${CTD_ALL_SAMPLES} (1 = 消融 A1 全量扰动)"
echo "CTD_TARGET_STRATEGY=${CTD_TARGET_STRATEGY} (legacy | gamut_aware)"
echo "MAX_TRAIN_SAMPLES=${MAX_TRAIN_SAMPLES}"
echo "NUM_GPUS=${NUM_GPUS}, NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS}"
echo "--------------------------------------------"
echo "D0 判据: train/ctd_rate 应在 ${CTD_PROB} x 0.878 的 +-10% 内,"
echo "         train/ctd_delta_e 均值 > 25。这条不过, 后面全部无意义。"
echo "D3 判据: 前 300 步 train/tcpm_residual_scale 或 train/tcpm_gate_mean"
echo "         应明显偏离 E5 的值。"
echo "============================================"
printf '%q ' "${CMD[@]}"
echo

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1, command not executed."
  exit 0
fi

"${CMD[@]}"
