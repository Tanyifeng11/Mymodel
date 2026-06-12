#!/bin/bash
set -euo pipefail

# Run Phase 1 E0/E1 only. Texture Adapter is assumed to be trained already.

DATA_JSON_DIR="data"
DATA_ROOT_PATH="/share/home/u2515283058/datasets/BF/training"

TRAIN_JSON="${DATA_JSON_DIR}/bf_fashion_train.json"
SD_MODEL="/share/home/u2515283058/Mymodel/models/stable-diffusion-v1-5"
VAE_MODEL="/share/home/u2515283058/Mymodel/models/stable-diffusion-v1-5/vae"
CLIP_MODEL="/share/home/u2515283058/Mymodel/models/clip"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-0}"

NUM_GPUS=4
MIXED_PRECISION="fp16"
WIDTH=384
HEIGHT=512

TEXTURE_ADAPTER_DIR="output/texture_adapter_bf_e20"
E0_DIR="output/phase1_e0_baseline_e5"
E1_DIR="output/phase1_e1_grouped_e5"

find_texture_ckpt() {
    local base_dir="$1"
    local ckpt=""

    if [ -f "${base_dir}/checkpoint-final/texture_adapter.bin" ]; then
        ckpt="${base_dir}/checkpoint-final/texture_adapter.bin"
    else
        ckpt=$(find "${base_dir}" -maxdepth 2 -path "*/texture_adapter.bin" -type f \
            | grep -E "checkpoint-epoch-[0-9]+/texture_adapter.bin$" \
            | sort -V \
            | tail -1 || true)
    fi

    if [ -z "${ckpt}" ]; then
        ckpt=$(find "${base_dir}" -maxdepth 2 -path "*/texture_adapter.bin" -type f \
            | grep -E "checkpoint-[0-9]+/texture_adapter.bin$" \
            | sort -V \
            | tail -1 || true)
    fi

    echo "${ckpt}"
}

find_latest_gam_ckpt() {
    local base_dir="$1"
    find "${base_dir}" -maxdepth 2 -path "*/joint_model.pt" -type f \
        | sort -V \
        | tail -1 || true
}

TEXTURE_CKPT=$(find_texture_ckpt "${TEXTURE_ADAPTER_DIR}")
if [ -z "${TEXTURE_CKPT}" ]; then
    echo "[ERROR] Cannot find texture_adapter.bin under ${TEXTURE_ADAPTER_DIR}"
    exit 1
fi

echo "============================================"
echo " Phase 1 E0/E1 training"
echo " Texture checkpoint: ${TEXTURE_CKPT}"
echo " GPU: ${NUM_GPUS}, precision: ${MIXED_PRECISION}"
echo " Resolution: ${WIDTH}x${HEIGHT}"
echo "============================================"

run_gam_train() {
    local output_dir="$1"
    local layer_group_enabled="$2"
    local run_name="$3"

    local existing_ckpt
    existing_ckpt=$(find_latest_gam_ckpt "${output_dir}")
    if [ -n "${existing_ckpt}" ]; then
        echo "[SKIP] ${run_name} already has checkpoint: ${existing_ckpt}"
        return
    fi

    echo ""
    echo "=== Training ${run_name} ==="
    accelerate launch --num_processes=${NUM_GPUS} --main_process_port ${MAIN_PROCESS_PORT} --mixed_precision=${MIXED_PRECISION} \
        train_GAM_texture_joint.py \
        --pretrained_model_name_or_path ${SD_MODEL} \
        --pretrained_vae_model_path ${VAE_MODEL} \
        --image_encoder_path ${CLIP_MODEL} \
        --dataset_json_path ${TRAIN_JSON} \
        --data_root_path ${DATA_ROOT_PATH} \
        --texture_adapter_ckpt ${TEXTURE_CKPT} \
        --output_dir ${output_dir} \
        --texture_condition_mode token \
        --layer_group_enabled ${layer_group_enabled} \
        --texture_mode patch_resampled \
        --texture_preprocess_mode plain_resize \
        --bf_num_tokens 16 \
        --train_batch_size 1 \
        --gradient_accumulation_steps 2 \
        --num_train_epochs 5 \
        --checkpointing_epochs 1 \
        --learning_rate 5e-5 \
        --num_warmup_steps 500 \
        --max_grad_norm 1.0 \
        --width ${WIDTH} --height ${HEIGHT} \
        --lambda_style 1.0 \
        --lambda_edge 0.05 \
        --lambda_texture_color 0.1 \
        --style_loss_type gram \
        --joint_t_drop_rate 0.2 \
        --joint_i_drop_rate 0.05 \
        --joint_ti_drop_rate 0.05 \
        --val_vis_steps 0 \
        --report_to wandb \
        --wandb_project Mymodel \
        --wandb_run_name ${run_name}

    echo "[DONE] ${run_name}: ${output_dir}"
}

run_gam_train "${E0_DIR}" 0 "phase1_e0_baseline_e5"
run_gam_train "${E1_DIR}" 1 "phase1_e1_grouped_e5"

echo ""
echo "============================================"
echo " E0/E1 training finished"
echo " E0 latest: $(find_latest_gam_ckpt "${E0_DIR}")"
echo " E1 latest: $(find_latest_gam_ckpt "${E1_DIR}")"
echo "============================================"