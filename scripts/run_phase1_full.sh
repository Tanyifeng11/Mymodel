#!/bin/bash
# ===========================================================================
# Phase 1: Ti-MGD Layer-Grouped Texture Control — 一键训练 + 评估脚本
# 硬件: 3 × NVIDIA A30 (24 GB each)
# 数据: BF-Fashion (约 45k 服装图像)
# ===========================================================================
set -euo pipefail

# ===========================================================================
# 0. 路径配置 — 修改这里
# ===========================================================================
DATA_ROOT="data"
TRAIN_JSON="${DATA_ROOT}/bf_fashion_train.json"
VAL_JSON="${DATA_ROOT}/bf_fashion_val.json"
REAL_IMG_DIR="${DATA_ROOT}/test_real_images"           # ground truth 图片目录，用于 FID/CLIP-I
SD_MODEL="runwayml/stable-diffusion-v1-5"
VAE_MODEL="stabilityai/sd-vae-ft-mse"
CLIP_MODEL="openai/clip-vit-large-patch14"

OUTPUT_BASE="output"
EVAL_BASE="eval_outputs"

# 硬件
NUM_GPUS=3
MIXED_PRECISION="fp16"

# 分辨率 (A30 24GB 安全上限)
WIDTH=384
HEIGHT=512

echo "============================================"
echo " Phase 1: Texture Adapter + GAM 训练流水线"
echo " GPU: ${NUM_GPUS}x A30 24GB, ${MIXED_PRECISION}"
echo " 分辨率: ${WIDTH}x${HEIGHT}"
echo "============================================"

# ===========================================================================
# 1. 训练 Texture Adapter (只做一次)
# ===========================================================================
TEXTURE_ADAPTER_DIR="${OUTPUT_BASE}/texture_adapter_bf_e20"

if [ -f "${TEXTURE_ADAPTER_DIR}/checkpoint-final/texture_adapter.bin" ]; then
    echo "[SKIP] Texture Adapter 已存在: ${TEXTURE_ADAPTER_DIR}"
else
    echo ""
    echo "=== Step 1/4: 训练 Texture Adapter (20 epoch, 约 8 小时) ==="
    accelerate launch --num_processes=${NUM_GPUS} --mixed_precision=${MIXED_PRECISION} \
        train_texture_adapter.py \
        --pretrained_model_name_or_path ${SD_MODEL} \
        --data_json_file ${TRAIN_JSON} \
        --data_root_path ${DATA_ROOT} \
        --image_encoder_path ${CLIP_MODEL} \
        --output_dir ${TEXTURE_ADAPTER_DIR} \
        --resolution 512 \
        --width ${WIDTH} --height ${HEIGHT} \
        --num_train_epochs 20 \
        --train_batch_size 4 \
        --gradient_accumulation_steps 1 \
        --learning_rate 1e-4 \
        --lr_scheduler cosine \
        --lr_warmup_steps 500 \
        --mixed_precision ${MIXED_PRECISION} \
        --bf_num_tokens 16 \
        --bf_base_channels 32 \
        --texture_mode patch_resampled \
        --texture_preprocess_mode plain_resize \
        --lambda_texture_style 0.1 \
        --lambda_texture_global 0.05 \
        --texture_loss_target_mode conditioned_texture \
        --i_drop_rate 0.05 \
        --t_drop_rate 0.2 \
        --ti_drop_rate 0.05 \
        --loss_type huber \
        --huber_c 0.1 \
        --unfreeze_mid_block \
        --unfreeze_up_blocks 2 \
        --unfreeze_attention_only \
        --save_steps 0 \
        --validation_steps 2000 \
        --validation_num_textures 4 \
        --report_to wandb \
        --wandb_project Mymodel \
        --wandb_run_name texture_adapter_bf_e20

    echo "[DONE] Texture Adapter 训练完成: ${TEXTURE_ADAPTER_DIR}"
fi

# 找到最新的 texture_adapter checkpoint
TEXTURE_CKPT=$(ls -t ${TEXTURE_ADAPTER_DIR}/checkpoint-*/texture_adapter.bin 2>/dev/null | head -1)
if [ -z "${TEXTURE_CKPT}" ]; then
    TEXTURE_CKPT=$(ls -t ${TEXTURE_ADAPTER_DIR}/checkpoint-final/texture_adapter.bin 2>/dev/null | head -1)
fi
echo "Texture Adapter checkpoint: ${TEXTURE_CKPT}"

# ===========================================================================
# 2. 训练 E0 Baseline (无分组, layer_group_enabled=0)
# ===========================================================================
E0_DIR="${OUTPUT_BASE}/phase1_e0_baseline_e5"

if [ -f "${E0_DIR}/checkpoint-epoch-5/joint_model.pt" ]; then
    echo "[SKIP] E0 baseline 已存在: ${E0_DIR}"
else
    echo ""
    echo "=== Step 2/4: 训练 E0 Baseline (无分组, 5 epoch, 约 10 小时) ==="
    accelerate launch --num_processes=${NUM_GPUS} --mixed_precision=${MIXED_PRECISION} \
        train_GAM_texture_joint.py \
        --pretrained_model_name_or_path ${SD_MODEL} \
        --pretrained_vae_model_path ${VAE_MODEL} \
        --image_encoder_path ${CLIP_MODEL} \
        --dataset_json_path ${TRAIN_JSON} \
        --data_root_path ${DATA_ROOT} \
        --texture_adapter_ckpt ${TEXTURE_CKPT} \
        --output_dir ${E0_DIR} \
        --texture_condition_mode token \
        --layer_group_enabled 0 \
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
        --wandb_run_name phase1_e0_baseline_e5

    echo "[DONE] E0 Baseline 训练完成: ${E0_DIR}"
fi

# ===========================================================================
# 3. 训练 E1 Grouped (有分组, layer_group_enabled=1)
# ===========================================================================
E1_DIR="${OUTPUT_BASE}/phase1_e1_grouped_e5"

if [ -f "${E1_DIR}/checkpoint-epoch-5/joint_model.pt" ]; then
    echo "[SKIP] E1 Grouped 已存在: ${E1_DIR}"
else
    echo ""
    echo "=== Step 3/4: 训练 E1 Grouped (有分组, 5 epoch, 约 10 小时) ==="
    accelerate launch --num_processes=${NUM_GPUS} --mixed_precision=${MIXED_PRECISION} \
        train_GAM_texture_joint.py \
        --pretrained_model_name_or_path ${SD_MODEL} \
        --pretrained_vae_model_path ${VAE_MODEL} \
        --image_encoder_path ${CLIP_MODEL} \
        --dataset_json_path ${TRAIN_JSON} \
        --data_root_path ${DATA_ROOT} \
        --texture_adapter_ckpt ${TEXTURE_CKPT} \
        --output_dir ${E1_DIR} \
        --texture_condition_mode token \
        --layer_group_enabled 1 \
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
        --wandb_run_name phase1_e1_grouped_e5

    echo "[DONE] E1 Grouped 训练完成: ${E1_DIR}"
fi

# ===========================================================================
# 4. 评估 — 生成论文消融表格
# ===========================================================================
echo ""
echo "=== Step 4/4: 评估全部实验，生成论文表格 ==="

E0_CKPT="${E0_DIR}/checkpoint-epoch-5/joint_model.pt"
E1_CKPT="${E1_DIR}/checkpoint-epoch-5/joint_model.pt"

# 4a. 每个实验独立跑 fixed benchmark
for EXP in e0_baseline e1_grouped; do
    if [ "${EXP}" = "e0_baseline" ]; then
        CKPT_PATH="${E0_CKPT}"
    else
        CKPT_PATH="${E1_CKPT}"
    fi

    if [ ! -f "${EVAL_BASE}/${EXP}/summary_metrics.json" ]; then
        echo "  评估 ${EXP} ..."
        python tools/run_fixed_benchmark.py \
            --dataset_json ${VAL_JSON} \
            --data_root ${DATA_ROOT} \
            --gam_ckpt ${CKPT_PATH} \
            --texture_ckpt ${TEXTURE_CKPT} \
            --modes token \
            --num_samples 100 \
            --seed 42 \
            --texture_preprocess_mode plain_resize \
            --output_dir ${EVAL_BASE} \
            --run_name ${EXP}
    else
        echo "  [SKIP] ${EXP} 评估结果已存在"
    fi
done

# 4b. 汇总生成论文表格
REPORT_DIR="${EVAL_BASE}/report"
python -m eval.ablation_report \
    --experiments_dir ${EVAL_BASE} \
    --real_images_dir ${REAL_IMG_DIR} \
    --output_dir ${REPORT_DIR} \
    --experiment_names e0_baseline,e1_grouped

echo ""
echo "============================================"
echo " 全部完成!"
echo "============================================"
echo ""
echo "产物位置:"
echo "  Texture Adapter: ${TEXTURE_ADAPTER_DIR}"
echo "  E0 Baseline:     ${E0_DIR}"
echo "  E1 Grouped:      ${E1_DIR}"
echo ""
echo "论文表格:"
echo "  综合表格: ${REPORT_DIR}/comprehensive_table.md"
echo "  分类表格: ${REPORT_DIR}/ablation_tables.md"
echo "  CSV:      ${REPORT_DIR}/ablation_results.csv"
echo "  雷达图:   ${REPORT_DIR}/radar_chart.html"
echo ""
echo " 打开 comprehensive_table.md 即可看到类似这样的表格:"
echo ""
echo " | Experiment      | FID ↓ | CLIP-I ↑ | TCF-LAB ↓ | TPF-Patch ↑ | LR-Colored ↓ | Edge F1 ↑ | ... |"
echo " | ---             | ---   | ---      | ---       | ---         | ---          | ---       | --- |"
echo " | e0_baseline     | 28.43 | 0.8214   | 18.32     | 0.6734      | 0.0421       | 0.6234    | ... |"
echo " | e1_grouped      | 27.98 | 0.8301   | 14.21     | 0.7102      | 0.0512       | 0.6312    | ... |"
echo ""
echo " 箭头说明: ↓ = 越低越好, ↑ = 越高越好"
echo "============================================"
