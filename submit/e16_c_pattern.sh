#!/bin/bash
#SBATCH -J E16_C_pattern
#SBATCH -p gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH -o log_e16_c_%j.log
#SBATCH -e log_e16_c_%j.err
# E16-C：只增加一个纹样重建辅助监督，其余（64 spatial token、步数、lr、数据顺序）与 B 完全一致。
#
# 监督含义：resampler 输出必须能重建 fused 里 stage3 的 8x8 局部纹样块（余弦损失），
# 因此 token 必须携带位置相关的纹样信息，而不能只做全局外观摘要。这是单变量对照，
# 不再叠加 FFT / Gram / 对比 / 分类等其它损失。
set -eo pipefail
if [[ "${DRY_RUN:-0}" != 1 ]]; then
 source /share/apps/anaconda3/etc/profile.d/conda.sh
 conda activate Mymodel
fi
set -u
PROJECT_ROOT="${PROJECT_ROOT:-/share/home/u2515283058/Mymodel}"
SOURCE="${SOURCE:-${PROJECT_ROOT}/output/e16_b_spatial64/checkpoint-final}"
E16_ROOT="${E16_ROOT:-${PROJECT_ROOT}/e16}"
STEPS="${STEPS:-2000}"
SEED="${SEED:-1234}"
PREVIOUS_REPORT="${PREVIOUS_REPORT:-${PROJECT_ROOT}/eval_e14/causal_suite/113817/denoising_report.json}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "${PROJECT_ROOT}"
run() { printf '%q ' "$@"; printf '\n'; if [[ "${DRY_RUN:-0}" != 1 ]]; then "$@"; fi; }
if [[ "${DRY_RUN:-0}" != 1 ]]; then
 [[ -f "${SOURCE}/pytorch_model.bin" ]] || { echo '缺少完整pytorch_model.bin'; exit 1; }
 [[ -f "${PREVIOUS_REPORT}" ]] || { echo '缺少 E15 报告'; exit 1; }
fi
run mkdir -p "${E16_ROOT}"

train_arm() {  # $1=输出子目录 $2=token 数
 local tag="$1" tokens="$2"
 run python train_texture_adapter.py \
  --pretrained_model_name_or_path "${PROJECT_ROOT}/models/stable-diffusion-v1-5" \
  --image_encoder_path "${PROJECT_ROOT}/models/clip" \
  --data_json_file "${TRAIN_JSON:-${PROJECT_ROOT}/data/train_bf_texture.json}" \
  --data_root_path "${DATA_ROOT:-/share/home/u2515283058/datasets/BF/training}" \
  --output_dir "${PROJECT_ROOT}/output/${tag}" \
  --warmstart_full_model "${SOURCE}/pytorch_model.bin" \
  --max_train_steps "${STEPS}" --train_batch_size 2 --gradient_accumulation_steps 2 \
  --dataloader_num_workers 0 --log_every_n_steps 50 \
  --learning_rate 1e-4 --lr_scheduler cosine --lr_warmup_steps 500 --weight_decay 1e-2 \
  --loss_type huber --huber_c 0.1 --max_grad_norm 1.0 \
  --i_drop_rate 0.05 --t_drop_rate 0.2 --ti_drop_rate 0.05 \
  --lambda_texture_style 0.1 --lambda_texture_global 0.05 \
  --texture_mode patch_resampled --texture_preprocess_mode plain_resize \
  --texture_loss_target_mode conditioned_texture \
  --unfreeze_mid_block --unfreeze_up_blocks 2 --unfreeze_attention_only \
  --bf_num_tokens "${tokens}" --bf_query_layout spatial --bf_pattern_loss_weight "${PATTERN_W:-0.2}" \
  --width 384 --height 512 --mixed_precision fp16 \
  --fixed_seed "${SEED}" --training_seed "${SEED}" \
  --report_to none --save_steps 0 --validation_steps 0
}

train_arm e16_c_pattern64 64

# 复用 E15-D1/D2：token 数由权重决定，64 token 检查点直接走同一套诊断。
for PAIR in "c_pattern64:e16_c_pattern64"; do
 TAG="${PAIR%%:*}"; ARM="${PAIR##*:}"
 CMD=(python -m tools.e15_diagnosis
  --manifest "${TRAIN_JSON:-${PROJECT_ROOT}/data/train_bf_texture.json}"
  --data-root "${DATA_ROOT:-/share/home/u2515283058/datasets/BF/training}"
  --checkpoint "${PROJECT_ROOT}/output/${ARM}/checkpoint-final/pytorch_model.bin"
  --base-model "${PROJECT_ROOT}/models/stable-diffusion-v1-5"
  --clip-model "${PROJECT_ROOT}/models/clip"
  --mask-root "${MASK_ROOT:-${PROJECT_ROOT}/eval_outputs/e14_causal_inputs/regions}"
  --previous-report "${PREVIOUS_REPORT}"
  --count "${COUNT:-32}"
  --stages d1,d2
  --device cuda:0
  --output "${E16_ROOT}/${TAG}_d1d2")
 if [[ "${DRY_RUN:-0}" != 1 ]]; then "${CMD[@]}"; else printf '%q ' "${CMD[@]}"; printf '\n'; fi
done
echo "完成：${E16_ROOT}。对比 ${E16_ROOT}/*_d1d2/SUMMARY.md 与 baseline ${PROJECT_ROOT}/e15/SUMMARY.md。"
