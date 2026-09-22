#!/bin/bash
#SBATCH -J E14_dropout_ab
#SBATCH -p gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH -o log_e14_dropout_ab_%j.log
#SBATCH -e log_e14_dropout_ab_%j.err
set -eo pipefail
if [[ "${DRY_RUN:-0}" != 1 ]]; then
 source /share/apps/anaconda3/etc/profile.d/conda.sh
 conda activate Mymodel
fi
set -u
PROJECT_ROOT="${PROJECT_ROOT:-/share/home/u2515283058/Mymodel}"
SOURCE="${SOURCE:-${PROJECT_ROOT}/output/texture_adapter_bf_e20/checkpoint-final}"
OUT="${EVAL_ROOT:-${PROJECT_ROOT}/eval_outputs/e14_dropout_ab/${SLURM_JOB_ID:-local}}"
INPUTS="${INPUTS_ROOT:-${PROJECT_ROOT}/eval_outputs/e14_real_orientation_inputs}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "${PROJECT_ROOT}"
run() { printf '%q ' "$@"; printf '\n'; if [[ "${DRY_RUN:-0}" != 1 ]]; then "$@"; fi; }
if [[ "${DRY_RUN:-0}" != 1 ]]; then
 [[ -f "${SOURCE}/pytorch_model.bin" ]] || { echo '缺少完整pytorch_model.bin'; exit 1; }
 [[ ! -e "${OUT}" ]] || { echo '请使用新输出目录'; exit 1; }
fi
run mkdir -p "${OUT}"
run cp -r "${INPUTS}" "${OUT}/inputs"
run cp "${TRAIN_JSON:-${PROJECT_ROOT}/data/train_bf_texture.json}" "${OUT}/train_manifest.json"
for MODE in legacy_pooled zero_final_tokens; do
 run python train_texture_adapter.py \
  --pretrained_model_name_or_path "${PROJECT_ROOT}/models/stable-diffusion-v1-5" \
  --image_encoder_path "${PROJECT_ROOT}/models/clip" \
  --data_json_file "${OUT}/train_manifest.json" \
  --data_root_path "${DATA_ROOT:-/share/home/u2515283058/datasets/BF/training}" \
  --warmstart_full_model "${SOURCE}/pytorch_model.bin" \
  --output_dir "${OUT}/${MODE}" --image_dropout_mode "${MODE}" \
  --training_seed 42 --fixed_seed 42 --max_train_steps "${STEPS:-200}" \
  --train_batch_size 1 --gradient_accumulation_steps 1 --dataloader_num_workers 0 \
  --learning_rate 2e-6 --lr_scheduler constant --lr_warmup_steps 0 \
  --width 384 --height 512 --bf_num_tokens 16 --texture_mode patch_resampled \
  --texture_preprocess_mode plain_resize --texture_loss_target_mode conditioned_texture \
  --i_drop_rate 0.05 --ti_drop_rate 0.05 --t_drop_rate 0.05 \
  --lambda_texture_style 0.1 --lambda_texture_global 0 \
  --mixed_precision fp16 --report_to none --save_steps 0 --validation_steps 0
done
for STAGE in baseline legacy_pooled zero_final_tokens; do
 CKPT="${SOURCE}/texture_adapter.bin"
 if [[ "${STAGE}" != baseline ]]; then CKPT="${OUT}/${STAGE}/checkpoint-final/texture_adapter.bin"; fi
 run python -m tools.e14_pattern_probe extract --checkpoint "${CKPT}" --bf-only \
  --preprocess-protocol texture_train --allow-unmatched-extraction \
  --labels "${OUT}/inputs/feature_labels.csv" --data-root "${OUT}/inputs" \
  --base-model "${PROJECT_ROOT}/models/stable-diffusion-v1-5" --clip-model "${PROJECT_ROOT}/models/clip" \
  --output "${OUT}/${STAGE}/representations"
done
echo "完成：${OUT}。下载representations和training_metrics.jsonl；大权重留在服务器。"
