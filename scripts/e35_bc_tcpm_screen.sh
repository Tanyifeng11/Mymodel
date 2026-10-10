#!/usr/bin/env bash
set -euo pipefail

# BC-TCPM 配对筛选：同一 checkpoint、同一 split、同一生成 seed，
# 仅切换 TCPM 的推理作用域。所有产物集中到 output_eval/e35_bc_tcpm_20261010。
PROJECT_ROOT="${PROJECT_ROOT:-/share/home/u2515283058/Mymodel}"
DATA_ROOT="${DATA_ROOT:-/share/home/u2515283058/datasets/BF/validation}"
DATASET_JSON="${DATASET_JSON:-${PROJECT_ROOT}/data/bf_validation.json}"
SPLIT_PATH="${SPLIT_PATH:-${PROJECT_ROOT}/eval/benchmarks/bf_validation_dev500_split.json}"
TEXTURE_CKPT="${TEXTURE_CKPT:-${PROJECT_ROOT}/output/texture_adapter_bf_e20/checkpoint-final/texture_adapter.bin}"
GAM_CKPT="${GAM_CKPT:-${PROJECT_ROOT}/output/phase1_e5_tcpm_lite_e12/checkpoint-final/joint_model.pt}"
CLIP_MODEL="${CLIP_MODEL:-${PROJECT_ROOT}/models/clip}"
EVAL_ROOT="${EVAL_ROOT:-${PROJECT_ROOT}/output_eval/e35_bc_tcpm_20261010}"
NUM_SAMPLES="${NUM_SAMPLES:-128}"
GENERATION_SEED="${GENERATION_SEED:-42}"
DEVICE="${DEVICE:-cuda:0}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

for path in "${DATA_ROOT}" "${DATASET_JSON}" "${SPLIT_PATH}" "${TEXTURE_CKPT}" "${GAM_CKPT}" "${CLIP_MODEL}"; do
  [[ -e "${path}" ]] || { echo "[ERROR] missing path: ${path}" >&2; exit 1; }
done

python tools/build_bf_test_manifest.py \
  --data_root "${DATA_ROOT}" \
  --dataset_json "${DATASET_JSON}" \
  --split_path "${SPLIT_PATH}" \
  --layout flat \
  --split_count "${NUM_SAMPLES}" \
  --seed 42 \
  --train_json "${PROJECT_ROOT}/data/train_bf_texture.json"

run_arm() {
  local name="$1" mode="$2"
  local out="${EVAL_ROOT}/seed_${GENERATION_SEED}"
  mkdir -p "${out}"
  python tools/run_fixed_benchmark.py \
    --dataset_json "${DATASET_JSON}" \
    --data_root "${DATA_ROOT}" \
    --split_path "${SPLIT_PATH}" \
    --num_samples "${NUM_SAMPLES}" \
    --sample_id_start 0 --sample_id_end "${NUM_SAMPLES}" \
    --generation_seed "${GENERATION_SEED}" \
    --resume_generation 1 --skip_existing 1 --overwrite 0 \
    --gam_ckpt "${GAM_CKPT}" \
    --texture_ckpt "${TEXTURE_CKPT}" \
    --device "${DEVICE}" \
    --modes token \
    --texture_preprocess_mode plain_resize \
    --clip_model_path "${CLIP_MODEL}" \
    --output_dir "${out}" \
    --run_name "${name}" \
    --evaluation_protocol original_image_size \
    --mask_policy sketch_only \
    --fail_on_empty_masks 1 \
    --compute_fid 1 --compute_kid 1 \
    --use_texture_gate 1 --use_tcpm_lite 1 \
    --tcpm_mask_mode "${mode}" \
    --tcpm_mask_kernel_size 9
}

run_arm legacy legacy
run_arm bc_tcpm consistent

python tools/validate_benchmark_outputs.py \
  --experiments_dir "${EVAL_ROOT}/seed_${GENERATION_SEED}" \
  --experiment_names legacy,bc_tcpm \
  --expected_count "${NUM_SAMPLES}"

echo "[done] BC-TCPM screen: ${EVAL_ROOT}/seed_${GENERATION_SEED}"
