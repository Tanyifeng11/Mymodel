#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/share/home/u2515283058/Mymodel}"
DATASETS_ROOT="${DATASETS_ROOT:-/share/home/u2515283058/datasets}"
DATA_ROOT_PATH="${DATA_ROOT_PATH:-${DATASETS_ROOT}/BF/training}"
DATASET_JSON="${DATASET_JSON:-${PROJECT_ROOT}/data/train_bf_texture.json}"
SPLIT_PATH="${SPLIT_PATH:-${PROJECT_ROOT}/eval/benchmarks/phase1_bf_val_split.json}"
OUTPUT_BASE="${OUTPUT_BASE:-${PROJECT_ROOT}/output}"
TEXTURE_CKPT="${TEXTURE_CKPT:-${OUTPUT_BASE}/texture_adapter_bf_e20/checkpoint-final/texture_adapter.bin}"
E5_CKPT="${E5_CKPT:-${OUTPUT_BASE}/phase1_e5_tcpm_lite_e3/checkpoint-final/joint_model.pt}"
E7A_CKPT="${E7A_CKPT:-${OUTPUT_BASE}/phase1_e7a/joint_model.pt}"
CLIP_MODEL="${CLIP_MODEL:-${PROJECT_ROOT}/models/clip}"

EVAL_ROOT="${E7A_CONTROL_EVAL_ROOT:-${PROJECT_ROOT}/eval_outputs/e7a_control/sketch_only}"
AUTO_ROOT="${E7A_CONTROL_AUTO_ROOT:-${PROJECT_ROOT}/eval_outputs/e7a_control/auto}"
REPORT_ROOT="${E7A_CONTROL_REPORT_ROOT:-${PROJECT_ROOT}/eval_outputs/e7a_control/report}"
NUM_SAMPLES="${NUM_SAMPLES:-500}"
EVAL_SEED="${EVAL_SEED:-42}"
GENERATION_SEEDS="${GENERATION_SEEDS:-42,123,2026}"
DEVICE="${DEVICE:-cuda:0}"
KID_SUBSETS="${KID_SUBSETS:-50}"
KID_SUBSET_SIZE="${KID_SUBSET_SIZE:-100}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-2000}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
cd "${PROJECT_ROOT}"

for path in "${DATASET_JSON}" "${DATA_ROOT_PATH}" "${SPLIT_PATH}" "${TEXTURE_CKPT}" "${E5_CKPT}" "${E7A_CKPT}" "${CLIP_MODEL}"; do
  [[ -e "${path}" ]] || { echo "[ERROR] missing required path: ${path}" >&2; exit 1; }
done

common_args=(
  --dataset_json "${DATASET_JSON}" --data_root "${DATA_ROOT_PATH}" --split_path "${SPLIT_PATH}"
  --num_samples "${NUM_SAMPLES}" --seed "${EVAL_SEED}" --sample_id_start 0 --sample_id_end "${NUM_SAMPLES}"
  --resume_generation 1 --skip_existing 1 --overwrite 0 --texture_ckpt "${TEXTURE_CKPT}"
  --device "${DEVICE}" --modes token --texture_preprocess_mode plain_resize
  --use_texture_gate 1 --use_tcpm_lite 1 --layer_group_enabled 1 --clip_model_path "${CLIP_MODEL}"
  --evaluation_protocol original_image_size --compute_kid 1 --kid_subsets "${KID_SUBSETS}" --kid_subset_size "${KID_SUBSET_SIZE}"
  --write_text_sidecars 1
)

run_main() {
  local seed="$1" name="$2" ckpt="$3" use_aa="$4"
  python tools/run_fixed_benchmark.py "${common_args[@]}" \
    --generation_seed "${seed}" --gam_ckpt "${ckpt}" --output_dir "${EVAL_ROOT}/seed_${seed}" \
    --run_name "${name}" --mask_policy sketch_only --use_aa_tcr_fuse "${use_aa}"
}

run_auto_metrics() {
  local seed="$1" name="$2" ckpt="$3" use_aa="$4"
  python tools/run_fixed_benchmark.py "${common_args[@]}" \
    --generation_seed "${seed}" --gam_ckpt "${ckpt}" --output_dir "${EVAL_ROOT}/seed_${seed}" \
    --metrics_output_dir "${AUTO_ROOT}/seed_${seed}/${name}" --run_name "${name}" \
    --mask_policy auto --use_aa_tcr_fuse "${use_aa}" --metrics_only
}

IFS=',' read -r -a seeds <<< "${GENERATION_SEEDS}"
for seed in "${seeds[@]}"; do
  echo "========== seed ${seed}: sketch_only generation + metrics =========="
  run_main "${seed}" e5 "${E5_CKPT}" 0
  run_main "${seed}" e7a_on "${E7A_CKPT}" 1
  run_main "${seed}" e7a_off "${E7A_CKPT}" 0

  echo "========== seed ${seed}: auto metrics only =========="
  run_auto_metrics "${seed}" e5 "${E5_CKPT}" 0
  run_auto_metrics "${seed}" e7a_on "${E7A_CKPT}" 1
  run_auto_metrics "${seed}" e7a_off "${E7A_CKPT}" 0

  python tools/validate_benchmark_outputs.py --experiments_dir "${EVAL_ROOT}/seed_${seed}" \
    --experiment_names e5,e7a_on,e7a_off --expected_count "${NUM_SAMPLES}"
done

python tools/report_e7a_control.py --eval_root "${EVAL_ROOT}" --output_dir "${REPORT_ROOT}/sketch_only" \
  --generation_seeds "${GENERATION_SEEDS}" --bootstrap_samples "${BOOTSTRAP_SAMPLES}"
python tools/report_e7a_control.py --eval_root "${AUTO_ROOT}" --output_dir "${REPORT_ROOT}/auto" \
  --generation_seeds "${GENERATION_SEEDS}" --bootstrap_samples "${BOOTSTRAP_SAMPLES}"

echo "[done] primary report: ${REPORT_ROOT}/sketch_only"
echo "[done] auxiliary report: ${REPORT_ROOT}/auto"
