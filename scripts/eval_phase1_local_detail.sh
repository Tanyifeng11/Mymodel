#!/usr/bin/env bash
set -euo pipefail

# 固定官方 validation 样本与 seed，先验证冻结和 off 等价，再比较 A/B。
PROJECT_ROOT="${PROJECT_ROOT:-/share/home/u2515283058/Mymodel}"
DATASETS_ROOT="${DATASETS_ROOT:-/share/home/u2515283058/datasets}"
BF_SPLIT="${BF_SPLIT:-validation}"
case "${BF_SPLIT}" in validation|test) ;; *) echo "BF_SPLIT 必须为 validation 或 test" >&2; exit 1 ;; esac
NUM_SAMPLES="${NUM_SAMPLES:-64}"
DATA_ROOT_PATH="${DATA_ROOT_PATH:-${DATASETS_ROOT}/BF/${BF_SPLIT}}"
DATASET_JSON="${DATASET_JSON:-${PROJECT_ROOT}/data/bf_${BF_SPLIT}_e9.json}"
SPLIT_PATH="${SPLIT_PATH:-${PROJECT_ROOT}/eval/benchmarks/bf_${BF_SPLIT}_e9_${NUM_SAMPLES}.json}"
OUTPUT_BASE="${OUTPUT_BASE:-${PROJECT_ROOT}/output}"
E5_CKPT="${E5_CKPT:-${OUTPUT_BASE}/phase1_e5_tcpm_lite_e3/checkpoint-final/joint_model.pt}"
E9_A_CKPT="${E9_A_CKPT:-${OUTPUT_BASE}/phase1_e9_resampled/checkpoint-final/joint_model.pt}"
E9_B_CKPT="${E9_B_CKPT:-${OUTPUT_BASE}/phase1_e9_local/checkpoint-final/joint_model.pt}"
TEXTURE_CKPT="${TEXTURE_CKPT:-${OUTPUT_BASE}/texture_adapter_bf_e20/checkpoint-final/texture_adapter.bin}"
CLIP_MODEL="${CLIP_MODEL:-${PROJECT_ROOT}/models/clip}"
EVAL_RUN_ID="${EVAL_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
EVAL_ROOT="${EVAL_ROOT:-${PROJECT_ROOT}/eval_outputs/local_detail_${BF_SPLIT}_${NUM_SAMPLES}/${EVAL_RUN_ID}}"
GENERATION_SEEDS="${GENERATION_SEEDS:-42}"
EXPERIMENTS="e5,e9_a_off,e9_a,e9_b_off,e9_b"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false
cd "${PROJECT_ROOT}"
run() {
  printf '%q ' "$@"
  printf '\n'
  if [[ "${DRY_RUN:-0}" != "1" ]]; then "$@"; fi
}
if [[ "${DRY_RUN:-0}" != "1" ]]; then
  for manifest in "${EVAL_ROOT}"/seed_*/*/experiment_manifest.json; do
    [[ ! -f "${manifest}" ]] || { echo "评测目录已有结果：${manifest}；请设置新的 EVAL_RUN_ID 或 EVAL_ROOT。" >&2; exit 1; }
  done
  for path in "${DATASET_JSON}" "${SPLIT_PATH}"; do
    [[ -f "${path}" ]] || { echo "缺少 E9 图案初筛文件：${path}；请同步已固定的清单，或使用 tools/prepare_e9_pattern_split.py 重建。" >&2; exit 1; }
  done
fi
run python tools/prepare_e9_pattern_split.py --check-only --dataset-json "${DATASET_JSON}" \
  --split-path "${SPLIT_PATH}" --data-root "${DATA_ROOT_PATH}" --expected-count "${NUM_SAMPLES}" \
  --train-json "${TRAIN_JSON:-${PROJECT_ROOT}/data/train_bf_texture.json}"
run python tools/check_e9_frozen.py --e5-ckpt "${E5_CKPT}" --candidate-ckpt "${E9_A_CKPT}" \
  --expected-source resampled --output-dir "${EVAL_ROOT}/frozen_check_a"
run python tools/check_e9_frozen.py --e5-ckpt "${E5_CKPT}" --candidate-ckpt "${E9_B_CKPT}" \
  --expected-source local --paired-report "${EVAL_ROOT}/frozen_check_a/frozen_check.json" \
  --output-dir "${EVAL_ROOT}/frozen_check_b"

common=(
  --dataset_json "${DATASET_JSON}" --data_root "${DATA_ROOT_PATH}" --split_path "${SPLIT_PATH}"
  --num_samples "${NUM_SAMPLES}" --sample_id_start 0 --sample_id_end "${NUM_SAMPLES}" --seed 42
  --texture_ckpt "${TEXTURE_CKPT}" --clip_model_path "${CLIP_MODEL}" --device "${DEVICE:-cuda:0}"
  --modes token --texture_preprocess_mode plain_resize --use_tcpm_lite 1 --use_texture_gate 1
  --layer_group_enabled 1 --use_aa_tcr_fuse 0 --use_palette_tokens 0 --use_text_guided_resampler 0
  --mask_policy sketch_only --evaluation_protocol original_image_size
  --compute_kid 1 --kid_subset_size "${NUM_SAMPLES}" --kid_subsets 50
  --write_text_sidecars 1 --resume_generation 1 --skip_existing 1 --overwrite 0
)
IFS=',' read -r -a seeds <<< "${GENERATION_SEEDS}"
IFS=',' read -r -a experiments <<< "${EXPERIMENTS}"
for seed in "${seeds[@]}"; do
  for experiment in "${experiments[@]}"; do
    case "${experiment}" in
      e5) ckpt="${E5_CKPT}"; enabled=0 ;;
      e9_a_off) ckpt="${E9_A_CKPT}"; enabled=0 ;;
      e9_a) ckpt="${E9_A_CKPT}"; enabled=1 ;;
      e9_b_off) ckpt="${E9_B_CKPT}"; enabled=0 ;;
      e9_b) ckpt="${E9_B_CKPT}"; enabled=1 ;;
    esac
    run python tools/run_fixed_benchmark.py "${common[@]}" \
      --generation_seed "${seed}" --gam_ckpt "${ckpt}" --run_name "${experiment}" \
      --use_local_detail_adapter "${enabled}" --output_dir "${EVAL_ROOT}/seed_${seed}"
  done
  run python tools/validate_benchmark_outputs.py --experiments_dir "${EVAL_ROOT}/seed_${seed}" \
    --experiment_names "${EXPERIMENTS}" --expected_count "${NUM_SAMPLES}"
  run python tools/check_e9_images.py --experiments-dir "${EVAL_ROOT}/seed_${seed}" \
    --expected-count "${NUM_SAMPLES}" --output-dir "${EVAL_ROOT}/seed_${seed}/image_check"
done
report_args=()
if [[ "${#seeds[@]}" == 1 ]]; then report_args+=(--single_seed_reference); fi
run python tools/report_e7a_control.py --eval_root "${EVAL_ROOT}" --output_dir "${EVAL_ROOT}/report" \
  --experiments "${EXPERIMENTS}" --generation_seeds "${GENERATION_SEEDS}" \
  --comparisons e9_a:e5,e9_b:e5,e9_b:e9_a,e9_a_off:e5,e9_b_off:e5 "${report_args[@]}"
