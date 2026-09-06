#!/usr/bin/env bash
set -euo pipefail

# 同一官方数据划分、同一 100 张样本、同一生成 seed，比较 A/B/C。
PROJECT_ROOT="${PROJECT_ROOT:-/share/home/u2515283058/Mymodel}"
DATASETS_ROOT="${DATASETS_ROOT:-/share/home/u2515283058/datasets}"
BF_SPLIT="${BF_SPLIT:-validation}"
case "${BF_SPLIT}" in validation|test) ;; *) echo "BF_SPLIT 必须为 validation 或 test" >&2; exit 1 ;; esac
NUM_SAMPLES="${NUM_SAMPLES:-100}"
DATA_ROOT_PATH="${DATA_ROOT_PATH:-${DATASETS_ROOT}/BF/${BF_SPLIT}}"
DATASET_JSON="${DATASET_JSON:-${PROJECT_ROOT}/data/bf_${BF_SPLIT}_resampler.json}"
SPLIT_PATH="${SPLIT_PATH:-${PROJECT_ROOT}/eval/benchmarks/bf_${BF_SPLIT}_resampler_${NUM_SAMPLES}.json}"
OUTPUT_BASE="${OUTPUT_BASE:-${PROJECT_ROOT}/output}"
E5_CKPT="${E5_CKPT:-${OUTPUT_BASE}/phase1_e5_tcpm_lite_e3/checkpoint-final/joint_model.pt}"
VISUAL_CKPT="${VISUAL_CKPT:-${OUTPUT_BASE}/phase1_resampler_visual/checkpoint-final/joint_model.pt}"
TEXT_CKPT="${TEXT_CKPT:-${OUTPUT_BASE}/phase1_resampler_text/checkpoint-final/joint_model.pt}"
TEXTURE_CKPT="${TEXTURE_CKPT:-${OUTPUT_BASE}/texture_adapter_bf_e20/checkpoint-final/texture_adapter.bin}"
CLIP_MODEL="${CLIP_MODEL:-${PROJECT_ROOT}/models/clip}"
# 默认每次评测单独落盘，避免 checkpoint 更新后误复用旧生成图。
EVAL_RUN_ID="${EVAL_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
EVAL_ROOT="${EVAL_ROOT:-${PROJECT_ROOT}/eval_outputs/resampler_${BF_SPLIT}_${NUM_SAMPLES}/${EVAL_RUN_ID}}"
GENERATION_SEEDS="${GENERATION_SEEDS:-42}"
EXPERIMENTS="${EXPERIMENTS:-e5,resampler_visual,resampler_text}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false
cd "${PROJECT_ROOT}"
run() {
  printf '%q ' "$@"
  printf '\n'
  if [[ "${DRY_RUN:-0}" != "1" ]]; then "$@"; fi
}
if [[ ! -f "${DATASET_JSON}" ]]; then
  run python tools/build_bf_test_manifest.py --data_root "${DATA_ROOT_PATH}" \
    --dataset_json "${DATASET_JSON}" --split_path "${SPLIT_PATH}" --split_count "${NUM_SAMPLES}" \
    --seed 42 --train_json "${TRAIN_JSON:-${PROJECT_ROOT}/data/train_bf_texture.json}"
fi

common=(
  --dataset_json "${DATASET_JSON}" --data_root "${DATA_ROOT_PATH}" --split_path "${SPLIT_PATH}"
  --num_samples "${NUM_SAMPLES}" --sample_id_start 0 --sample_id_end "${NUM_SAMPLES}" --seed 42
  --texture_ckpt "${TEXTURE_CKPT}" --clip_model_path "${CLIP_MODEL}" --device "${DEVICE:-cuda:0}"
  --modes token --texture_preprocess_mode plain_resize --use_tcpm_lite 1 --use_texture_gate 1
  --layer_group_enabled 1 --use_aa_tcr_fuse 0 --mask_policy sketch_only
  --evaluation_protocol original_image_size --compute_kid 1 --kid_subset_size "${NUM_SAMPLES}"
  --kid_subsets 50 --write_text_sidecars 1 --resume_generation 1 --skip_existing 1 --overwrite 0
)
IFS=',' read -r -a seeds <<< "${GENERATION_SEEDS}"
IFS=',' read -r -a experiments <<< "${EXPERIMENTS}"
for seed in "${seeds[@]}"; do
  for experiment in "${experiments[@]}"; do
    case "${experiment}" in
      e5) ckpt="${E5_CKPT}"; guidance=0 ;;
      resampler_visual) ckpt="${VISUAL_CKPT}"; guidance=0 ;;
      resampler_text) ckpt="${TEXT_CKPT}"; guidance=1 ;;
      *) echo "未知实验：${experiment}" >&2; exit 1 ;;
    esac
    run python tools/run_fixed_benchmark.py "${common[@]}" \
      --generation_seed "${seed}" --gam_ckpt "${ckpt}" --run_name "${experiment}" \
      --use_text_guided_resampler "${guidance}" --output_dir "${EVAL_ROOT}/seed_${seed}"
  done
  run python tools/validate_benchmark_outputs.py --experiments_dir "${EVAL_ROOT}/seed_${seed}" \
    --experiment_names "${EXPERIMENTS}" --expected_count "${NUM_SAMPLES}"
done
if [[ "${EXPERIMENTS}" == "e5,resampler_visual,resampler_text" ]]; then
  report_args=()
  if [[ "${#seeds[@]}" == 1 ]]; then report_args+=(--single_seed_reference); fi
  run python tools/report_e7a_control.py --eval_root "${EVAL_ROOT}" --output_dir "${EVAL_ROOT}/report" \
    --experiments "${EXPERIMENTS}" --generation_seeds "${GENERATION_SEEDS}" \
    --comparisons resampler_visual:e5,resampler_text:e5,resampler_text:resampler_visual "${report_args[@]}"
fi
