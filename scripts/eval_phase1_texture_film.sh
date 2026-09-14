#!/usr/bin/env bash
set -euo pipefail

# 先固定已核对文本与训练重叠的清单；本脚本不自动生成评测样本。
: "${DATASET_JSON:?请设置已清洗的验证或测试 DATASET_JSON}"
: "${SPLIT_PATH:?请设置与 DATASET_JSON 对应的固定 SPLIT_PATH}"
: "${FILM_CKPT:?请设置本次训练得到的 FiLM joint_model.pt}"
PROJECT_ROOT="${PROJECT_ROOT:-/share/home/u2515283058/Mymodel}"
DATASETS_ROOT="${DATASETS_ROOT:-/share/home/u2515283058/datasets}"
BF_SPLIT="${BF_SPLIT:-validation}"
case "${BF_SPLIT}" in validation|test) ;; *) echo "BF_SPLIT 必须为 validation 或 test" >&2; exit 1 ;; esac
# 新清单中的相对路径已含 validation/ 或 test/，根目录必须指向 BF。
DATA_ROOT_PATH="${DATA_ROOT_PATH:-${DATASETS_ROOT}/BF}"
NUM_SAMPLES="${NUM_SAMPLES:-200}"
E5_CKPT="${E5_CKPT:-${PROJECT_ROOT}/output/phase1_e5_tcpm_lite_e3/checkpoint-final/joint_model.pt}"
TEXTURE_CKPT="${TEXTURE_CKPT:-${PROJECT_ROOT}/output/texture_adapter_bf_e20/checkpoint-final/texture_adapter.bin}"
CLIP_MODEL="${CLIP_MODEL:-${PROJECT_ROOT}/models/clip}"
EVAL_RUN_ID="${EVAL_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
EVAL_ROOT="${EVAL_ROOT:-${PROJECT_ROOT}/eval_outputs/texture_film_${BF_SPLIT}_${NUM_SAMPLES}/${EVAL_RUN_ID}}"
GENERATION_SEEDS="${GENERATION_SEEDS:-42}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false
cd "${PROJECT_ROOT}"
for path in "${DATASET_JSON}" "${SPLIT_PATH}"; do
  [[ -f "${path}" ]] || { echo "评测清单不存在：${path}" >&2; exit 1; }
done
run() {
  printf '%q ' "$@"
  printf '\n'
  if [[ "${DRY_RUN:-0}" != "1" ]]; then "$@"; fi
}

# 汇总器需要 FID/KID；小样本 FID 仅作参考，不能凭小幅波动判定 FiLM 有效。
common=(
  --dataset_json "${DATASET_JSON}" --data_root "${DATA_ROOT_PATH}" --split_path "${SPLIT_PATH}"
  --num_samples "${NUM_SAMPLES}" --sample_id_start 0 --sample_id_end "${NUM_SAMPLES}" --seed 42
  --texture_ckpt "${TEXTURE_CKPT}" --clip_model_path "${CLIP_MODEL}" --device "${DEVICE:-cuda:0}"
  --modes token --texture_preprocess_mode plain_resize --use_tcpm_lite 1 --use_texture_gate 1
  --layer_group_enabled 1 --use_aa_tcr_fuse 0 --use_text_guided_resampler 0 --use_local_detail_adapter 0
  --mask_policy sketch_only --evaluation_protocol original_image_size
  --compute_fid 1 --compute_kid 1 --kid_subset_size "${NUM_SAMPLES}" --kid_subsets 50
  --write_text_sidecars 1 --resume_generation 1 --skip_existing 1 --overwrite 0
)
IFS=',' read -r -a seeds <<< "${GENERATION_SEEDS}"
for seed in "${seeds[@]}"; do
  for experiment in e5 film_off film_on; do
    film_args=()
    case "${experiment}" in
      e5) ckpt="${E5_CKPT}"; film_args+=(--disable_texture_film) ;;
      film_off) ckpt="${FILM_CKPT}"; film_args+=(--disable_texture_film) ;;
      film_on) ckpt="${FILM_CKPT}" ;;
    esac
    run python tools/run_fixed_benchmark.py "${common[@]}" "${film_args[@]}" \
      --generation_seed "${seed}" --gam_ckpt "${ckpt}" --run_name "${experiment}" \
      --output_dir "${EVAL_ROOT}/seed_${seed}"
  done
  run python tools/validate_benchmark_outputs.py --experiments_dir "${EVAL_ROOT}/seed_${seed}" \
    --experiment_names e5,film_off,film_on --expected_count "${NUM_SAMPLES}"
done
report_args=()
if [[ "${#seeds[@]}" == 1 ]]; then report_args+=(--single_seed_reference); fi
run python tools/report_e7a_control.py --eval_root "${EVAL_ROOT}" --output_dir "${EVAL_ROOT}/report" \
  --experiments e5,film_off,film_on --generation_seeds "${GENERATION_SEEDS}" \
  --comparisons film_off:e5,film_on:e5,film_on:film_off "${report_args[@]}"
