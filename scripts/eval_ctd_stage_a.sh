#!/usr/bin/env bash
# CTD Stage A 正式评测：S1 原始集护栏、S2 训练同族、S3 保留族；每组均比较 E5 与 CTD。
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/share/home/u2515283058/Mymodel}"
DATASETS_ROOT="${DATASETS_ROOT:-/share/home/u2515283058/datasets}"
DATA_ROOT_PATH="${DATA_ROOT_PATH:-${DATASETS_ROOT}/BF/training}"
OUTPUT_BASE="${OUTPUT_BASE:-${PROJECT_ROOT}/output}"
TEXTURE_CKPT="${TEXTURE_CKPT:-${OUTPUT_BASE}/texture_adapter_bf_e20/checkpoint-final/texture_adapter.bin}"
E5_CKPT="${E5_CKPT:-${OUTPUT_BASE}/phase1_e5_tcpm_lite_e3/checkpoint-final/joint_model.pt}"
CTD_CKPT="${CTD_CKPT:-${OUTPUT_BASE}/phase1_ctd_stage_a_gamut_p030_full/checkpoint-final/joint_model.pt}"
CLIP_MODEL="${CLIP_MODEL:-${PROJECT_ROOT}/models/clip}"
PER_SAMPLE_CSV="${PER_SAMPLE_CSV:-${PROJECT_ROOT}/eval_outputs/phase1_e7a_500/report_e7a/metrics_per_sample.csv}"

EVAL_ROOT="${CTD_EVAL_ROOT:-${PROJECT_ROOT}/output_eval/ctd_stage_a_gamut_p030_full_eval}"
SET_ROOT="${CTD_EVAL_SET_ROOT:-${EVAL_ROOT}/sets}"
SPLIT_ROOT="${CTD_EVAL_SPLIT_ROOT:-${EVAL_ROOT}/splits}"
REPORT_ROOT="${CTD_EVAL_REPORT_ROOT:-${EVAL_ROOT}/report}"
GENERATION_SEEDS="${GENERATION_SEEDS:-42,123,2026}"
EVAL_SEED="${EVAL_SEED:-42}"
DEVICE="${DEVICE:-cuda:0}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-2000}"
KID_SUBSETS="${KID_SUBSETS:-50}"
KID_SUBSET_SIZE="${KID_SUBSET_SIZE:-100}"
CTD_TARGET_STRATEGY="${CTD_TARGET_STRATEGY:-gamut_aware}"
CTD_EVAL_PAIR_MODE="${CTD_EVAL_PAIR_MODE:-independent}"
CTD_SEED="${CTD_SEED:-42}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

for path in "${DATA_ROOT_PATH}" "${TEXTURE_CKPT}" "${E5_CKPT}" "${CTD_CKPT}" "${CLIP_MODEL}" "${PER_SAMPLE_CSV}"; do
  [[ -e "${path}" ]] || { echo "[ERROR] missing required path: ${path}" >&2; exit 1; }
done

mkdir -p "${SET_ROOT}" "${SPLIT_ROOT}" "${REPORT_ROOT}"

echo "========== [1/3] 构造 S1/S2/S3 固定评测集 =========="
pushd "${SET_ROOT}" >/dev/null
python "${PROJECT_ROOT}/tools/prepare_ctd_eval_sets.py" \
  --per_sample_csv "${PER_SAMPLE_CSV}" \
  --data_root "${DATA_ROOT_PATH}" \
  --out_dir "${SET_ROOT}/rendered" \
  --ctd_seed "${CTD_SEED}" \
  --ctd_target_strategy "${CTD_TARGET_STRATEGY}" \
  --ctd_eval_pair_mode "${CTD_EVAL_PAIR_MODE}"
popd >/dev/null

dataset_count() {
  python - "$1" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as f:
    print(len(json.load(f)))
PY
}

common_args=(
  --data_root "${DATA_ROOT_PATH}" --seed "${EVAL_SEED}"
  --resume_generation 1 --skip_existing 1 --overwrite 0
  --texture_ckpt "${TEXTURE_CKPT}" --device "${DEVICE}" --modes token
  --texture_preprocess_mode plain_resize --use_texture_gate 1 --use_tcpm_lite 1
  --layer_group_enabled 1 --clip_model_path "${CLIP_MODEL}"
  --evaluation_protocol original_image_size --mask_policy sketch_only
  --compute_kid 1 --kid_subsets "${KID_SUBSETS}" --kid_subset_size "${KID_SUBSET_SIZE}"
  --write_text_sidecars 1
)

run_set() {
  local set_name="$1" dataset_json="$2"
  local sample_count split_path generation_seed seed_dir
  sample_count="$(dataset_count "${dataset_json}")"
  [[ "${sample_count}" -gt 0 ]] || { echo "[ERROR] ${set_name} has no samples" >&2; exit 1; }
  split_path="${SPLIT_ROOT}/${set_name}.json"

  echo "========== [2/3] ${set_name}: ${sample_count} samples, E5 vs CTD =========="
  IFS=',' read -r -a seed_list <<< "${GENERATION_SEEDS}"
  for generation_seed in "${seed_list[@]}"; do
    generation_seed="${generation_seed//[[:space:]]/}"
    seed_dir="${EVAL_ROOT}/${set_name}/seed_${generation_seed}"
    mkdir -p "${seed_dir}"
    for experiment in e5 ctd; do
      if [[ "${experiment}" == "e5" ]]; then
        gam_ckpt="${E5_CKPT}"
      else
        gam_ckpt="${CTD_CKPT}"
      fi
      python "${PROJECT_ROOT}/tools/run_fixed_benchmark.py" \
        "${common_args[@]}" \
        --dataset_json "${dataset_json}" --split_path "${split_path}" \
        --num_samples "${sample_count}" --sample_id_start 0 --sample_id_end "${sample_count}" \
        --generation_seed "${generation_seed}" --gam_ckpt "${gam_ckpt}" \
        --output_dir "${seed_dir}" --run_name "${experiment}"
    done
    python "${PROJECT_ROOT}/tools/validate_benchmark_outputs.py" \
      --experiments_dir "${seed_dir}" --experiment_names e5,ctd \
      --expected_count "${sample_count}"
  done

  python "${PROJECT_ROOT}/tools/report_e7a_control.py" \
    --eval_root "${EVAL_ROOT}/${set_name}" \
    --output_dir "${REPORT_ROOT}/${set_name}" \
    --experiments e5,ctd --comparisons ctd:e5 \
    --generation_seeds "${GENERATION_SEEDS}" \
    --bootstrap_samples "${BOOTSTRAP_SAMPLES}"
}

run_set S1 "${SET_ROOT}/data/ctd_eval_setS1.json"
run_set S2 "${SET_ROOT}/data/ctd_eval_setA.json"
run_set S3 "${SET_ROOT}/data/ctd_eval_setB.json"

echo "========== [3/3] 完成 =========="
echo "[done] S1 护栏报告: ${REPORT_ROOT}/S1"
echo "[done] S2 主评测报告: ${REPORT_ROOT}/S2"
echo "[done] S3 泛化报告: ${REPORT_ROOT}/S3"
