#!/usr/bin/env bash
# CTD Stage A 正式评测：S1 原始集护栏、S2 训练同族、S3 保留族。
# 默认比较 E5 与 CTD；A0 对照应通过 CONTROL_NAME=a0 与 CONTROL_CKPT=<A0 checkpoint>
# 覆盖，确保“对照组”与 CTD 仅相差条件端色度扰动。
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/share/home/u2515283058/Mymodel}"
DATASETS_ROOT="${DATASETS_ROOT:-/share/home/u2515283058/datasets}"
DATA_ROOT_PATH="${DATA_ROOT_PATH:-${DATASETS_ROOT}/BF/training}"
OUTPUT_BASE="${OUTPUT_BASE:-${PROJECT_ROOT}/output}"
TEXTURE_CKPT="${TEXTURE_CKPT:-${OUTPUT_BASE}/texture_adapter_bf_e20/checkpoint-final/texture_adapter.bin}"
E5_CKPT="${E5_CKPT:-${OUTPUT_BASE}/phase1_e5_tcpm_lite_e3/checkpoint-final/joint_model.pt}"
CTD_CKPT="${CTD_CKPT:-${OUTPUT_BASE}/phase1_ctd_stage_a_gamut_p030_full/checkpoint-final/joint_model.pt}"
CONTROL_NAME="${CONTROL_NAME:-e5}"
CONTROL_CKPT="${CONTROL_CKPT:-${E5_CKPT}}"
CLIP_MODEL="${CLIP_MODEL:-${PROJECT_ROOT}/models/clip}"
PER_SAMPLE_CSV="${PER_SAMPLE_CSV:-${PROJECT_ROOT}/eval_outputs/phase1_e7a_500/report_e7a/metrics_per_sample.csv}"

EVAL_ROOT="${CTD_EVAL_ROOT:-${PROJECT_ROOT}/output_eval/ctd_stage_a_gamut_p030_full_eval}"
SET_ROOT="${CTD_EVAL_SET_ROOT:-${EVAL_ROOT}/sets}"
SPLIT_ROOT="${CTD_EVAL_SPLIT_ROOT:-${EVAL_ROOT}/splits}"
REPORT_ROOT="${CTD_EVAL_REPORT_ROOT:-${EVAL_ROOT}/report}"
CTD_EVAL_MODE="${CTD_EVAL_MODE:-full}" # full: S1/S2/S3 x 3 seeds; screen: S2/S3 x 1 seed; screen_all: S1/S2/S3 x 1 seed
SINGLE_SEED_REFERENCE_REPORT="${SINGLE_SEED_REFERENCE_REPORT:-0}"
EVAL_SEED="${EVAL_SEED:-42}"
DEVICE="${DEVICE:-cuda:0}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-2000}"
KID_SUBSETS="${KID_SUBSETS:-50}"
KID_SUBSET_SIZE="${KID_SUBSET_SIZE:-100}"
CTD_TARGET_STRATEGY="${CTD_TARGET_STRATEGY:-gamut_aware}"
CTD_EVAL_PAIR_MODE="${CTD_EVAL_PAIR_MODE:-independent}"
CTD_SEED="${CTD_SEED:-42}"

case "${CTD_EVAL_MODE}" in
  full)
    EVAL_SETS="S1,S2,S3"
    GENERATION_SEEDS="${GENERATION_SEEDS:-42,123,2026}"
    ;;
  screen)
    EVAL_SETS="S2,S3"
    GENERATION_SEEDS="${GENERATION_SEEDS:-42}"
    ;;
  screen_all)
    EVAL_SETS="S1,S2,S3"
    GENERATION_SEEDS="${GENERATION_SEEDS:-42}"
    ;;
  *)
    echo "[ERROR] CTD_EVAL_MODE must be full, screen, or screen_all, got: ${CTD_EVAL_MODE}" >&2
    exit 1
    ;;
esac

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

[[ "${CONTROL_NAME}" =~ ^[A-Za-z0-9_]+$ ]] || {
  echo "[ERROR] CONTROL_NAME must contain only letters, digits, and underscores: ${CONTROL_NAME}" >&2
  exit 1
}
[[ "${CONTROL_NAME}" != "ctd" ]] || {
  echo "[ERROR] CONTROL_NAME must not be ctd." >&2
  exit 1
}

for path in "${DATA_ROOT_PATH}" "${TEXTURE_CKPT}" "${CONTROL_CKPT}" "${CTD_CKPT}" "${CLIP_MODEL}" "${PER_SAMPLE_CSV}"; do
  [[ -e "${path}" ]] || { echo "[ERROR] missing required path: ${path}" >&2; exit 1; }
done

mkdir -p "${SET_ROOT}" "${SPLIT_ROOT}" "${REPORT_ROOT}"

echo "[CTD eval] control=${CONTROL_NAME}, mode=${CTD_EVAL_MODE}, sets=${EVAL_SETS}, generation_seeds=${GENERATION_SEEDS}"

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

  echo "========== [2/3] ${set_name}: ${sample_count} samples, ${CONTROL_NAME} vs CTD =========="
  IFS=',' read -r -a seed_list <<< "${GENERATION_SEEDS}"
  for generation_seed in "${seed_list[@]}"; do
    generation_seed="${generation_seed//[[:space:]]/}"
    seed_dir="${EVAL_ROOT}/${set_name}/seed_${generation_seed}"
    mkdir -p "${seed_dir}"
    for experiment in "${CONTROL_NAME}" ctd; do
      if [[ "${experiment}" == "${CONTROL_NAME}" ]]; then
        gam_ckpt="${CONTROL_CKPT}"
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
      --experiments_dir "${seed_dir}" --experiment_names "${CONTROL_NAME},ctd" \
      --expected_count "${sample_count}"
  done

  if [[ "${CTD_EVAL_MODE}" == "full" ]]; then
    python "${PROJECT_ROOT}/tools/report_e7a_control.py" \
      --eval_root "${EVAL_ROOT}/${set_name}" \
      --output_dir "${REPORT_ROOT}/${set_name}" \
      --experiments "${CONTROL_NAME},ctd" --comparisons "ctd:${CONTROL_NAME}" \
      --generation_seeds "${GENERATION_SEEDS}" \
      --bootstrap_samples "${BOOTSTRAP_SAMPLES}"
  elif [[ "${SINGLE_SEED_REFERENCE_REPORT}" == "1" ]]; then
    python "${PROJECT_ROOT}/tools/report_e7a_control.py" \
      --eval_root "${EVAL_ROOT}/${set_name}" \
      --output_dir "${REPORT_ROOT}/${set_name}" \
      --experiments "${CONTROL_NAME},ctd" --comparisons "ctd:${CONTROL_NAME}" \
      --generation_seeds "${GENERATION_SEEDS}" \
      --bootstrap_samples "${BOOTSTRAP_SAMPLES}" \
      --single_seed_reference
  fi
}

for eval_set in ${EVAL_SETS//,/ }; do
  case "${eval_set}" in
    S1) run_set S1 "${SET_ROOT}/data/ctd_eval_setS1.json" ;;
    S2) run_set S2 "${SET_ROOT}/data/ctd_eval_setA.json" ;;
    S3) run_set S3 "${SET_ROOT}/data/ctd_eval_setB.json" ;;
  esac
done

echo "========== [3/3] 完成 =========="
if [[ "${CTD_EVAL_MODE}" == "full" ]]; then
  echo "[done] S1 护栏报告: ${REPORT_ROOT}/S1"
  echo "[done] S2 主评测报告: ${REPORT_ROOT}/S2"
  echo "[done] S3 泛化报告: ${REPORT_ROOT}/S3"
else
  if [[ "${SINGLE_SEED_REFERENCE_REPORT}" == "1" ]]; then
    echo "[done] single-seed reference reports: ${REPORT_ROOT} (not a formal multiseed result)."
  else
    echo "[done] screening metrics: ${EVAL_ROOT}; no multiseed bootstrap report was generated."
  fi
  echo "[next] CTD_EVAL_MODE=full sbatch submit/eval_ctd_stage_a.sh will reuse these generated images."
fi
