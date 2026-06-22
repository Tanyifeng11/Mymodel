#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/share/home/u2515283058/Mymodel}"
cd "${PROJECT_ROOT}"

module load anaconda3/4.12.0 || true
if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  # 如果项目实际环境名不是 mymodel，请修改 CONDA_ENV。
  CONDA_ENV="${CONDA_ENV:-mymodel}"
  conda activate "${CONDA_ENV}" || conda activate base
else
  echo "[WARN] conda command not found; using the current Python environment."
fi

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

DATASETS_ROOT="${DATASETS_ROOT:-/share/home/u2515283058/datasets}"
BF_ROOT="${BF_ROOT:-${DATASETS_ROOT}/BF}"
DATA_ROOT_PATH="${DATA_ROOT_PATH:-${BF_ROOT}/training}"
DATASET_JSON="${DATASET_JSON:-${PROJECT_ROOT}/data/train_bf_texture.json}"
SPLIT_PATH="${SPLIT_PATH:-${PROJECT_ROOT}/eval/benchmarks/phase1_bf_val_split.json}"

OUTPUT_BASE="${OUTPUT_BASE:-${PROJECT_ROOT}/output}"
TEXTURE_ADAPTER_CKPT="${TEXTURE_ADAPTER_CKPT:-${OUTPUT_BASE}/texture_adapter_bf_e20/checkpoint-final/texture_adapter.bin}"
E0_CKPT="${E0_CKPT:-${OUTPUT_BASE}/phase1_e0_baseline_e5/checkpoint-28365/joint_model.pt}"
E1_CKPT="${E1_CKPT:-${OUTPUT_BASE}/phase1_e1_grouped_e5/checkpoint-final/joint_model.pt}"
E2A_CKPT="${E2A_CKPT:-${OUTPUT_BASE}/phase1_e2a_region_e5/checkpoint-final/joint_model.pt}"

OLD_EVAL_BASE="${OLD_EVAL_BASE:-${PROJECT_ROOT}/eval_outputs/phase1_full}"
EVAL_BASE="${EVAL_BASE:-${PROJECT_ROOT}/eval_outputs/phase1_full_500}"
RESIZED_BASE="${RESIZED_BASE:-${PROJECT_ROOT}/eval_outputs/phase1_full_500_resized256}"
REPORT_NORMAL="${REPORT_NORMAL:-${EVAL_BASE}/report_normal}"
REPORT_RESIZE256="${REPORT_RESIZE256:-${EVAL_BASE}/report_resize256}"

NUM_SAMPLES="${NUM_SAMPLES:-500}"
EVAL_SEED="${EVAL_SEED:-42}"
EVAL_MODES="${EVAL_MODES:-token}"
TEXTURE_PREPROCESS_MODE="${TEXTURE_PREPROCESS_MODE:-plain_resize}"
CLIP_MODEL="${CLIP_MODEL:-${PROJECT_ROOT}/models/clip}"
E0_DEVICE="${E0_DEVICE:-cuda:0}"
E1_DEVICE="${E1_DEVICE:-cuda:0}"
E2A_DEVICE="${E2A_DEVICE:-cuda:0}"
REPORT_DEVICE="${REPORT_DEVICE:-cuda:0}"
EXPERIMENT_NAMES="e0_baseline,e1_grouped,e2a_region"

find_latest_gam_ckpt() {
  local base_dir="$1"
  if [[ -f "${base_dir}/checkpoint-final/joint_model.pt" ]]; then
    echo "${base_dir}/checkpoint-final/joint_model.pt"
    return
  fi
  find "${base_dir}" -maxdepth 2 -path '*/joint_model.pt' -type f 2>/dev/null | sort -V | tail -1 || true
}

resolve_checkpoint() {
  local configured="$1"
  local fallback_dir="$2"
  if [[ -f "${configured}" ]]; then
    echo "${configured}"
    return
  fi
  find_latest_gam_ckpt "${fallback_dir}"
}

E0_CKPT="$(resolve_checkpoint "${E0_CKPT}" "${OUTPUT_BASE}/phase1_e0_baseline_e5")"
E1_CKPT="$(resolve_checkpoint "${E1_CKPT}" "${OUTPUT_BASE}/phase1_e1_grouped_e5")"
E2A_CKPT="$(resolve_checkpoint "${E2A_CKPT}" "${OUTPUT_BASE}/phase1_e2a_region_e5")"

for required_path in \
  "${DATASET_JSON}" \
  "${DATA_ROOT_PATH}" \
  "${TEXTURE_ADAPTER_CKPT}" \
  "${E0_CKPT}" \
  "${E1_CKPT}" \
  "${E2A_CKPT}"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "[ERROR] Required path does not exist: ${required_path}" >&2
    exit 1
  fi
done

mkdir -p "${EVAL_BASE}" "${RESIZED_BASE}" "${REPORT_NORMAL}" "${REPORT_RESIZE256}"

run_normal_benchmark() {
  local run_name="$1"
  local gam_ckpt="$2"
  local device="$3"
  local reuse_dir="${OLD_EVAL_BASE}/${run_name}"

  python tools/run_fixed_benchmark.py \
    --dataset_json "${DATASET_JSON}" \
    --data_root "${DATA_ROOT_PATH}" \
    --split_path "${SPLIT_PATH}" \
    --num_samples "${NUM_SAMPLES}" \
    --seed "${EVAL_SEED}" \
    --sample_id_start 0 \
    --sample_id_end "${NUM_SAMPLES}" \
    --resume_generation 1 \
    --skip_existing 1 \
    --overwrite 0 \
    --reuse_from_dir "${reuse_dir}" \
    --gam_ckpt "${gam_ckpt}" \
    --texture_ckpt "${TEXTURE_ADAPTER_CKPT}" \
    --device "${device}" \
    --modes "${EVAL_MODES}" \
    --texture_preprocess_mode "${TEXTURE_PREPROCESS_MODE}" \
    --clip_model_path "${CLIP_MODEL}" \
    --output_dir "${EVAL_BASE}" \
    --run_name "${run_name}" \
    --evaluation_protocol original_image_size
}

run_resize256_benchmark() {
  local run_name="$1"
  local gam_ckpt="$2"
  local device="$3"

  python tools/run_fixed_benchmark.py \
    --dataset_json "${DATASET_JSON}" \
    --data_root "${DATA_ROOT_PATH}" \
    --split_path "${SPLIT_PATH}" \
    --num_samples "${NUM_SAMPLES}" \
    --seed "${EVAL_SEED}" \
    --sample_id_start 0 \
    --sample_id_end "${NUM_SAMPLES}" \
    --resume_generation 1 \
    --skip_existing 1 \
    --overwrite 0 \
    --gam_ckpt "${gam_ckpt}" \
    --texture_ckpt "${TEXTURE_ADAPTER_CKPT}" \
    --device "${device}" \
    --modes "${EVAL_MODES}" \
    --texture_preprocess_mode "${TEXTURE_PREPROCESS_MODE}" \
    --clip_model_path "${CLIP_MODEL}" \
    --output_dir "${EVAL_BASE}" \
    --run_name "${run_name}" \
    --metrics_output_dir "${RESIZED_BASE}/${run_name}" \
    --evaluation_resize 256 \
    --evaluation_protocol resize_generated_real_to_256 \
    --metrics_only
}

echo "============================================"
echo "500-sample E0/E1/E2A evaluation"
echo "DATASET_JSON=${DATASET_JSON}"
echo "SPLIT_PATH=${SPLIT_PATH}"
echo "OLD_EVAL_BASE=${OLD_EVAL_BASE}"
echo "EVAL_BASE=${EVAL_BASE}"
echo "RESIZED_BASE=${RESIZED_BASE}"
echo "E0_CKPT=${E0_CKPT}"
echo "E1_CKPT=${E1_CKPT}"
echo "E2A_CKPT=${E2A_CKPT}"
echo "============================================"

echo "=== 1/7 E0 resume generation and normal evaluation ==="
run_normal_benchmark "e0_baseline" "${E0_CKPT}" "${E0_DEVICE}"

echo "=== 2/7 E1 resume generation and normal evaluation ==="
run_normal_benchmark "e1_grouped" "${E1_CKPT}" "${E1_DEVICE}"

echo "=== 3/7 E2A resume generation and normal evaluation ==="
run_normal_benchmark "e2a_region" "${E2A_CKPT}" "${E2A_DEVICE}"

echo "=== 4/7 Validate normal evaluation inputs ==="
python tools/validate_benchmark_outputs.py \
  --experiments_dir "${EVAL_BASE}" \
  --experiment_names "${EXPERIMENT_NAMES}" \
  --expected_count "${NUM_SAMPLES}"

echo "=== 5/7 Build normal evaluation report ==="
python -m eval.ablation_report \
  --experiments_dir "${EVAL_BASE}" \
  --output_dir "${REPORT_NORMAL}" \
  --experiment_names "${EXPERIMENT_NAMES}" \
  --device "${REPORT_DEVICE}" \
  --evaluation_protocol original_image_size \
  --num_samples "${NUM_SAMPLES}" \
  --resume_generation 1 \
  --existing_samples_skipped 1

echo "=== 6/7 Run resize256 evaluation ==="
run_resize256_benchmark "e0_baseline" "${E0_CKPT}" "${E0_DEVICE}"
run_resize256_benchmark "e1_grouped" "${E1_CKPT}" "${E1_DEVICE}"
run_resize256_benchmark "e2a_region" "${E2A_CKPT}" "${E2A_DEVICE}"

python tools/validate_benchmark_outputs.py \
  --experiments_dir "${RESIZED_BASE}" \
  --experiment_names "${EXPERIMENT_NAMES}" \
  --expected_count "${NUM_SAMPLES}" \
  --required_size 256

echo "=== 7/7 Build resize256 evaluation report ==="
python -m eval.ablation_report \
  --experiments_dir "${RESIZED_BASE}" \
  --output_dir "${REPORT_RESIZE256}" \
  --experiment_names "${EXPERIMENT_NAMES}" \
  --device "${REPORT_DEVICE}" \
  --evaluation_protocol resize_generated_real_to_256 \
  --num_samples "${NUM_SAMPLES}" \
  --resume_generation 1 \
  --existing_samples_skipped 1

echo "============================================"
echo "Evaluation completed."
echo "Normal report:    ${REPORT_NORMAL}"
echo "Resize256 report: ${REPORT_RESIZE256}"
echo "Generated images: ${EVAL_BASE}/{e0_baseline,e1_grouped,e2a_region}"
echo "Resized images:   ${RESIZED_BASE}/{e0_baseline,e1_grouped,e2a_region}"
echo "============================================"
