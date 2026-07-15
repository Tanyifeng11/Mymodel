#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/share/home/u2515283058/Mymodel}"
DATASETS_ROOT="${DATASETS_ROOT:-/share/home/u2515283058/datasets}"
BF_VAL_ROOT="${BF_VAL_ROOT:-${DATASETS_ROOT}/BF/validation}"

DATASET_JSON="${DATASET_JSON:-${PROJECT_ROOT}/data/bf_validation.json}"
TRAIN_JSON="${TRAIN_JSON:-${PROJECT_ROOT}/data/train_bf_texture.json}"
SPLIT_PATH="${SPLIT_PATH:-${PROJECT_ROOT}/eval/benchmarks/bf_validation_dev500_split.json}"

OUTPUT_BASE="${OUTPUT_BASE:-${PROJECT_ROOT}/output}"
TEXTURE_ADAPTER_CKPT="${TEXTURE_ADAPTER_CKPT:-${OUTPUT_BASE}/texture_adapter_bf_e20/checkpoint-final/texture_adapter.bin}"
E2A_CKPT="${E2A_CKPT:-${OUTPUT_BASE}/phase1_e2a_region_e5/checkpoint-final/joint_model.pt}"
E2B_COLOR_SAFE_CKPT="${E2B_COLOR_SAFE_CKPT:-${OUTPUT_BASE}/phase1_e2b_color_safe_gate_e3/checkpoint-final/joint_model.pt}"
E5_CKPT="${E5_CKPT:-${OUTPUT_BASE}/phase1_e5_tcpm_lite_e12/checkpoint-final/joint_model.pt}"
CLIP_MODEL="${CLIP_MODEL:-${PROJECT_ROOT}/models/clip}"

EVAL_ROOT="${EVAL_ROOT:-${PROJECT_ROOT}/eval_outputs/a0_bf_validation_500}"
REPORT_DIR="${REPORT_DIR:-${EVAL_ROOT}/report}"
NUM_SAMPLES="${NUM_SAMPLES:-500}"
SPLIT_SEED="${SPLIT_SEED:-42}"
GENERATION_SEEDS="${GENERATION_SEEDS:-42,123,2026}"
EVAL_DEVICES="${EVAL_DEVICES:-${EVAL_DEVICE:-cuda:0}}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-2000}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

cd "${PROJECT_ROOT}"

for required_path in \
  "${BF_VAL_ROOT}" \
  "${TRAIN_JSON}" \
  "${TEXTURE_ADAPTER_CKPT}" \
  "${E2A_CKPT}" \
  "${E2B_COLOR_SAFE_CKPT}" \
  "${E5_CKPT}" \
  "${CLIP_MODEL}"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "[ERROR] Required path does not exist: ${required_path}" >&2
    exit 1
  fi
done

python tools/build_bf_test_manifest.py \
  --data_root "${BF_VAL_ROOT}" \
  --dataset_json "${DATASET_JSON}" \
  --split_path "${SPLIT_PATH}" \
  --layout flat \
  --split_count "${NUM_SAMPLES}" \
  --seed "${SPLIT_SEED}" \
  --train_json "${TRAIN_JSON}"

if [[ "${PREPARE_ONLY:-0}" == "1" ]]; then
  echo "[A0] PREPARE_ONLY=1, manifest prepared; generation skipped."
  exit 0
fi

experiments=("e2a_region" "e2b_color_safe_gate" "e5_tcpm_lite")
checkpoints=("${E2A_CKPT}" "${E2B_COLOR_SAFE_CKPT}" "${E5_CKPT}")
IFS=',' read -r -a generation_seed_list <<< "${GENERATION_SEEDS}"
IFS=',' read -r -a eval_device_list <<< "${EVAL_DEVICES}"

devices=()
declare -A seen_devices=()
for device in "${eval_device_list[@]}"; do
  device="${device//[[:space:]]/}"
  if [[ -z "${device}" ]]; then
    continue
  fi
  if [[ -n "${seen_devices[$device]:-}" ]]; then
    echo "[ERROR] Duplicate device in EVAL_DEVICES: ${device}" >&2
    exit 1
  fi
  seen_devices["${device}"]=1
  devices+=("${device}")
done
if [[ "${#devices[@]}" -eq 0 ]]; then
  echo "[ERROR] EVAL_DEVICES does not contain a usable device." >&2
  exit 1
fi

job_experiments=()
job_checkpoints=()
job_seeds=()
for generation_seed in "${generation_seed_list[@]}"; do
  generation_seed="${generation_seed//[[:space:]]/}"
  if [[ -z "${generation_seed}" ]]; then
    continue
  fi
  for index in "${!experiments[@]}"; do
    job_experiments+=("${experiments[$index]}")
    job_checkpoints+=("${checkpoints[$index]}")
    job_seeds+=("${generation_seed}")
  done
done
if [[ "${#job_experiments[@]}" -eq 0 ]]; then
  echo "[ERROR] GENERATION_SEEDS does not contain a usable seed." >&2
  exit 1
fi

mkdir -p "${EVAL_ROOT}/logs"

run_a0_job() {
  local device="$1"
  local experiment="$2"
  local checkpoint="$3"
  local generation_seed="$4"
  local seed_dir="${EVAL_ROOT}/seed_${generation_seed}"

  mkdir -p "${seed_dir}"
  echo "[A0] start experiment=${experiment} seed=${generation_seed} device=${device}"

  if ! python tools/run_fixed_benchmark.py \
    --dataset_json "${DATASET_JSON}" \
    --data_root "${BF_VAL_ROOT}" \
    --split_path "${SPLIT_PATH}" \
    --num_samples "${NUM_SAMPLES}" \
    --seed "${SPLIT_SEED}" \
    --generation_seed "${generation_seed}" \
    --sample_id_start 0 \
    --sample_id_end "${NUM_SAMPLES}" \
    --resume_generation 1 \
    --skip_existing 1 \
    --overwrite 0 \
    --gam_ckpt "${checkpoint}" \
    --texture_ckpt "${TEXTURE_ADAPTER_CKPT}" \
    --device "${device}" \
    --modes token \
    --texture_preprocess_mode plain_resize \
    --clip_model_path "${CLIP_MODEL}" \
    --output_dir "${seed_dir}" \
    --run_name "${experiment}" \
    --evaluation_protocol original_image_size \
    --mask_policy sketch_only \
    --fail_on_empty_masks 1 \
    --debug_save_masks 10 \
    --compute_fid 1 \
    --compute_kid 1; then
    return 1
  fi

  if ! python tools/validate_benchmark_outputs.py \
    --experiments_dir "${seed_dir}" \
    --experiment_names "${experiment}" \
    --expected_count "${NUM_SAMPLES}"; then
    return 1
  fi

  echo "[A0] done experiment=${experiment} seed=${generation_seed} device=${device}"
}

run_a0_worker() {
  local worker_index="$1"
  local device="${devices[$worker_index]}"
  local job_index
  local log_path

  for ((job_index=worker_index; job_index<${#job_experiments[@]}; job_index+=${#devices[@]})); do
    log_path="${EVAL_ROOT}/logs/${job_experiments[$job_index]}_seed_${job_seeds[$job_index]}.log"
    if ! run_a0_job \
      "${device}" \
      "${job_experiments[$job_index]}" \
      "${job_checkpoints[$job_index]}" \
      "${job_seeds[$job_index]}" \
      >"${log_path}" 2>&1; then
      echo "[ERROR] A0 job failed on ${device}. See ${log_path}" >&2
      return 1
    fi
  done
}

echo "[A0] devices=${devices[*]} jobs=${#job_experiments[@]}"
worker_pids=()
worker_devices=()
worker_count="${#devices[@]}"
if (( worker_count > ${#job_experiments[@]} )); then
  worker_count="${#job_experiments[@]}"
fi

for ((worker_index=0; worker_index<worker_count; worker_index++)); do
  run_a0_worker "${worker_index}" &
  worker_pids+=("$!")
  worker_devices+=("${devices[$worker_index]}")
done

worker_failed=0
for index in "${!worker_pids[@]}"; do
  if ! wait "${worker_pids[$index]}"; then
    echo "[ERROR] A0 worker failed: ${worker_devices[$index]}" >&2
    worker_failed=1
  fi
done
if [[ "${worker_failed}" -ne 0 ]]; then
  echo "[ERROR] A0 generation failed; aggregation was skipped." >&2
  exit 1
fi

python tools/aggregate_a0_multiseed.py \
  --eval_root "${EVAL_ROOT}" \
  --experiments "e2a_region,e2b_color_safe_gate,e5_tcpm_lite" \
  --generation_seeds "${GENERATION_SEEDS}" \
  --reference e2b_color_safe_gate \
  --output_dir "${REPORT_DIR}" \
  --bootstrap_samples "${BOOTSTRAP_SAMPLES}"

echo "[A0] completed. Report: ${REPORT_DIR}/a0_multiseed_summary.md"
