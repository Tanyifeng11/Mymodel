#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/share/home/u2515283058/Mymodel}"
DATASETS_ROOT="${DATASETS_ROOT:-/share/home/u2515283058/datasets}"
DATA_ROOT_PATH="${DATA_ROOT_PATH:-${DATASETS_ROOT}/BF/training}"
DATASET_JSON="${DATASET_JSON:-${PROJECT_ROOT}/data/train_bf_texture.json}"
SPLIT_PATH="${SPLIT_PATH:-${PROJECT_ROOT}/eval/benchmarks/phase1_bf_val_split.json}"
VAE_MODEL_PATH="${VAE_MODEL_PATH:-${PROJECT_ROOT}/models/stable-diffusion-v1-5/vae}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-${PROJECT_ROOT}/eval_outputs/phase1_full_500/e6b_unet_late_distribution}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/eval_outputs/phase2_diagnostics}"

# 仅使用调度系统分配给作业的可见设备，不绑定物理 GPU 编号。
DEVICE="${DEVICE:-cuda}"
NUM_SAMPLES="${NUM_SAMPLES:-500}"
SEED="${SEED:-42}"
CLEAN_FID="${CLEAN_FID:-0}"
TASKS="${TASKS:-vae,audit,background}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_DIR}"

run_task() {
  case "$1" in
    vae)
      python tools/eval_vae_reconstruction.py \
        --dataset_json "${DATASET_JSON}" \
        --data_root "${DATA_ROOT_PATH}" \
        --split_path "${SPLIT_PATH}" \
        --vae_model_path "${VAE_MODEL_PATH}" \
        --output_dir "${OUTPUT_DIR}/vae_reconstruction" \
        --num_samples "${NUM_SAMPLES}" \
        --seed "${SEED}" \
        --device "${DEVICE}" \
        --clean_fid "${CLEAN_FID}"
      ;;
    audit)
      python tools/audit_multimodal_dataset.py \
        --dataset_json "${DATASET_JSON}" \
        --data_root "${DATA_ROOT_PATH}" \
        --split_path "${SPLIT_PATH}" \
        --output_dir "${OUTPUT_DIR}/dataset_audit" \
        --num_samples "${NUM_SAMPLES}" \
        --seed "${SEED}"
      ;;
    background)
      for image_dir in "${EXPERIMENT_DIR}/real" "${EXPERIMENT_DIR}/generated"; do
        if [[ ! -d "${image_dir}" ]]; then
          echo "[ERROR] Background FID image directory does not exist: ${image_dir}" >&2
          exit 1
        fi
      done
      python tools/eval_background_fid.py \
        --experiment_dir "${EXPERIMENT_DIR}" \
        --output_dir "${OUTPUT_DIR}/background_fid" \
        --num_samples "${NUM_SAMPLES}" \
        --seed "${SEED}" \
        --device "${DEVICE}" \
        --clean_fid "${CLEAN_FID}"
      ;;
    split)
      python tools/build_disjoint_dataset_split.py \
        --master_json "${DATASET_JSON}" \
        --output_train_json "${OUTPUT_DIR}/disjoint_split/train.json" \
        --output_val_json "${OUTPUT_DIR}/disjoint_split/validation.json" \
        --output_split_json "${OUTPUT_DIR}/disjoint_split/validation_benchmark.json" \
        --report_json "${OUTPUT_DIR}/disjoint_split/report.json" \
        --existing_benchmark_split "${SPLIT_PATH}" \
        --seed "${SEED}"
      ;;
    *)
      echo "[ERROR] Unknown task: $1 (valid: vae,audit,background,split)" >&2
      exit 1
      ;;
  esac
}

IFS=',' read -r -a selected_tasks <<< "${TASKS}"
for task in "${selected_tasks[@]}"; do
  echo "[phase2] start ${task} on DEVICE=${DEVICE}"
  run_task "${task}"
done

echo "[phase2] done: ${OUTPUT_DIR}"
