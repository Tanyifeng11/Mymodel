#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/share/home/u2515283058/Mymodel}"
BF_TEST_ROOT="${BF_TEST_ROOT:-/share/home/u2515283058/datasets/BF/test}"
SPLIT_PATH="${SPLIT_PATH:-${PROJECT_ROOT}/eval/benchmarks/bf_test_no_bag_full_split.json}"
OUTPUT_BASE="${OUTPUT_BASE:-${PROJECT_ROOT}/output}"
E5_CKPT="${E5_CKPT:-${OUTPUT_BASE}/phase1_e5_tcpm_lite_e3/checkpoint-final/joint_model.pt}"
TEXTURE_CKPT="${TEXTURE_CKPT:-${OUTPUT_BASE}/texture_adapter_bf_e20/checkpoint-final/texture_adapter.bin}"
EVAL_BASE="${EVAL_BASE:-${PROJECT_ROOT}/eval_outputs/e5_bf_full_test}"
RUN_NAME="${RUN_NAME:-e5_tcpm_lite_e3_bf_test}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_ID="${SHARD_ID:-${SLURM_ARRAY_TASK_ID:-}}"
GENERATION_SEED="${GENERATION_SEED:-42}"
DEVICE="${DEVICE:-cuda}"

if [[ -z "${SHARD_ID}" ]]; then
  echo "[ERROR] Set SHARD_ID or submit as a Slurm array job." >&2
  exit 1
fi

for path in "${BF_TEST_ROOT}" "${SPLIT_PATH}" "${E5_CKPT}" "${TEXTURE_CKPT}"; do
  if [[ ! -e "${path}" ]]; then
    echo "[ERROR] Required path does not exist: ${path}" >&2
    exit 1
  fi
done

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

python tools/generate_bf_test_shard.py \
  --project_root "${PROJECT_ROOT}" \
  --data_root "${BF_TEST_ROOT}" \
  --split_path "${SPLIT_PATH}" \
  --output_dir "${EVAL_BASE}" \
  --run_name "${RUN_NAME}" \
  --gam_ckpt "${E5_CKPT}" \
  --texture_ckpt "${TEXTURE_CKPT}" \
  --num_shards "${NUM_SHARDS}" \
  --shard_id "${SHARD_ID}" \
  --generation_seed "${GENERATION_SEED}" \
  --device "${DEVICE}"
