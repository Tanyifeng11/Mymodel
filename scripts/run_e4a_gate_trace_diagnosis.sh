#!/usr/bin/env bash
set -euo pipefail

# Only runs E4a inference diagnostics for runtime gate traces.
# It does not train any model and does not modify checkpoints.

PROJECT_ROOT="${PROJECT_ROOT:-/share/home/u2515283058/Mymodel}"
DATASETS_ROOT="${DATASETS_ROOT:-/share/home/u2515283058/datasets}"
BF_TRAIN_ROOT="${BF_TRAIN_ROOT:-${DATASETS_ROOT}/BF/training}"

DATASET_JSON="${DATASET_JSON:-${PROJECT_ROOT}/data/train_bf_texture.json}"
SPLIT_PATH="${SPLIT_PATH:-${PROJECT_ROOT}/eval/benchmarks/phase1_bf_val_split.json}"
DATA_ROOT="${DATA_ROOT:-${BF_TRAIN_ROOT}}"

E4A_CKPT="${E4A_CKPT:-${PROJECT_ROOT}/output/phase1_e4a_balanced_gate_e3/checkpoint-final/joint_model.pt}"
TEXTURE_CKPT="${TEXTURE_CKPT:-${PROJECT_ROOT}/output/texture_adapter_bf_e20/checkpoint-final/texture_adapter.bin}"

TRACE_OUTPUT_DIR="${TRACE_OUTPUT_DIR:-${PROJECT_ROOT}/eval_outputs/phase1_full_500_gate_trace}"
DIAG_OUT_DIR="${DIAG_OUT_DIR:-${PROJECT_ROOT}/eval_outputs/e4a_diagnosis}"
MERGED_VIEW_DIR="${MERGED_VIEW_DIR:-${TRACE_OUTPUT_DIR}/e4a_balanced_gate_merged}"
MERGE_VIEW="${MERGE_VIEW:-1}"

DEVICE="${DEVICE:-cuda:0}"
DEVICES="${DEVICES:-cuda:0,cuda:1,cuda:2}"
NUM_SAMPLES="${NUM_SAMPLES:-500}"
SAMPLE_ID_START="${SAMPLE_ID_START:-0}"
SAMPLE_ID_END="${SAMPLE_ID_END:-500}"
SEED="${SEED:-42}"

TEXTURE_PREPROCESS_MODE="${TEXTURE_PREPROCESS_MODE:-plain_resize}"
BALANCED_GATE_HIDDEN_DIM="${BALANCED_GATE_HIDDEN_DIM:-64}"
BALANCED_GATE_SCALE="${BALANCED_GATE_SCALE:-0.2}"
BALANCED_GATE_MIN="${BALANCED_GATE_MIN:-0.8}"
BALANCED_GATE_MAX="${BALANCED_GATE_MAX:-1.2}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

cd "${PROJECT_ROOT}"

for required_path in \
  "${DATASET_JSON}" \
  "${SPLIT_PATH}" \
  "${DATA_ROOT}" \
  "${E4A_CKPT}" \
  "${TEXTURE_CKPT}"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "[ERROR] Required path does not exist: ${required_path}" >&2
    exit 1
  fi
done

mkdir -p "${TRACE_OUTPUT_DIR}" "${DIAG_OUT_DIR}"

IFS=',' read -r -a DEVICE_LIST <<< "${DEVICES}"
if [[ "${#DEVICE_LIST[@]}" -lt 1 ]]; then
  echo "[ERROR] DEVICES is empty. Example: DEVICES=cuda:0,cuda:1,cuda:2" >&2
  exit 1
fi

build_benchmark_cmd() {
  local shard_output_dir="$1"
  local shard_start="$2"
  local shard_end="$3"
  local shard_device="$4"

  BENCHMARK_CMD=(
    python tools/run_fixed_benchmark.py
    --dataset_json "${DATASET_JSON}"
    --data_root "${DATA_ROOT}"
    --split_path "${SPLIT_PATH}"
    --gam_ckpt "${E4A_CKPT}"
    --texture_ckpt "${TEXTURE_CKPT}"
    --output_dir "${shard_output_dir}"
    --run_name e4a_balanced_gate
    --modes token
    --num_samples "${NUM_SAMPLES}"
    --sample_id_start "${shard_start}"
    --sample_id_end "${shard_end}"
    --seed "${SEED}"
    --resume_generation 0
    --skip_existing 0
    --overwrite 1
    --texture_preprocess_mode "${TEXTURE_PREPROCESS_MODE}"
    --layer_group_enabled 1
    --use_texture_gate 0
    --use_palette_tokens 1
    --num_palette_tokens 4
    --use_balanced_fusion_gate 1
    --balanced_gate_hidden_dim "${BALANCED_GATE_HIDDEN_DIM}"
    --balanced_gate_scale "${BALANCED_GATE_SCALE}"
    --balanced_gate_min "${BALANCED_GATE_MIN}"
    --balanced_gate_max "${BALANCED_GATE_MAX}"
    --save_balanced_gate_trace 1
    --device "${shard_device}"
  )
}

STATS_CMD=(
  python tools/build_e4a_gate_stats.py
  --search_root "${TRACE_OUTPUT_DIR}"
  --out_dir "${DIAG_OUT_DIR}"
)

link_or_copy() {
  local src="$1"
  local dst="$2"
  if [[ -e "${dst}" || -L "${dst}" ]]; then
    return 0
  fi
  ln -s "${src}" "${dst}" 2>/dev/null || cp -R "${src}" "${dst}"
}

merge_shard_view() {
  if [[ "${MERGE_VIEW}" != "1" ]]; then
    return 0
  fi
  mkdir -p "${MERGED_VIEW_DIR}/generated" "${MERGED_VIEW_DIR}/real" "${MERGED_VIEW_DIR}/token"
  : > "${MERGED_VIEW_DIR}/gate_trace_all.jsonl"
  for i in "${!DEVICE_LIST[@]}"; do
    shard_dir="${TRACE_OUTPUT_DIR}/shard_$(printf '%02d' "${i}")/e4a_balanced_gate"
    if [[ ! -d "${shard_dir}" ]]; then
      continue
    fi
    if [[ -d "${shard_dir}/generated" ]]; then
      find "${shard_dir}/generated" -mindepth 1 -maxdepth 1 -print0 | while IFS= read -r -d '' item; do
        link_or_copy "${item}" "${MERGED_VIEW_DIR}/generated/$(basename "${item}")"
      done
    fi
    if [[ -d "${shard_dir}/real" ]]; then
      find "${shard_dir}/real" -mindepth 1 -maxdepth 1 -print0 | while IFS= read -r -d '' item; do
        link_or_copy "${item}" "${MERGED_VIEW_DIR}/real/$(basename "${item}")"
      done
    fi
    if [[ -d "${shard_dir}/token" ]]; then
      find "${shard_dir}/token" -mindepth 1 -maxdepth 1 -print0 | while IFS= read -r -d '' item; do
        link_or_copy "${item}" "${MERGED_VIEW_DIR}/token/$(basename "${item}")"
      done
      find "${shard_dir}/token" -name gate_trace.jsonl -print0 | while IFS= read -r -d '' trace_file; do
        cat "${trace_file}" >> "${MERGED_VIEW_DIR}/gate_trace_all.jsonl"
      done
    fi
  done
}

echo "============================================"
echo "E4a gate trace diagnosis"
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "E4A_CKPT=${E4A_CKPT}"
echo "TEXTURE_CKPT=${TEXTURE_CKPT}"
echo "TRACE_OUTPUT_DIR=${TRACE_OUTPUT_DIR}"
echo "DIAG_OUT_DIR=${DIAG_OUT_DIR}"
echo "MERGED_VIEW_DIR=${MERGED_VIEW_DIR}"
echo "NUM_SAMPLES=${NUM_SAMPLES}, SAMPLE_ID_RANGE=[${SAMPLE_ID_START}, ${SAMPLE_ID_END})"
echo "DEVICES=${DEVICES}"
echo "============================================"
total_range=$((SAMPLE_ID_END - SAMPLE_ID_START))
if [[ "${total_range}" -le 0 ]]; then
  echo "[ERROR] invalid sample range: [${SAMPLE_ID_START}, ${SAMPLE_ID_END})" >&2
  exit 1
fi

for i in "${!DEVICE_LIST[@]}"; do
  shard_start=$((SAMPLE_ID_START + i * total_range / ${#DEVICE_LIST[@]}))
  shard_end=$((SAMPLE_ID_START + (i + 1) * total_range / ${#DEVICE_LIST[@]}))
  shard_dir="${TRACE_OUTPUT_DIR}/shard_$(printf '%02d' "${i}")"
  build_benchmark_cmd "${shard_dir}" "${shard_start}" "${shard_end}" "${DEVICE_LIST[$i]}"
  echo "[SHARD ${i}] device=${DEVICE_LIST[$i]} range=[${shard_start}, ${shard_end}) output=${shard_dir}"
  printf '%q ' "${BENCHMARK_CMD[@]}"
  echo
done
printf '%q ' "${STATS_CMD[@]}"
echo

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1, commands not executed."
  exit 0
fi

pids=()
names=()
for i in "${!DEVICE_LIST[@]}"; do
  shard_start=$((SAMPLE_ID_START + i * total_range / ${#DEVICE_LIST[@]}))
  shard_end=$((SAMPLE_ID_START + (i + 1) * total_range / ${#DEVICE_LIST[@]}))
  shard_dir="${TRACE_OUTPUT_DIR}/shard_$(printf '%02d' "${i}")"
  build_benchmark_cmd "${shard_dir}" "${shard_start}" "${shard_end}" "${DEVICE_LIST[$i]}"
  "${BENCHMARK_CMD[@]}" &
  pids+=("$!")
  names+=("shard_${i}:${DEVICE_LIST[$i]}:[${shard_start},${shard_end})")
done

for i in "${!pids[@]}"; do
  status=0
  wait "${pids[$i]}" || status=$?
  if [[ "${status}" != "0" ]]; then
    echo "[ERROR] ${names[$i]} failed with status=${status}" >&2
    exit "${status}"
  fi
done

merge_shard_view
"${STATS_CMD[@]}"

echo "Done."
if [[ "${MERGE_VIEW}" == "1" ]]; then
  echo "Merged view:"
  echo "  ${MERGED_VIEW_DIR}"
fi
echo "Gate stats:"
echo "  ${DIAG_OUT_DIR}/gate_stats_e4a.json"
echo "  ${DIAG_OUT_DIR}/gate_stats_e4a.csv"
