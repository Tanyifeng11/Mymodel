#!/usr/bin/env bash
set -euo pipefail

# Summarize existing E4a runtime gate traces only.
# It does not run inference, does not train, and does not modify checkpoints.

PROJECT_ROOT="${PROJECT_ROOT:-/share/home/u2515283058/Mymodel}"
TRACE_OUTPUT_DIR="${TRACE_OUTPUT_DIR:-${PROJECT_ROOT}/eval_outputs/phase1_full_500_gate_trace}"
DIAG_OUT_DIR="${DIAG_OUT_DIR:-${PROJECT_ROOT}/eval_outputs/e4a_diagnosis}"
MERGED_VIEW_DIR="${MERGED_VIEW_DIR:-${TRACE_OUTPUT_DIR}/e4a_balanced_gate_merged}"
MERGE_VIEW="${MERGE_VIEW:-1}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

cd "${PROJECT_ROOT}"

if [[ ! -d "${TRACE_OUTPUT_DIR}" ]]; then
  echo "[ERROR] TRACE_OUTPUT_DIR does not exist: ${TRACE_OUTPUT_DIR}" >&2
  echo "Set TRACE_OUTPUT_DIR to the directory that contains shard_*/e4a_balanced_gate/token/*/gate_trace.jsonl" >&2
  exit 1
fi

mapfile -t TRACE_FILES < <(find "${TRACE_OUTPUT_DIR}" -name gate_trace.jsonl -type f | sort)
if [[ "${#TRACE_FILES[@]}" -eq 0 ]]; then
  echo "[ERROR] No gate_trace.jsonl found under: ${TRACE_OUTPUT_DIR}" >&2
  echo "This script only summarizes existing traces. It will not rerun inference." >&2
  exit 1
fi

mkdir -p "${DIAG_OUT_DIR}"

if [[ "${MERGE_VIEW}" == "1" ]]; then
  mkdir -p "${MERGED_VIEW_DIR}/generated" "${MERGED_VIEW_DIR}/real" "${MERGED_VIEW_DIR}/token"
  : > "${MERGED_VIEW_DIR}/gate_trace_all.jsonl"

  while IFS= read -r -d '' exp_dir; do
    if [[ -d "${exp_dir}/generated" ]]; then
      find "${exp_dir}/generated" -mindepth 1 -maxdepth 1 -print0 | while IFS= read -r -d '' item; do
        dst="${MERGED_VIEW_DIR}/generated/$(basename "${item}")"
        [[ -e "${dst}" || -L "${dst}" ]] || ln -s "${item}" "${dst}" 2>/dev/null || cp -R "${item}" "${dst}"
      done
    fi
    if [[ -d "${exp_dir}/real" ]]; then
      find "${exp_dir}/real" -mindepth 1 -maxdepth 1 -print0 | while IFS= read -r -d '' item; do
        dst="${MERGED_VIEW_DIR}/real/$(basename "${item}")"
        [[ -e "${dst}" || -L "${dst}" ]] || ln -s "${item}" "${dst}" 2>/dev/null || cp -R "${item}" "${dst}"
      done
    fi
    if [[ -d "${exp_dir}/token" ]]; then
      find "${exp_dir}/token" -mindepth 1 -maxdepth 1 -print0 | while IFS= read -r -d '' item; do
        dst="${MERGED_VIEW_DIR}/token/$(basename "${item}")"
        [[ -e "${dst}" || -L "${dst}" ]] || ln -s "${item}" "${dst}" 2>/dev/null || cp -R "${item}" "${dst}"
      done
    fi
  done < <(find "${TRACE_OUTPUT_DIR}" -path "*/e4a_balanced_gate" -type d -print0 | sort -z)

  for trace_file in "${TRACE_FILES[@]}"; do
    cat "${trace_file}" >> "${MERGED_VIEW_DIR}/gate_trace_all.jsonl"
  done
fi

echo "============================================"
echo "Summarize E4a gate traces"
echo "TRACE_OUTPUT_DIR=${TRACE_OUTPUT_DIR}"
echo "DIAG_OUT_DIR=${DIAG_OUT_DIR}"
echo "TRACE_FILES=${#TRACE_FILES[@]}"
if [[ "${MERGE_VIEW}" == "1" ]]; then
  echo "MERGED_VIEW_DIR=${MERGED_VIEW_DIR}"
fi
echo "============================================"

python tools/build_e4a_gate_stats.py \
  --search_root "${TRACE_OUTPUT_DIR}" \
  --out_dir "${DIAG_OUT_DIR}"

echo "Done."
echo "Gate stats:"
echo "  ${DIAG_OUT_DIR}/gate_stats_e4a.json"
echo "  ${DIAG_OUT_DIR}/gate_stats_e4a.csv"
if [[ "${MERGE_VIEW}" == "1" ]]; then
  echo "Merged trace:"
  echo "  ${MERGED_VIEW_DIR}/gate_trace_all.jsonl"
fi
