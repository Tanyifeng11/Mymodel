#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/share/home/u2515283058/Mymodel}"
BF_TEST_ROOT="${BF_TEST_ROOT:-/share/home/u2515283058/datasets/BF/test}"
DATASET_JSON="${DATASET_JSON:-${PROJECT_ROOT}/data/bf_test_no_bag.json}"
SPLIT_PATH="${SPLIT_PATH:-${PROJECT_ROOT}/eval/benchmarks/bf_test_no_bag_full_split.json}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

python tools/build_bf_test_manifest.py \
  --data_root "${BF_TEST_ROOT}" \
  --dataset_json "${DATASET_JSON}" \
  --split_path "${SPLIT_PATH}" \
  --classes top outwear pants dress
