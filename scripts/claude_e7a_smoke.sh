#!/usr/bin/env bash
set -euo pipefail

export DRY_RUN="${DRY_RUN:-1}"
export PROJECT_ROOT="${PROJECT_ROOT:-/mnt/f/fuxian/Mymodel}"
export DATASETS_ROOT="${DATASETS_ROOT:-/mnt/f/fuxian/dataset/datasets}"

echo "[INFO] PROJECT_ROOT=$PROJECT_ROOT"
echo "[INFO] DATASETS_ROOT=$DATASETS_ROOT"
echo "[INFO] DRY_RUN=$DRY_RUN"

if [[ ! -d "$DATASETS_ROOT" ]]; then
    echo "[ERROR] DATASETS_ROOT does not exist: $DATASETS_ROOT" >&2
    exit 1
fi

exec bash "$PROJECT_ROOT/scripts/train_phase1_e7a_smoke.sh"
