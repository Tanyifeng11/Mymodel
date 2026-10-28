#!/bin/bash
set -euo pipefail
PROJECT=/share/home/u2515283058/LOTS
cd "$PROJECT"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_DISABLE_XET=1 HF_HOME="$PROJECT/cache/huggingface"
export HF_ENDPOINT=https://huggingface.co HF_HUB_ETAG_TIMEOUT=30 HF_HUB_DOWNLOAD_TIMEOUT=60
if [[ "$1" == "fetch" ]]; then
  # 沿用集群既有HTTP代理，供官方PyPI/HuggingFace下载使用。
  export HTTP_PROXY=http://211.67.63.75:3128 HTTPS_PROXY=http://211.67.63.75:3128
  export http_proxy="$HTTP_PROXY" https_proxy="$HTTPS_PROXY"
  exec 9>"$PROJECT/.fetch.lock"
  if ! flock -n 9; then
    echo "LOTS_FETCH_ALREADY_RUNNING"
    exit 0
  fi
  unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
  exec /share/home/u2515283058/.conda/envs/texfusion-dit/bin/python run_experiment.py fetch
else
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
  exec "$PROJECT/.venv/bin/python" run_experiment.py "$@"
fi
