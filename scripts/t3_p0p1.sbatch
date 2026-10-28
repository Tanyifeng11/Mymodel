#!/bin/bash
set -euo pipefail
RUN=/share/home/u2515283058/Mymodel/output_eval/t3_p0_p1_20261028
cd "$RUN/source"
export PATH=/share/home/u2515283058/.conda/envs/Mymodel/bin:$PATH
export PYTHONPATH="$PWD:$PWD/tools"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
case "$1" in
  generate)
    shift
    if [[ "$1" == "BF-Fashion" ]]; then
      export PATH=/share/home/u2515283058/.conda/envs/ldm/bin:$PATH
    fi
    python -m tools.t3_p0p1_generate "$@" ;;
  lots_fetch)
    unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
    /share/home/u2515283058/.conda/envs/texfusion-dit/bin/python -m tools.t3_p0p1_lots fetch ;;
  lots_smoke)
    "$RUN/lots_env/bin/python" -m tools.t3_p0p1_lots smoke ;;
  review) python -m tools.t3_p0p1_review --project /share/home/u2515283058/Mymodel ;;
  human) python -m tools.t3_p0p1_human --project /share/home/u2515283058/Mymodel ;;
  evaluate) shift; python -m tools.t3_p0p1_evaluate "$@" ;;
  *) python -m tools.t3_p0p1 "$@" ;;
esac
