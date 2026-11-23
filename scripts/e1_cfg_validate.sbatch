#!/bin/bash
#SBATCH --nodes=1
set -euo pipefail
RUN=/share/home/u2515283058/Mymodel/output_eval/e1_cfg_validate_20261123
cd "$RUN/source"
export PATH=/share/home/u2515283058/.conda/envs/Mymodel/bin:$PATH
export PYTHONPATH="/share/home/u2515283058/Mymodel/output_eval/sd_bfm_fresh_20261113/dependencies:$PWD:$PWD/tools"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
python -m tools.e1_cfg_validate prepare
if [[ ! -f "$RUN/audit/preflight_passed.json" ]]; then
    python -m tools.e1_cfg_validate smoke
fi
python -m tools.e1_cfg_validate formal
python -m tools.e1_cfg_validate evaluate
