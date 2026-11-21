#!/bin/bash
#SBATCH --nodes=1
set -euo pipefail
RUN=/share/home/u2515283058/Mymodel/output_eval/e1_bfm_continue20_20261121
cd "$RUN/source"
export PATH=/share/home/u2515283058/.conda/envs/Mymodel/bin:$PATH
export PYTHONPATH="/share/home/u2515283058/Mymodel/output_eval/sd_bfm_fresh_20261113/dependencies:$PWD:$PWD/tools"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
if [[ "$1" == run ]]; then
    python -m tools.e1_bfm_continue20 prepare
    if [[ ! -f "$RUN/audit/preflight_passed.json" ]]; then
        python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 -m tools.e1_bfm_continue20 smoke
    fi
    for epoch in $(seq 11 20); do
        python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 -m tools.e1_bfm_continue20 train --epoch "$epoch"
        python -m tools.e1_bfm_continue20 evaluate --epoch "$epoch"
    done
    python -m tools.e1_bfm_continue20 report
else
    python -m tools.e1_bfm_continue20 "$@"
fi
