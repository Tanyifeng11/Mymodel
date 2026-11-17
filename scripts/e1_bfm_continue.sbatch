#!/bin/bash
#SBATCH --nodes=1
set -euo pipefail
RUN=/share/home/u2515283058/Mymodel/output_eval/e1_bfm_continue_20261117
cd "$RUN/source"
export PATH=/share/home/u2515283058/.conda/envs/Mymodel/bin:$PATH
export PYTHONPATH="/share/home/u2515283058/Mymodel/output_eval/sd_bfm_fresh_20261113/dependencies:$PWD:$PWD/tools"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
if [[ "$1" == run ]]; then
    for epoch in 6 7 8 9 10; do
        python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 -m tools.e1_bfm_continue train --epoch "$epoch"
        if [[ ! -f "$RUN/epochs/epoch_$(printf '%02d' "$epoch")/evaluation_complete.json" ]]; then
            python -m tools.e1_bfm_continue evaluate --epoch "$epoch"
        fi
    done
    python -m tools.e1_bfm_continue report
elif [[ "$1" == train ]]; then
    python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 -m tools.e1_bfm_continue "$@"
else
    python -m tools.e1_bfm_continue "$@"
fi
