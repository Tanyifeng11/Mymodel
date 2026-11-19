#!/bin/bash
#SBATCH --nodes=1
set -euo pipefail
RUN=/share/home/u2515283058/Mymodel/output_eval/e1_bfm_reference_region_20261119
cd "$RUN/source"
export PATH=/share/home/u2515283058/.conda/envs/Mymodel/bin:$PATH
export PYTHONPATH="/share/home/u2515283058/Mymodel/output_eval/sd_bfm_fresh_20261113/dependencies:$PWD:$PWD/tools"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
if [[ "$1" == prepare ]]; then
    python -m tools.e1_bfm_reference_region prepare
elif [[ "$1" == preflight ]]; then
    python -m tools.e1_bfm_reference_region calibrate
    if [[ ! -f "$RUN/audit/preflight_passed.json" ]]; then
        for arm in control texture; do
            python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 -m tools.e1_bfm_reference_region smoke --arm "$arm"
        done
        python -m tools.e1_bfm_reference_region confirm_smoke
    fi
else
    arm="$1"
    [[ "$arm" == control || "$arm" == texture ]]
    for epoch in 11 12; do
        python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 -m tools.e1_bfm_reference_region train --arm "$arm" --epoch "$epoch"
        if [[ ! -f "$RUN/arms/$arm/epoch_$(printf '%02d' "$epoch")/validation/evaluation_complete.json" ]]; then
            python -m tools.e1_bfm_reference_region validation --arm "$arm" --epoch "$epoch"
        fi
    done
    if [[ "$arm" == texture ]]; then
        if [[ ! -f "$RUN/final_comparison/evaluation_complete.json" ]]; then
            python -m tools.e1_bfm_reference_region formal
        fi
        python -m tools.e1_bfm_reference_region report
    fi
fi
