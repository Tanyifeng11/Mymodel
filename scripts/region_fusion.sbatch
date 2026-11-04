#!/bin/bash
#SBATCH --nodes=1
set -euo pipefail
RUN=/share/home/u2515283058/Mymodel/output_eval/region_asymmetric_fusion_20261104
cd "$RUN/source"
export PATH=/share/home/u2515283058/.conda/envs/Mymodel/bin:$PATH
export PYTHONPATH="$PWD:$PWD/tools"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
ACTION="$1"
if [[ "$ACTION" == train ]]; then
  test -f "$RUN/audit/preflight_complete.json"
  ARM="$2"
  GPUS="$3"
  python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$GPUS" \
    -m tools.train_region_fusion train --variant "$ARM" --output-dir "$RUN/training" \
    --project-root /share/home/u2515283058/Mymodel --initialization-root "$RUN" \
    --accumulation "$((8/GPUS))"
elif [[ "$ACTION" == generate ]]; then
  ARM="$2"
  python -m tools.train_region_fusion generate --variant "$ARM" --output-dir "$RUN/training" \
    --project-root /share/home/u2515283058/Mymodel --initialization-root "$RUN" \
    --checkpoint "$RUN/training/$ARM/step28365.pt" --eval-manifest "$RUN/evaluation.json"
else
  python -m tools.region_fusion_experiment "$@"
fi
