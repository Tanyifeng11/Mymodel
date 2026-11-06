#!/bin/bash
#SBATCH --nodes=1
set -euo pipefail
RUN=/share/home/u2515283058/Mymodel/output_eval/e2_fft_20261106
cd "$RUN/source"
export PATH=/share/home/u2515283058/.conda/envs/Mymodel/bin:$PATH
export PYTHONPATH="$PWD:$PWD/tools"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export CUBLAS_WORKSPACE_CONFIG=:4096:8
if [[ "$1" == baseline ]]; then
  python -m tools.e2_fft_experiment prepare
  python -m tools.e2_fft_experiment regression
  python -m tools.e2_fft_experiment generate-regression
elif [[ "$1" == components ]]; then
  mkdir -p "$RUN/tests"
  python -m pytest tests/test_e2_compat.py tests/test_e2_geometry_fft.py -q
  python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 \
    -m tools.e2_fft_ddp_probe --out "$RUN/tests/ddp_result.json"
  python -m tools.e2_fft_experiment prepare-protocol
elif [[ "$1" == pilot ]]; then
  python -m tools.e2_fft_experiment calibrate
  python -m tools.e2_fft_experiment smoke
  python -m tools.e2_fft_experiment prepare-comparison
elif [[ "$1" == train ]]; then
  ARM="$2"
  GPUS="$3"
  python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$GPUS" \
    -m tools.e2_fft_experiment train --config "$RUN/comparison/$ARM.json"
else
  python -m tools.e2_fft_experiment "$@"
fi
