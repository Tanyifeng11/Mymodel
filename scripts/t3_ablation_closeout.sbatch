#!/bin/bash
set -euo pipefail
cd /share/home/u2515283058/Mymodel/output_eval/t3_reference_response_20261027/source
export PATH=/share/home/u2515283058/.conda/envs/Mymodel/bin:$PATH
export PYTHONPATH="$PWD:$PWD/tools"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
python -u -m tools.t3_ablation_closeout
