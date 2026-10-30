#!/bin/bash
set -euo pipefail
cd /share/home/u2515283058/Mymodel/output_eval/t3_p0_p1_20261028/source
export PATH=/share/home/u2515283058/.conda/envs/Mymodel/bin:$PATH
export PYTHONPATH="$PWD:$PWD/tools" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
python -m tools.t3_controlled_package
