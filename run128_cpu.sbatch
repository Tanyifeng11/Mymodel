#!/bin/bash
set -euo pipefail
cd /share/home/u2515283058/TextureGAN
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
PYTHON=/share/home/u2515283058/.conda/envs/Mymodel/bin/python
"$PYTHON" compare128.py replay_cpu
"$PYTHON" compare128.py generate --device cpu
"$PYTHON" compare128.py evaluate_cpu
