#!/usr/bin/env bash
set -euo pipefail

cd /share/home/u2515283058/Mymodel

module load anaconda3/4.12.0 || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate Mymodel || conda activate mymodel || conda activate base

bash scripts/train_phase1_e3a.sh
