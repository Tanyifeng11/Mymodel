#!/bin/bash
#SBATCH -J E30A2Prep
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH -o /share/home/u2515283058/Mymodel/output_eval/e30_a2_affinity_20260930/prepare_%j.log
#SBATCH -e /share/home/u2515283058/Mymodel/output_eval/e30_a2_affinity_20260930/prepare_%j.err
set -eo pipefail
source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate Mymodel
cd /share/home/u2515283058/Mymodel
export PYTHONPATH="$PWD:${PYTHONPATH:-}" OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
for module in attention block; do
  file="$HOME/.cache/dinov2-e30/dinov2/layers/$module.py"
  grep -q '^from __future__ import annotations' "$file" || sed -i '1i from __future__ import annotations' "$file"
done
out="$PWD/output_eval/e30_a2_affinity_20260930"
python -m tools.e30_a2_affinity prepare --out "$out" \
  --weights "$PWD/output_eval/e30_apacc_20260930/dinov2_vits14_pretrain.pth"
python -m tools.e30_a2_affinity frozen --out "$out"
