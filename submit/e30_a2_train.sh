#!/bin/bash
#SBATCH -J E30A2Train
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -o /share/home/u2515283058/Mymodel/output_eval/e30_a2_affinity_20260930/train_%j.log
#SBATCH -e /share/home/u2515283058/Mymodel/output_eval/e30_a2_affinity_20260930/train_%j.err
set -eo pipefail
source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate Mymodel
cd /share/home/u2515283058/Mymodel
export PYTHONPATH="$PWD:${PYTHONPATH:-}" OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
out="$PWD/output_eval/e30_a2_affinity_20260930"
for seed in 42 43 44; do
  python -m tools.e30_a2_affinity train --out "$out" --seed "$seed"
done
python -m tools.e30_a2_affinity audit --out "$out"
