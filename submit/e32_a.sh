#!/bin/bash
#SBATCH -J E32A
#SBATCH -p gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -o /share/home/u2515283058/Mymodel/output_eval/e32_target_supervised_pattern_field_20261001/A_%j.log
#SBATCH -e /share/home/u2515283058/Mymodel/output_eval/e32_target_supervised_pattern_field_20261001/A_%j.err
set -eo pipefail
source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate Mymodel
cd /share/home/u2515283058/Mymodel
export PYTHONPATH="$PWD:${PYTHONPATH:-}" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -m tools.e32_a_geometry_train prepare
for seed in 42 43 44; do
    python -m tools.e32_a_geometry_train train --seed "$seed"
    python -m tools.e32_a_geometry_train eval --seed "$seed"
done
for variant in A_no_target_supervision E_no_interior G_no_E26 G_dino_only; do
    python -m tools.e32_a_geometry_train train --seed 42 --variant "$variant"
    python -m tools.e32_a_geometry_train eval --seed 42 --variant "$variant"
done
python -m tools.e32_a_geometry_train decide
