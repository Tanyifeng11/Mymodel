#!/bin/bash
#SBATCH -J E43_BF_audit
#SBATCH -p cpu
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 00:10:00
#SBATCH -o /share/home/u2515283058/Mymodel/output_eval/e43_rgb_utility_20261011/job_%j.log
#SBATCH -e /share/home/u2515283058/Mymodel/output_eval/e43_rgb_utility_20261011/job_%j.err
set -eo pipefail
source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate Mymodel
cd /share/home/u2515283058/Mymodel
export PYTHONPATH="$PWD" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
python -m tools.e43_bf_audit
