#!/bin/bash
#SBATCH -J E53_CPU
#SBATCH -p cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH -o /share/home/u2515283058/Mymodel/output_eval/e53_b1_oracle_20261016/resources/cpu_%j.log
#SBATCH -e /share/home/u2515283058/Mymodel/output_eval/e53_b1_oracle_20261016/resources/cpu_%j.err
set -eo pipefail
source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate Mymodel
cd /share/home/u2515283058/Mymodel
export PYTHONPATH="$PWD:${PYTHONPATH:-}" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
python -m tools.e53_run "$@"
python -m tools.e53_dispatch advance || true
