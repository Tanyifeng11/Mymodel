#!/bin/bash
#SBATCH -J e48_holdout_audit
#SBATCH -p cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH -o /share/home/u2515283058/Mymodel/output_eval/e48_b1_structure_20261014/cpu_%j.log
#SBATCH -e /share/home/u2515283058/Mymodel/output_eval/e48_b1_structure_20261014/cpu_%j.err
set -eo pipefail
source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate Mymodel
cd /share/home/u2515283058/Mymodel
export PYTHONPATH="$PWD:${PYTHONPATH:-}" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m tools.e48_holdout --workers "${SLURM_CPUS_PER_TASK:-8}"
