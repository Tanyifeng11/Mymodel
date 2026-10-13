#!/bin/bash
#SBATCH -J e47_P0_CPU
#SBATCH -p cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH -o /share/home/u2515283058/Mymodel/output_eval/e47_tcpm_b1_factorial_20261013/cpu_%j.log
#SBATCH -e /share/home/u2515283058/Mymodel/output_eval/e47_tcpm_b1_factorial_20261013/cpu_%j.err
set -eo pipefail
source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate Mymodel
cd /share/home/u2515283058/Mymodel
export PYTHONPATH="$PWD:${PYTHONPATH:-}" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
python -m tools.e47_audit --workers "${SLURM_CPUS_PER_TASK:-8}"
