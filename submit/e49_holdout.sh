#!/bin/bash
#SBATCH -J e49_holdout
#SBATCH -p cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH -o /share/home/u2515283058/Mymodel/output_eval/e49_ssvm_20261014/cpu_%j.log
#SBATCH -e /share/home/u2515283058/Mymodel/output_eval/e49_ssvm_20261014/cpu_%j.err
set -eo pipefail
source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate Mymodel
cd /share/home/u2515283058/Mymodel
export PYTHONPATH="$PWD:${PYTHONPATH:-}" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
python -m tools.e49_confirm audit --workers 4
