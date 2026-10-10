#!/bin/bash
#SBATCH -J E38_PRIOR
#SBATCH -p cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH -o /share/home/u2515283058/Mymodel/output_eval/e38_structure_prior_20261011/job_%j.log
#SBATCH -e /share/home/u2515283058/Mymodel/output_eval/e38_structure_prior_20261011/job_%j.err
set -eo pipefail
source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate Mymodel
cd /share/home/u2515283058/Mymodel
export PYTHONPATH="$PWD:${PYTHONPATH:-}" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m "${E38_MODULE:-tools.e38_run}" "$@"
