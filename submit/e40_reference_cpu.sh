#!/bin/bash
#SBATCH -J E40_ref_FFT
#SBATCH -p cpu
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -t 00:15:00
#SBATCH -o /share/home/u2515283058/Mymodel/output_eval/e40_pattern_identity_20261011/job_%j.log
#SBATCH -e /share/home/u2515283058/Mymodel/output_eval/e40_pattern_identity_20261011/job_%j.err
set -eo pipefail
source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate Mymodel
cd /share/home/u2515283058/Mymodel
export PYTHONPATH="$PWD" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
python -m tools.e40_reference_cpu
