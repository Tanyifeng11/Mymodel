#!/bin/bash
#SBATCH -J E40_analysis
#SBATCH -p cpu
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 00:25:00
#SBATCH -o /share/home/u2515283058/Mymodel/output_eval/e40_pattern_identity_20261011/job_%j.log
#SBATCH -e /share/home/u2515283058/Mymodel/output_eval/e40_pattern_identity_20261011/job_%j.err
set -eo pipefail
source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate Mymodel
cd /share/home/u2515283058/Mymodel
export PYTHONPATH="$PWD" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
python -m tools.e40_analyze
out=output_eval/e40_pattern_identity_20261011
touch "$out/e40_review_results.tar.gz"
tar -czf "$out/e40_review_results.tar.gz" --exclude='*.npz' --exclude='*.tar.gz*' --exclude='job_*' -C "$out" .
sha256sum "$out/e40_review_results.tar.gz" > "$out/e40_review_results.tar.gz.sha256"
