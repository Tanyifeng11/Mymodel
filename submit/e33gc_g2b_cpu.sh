#!/bin/bash
#SBATCH -J E33GC_G2b_CPU
#SBATCH -p cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH -o /share/home/u2515283058/Mymodel/output_eval/e33_gc_g2b_20261009/job_%j.log
#SBATCH -e /share/home/u2515283058/Mymodel/output_eval/e33_gc_g2b_20261009/job_%j.err
set -eo pipefail
source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate Mymodel
cd /share/home/u2515283058/Mymodel
export PYTHONPATH="$PWD:${PYTHONPATH:-}" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
printf 'MODULE %s ARGS' "${G2B_MODULE:-tools.e33gc_g2b_prepare}"
printf ' %q' "$@"
printf '\n'
python -m "${G2B_MODULE:-tools.e33gc_g2b_prepare}" "$@"
