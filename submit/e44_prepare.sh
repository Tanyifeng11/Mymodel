#!/bin/bash
#SBATCH -J E44_PREPARE
#SBATCH -p cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH -o /share/home/u2515283058/Mymodel/output_eval/e44_mimicbrush_drra_20261012/prepare_%j.log
#SBATCH -e /share/home/u2515283058/Mymodel/output_eval/e44_mimicbrush_drra_20261012/prepare_%j.err
set -eo pipefail
source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate Mymodel
cd /share/home/u2515283058/Mymodel
export PYTHONPATH="$PWD:${PYTHONPATH:-}" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
python -m tools.e44_protocol
