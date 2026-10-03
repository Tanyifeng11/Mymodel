#!/bin/bash
#SBATCH -J E33RCpreview
#SBATCH -p gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH -o /share/home/u2515283058/Mymodel/output_eval/e33_rc_real_rotation_20261003/preview_%j.log
#SBATCH -e /share/home/u2515283058/Mymodel/output_eval/e33_rc_real_rotation_20261003/preview_%j.err
set -eo pipefail
source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate Mymodel
cd /share/home/u2515283058/Mymodel
export PYTHONPATH="$PWD:${PYTHONPATH:-}" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
python -m tools.e33rc_preview "$@"
