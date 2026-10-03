#!/bin/bash
#SBATCH -J E34audit
#SBATCH -p cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH -o /share/home/u2515283058/Mymodel/output_eval/e34_capf_20261004/stage0_%j.log
#SBATCH -e /share/home/u2515283058/Mymodel/output_eval/e34_capf_20261004/stage0_%j.err
set -eo pipefail
source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate Mymodel
cd "${E34_PROJECT_ROOT:-/share/home/u2515283058/Mymodel}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
python -m tools.e34_stage0 "$@"
