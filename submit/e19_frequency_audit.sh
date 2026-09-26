#!/bin/bash
#SBATCH -J E19_B_audit
#SBATCH -p gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH -o /share/home/u2515283058/Mymodel/e19/e19_audit_%j.log
#SBATCH -e /share/home/u2515283058/Mymodel/e19/e19_audit_%j.err
set -eo pipefail
source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate Mymodel
ROOT="${PROJECT_ROOT:-/share/home/u2515283058/Mymodel}"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
python -m tools.e19_frequency_audit --root "${ROOT}"
