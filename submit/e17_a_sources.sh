#!/bin/bash
#SBATCH -J E17_A_sources
#SBATCH -p gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH -o /share/home/u2515283058/Mymodel/e17/e17_sources_%j.log
#SBATCH -e /share/home/u2515283058/Mymodel/e17/e17_sources_%j.err
set -eo pipefail
source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate Mymodel
PROJECT_ROOT="${PROJECT_ROOT:-/share/home/u2515283058/Mymodel}"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
for TAG in ${E17_SOURCE_TAGS:-baseline c_pattern64 gam}; do
  python -m tools.e17_source_probe --probe-dir "${PROJECT_ROOT}/e17/a_${TAG}" --device cuda:0
done
