#!/bin/bash
#SBATCH -J E33TMOC0
#SBATCH -p cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH -o /share/home/u2515283058/Mymodel/output_eval/e33_tm_oc_carrier_construction_20261006/job_%j.log
#SBATCH -e /share/home/u2515283058/Mymodel/output_eval/e33_tm_oc_carrier_construction_20261006/job_%j.err
set -eo pipefail
source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate Mymodel
cd /share/home/u2515283058/Mymodel
export PYTHONPATH="$PWD:${PYTHONPATH:-}" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -m tools.e33tmoc_geometry_audit
