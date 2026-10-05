#!/bin/bash
#SBATCH -J E33TMgen
#SBATCH -p gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH -o /share/home/u2515283058/Mymodel/output_eval/e33_tm_trimodal_validation_20261005/generation_%j.log
#SBATCH -e /share/home/u2515283058/Mymodel/output_eval/e33_tm_trimodal_validation_20261005/generation_%j.err
set -eo pipefail
source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate Mymodel
cd /share/home/u2515283058/Mymodel
export PYTHONPATH="$PWD:${PYTHONPATH:-}" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
stage="${1:-smoke}"
if [ "$#" -gt 0 ]; then shift; fi
python -m tools.e33tm_generate --stage "$stage" "$@"
