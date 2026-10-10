#!/bin/bash
#SBATCH -J E35_FreeU
#SBATCH -p gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH -o /share/home/u2515283058/Mymodel/output_eval/e35_freeu_skip_20261010/job_%j.log
#SBATCH -e /share/home/u2515283058/Mymodel/output_eval/e35_freeu_skip_20261010/job_%j.err
set -eo pipefail
source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate Mymodel
cd /share/home/u2515283058/Mymodel
export PYTHONPATH="$PWD:${PYTHONPATH:-}" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
python -m "${E35_MODULE:-tools.e35_freeu_run}" "$@"
