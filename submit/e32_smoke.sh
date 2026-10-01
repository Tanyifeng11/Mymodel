#!/bin/bash
#SBATCH -J E32amp
#SBATCH -p gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH -o /share/home/u2515283058/Mymodel/output_eval/e32_target_supervised_pattern_field_20261001/amp_%j.log
#SBATCH -e /share/home/u2515283058/Mymodel/output_eval/e32_target_supervised_pattern_field_20261001/amp_%j.err
set -eo pipefail
source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate Mymodel
cd /share/home/u2515283058/Mymodel
export PYTHONPATH="$PWD:${PYTHONPATH:-}" OMP_NUM_THREADS=2
python -m unittest discover -s tests -p test_e32_integrity.py -v
