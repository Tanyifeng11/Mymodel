#!/bin/bash
#SBATCH -J E40_prepare
#SBATCH -p cpu
#SBATCH -c 2
#SBATCH --mem=12G
#SBATCH -t 00:15:00
#SBATCH -o /share/home/u2515283058/Mymodel/output_eval/e40_pattern_identity_20261011/job_%j.log
#SBATCH -e /share/home/u2515283058/Mymodel/output_eval/e40_pattern_identity_20261011/job_%j.err
set -eo pipefail
source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate Mymodel
cd /share/home/u2515283058/Mymodel
export PYTHONPATH="$PWD" OMP_NUM_THREADS=2
python -m tools.e40_protocol
out=output_eval/e40_pattern_identity_20261011
touch "$out/input_review.tar.gz"
tar -czf "$out/input_review.tar.gz" -C "$out" protocol.json references.json real_color_pairs.json generation.json review
