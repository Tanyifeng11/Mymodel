#!/bin/bash
#SBATCH -J E41_prepare
#SBATCH -p cpu
#SBATCH -c 2
#SBATCH --mem=12G
#SBATCH -t 00:20:00
#SBATCH -o /share/home/u2515283058/Mymodel/output_eval/e41_attribute_bottleneck_20261011/job_%j.log
#SBATCH -e /share/home/u2515283058/Mymodel/output_eval/e41_attribute_bottleneck_20261011/job_%j.err
set -eo pipefail
source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate Mymodel
cd /share/home/u2515283058/Mymodel
export PYTHONPATH="$PWD" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
if [ "${E41_REPAIR_ROTATIONS:-0}" = 1 ]; then
  python -m tools.e41_protocol --repair-real-rotations
else
  python -m tools.e41_protocol
fi
out=output_eval/e41_attribute_bottleneck_20261011
tar -czf "$out/input_review.tar.gz" -C "$out" protocol.json calibration.json generation.json references.json review
sha256sum "$out/input_review.tar.gz" > "$out/input_review.tar.gz.sha256"
