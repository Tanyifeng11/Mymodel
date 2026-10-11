#!/bin/bash
#SBATCH -J E41_pack
#SBATCH -p cpu
#SBATCH -c 2
#SBATCH --mem=4G
#SBATCH -t 00:15:00
#SBATCH -o /share/home/u2515283058/Mymodel/output_eval/e41_attribute_bottleneck_20261011/job_%j.log
#SBATCH -e /share/home/u2515283058/Mymodel/output_eval/e41_attribute_bottleneck_20261011/job_%j.err
set -eo pipefail
source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate Mymodel
cd /share/home/u2515283058/Mymodel
export PYTHONPATH="$PWD" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m tools.e41_finalize
