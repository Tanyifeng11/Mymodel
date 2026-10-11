#!/bin/bash
#SBATCH -J E42_explore
#SBATCH -p cpu
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 02:00:00
#SBATCH -o /share/home/u2515283058/Mymodel/output_eval/e42_real_fabric_repeat_gate_20261011/exploratory_override_20261011/job_%j.log
#SBATCH -e /share/home/u2515283058/Mymodel/output_eval/e42_real_fabric_repeat_gate_20261011/exploratory_override_20261011/job_%j.err
set -eo pipefail
source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate Mymodel
cd /share/home/u2515283058/Mymodel
export PYTHONPATH="$PWD" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
if [[ "${E42_RESUME:-0}" == 1 ]]; then
  python -m tools.e42_explore --resume
else
  python -m tools.e42_explore
fi
