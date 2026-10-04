#!/bin/bash
#SBATCH --job-name=E34-pilot-audit
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A30:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/share/home/u2515283058/Mymodel/output_eval/e34_capf_20261004/evidence_pilot_v2/audit_%j.log
#SBATCH --error=/share/home/u2515283058/Mymodel/output_eval/e34_capf_20261004/evidence_pilot_v2/audit_%j.err
set -eo pipefail
source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate Mymodel
cd /share/home/u2515283058/Mymodel_e34_capf
python -u -m tools.e34_pilot_audit
