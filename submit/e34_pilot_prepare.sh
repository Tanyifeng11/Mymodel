#!/bin/bash
#SBATCH --job-name=E34-pilot-pack
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/share/home/u2515283058/Mymodel/output_eval/e34_capf_20261004/evidence_pilot_v2/prepare_%j.log
#SBATCH --error=/share/home/u2515283058/Mymodel/output_eval/e34_capf_20261004/evidence_pilot_v2/prepare_%j.err
set -eo pipefail
source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate Mymodel
cd /share/home/u2515283058/Mymodel_e34_capf
python -u -m tools.e34_pilot_prepare
