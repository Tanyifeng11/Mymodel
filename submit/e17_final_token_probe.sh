#!/bin/bash
#SBATCH -J E17_token_probe
#SBATCH -p gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH -o /share/home/u2515283058/Mymodel/e17/e17_token_probe_%j.log
#SBATCH -e /share/home/u2515283058/Mymodel/e17/e17_token_probe_%j.err
set -eo pipefail
source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate Mymodel
cd /share/home/u2515283058/Mymodel
export PYTHONPATH="$PWD:${PYTHONPATH:-}" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
for SUFFIX in "" "_select"; do
  python -m tools.e17_final_token_probe \
    --fused-dir e17/a_gam \
    --checkpoint "e17/direct_train${SUFFIX}/checkpoint-final/joint_model.pt" \
    --output "e17/final_token_probe${SUFFIX}" --device cuda:0
done
