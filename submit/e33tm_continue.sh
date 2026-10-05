#!/bin/bash
#SBATCH -J E33TMfull
#SBATCH -p gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH -o /share/home/u2515283058/Mymodel/output_eval/e33_tm_trimodal_validation_20261005/full_%j.log
#SBATCH -e /share/home/u2515283058/Mymodel/output_eval/e33_tm_trimodal_validation_20261005/full_%j.err
set -eo pipefail
source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate Mymodel
cd /share/home/u2515283058/Mymodel
export PYTHONPATH="$PWD:${PYTHONPATH:-}" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -m tools.e33tm_generate --stage seed42
if python -c "import json,sys; d=json.load(open('output_eval/e33_tm_trimodal_validation_20261005/decision_summary.json')); sys.exit(1 if d.get('hard_stop') else 0)"; then
    python -m tools.e33tm_generate --stage remaining
    python -m tools.e33tm_generate --stage robustness
    python -m tools.e33tm_generate --stage ablations
    python -m tools.e33tm_generate --stage diagnostics
fi
python -m tools.e33tm_visual_audit
python -m tools.e33tm_report
