#!/bin/bash
#SBATCH -J E33TMMS_M3
#SBATCH -p gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH -o /share/home/u2515283058/Mymodel/output_eval/e33_tm_ms_representation_search_20261007/job_%j.log
#SBATCH -e /share/home/u2515283058/Mymodel/output_eval/e33_tm_ms_representation_search_20261007/job_%j.err
set -eo pipefail
source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate Mymodel
cd /share/home/u2515283058/Mymodel
export PYTHONPATH="$PWD:${PYTHONPATH:-}" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
if [ "${1:-train}" = train ]; then
    python -m tools.e33tmms_m3_train
else
    initialization="${2:-pure-noise}"
    python -m tools.e33tmms_m3_eval diagnostic64 --initialization "$initialization"
    if python -c 'from tools.e33tmms_protocol import OUT,read; import sys; sys.exit(0 if read(OUT/"decision_summary.json")["M3_feature_pass"] else 1)'; then
        python -m tools.e33tmms_m3_eval confirmation64 --initialization "$initialization"
    fi
    python -m tools.e33tmms_finalize
fi
