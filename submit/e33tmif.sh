#!/bin/bash
#SBATCH -J E33TMIF
#SBATCH -p gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH -o /share/home/u2515283058/Mymodel/output_eval/e33_tm_if_interface_localization_20261006/job_%j.log
#SBATCH -e /share/home/u2515283058/Mymodel/output_eval/e33_tm_if_interface_localization_20261006/job_%j.err
set -eo pipefail
source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate Mymodel
cd /share/home/u2515283058/Mymodel
export PYTHONPATH="$PWD:${PYTHONPATH:-}" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
action="${1:-smoke}"
shift || true
if [ "$action" = aggregate ]; then
    python -m tools.e33tmif_report "$@"
elif [ "$action" = visual ]; then
    python -m tools.e33tmif_visual "$@"
else
    python -m tools.e33tmif_run --work "$action" "$@"
    if [ "$action" = smoke ]; then python -m tools.e33tmif_visual --smoke; fi
fi
