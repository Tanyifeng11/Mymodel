#!/bin/bash
#SBATCH -J E39_analysis
#SBATCH -p cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH -o /share/home/u2515283058/Mymodel/output_eval/e39_reference_trajectory_20261011/job_%j.log
#SBATCH -e /share/home/u2515283058/Mymodel/output_eval/e39_reference_trajectory_20261011/job_%j.err
set -eo pipefail
source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate Mymodel
cd /share/home/u2515283058/Mymodel
export PYTHONPATH="$PWD:${PYTHONPATH:-}" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m tools.e39_analyze
out=output_eval/e39_reference_trajectory_20261011
sacct -j 116752,116755,116756 -X -P -o JobID,State,ElapsedRaw,AllocTRES,ExitCode,NodeList > "$out/slurm_accounting.txt"
# 原始 NPZ 全留服务器；本地复核包包含图像、逐项数值、协议和哈希。
tar --exclude='*.npz' --exclude='*.tar.gz' --exclude='*.tar.gz.sha256' -czf "$out/e39_review_results.tar.gz" -C "$out" .
sha256sum "$out/e39_review_results.tar.gz" > "$out/e39_review_results.tar.gz.sha256"
