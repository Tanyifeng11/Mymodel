#!/bin/bash
set -euo pipefail
cd /share/home/u2515283058/Mymodel/output_eval/t3_p0_p1_20261028/source
export PATH=/share/home/u2515283058/.conda/envs/Mymodel/bin:$PATH
export PYTHONPATH="$PWD:$PWD/tools" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
R=/share/home/u2515283058/Mymodel/output_eval/t3_controlled_continue_20261030
for stage in C0250 C0500 C1000; do
  python -m tools.t3_controlled_continue evaluate --stage "$stage"
done
# 此时早先依赖已结束，提交数有余量；评价成功后才启动CPU打包。
PACKAGE=$(sbatch --parsable -p cpu -J t3cont_package --cpus-per-task=2 --mem=8G \
  -t 00:05:00 --dependency=afterok:"$SLURM_JOB_ID" --kill-on-invalid-dep=yes \
  -o "$R/logs/package_%j.log" scripts/t3_controlled_package.sbatch)
printf 'package,%s\n' "$PACKAGE" >> "$R/audit/jobs.csv"
echo "PACKAGE_JOB=$PACKAGE"
