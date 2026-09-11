#!/bin/bash
# sbatch submit/recheck_e9_output_block_100.sh
# 三组固定 100 张，seed 42；输出端阻断额外计算同 latent 的关闭旁路预测。
#SBATCH -J E9_output_block
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -o /share/home/u2515283058/Mymodel/log_e9_output_block_%j.log
#SBATCH -e /share/home/u2515283058/Mymodel/log_e9_output_block_%j.err
set -eo pipefail
export PROJECT_ROOT="${PROJECT_ROOT:-/share/home/u2515283058/Mymodel}"
export RECHECK_RUN_ID="${RECHECK_RUN_ID:-${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}"
export RECHECK_OUTPUT_DIR="${RECHECK_OUTPUT_DIR:-${PROJECT_ROOT}/eval_outputs/e9_output_block_validation_100/${RECHECK_RUN_ID}}"
export RECHECK_VARIANTS=alpha_000,alpha_100,output_block
export RUN_OUTPUT_BLOCK_AUDIT=1
# 复用已验证的 Conda 激活、权重冻结检查、固定划分和评测参数。
bash "${PROJECT_ROOT}/submit/recheck_e9_b_schedule_100.sh"
