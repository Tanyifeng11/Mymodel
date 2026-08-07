#!/bin/bash

#SBATCH -J E7a
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --gres=gpu:1
# Slurm 在启动脚本前打开日志文件，因此不要依赖尚未创建的子目录。
# %j 会展开为当前作业号，便于区分多次提交。
#SBATCH -o /share/home/u2515283058/Mymodel/log_e7a_%j.log
#SBATCH -e /share/home/u2515283058/Mymodel/log_e7a_%j.err

# Keep nounset disabled while conda activation runs; activate.d scripts may
# reference optional variables. Enable it after the environment is active.
set -eo pipefail

source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate Mymodel
set -u

export HTTP_PROXY="${HTTP_PROXY:-http://211.67.63.75:3128}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://211.67.63.75:3128}"
export http_proxy="${http_proxy:-${HTTP_PROXY}}"
export https_proxy="${https_proxy:-${HTTPS_PROXY}}"

cd /share/home/u2515283058/Mymodel

NUM_GPUS="${NUM_GPUS:-1}" \
DRY_RUN="${DRY_RUN:-0}" \
bash scripts/train_phase1_e7a.sh
