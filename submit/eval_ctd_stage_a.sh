#!/bin/bash
# CTD Stage A 正式 S1/S2/S3 多 seed 评测。
# 提交：sbatch submit/eval_ctd_stage_a.sh

#SBATCH -J ctd_a_eval
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --gres=gpu:1
#SBATCH -o /share/home/u2515283058/Mymodel/log_ctd_eval_%j.log
#SBATCH -e /share/home/u2515283058/Mymodel/log_ctd_eval_%j.err

set -eo pipefail
source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate Mymodel
set -u

export HTTP_PROXY="${HTTP_PROXY:-http://211.67.63.75:3128}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://211.67.63.75:3128}"
export http_proxy="${http_proxy:-${HTTP_PROXY}}"
export https_proxy="${https_proxy:-${HTTPS_PROXY}}"

cd /share/home/u2515283058/Mymodel
bash scripts/eval_ctd_stage_a.sh
