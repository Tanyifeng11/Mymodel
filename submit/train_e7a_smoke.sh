#!/bin/bash

#SBATCH -J E7a_smoke            ### E7a smoke test
#SBATCH -p gpu                  ### 武汉纺织大学 gpu 队列
#SBATCH -N 1                    ### 1 个计算节点
#SBATCH -n 4                    ### 4 个 CPU 核心
#SBATCH --gres=gpu:1            ### 只用 1 张卡（smoke test 数据少）
#SBATCH -o /share/home/u2515283058/Mymodel/log/e7a_smoke/train_cluster.log
#SBATCH -e /share/home/u2515283058/Mymodel/log/e7a_smoke/train_cluster.err

set -euo pipefail

# 1. 加载底层 Anaconda
source /share/apps/anaconda3/etc/profile.d/conda.sh

# 2. 激活环境
conda activate Mymodel

# 3. 代理（如果需要）
export HTTP_PROXY=http://211.67.63.75:3128
export HTTPS_PROXY=http://211.67.63.75:3128
export http_proxy=http://211.67.63.75:3128
export https_proxy=http://211.67.63.75:3128

# 4. 切换到项目目录
cd /share/home/u2515283058/Mymodel

# 5. 启动 E7a smoke test
NUM_GPUS=1 \
MAX_TRAIN_STEPS=200 \
E7A_SMOKE_TEST=1 \
bash scripts/train_phase1_e7a_smoke.sh
