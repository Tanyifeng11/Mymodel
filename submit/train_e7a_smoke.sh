#!/bin/bash

#SBATCH -J E7a_smoke            ### E7a smoke test
#SBATCH -p gpu                  ### 武汉纺织大学 gpu 队列
#SBATCH -N 1                    ### 1 个计算节点
#SBATCH -n 4                    ### 4 个 CPU 核心
#SBATCH --gres=gpu:1            ### 只用 1 张卡（smoke test 数据少）
#SBATCH -o /share/home/u2515283058/Mymodel/log/e7a_smoke/train_cluster.log
#SBATCH -e /share/home/u2515283058/Mymodel/log/e7a_smoke/train_cluster.err

# 注意: 不要在这里开 `set -u`。conda 的 activate.d 脚本(binutils 那几个)
# 内部会引用 ADDR2LINE 等未设置的变量, 开了 -u 会直接崩在激活阶段,
# 训练根本起不来。等 conda activate 之后再开。
set -eo pipefail

# 1. 加载底层 Anaconda
source /share/apps/anaconda3/etc/profile.d/conda.sh

# 2. 激活环境
conda activate Mymodel

# 激活完成, 现在可以安全地开 -u 了
set -u

# 3. 代理（如果需要）
export HTTP_PROXY=http://211.67.63.75:3128
export HTTPS_PROXY=http://211.67.63.75:3128
export http_proxy=http://211.67.63.75:3128
export https_proxy=http://211.67.63.75:3128

# 4. 切换到项目目录
cd /share/home/u2515283058/Mymodel

# 5. 启动 E7a smoke test
#    先 DRY_RUN=1 确认路径校验和命令拼接, 确认无误后把这行改成 DRY_RUN=0
NUM_GPUS=1 \
MAX_TRAIN_STEPS=200 \
DRY_RUN="${DRY_RUN:-0}" \
bash scripts/train_phase1_e7a_smoke.sh
