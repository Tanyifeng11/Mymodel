#!/bin/bash
#SBATCH --job-name=E2B_CS_R42
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00

set -euo pipefail
RUN_ROOT=/share/home/u2515283058/Mymodel/output_eval/m54_b1_screen_20261017/e2b_color_safe_retrain_seed42_20261018
cd "${RUN_ROOT}/source"
export PATH=/share/home/u2515283058/.conda/envs/Mymodel/bin:$PATH
export PYTHONPATH="$PWD"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2

pick_e2b_port() {
  python -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()'
}

# 当前 Accelerate 会把显式端口0直接传给客户端，必须先取得真实非零端口。
E2B_MASTER_PORT=$(pick_e2b_port)
echo "[launch] 四卡通信预检 port=${E2B_MASTER_PORT} job=${SLURM_JOB_ID}"
accelerate launch --multi_gpu --num_processes=4 --num_machines=1 \
  --machine_rank=0 --main_process_ip=127.0.0.1 --main_process_port="${E2B_MASTER_PORT}" \
  --mixed_precision=fp16 tools/e2b_color_safe_retrain.py ddp-smoke

E2B_MASTER_PORT=$(pick_e2b_port)
echo "[launch] 原配方训练 port=${E2B_MASTER_PORT} job=${SLURM_JOB_ID}"
accelerate launch --multi_gpu --num_processes=4 --num_machines=1 \
  --machine_rank=0 --main_process_ip=127.0.0.1 --main_process_port="${E2B_MASTER_PORT}" --mixed_precision=fp16 \
  tools/e2b_color_safe_retrain.py train
python tools/e2b_color_safe_retrain.py verify
