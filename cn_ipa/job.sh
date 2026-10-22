#!/bin/bash
set -eo pipefail
ROOT=/share/home/u2515283058/cn_ipa_repro_20261022
source /share/apps/anaconda3/etc/profile.d/conda.sh
conda activate Mymodel
set -u
cd "$ROOT/repo"
export CNIPA_ROOT="$ROOT" PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
mkdir -p "$ROOT/cache/datasets" "$ROOT/cache/cuda"
export HF_DATASETS_CACHE="$ROOT/cache/datasets" CUDA_CACHE_PATH="$ROOT/cache/cuda"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
case "$1" in
  prepare) exec python -m cn_ipa.prepare ;;
  train_cn)
    shift
    exec accelerate launch --num_processes "${CNIPA_GPUS:-1}" --mixed_precision fp16 --num_cpu_threads_per_process 2 \
      -m cn_ipa.train_controlnet_garment \
      --pretrained_model_name_or_path /share/home/u2515283058/Mymodel/models/stable-diffusion-v1-5 \
      --controlnet_model_name_or_path "$ROOT/pretrained/controlnet" \
      --train_data_dir "$ROOT/cn_imagefolder" --image_column image \
      --cache_dir "$ROOT/cache/datasets" \
      --conditioning_image_column conditioning_image --caption_column text \
      --output_dir "$ROOT/checkpoints/cn_fashion" --height 512 --width 384 --resolution 512 \
      --train_batch_size 1 --gradient_accumulation_steps "${CNIPA_ACCUM:-4}" \
      --learning_rate 1e-5 --lr_scheduler constant --max_train_steps 5000 \
      --checkpointing_steps 1000 --gradient_checkpointing --mixed_precision fp16 \
      --seed 42 --report_to tensorboard --dataloader_num_workers 2 --set_grads_to_none "$@" ;;
  *) exec python -m cn_ipa.run "$@" ;;
esac
