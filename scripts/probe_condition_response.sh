#!/usr/bin/env bash
set -euo pipefail

# 沿用 E5 的 checkpoint/数据路径参数；输出到新目录，避免旧图跳过探针。
# 示例：bash scripts/probe_condition_response.sh --dataset_json ... --data_root ...
#   --split_path ... --gam_ckpt ... --texture_ckpt ... --clip_model_path ... --output_dir ...
cd "$(dirname "$0")/.."
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
python tools/run_fixed_benchmark.py \
  --run_name e5 --modes token --num_samples 32 --sample_id_start 0 --sample_id_end 32 \
  --generation_seed 42 --mask_policy sketch_only --texture_preprocess_mode plain_resize \
  --use_texture_gate 1 --use_tcpm_lite 1 --layer_group_enabled 1 \
  --condition_response_probe 1 --condition_response_probe_steps 0 5 15 25 49 \
  --condition_response_probe_fractions 0.1 0.2 --condition_response_probe_region_kernel 9 \
  --compute_fid 0 --compute_kid 0 "$@"
