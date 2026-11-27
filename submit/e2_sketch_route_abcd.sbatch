#!/bin/bash
#SBATCH --job-name=sketchABCD
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
set -euo pipefail
project=/share/home/u2515283058/Mymodel
run="$project/output_eval/e2_sketch_route_abcd_20261127"
cd "$project/output_eval/e1_cfg_validate_20261123/source"
export PATH=/share/home/u2515283058/.conda/envs/Mymodel/bin:$PATH
export PYTHONPATH="$project/output_eval/sd_bfm_fresh_20261113/dependencies:$PWD:$PWD/tools"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2

python "$run/e2_sketch_route_abcd_generate.py" --run "$run"
python "$run/e2_sketch_route_abcd_score.py" --manifest "$run/manifest.jsonl" --output "$run/scores.json"
archive="e2_sketch_route_abcd_results_${SLURM_JOB_ID}.tar.gz"
cd "$run"
tar -czf "$archive.partial" --exclude='*.tar.gz*' --exclude='*.out' .
mv -- "$archive.partial" "$archive"
sha256sum "$archive"
