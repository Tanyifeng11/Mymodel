#!/bin/bash
#SBATCH --nodes=1
set -euo pipefail
RUN=/share/home/u2515283058/Mymodel/output_eval/e2_a_ablation_20261110
cd "$RUN/source"
export PATH=/share/home/u2515283058/.conda/envs/Mymodel/bin:$PATH
export PYTHONPATH="$PWD:$PWD/tools"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export CUBLAS_WORKSPACE_CONFIG=:4096:8
if [[ "$1" == prepare ]]; then
  mkdir -p "$RUN/tests"
  python -m pytest tests/test_e2_compat.py tests/test_e2_geometry_fft.py tests/test_e2_followup.py tests/test_e2_local_texture.py tests/test_e2_structure.py tests/test_e2_a_metrics.py tests/test_e2_region_router.py tests/test_e2_c_metrics.py tests/test_e2_a_ablation.py -q -rP | tee "$RUN/tests/pytest.txt"
  python -m tools.e2_a_ablation prepare
elif [[ "$1" == pilot ]]; then
  python -m tools.e2_a_ablation regression
  python -m tools.e2_a_ablation smoke
elif [[ "$1" == train || "$1" == generate ]]; then
  arms=(A1 A2 A3 A4)
  arm=${2:-${arms[${SLURM_ARRAY_TASK_ID:-0}]}}
  python -m tools.e2_a_ablation "$1" --arm "$arm"
elif [[ "$1" == review ]]; then
  python -m tools.e2_a_ablation evaluate
  python -m tools.e2_a_ablation bundle
elif [[ "$1" == dispatch-tail ]]; then
  train_array=$2
  test ! -s "$RUN/audit/a4_job_id.txt"
  a4=$(sbatch --parsable -J E2A-A4 -p gpu --gres=gpu:A30:1 -c 2 --mem=32G -t 08:00:00 -o "$RUN/logs/train_A4_%j.out" -e "$RUN/logs/train_A4_%j.err" scripts/e2_a_ablation.sbatch train A4)
  printf '%s\n' "$a4" > "$RUN/audit/a4_job_id.txt"
  final=$(sbatch --parsable -J E2A-audit-final -p cpu -c 4 --mem=16G -t 01:00:00 --dependency="afterok:$train_array:$a4" -o "$RUN/logs/audit_final_%j.out" -e "$RUN/logs/audit_final_%j.err" scripts/e2_a_ablation.sbatch audit-final)
  printf '%s\n' "$final" > "$RUN/audit/final_audit_job_id.txt"
  next=$(sbatch --parsable -J E2A-gen-dispatch -p cpu -c 1 --mem=1G -t 00:10:00 --dependency="afterok:$final" -o "$RUN/logs/gen_dispatch_%j.out" -e "$RUN/logs/gen_dispatch_%j.err" scripts/e2_a_ablation.sbatch dispatch-generation)
  printf '%s\n' "$next" > "$RUN/audit/generate_dispatch_job_id.txt"
  printf 'A4 %s final_audit %s generation_dispatch %s\n' "$a4" "$final" "$next"
elif [[ "$1" == dispatch-generation ]]; then
  test ! -s "$RUN/audit/generate_job_id.txt"
  gen=$(sbatch --parsable -J E2A-generate -p gpu --gres=gpu:A30:1 -c 2 --mem=32G -t 01:00:00 --array=0-3%2 -o "$RUN/logs/generate_%A_%a.out" -e "$RUN/logs/generate_%A_%a.err" scripts/e2_a_ablation.sbatch generate)
  printf '%s\n' "$gen" > "$RUN/audit/generate_job_id.txt"
  next=$(sbatch --parsable -J E2A-review-dispatch -p cpu -c 1 --mem=1G -t 00:10:00 --dependency="afterok:$gen" -o "$RUN/logs/review_dispatch_%j.out" -e "$RUN/logs/review_dispatch_%j.err" scripts/e2_a_ablation.sbatch dispatch-review)
  printf '%s\n' "$next" > "$RUN/audit/review_dispatch_job_id.txt"
elif [[ "$1" == dispatch-review ]]; then
  test ! -s "$RUN/audit/review_job_id.txt"
  review=$(sbatch --parsable -J E2A-review -p gpu --gres=gpu:A30:1 -c 2 --mem=32G -t 01:00:00 -o "$RUN/logs/review_%j.out" -e "$RUN/logs/review_%j.err" scripts/e2_a_ablation.sbatch review)
  printf '%s\n' "$review" > "$RUN/audit/review_job_id.txt"
else
  python -m tools.e2_a_ablation "$@"
fi
