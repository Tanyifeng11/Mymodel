#!/bin/bash
# 依赖链确保准备和梯度验收成功后，才分配正式训练资源。
set -euo pipefail
RUN=/share/home/u2515283058/Mymodel/output_eval/bf_fusion_t3_three_arm_20261101
JOB="$RUN/source/scripts/bf_t3_three_arm.sbatch"
mkdir -p "$RUN/logs"
test ! -f "$RUN/submitted_jobs.json"
PREP=$(sbatch --parsable -p cpu -J bf3_prepare -N1 -n1 -c4 --mem=64G -t 00:30:00 -o "$RUN/logs/prepare_%j.log" "$JOB" prepare)
CHECK=$(sbatch --parsable -p gpu -J bf3_preflight -N1 -n1 -c4 --gres=gpu:1 --mem=64G -t 00:30:00 --dependency=afterok:"$PREP" -o "$RUN/logs/preflight_%j.log" "$JOB" preflight)
A=$(sbatch --parsable -p gpu -J bf3_base -N1 -n1 -c8 --gres=gpu:2 --mem=96G -t 3-00:00:00 --dependency=afterok:"$CHECK" -o "$RUN/logs/base_%j.log" "$JOB" train BF-SD)
B=$(sbatch --parsable -p gpu -J bf3_t3 -N1 -n1 -c8 --gres=gpu:2 --mem=96G -t 3-00:00:00 --dependency=afterok:"$CHECK" -o "$RUN/logs/t3_%j.log" "$JOB" train BF-SD-T3)
EVAL=$(sbatch --parsable -p gpu -J bf3_evaluate -N1 -n1 -c4 --gres=gpu:1 --mem=48G -t 01:00:00 --dependency=afterok:"$A":"$B" -o "$RUN/logs/evaluate_%j.log" "$JOB" closeout)
printf '{"prepare":"%s","preflight":"%s","BF-SD":"%s","BF-SD-T3":"%s","evaluate":"%s"}\n' "$PREP" "$CHECK" "$A" "$B" "$EVAL" > "$RUN/submitted_jobs.json"
cat "$RUN/submitted_jobs.json"
