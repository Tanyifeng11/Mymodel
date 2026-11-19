#!/bin/bash
# 分阶段保存作业号和产物标记。验收通过的组才提交另外 436 对。
set -euo pipefail
prior="$1"
out="$prior/real_paired_B_on_off_20261118/tail6_real_ab_20261119"
cd "$prior/source"
mkdir -p "$out"
exec 9>"$out/queue.lock"
flock -n 9
export PATH=/share/home/u2515283058/.conda/envs/Mymodel/bin:$PATH
runner=scripts/e2_f68_tail6_real_ab.sbatch
cpu=(--partition=cpu --gres=gpu:0)

stage() {
    local name="$1" marker="$2" job state record
    shift 2
    [[ ! -s "$out/$marker" ]] || return 0
    record="$out/$name.jobs"
    if [[ -s "$record" ]]; then
        job=$(tail -n 1 "$record")
        state=$(sacct -X -n -P -j "$job" -o State | head -n 1 | cut -d ' ' -f 1)
        case "$state" in
            TIMEOUT|NODE_FAIL|PREEMPTED)
                if [[ $(wc -l < "$record") -ge 2 ]]; then
                    echo "$(date -Is) $name retry exhausted at $job"
                    exit 1
                fi
                ;;
            COMPLETED|FAILED|OUT_OF_MEMORY|CANCELLED|BOOT_FAIL)
                echo "$(date -Is) $name $job $state without $marker"
                exit 1
                ;;
            *) return 0 ;;
        esac
    fi
    if job=$(sbatch --parsable -J "f68_tail_$name" -o "$out/$name-%j.out" "$@" 2>"$out/sbatch_error.log"); then
        echo "${job%%;*}" >> "$record"
        echo "$(date -Is) $name submitted $job"
    elif ! grep -Eqi 'QOSMaxSubmit|MaxSubmitJob' "$out/sbatch_error.log"; then
        cat "$out/sbatch_error.log" >&2
        exit 1
    fi
}

while (( SECONDS < 172800 )); do
    if [[ ! -s "$out/protocol.json" ]]; then
        stage prep protocol.json "${cpu[@]}" --time=00:10:00 "$runner" "$prior" prepare --device cpu
    elif [[ ! -s "$out/qualification.json" ]]; then
        stage qualify qualification.json --time=00:12:00 "$runner" "$prior" qualify
    elif [[ ! -s "$out/prefix_0.json" || ! -s "$out/prefix_1.json" ]]; then
        for shard in 0 1; do
            stage "prefix$shard" "prefix_$shard.json" "$runner" "$prior" cache --shard "$shard"
        done
    elif [[ ! -s "$out/A/training.json" || ! -s "$out/B/training.json" ]]; then
        for arm in A B; do
            stage "train$arm" "$arm/training.json" --time=01:30:00 "$runner" "$prior" train --arm "$arm"
        done
    elif [[ ! -s "$out/selection.json" ]]; then
        stage select selection.json "${cpu[@]}" --time=00:05:00 "$runner" "$prior" select --device cpu
    elif [[ ! -s "$out/gen64_A_0.json" || ! -s "$out/gen64_B_0.json" ]]; then
        for arm in A B; do
            stage "gen64$arm" "gen64_${arm}_0.json" "$runner" "$prior" gen64 --arm "$arm" --shards 1
        done
    elif [[ ! -s "$out/eval64/tail6_review.zip" ]]; then
        stage score64 eval64/tail6_review.zip "${cpu[@]}" --time=00:20:00 "$runner" "$prior" score64 --device cpu
    else
        accepted=$(python -c 'import json,sys; print(" ".join(k for k,v in json.load(open(sys.argv[1]))["acceptance"].items() if v["passed"]))' "$out/eval64/result.json")
        if [[ -z "$accepted" ]]; then
            echo "$(date -Is) Neither arm passed 64; 500 not submitted" | tee "$out/finished.txt"
            exit 0
        fi
        remain=0
        for arm in $accepted; do
            for shard in 0 1 2 3; do
                marker="gen500_${arm}_$shard.json"
                if [[ ! -s "$out/$marker" ]]; then
                    remain=1
                    stage "gen500$arm$shard" "$marker" "$runner" "$prior" gen500 --arm "$arm" --shard "$shard" --shards 4
                fi
            done
        done
        if (( remain == 0 )); then
            if [[ -s "$out/eval500/tail6_review.zip" ]]; then
                echo "$(date -Is) 500 scored: $accepted" | tee "$out/finished.txt"
                exit 0
            fi
            stage score500 eval500/tail6_review.zip "${cpu[@]}" --cpus-per-task=8 --time=00:30:00 "$runner" "$prior" score500 --device cpu
        fi
    fi
    sleep 12
done
exit 2
