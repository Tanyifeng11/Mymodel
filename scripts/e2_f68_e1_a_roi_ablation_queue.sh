#!/bin/bash
set -euo pipefail
prior="$1"
out="$prior/real_paired_B_on_off_20261118/tail6_real_ab_e1_epoch10_20261119/a_roi_frame_ablation_20261120"
cd "$prior/source"
mkdir -p "$out"
exec 9>"$out/queue.lock"
flock -n 9
export PATH=/share/home/u2515283058/.conda/envs/Mymodel/bin:$PATH
runner=scripts/e2_f68_e1_a_roi_ablation.sbatch
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
    if job=$(sbatch --parsable -J "a_roi_$name" -o "$out/$name-%j.out" "$@" 2>"$out/sbatch_error.log"); then
        echo "${job%%;*}" >> "$record"
        echo "$(date -Is) $name submitted $job"
    elif ! grep -Eqi 'QOSMaxSubmit|MaxSubmitJob' "$out/sbatch_error.log"; then
        cat "$out/sbatch_error.log" >&2
        exit 1
    fi
}

generate_and_score() {
    local n="$1" shards="$2" duration="$3" shard marker left=0
    for ((shard=0; shard<shards; shard++)); do
        marker="generate${n}_$shard.json"
        if [[ ! -s "$out/$marker" ]]; then
            left=1
            stage "gen${n}_$shard" "$marker" --time="$duration" "$runner" "$prior" "generate$n" --shard "$shard" --shards "$shards"
        fi
    done
    if (( left == 0 )); then
        stage "score$n" "eval$n/review.zip" "${cpu[@]}" --cpus-per-task=8 --time=00:45:00 "$runner" "$prior" "score$n" --device cpu --shards "$shards"
    fi
}

while true; do
    if [[ ! -s "$out/protocol.json" ]]; then
        stage prep protocol.json "${cpu[@]}" --time=00:10:00 "$runner" "$prior" prepare --device cpu
    elif [[ ! -s "$out/qualification.json" ]]; then
        stage qualify qualification.json --time=00:10:00 "$runner" "$prior" qualify
    elif [[ ! -s "$out/frame/training.json" || ! -s "$out/inside/training.json" || ! -s "$out/both/training.json" ]]; then
        for arm in frame inside both; do
            stage "$arm" "$arm/training.json" --time=00:25:00 "$runner" "$prior" train --arm "$arm"
        done
    elif [[ ! -s "$out/selection.json" ]]; then
        stage select selection.json "${cpu[@]}" --time=00:05:00 "$runner" "$prior" select --device cpu
    elif [[ ! -s "$out/eval64/review.zip" ]]; then
        generate_and_score 64 4 00:20:00
    else
        accepted=$(python -c 'import json,sys; print(" ".join(k for k,v in json.load(open(sys.argv[1]))["acceptance"].items() if v["passed"]))' "$out/eval64/result.json")
        if [[ -z "$accepted" ]]; then
            echo "$(date -Is) No arm passed 64; 500 not submitted" | tee "$out/finished.txt"
            exit 0
        fi
        if [[ -s "$out/eval500/review.zip" ]]; then
            echo "$(date -Is) 500 scored: $accepted" | tee "$out/finished.txt"
            exit 0
        fi
        generate_and_score 500 16 00:45:00
    fi
    sleep 30
done
