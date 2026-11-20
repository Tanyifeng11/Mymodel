#!/bin/bash
set -euo pipefail
prior="$1"
out="$prior/real_paired_B_on_off_20261118/tail6_real_ab_e1_epoch10_20261119/a_fused25_lstar_ab_20261121/retention_125_eval500_20261121"
cd "$prior/source"
mkdir -p "$out"
exec 9>"$out/queue.lock"
flock -n 9
runner=scripts/e2_f68_e1_retention500.sbatch
num=24
cpu=(--partition=cpu --gres=gpu:0)
stage() {
    local name="$1" marker="$2" job state record
    shift 2
    [[ ! -s "$out/$marker" ]] || return 0
    record="$out/$name.jobs"
    if [[ -s "$record" ]]; then
        job=$(tail -n1 "$record")
        state=$(sacct -X -n -P -j "$job" -o State | head -n1 | cut -d ' ' -f 1)
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
    if job=$(sbatch --parsable -J "retain500_$name" -o "$out/$name-%j.out" "$@" 2>"$out/sbatch_error.log"); then
        echo "${job%%;*}" >> "$record"
        echo "$(date -Is) $name submitted $job"
    elif ! grep -Eqi 'QOSMaxSubmit|MaxSubmitJob|AssocMaxSubmit' "$out/sbatch_error.log"; then
        cat "$out/sbatch_error.log" >&2
        exit 1
    fi
}

while true; do
    if [[ -s "$out/eval500/review.zip" ]]; then
        echo "$(date -Is) 500 and 436 scored" | tee "$out/finished.txt"
        exit 0
    fi
    if [[ ! -s "$out/protocol.json" ]]; then
        stage prepare protocol.json "${cpu[@]}" --time=00:12:00 "$runner" "$prior" prepare
    else
        remaining=0
        for ((shard=0; shard<num; shard++)); do
            marker="generate500_$shard.json"
            [[ ! -s "$out/$marker" ]] || continue
            remaining=$((remaining+1))
            stage "gen_$shard" "$marker" "$runner" "$prior" generate --shard "$shard" --shards "$num"
        done
        if (( remaining == 0 )); then
            stage score eval500/review.zip "${cpu[@]}" --cpus-per-task=12 --time=00:45:00 "$runner" "$prior" score --shards "$num"
        fi
    fi
    sleep 25
done
