#!/bin/bash
# 按依赖提交教师分片、仅头训练及两级验收。重复进入时保留已完成阶段。
set -euo pipefail
prior="$1"
out="$prior/real_paired_B_on_off_20261118/rgb_distill_a025_20261119"
cd "$prior/source"
mkdir -p "$out"
exec 9>"$out/queue.lock"
flock -n 9
export PATH=/share/home/u2515283058/.conda/envs/Mymodel/bin:$PATH
runner=scripts/e2_f68_rgb_distill.sbatch
cpu=(--partition=cpu --gres=gpu:0 --mem=24G)

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
                    echo "$(date -Is) $name exhausted retry at $job"
                    exit 1
                fi
                ;;
            COMPLETED|FAILED|OUT_OF_MEMORY|CANCELLED|BOOT_FAIL)
                echo "$(date -Is) $name $job $state without expected $marker"
                exit 1
                ;;
            *) return 0 ;;
        esac
    fi
    if job=$(sbatch --parsable -J "f68_rgb_$name" -o "$out/$name-%j.out" "$@" 2>"$out/sbatch_error.log"); then
        echo "${job%%;*}" >> "$record"
        echo "$(date -Is) $name submitted $job"
    elif ! grep -Eqi 'QOSMaxSubmit|MaxSubmitJob' "$out/sbatch_error.log"; then
        cat "$out/sbatch_error.log" >&2
        exit 1
    fi
}

while (( SECONDS < 172800 )); do
    if [[ ! -s "$out/protocol.json" ]]; then
        stage prep protocol.json "${cpu[@]}" --time=00:12:00 "$runner" "$prior" prepare --device cpu
    elif [[ ! -s "$out/generated_0_80.json" || ! -s "$out/generated_80_160.json" ]]; then
        stage gen0 generated_0_80.json "$runner" "$prior" generate --start 0 --end 80
        stage gen1 generated_80_160.json "$runner" "$prior" generate --start 80 --end 160
    elif [[ ! -s "$out/training.json" ]]; then
        stage train training.json "$runner" "$prior" train
    elif [[ ! -s "$out/eval64/review.zip" ]]; then
        stage score64 eval64/review.zip "${cpu[@]}" --time=00:18:00 "$runner" "$prior" evaluate64 --device cpu
    else
        accepted=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["acceptance"]["passed"])' "$out/eval64/result.json")
        if [[ "$accepted" != True ]]; then
            echo "$(date -Is) 64-pair gate failed. Full benchmark skipped."
            exit 0
        fi
        if [[ -s "$out/eval500/review.zip" ]]; then
            echo "$(date -Is) 500-pair scoring complete."
            exit 0
        fi
        stage score500 eval500/review.zip "${cpu[@]}" --cpus-per-task=8 --time=00:30:00 "$runner" "$prior" evaluate500 --device cpu
    fi
    sleep 30
done
echo 'Queue monitor reached 48 hours.'
exit 2
