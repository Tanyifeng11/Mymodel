#!/bin/bash
# 复用已提交及已完成分片，完整后由 CPU 作业统一评分。
set -euo pipefail
prior="$1"
out="$prior/real_paired_B_on_off_20261118/single_s0_gate_20261119"
cd "$prior/source"
mkdir -p "$out"
exec 9>"$out/queue.lock"
flock -n 9

runner=scripts/e2_f68_single_s0_gate.sbatch

submit() {
    local record="$1"
    shift
    local job
    if job=$(sbatch --parsable "$@" 2>"$out/sbatch_error.log"); then
        echo "${job%%;*}" >> "$record"
        echo "$(date -Is) submitted $job"
    elif grep -Eqi 'QOSMaxSubmit|MaxSubmitJob' "$out/sbatch_error.log"; then
        return 0
    else
        cat "$out/sbatch_error.log" >&2
        return 1
    fi
}

state() {
    sacct -X -n -P -j "$1" -o State | head -n 1 | cut -d ' ' -f 1
}

while (( SECONDS < 172800 )); do
    [[ ! -s "$out/review.zip" ]] || exit 0
    for shard in 0 1; do
        [[ ! -s "$out/shard$shard.json" ]] || continue
        record="$out/gen_jobs$shard.txt"
        extra=()
        if [[ -s "$record" ]]; then
            job=$(tail -n 1 "$record")
            case "$(state "$job")" in
                TIMEOUT|NODE_FAIL|PREEMPTED)
                    if [[ $(wc -l < "$record") -ge 2 ]]; then
                        echo "Shard $shard stopped after retry, see job $job"
                        exit 1
                    fi
                    extra=(--time=00:32:00)
                    ;;
                COMPLETED|FAILED|OUT_OF_MEMORY|CANCELLED|BOOT_FAIL)
                    echo "Shard $shard has no completion record, see job $job"
                    exit 1
                    ;;
                *) continue ;;
            esac
        fi
        submit "$record" "${extra[@]}" -J "f68_s0g$shard" -o "$out/gen-$shard-%j.out" "$runner" "$prior" generate --shard "$shard"
    done
    if [[ -s "$out/shard0.json" && -s "$out/shard1.json" ]]; then
        if [[ -s "$out/eval_job.txt" ]]; then
            job=$(cat "$out/eval_job.txt")
            case "$(state "$job")" in
                COMPLETED|FAILED|OUT_OF_MEMORY|CANCELLED|TIMEOUT|NODE_FAIL)
                    [[ ! -s "$out/review.zip" ]] || exit 0
                    echo "Evaluation has no archive, see job $job"
                    exit 1
                    ;;
            esac
        else
            submit "$out/eval_job.txt" --partition=cpu --gres=gpu:0 --cpus-per-task=4 --mem=16G \
                --time=00:15:00 -J f68_s0g_score -o "$out/eval-%j.out" "$runner" "$prior" evaluate
        fi
    fi
    sleep 60
done
echo '48-hour queue monitor finished; submitted jobs remain in Slurm.'
exit 2
