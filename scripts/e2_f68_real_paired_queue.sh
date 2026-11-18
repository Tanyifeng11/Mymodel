#!/bin/bash
# 分片受账户提交上限约束时接续提交，门户退出后由 Slurm 继续运行。
set -euo pipefail

prior="$1"
out="$prior/real_paired_B_on_off_20261118"
cd "$prior/source"
test -s "$out/protocol.json"
exec 9>"$out/queue.lock"
flock -n 9
touch "$out/generation_jobids.txt"

job_state() {
    sacct -X -n -P -j "$1" -o State | head -n 1 | cut -d ' ' -f 1
}

say() {
    printf '%s %s\n' "$(date -Is)" "$*"
}

submit() {
    local job
    if job=$(sbatch --parsable "$@" 2>"$out/submit_error.log"); then
        printf '%s\n' "${job%%;*}"
    else
        cat "$out/submit_error.log" >&2
        return 1
    fi
}

wait_for_slot() {
    local jobs
    jobs=$(squeue --me -h -o '%i') || return 1
    [[ $(sed '/^$/d' <<< "$jobs" | wc -l) -lt 4 ]]
}

say '接续剩余分片并等待完整指标'
while (( SECONDS < 172800 )); do
    ready=0
    for start in 0 125 250 375; do
        end=$((start+125))
        if [[ -s "$out/generated_${start}_${end}.json" ]]; then
            ready=$((ready+1))
            continue
        fi
        read -r count job < <(awk -v a="$start" -v b="$end" '$1==a && $2==b {n++; j=$3} END {print n+0, j}' "$out/generation_jobids.txt")
        if [[ -n "$job" ]]; then
            case "$(job_state "$job")" in
                TIMEOUT|NODE_FAIL|PREEMPTED)
                    if (( count >= 2 )); then say "分片 $start $end 再次中断，作业 $job"; exit 1; fi ;;
                FAILED|OUT_OF_MEMORY|CANCELLED|BOOT_FAIL|COMPLETED)
                    say "分片 $start $end 缺少完成记录，检查作业 $job"
                    exit 1 ;;
                *) continue ;;
            esac
        fi
        if wait_for_slot && job=$(submit -J "f68_pair_$start" -o "$out/gen_${start}_${end}-%j.out" \
                scripts/e2_f68_real_paired.sbatch "$prior" generate --start "$start" --end "$end"); then
            printf '%s %s %s\n' "$start" "$end" "$job" >> "$out/generation_jobids.txt"
            say "生成分片 $start $end 已提交 $job"
        fi
    done
    if (( ready == 4 )); then
        if [[ -s "$out/review.zip" ]]; then say '完整评测归档已生成'; exit 0; fi
        if [[ -s "$out/evaluate_jobid.txt" ]]; then
            job=$(cat "$out/evaluate_jobid.txt")
            case "$(job_state "$job")" in
                FAILED|OUT_OF_MEMORY|CANCELLED|NODE_FAIL|TIMEOUT|COMPLETED)
                    [[ ! -s "$out/review.zip" ]] || exit 0
                    say "评测未归档，检查作业 $job"; exit 1 ;;
            esac
        elif wait_for_slot && job=$(submit --partition=cpu --gres=gpu:0 --cpus-per-task=4 --mem=16G \
                --time=00:40:00 -J f68_pair_score -o "$out/evaluate-%j.out" \
                scripts/e2_f68_real_paired.sbatch "$prior" evaluate --device cpu); then
            printf '%s\n' "$job" > "$out/evaluate_jobid.txt"
            say "整图评测已提交 $job"
        fi
    fi
    sleep 60
done
say '接续等待达到 48 小时上限，已提交作业由 Slurm 继续处理'
exit 2
