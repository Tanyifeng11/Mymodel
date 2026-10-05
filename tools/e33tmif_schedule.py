"""登录节点串联已授权实验；每次至多两GPU作业，失败即停止下游。"""
import argparse
from tools.e33tm_schedule import command,submit,wait
from tools.e33tmif_protocol import *

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--wait-for');args=parser.parse_args()
    if args.wait_for:wait(args.wait_for)
    history=[]
    def phase(work,seed=42,shards=2,aggregate=None):
        jobs=[submit(['submit/e33tmif.sh',work,'--rf-seed',str(seed),'--shard-index',str(i),
                      '--shard-count',str(shards)]) for i in range(shards)]
        history.append(dict(work=work,seed=seed,jobs=jobs,complete=False))
        write(OUT/'scheduler_state.json',dict(phases=history,complete=False))
        for job in jobs:wait(job)
        if aggregate:
            job=submit(['-J','E33TMIFagg','submit/e33tmif.sh','aggregate',aggregate]);wait(job)
            history[-1]['aggregate_job']=job
        history[-1]['complete']=True;write(OUT/'scheduler_state.json',dict(phases=history,complete=False))
    assert read(OUT/'visual_audit/smoke_review.json')['pass']
    for seed in SEEDS:
        phase('fields',seed,1)
        phase('reproduce',seed)
    job=submit(['submit/e33tmif.sh','aggregate','reproduce']);wait(job)
    job=submit(['submit/e33tmif.sh','aggregate','localize']);wait(job)
    phase('conditions',aggregate='conditions')
    phase('strengths',aggregate='strengths')
    candidate=read(OUT/'strength_grid/candidate.json')
    if candidate['strength'] is not None:
        phase('fields',42,1)
        phase('confirmation',aggregate='confirmation')
    if read(OUT/'stage_survival/localization.json')['field_to_rgb_bottleneck']:
        phase('purity',aggregate='purity')
    pair=read(OUT/'stage_survival/localization.json')['stage_pair']
    if not pair or not all(s in ('S0','S1','S2','S4') for s in pair):
        for seed in (43,44):phase('seed_confirmation',seed)
    job=submit(['submit/e33tmif.sh','aggregate','seed_confirmation']);wait(job)
    job=submit(['submit/e33tmif.sh','visual']);wait(job)
    job=submit(['submit/e33tmif.sh','finalize']);wait(job)
    assert read(OUT/'completion_check.json')['experiment_execution_complete']
    write(OUT/'scheduler_state.json',dict(phases=history,final_job=job,complete=True))
    print('[IF schedule complete]',flush=True)

if __name__=='__main__':main()
