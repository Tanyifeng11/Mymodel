"""正式三seed成功结束后，按原seed42触发条件提交消融和数值报告作业。"""
import argparse,subprocess
from tools.e33r_common import *

def submit(arguments):
    value=subprocess.check_output(['sbatch','--parsable']+arguments,text=True).strip().split(';')[0]
    assert value.isdigit();return value

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full-jobs',required=True)
    parser.add_argument('--resume-existing',action='store_true')
    args=parser.parse_args();assert sanity_continuation()['allowed']
    for seed in PROTOCOL['seeds']:
        folder=OUT/'P1_controlled'/('seed%d'%seed)
        complete=read(folder/'phase_complete.json')
        assert complete['steps']==8000 and complete['checkpoint_sha256']==sha(folder/'checkpoint_final.pt')
    path=OUT/'continuation_jobs.json'
    full=read(OUT/'P1_controlled/seed42/dev/summary.json')
    trigger=full['gate_pass'] or full['near_gate']
    if path.exists():
        assert args.resume_existing,'调度记录已存在；先核对作业，再显式恢复缺失提交'
        state=read(path)
        assert state['full_jobs']==args.full_jobs and state['ablation_trigger']==trigger
    else:
        state=dict(full_jobs=args.full_jobs,ablation_trigger=trigger,ablations={},report_job=None,
            note='initial report requires subsequent manual visual review; completion remains false until review')
    write(path,state)
    if trigger:
        for variant in PROTOCOL['ablations']:
            if variant in state['ablations']:continue
            state['ablations'][variant]=submit(['submit/e33r_control.sh','--phase','ablation','--seed','42','--variant',variant])
            write(path,state)
    dependencies=list(state['ablations'].values())
    if state['report_job'] is None:
        state['report_job']=submit((['--dependency=afterok:'+':'.join(dependencies)] if dependencies else [])+['submit/e33r_report.sh'])
    write(path,state);print(state,flush=True)

if __name__=='__main__':main()
