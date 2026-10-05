"""登录节点轻量调度：每阶段只提交两个固定分片，成功后汇总。"""
import argparse
import subprocess
import time
from tools.e33tm_protocol import OUT, read, write


def command(args):
    return subprocess.check_output(args,text=True).strip()


def submit(args):
    job = command(['sbatch','--parsable']+args).split(';')[0]
    assert job.isdigit(), job
    print('[submitted]',job,args,flush=True)
    return job


def wait(job):
    while command(['squeue','-h','-j',job,'-o','%T']): time.sleep(15)
    # 等待调度系统的作业状态入账，失败时停止调度，保留结果供检查。
    result = ''
    while not result:
        result=command(['sacct','-n','-X','-j',job,'--format=State,ExitCode','--parsable2'])
        if not result: time.sleep(5)
    assert result.split('|')[:2]==['COMPLETED','0:0'], (job,result)
    print('[completed]',job,result,flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--wait-for',required=True)
    args=parser.parse_args()
    history=[]
    wait(args.wait_for)
    for work in ('robustness','ablations','diagnostics'):
        if read(OUT/'decision_summary.json').get('hard_stop'): break
        jobs=[submit(['submit/e33tm_generate.sh','worker','--work',work,'--rf-seed','42',
                      '--shard-index',str(i),'--shard-count','2']) for i in range(2)]
        history.append(dict(work=work,worker_jobs=jobs,completed=False))
        write(OUT/'scheduler_state.json',dict(phases=history,complete=False))
        for job in jobs: wait(job)
        aggregate=submit(['-J','E33TMagg','--dependency=afterok:'+':'.join(jobs),
            'submit/e33tm_generate.sh','aggregate','--work',work,'--rf-seed','42','--shard-count','2'])
        history[-1]['aggregate_job']=aggregate
        write(OUT/'scheduler_state.json',dict(phases=history,complete=False))
        wait(aggregate)
        history[-1]['completed']=True
        write(OUT/'scheduler_state.json',dict(phases=history,complete=False))
    final=submit(['submit/e33tm_finalize.sh'])
    write(OUT/'scheduler_state.json',dict(phases=history,final_job=final,complete=False))
    wait(final)
    assert read(OUT/'completion_check.json')['experiment_execution_complete']
    write(OUT/'scheduler_state.json',dict(phases=history,final_job=final,complete=True))
    print('[E33TM schedule complete]',flush=True)


if __name__=='__main__': main()
