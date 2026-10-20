"""单次、幂等的限额调度；科学任务仍由冻结入口和 Slurm 执行。"""
import csv
import fcntl
import subprocess
from tools.m12_gain_confirm import RUN,ARMS,read,write,sha,require,submit_job,accounting,utc


def advance():
    require()
    if (RUN/'resources/server_ready.json').exists():
        print('SERVER_COMPLETE_LOCAL_REVIEW_PENDING');return
    approval=RUN/'audit/input_review_approval.json'
    if not approval.exists():
        print('WAIT_INPUT_REVIEW');return
    review=read(approval)
    assert review['passed'] and review['identities_sha256']==sha(RUN/'protocol/identities.json')
    jobs=read(RUN/'resources/jobs.json')
    raw=subprocess.check_output(['sacct','-X','-P','-j',','.join(str(j['id']) for j in jobs),
        '--format=JobIDRaw,State,ExitCode'],text=True)
    states={int(r['JobIDRaw']):r['State'].split()[0] for r in csv.DictReader(raw.splitlines(),delimiter='|') if r['JobIDRaw'].isdigit()}
    by_action={j['action']:j for j in jobs}
    recovery_file=RUN/'resources/recovered_job_failures.json'
    recovered={r['failed_job_id'] for r in read(recovery_file)} if recovery_file.exists() else set()
    failed=[dict(job=j,state=states.get(j['id'])) for j in jobs if j['id'] not in recovered and states.get(j['id']) in ['FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED']]
    if failed:
        write(RUN/'resources/dispatch_attention.json',dict(failed=failed,utc=utc()))
        print('ENGINEERING_ATTENTION_REQUIRED',failed);return
    queue=subprocess.check_output(['squeue','-h','-u','u2515283058','-o','%i|%q|%T'],text=True)
    gpu_queued=[line for line in queue.splitlines() if len(line.split('|'))==3 and line.split('|')[1].startswith('gpu')]
    slots=max(0,4-len(gpu_queued))
    smoke_action='smoke_r1'
    if smoke_action not in by_action:
        if slots:submit_job(smoke_action,'submit/m12_confirm_gpu.sh',arguments=['smoke'])
        else:print('WAIT_QOS_GPU_SUBMISSION_SLOT')
        return
    smoke=by_action[smoke_action]
    if states.get(smoke['id'])!='COMPLETED':print('WAIT_SMOKE',smoke['id'],states.get(smoke['id']));return
    assert read(RUN/'smoke/complete.json')['passed']
    assert read(RUN/'smoke/complete.json')['commit']==read(RUN/'protocol/protocol.json')['commit']
    active=[]
    for arm in ARMS:
        action='generate_'+arm
        if action in by_action:
            job=by_action[action]
            if states.get(job['id'])!='COMPLETED':
                active.append(job['id'])
                # R0输入审核期间已hold的E5，在R1烟测和输入审核通过后才放行。
                if arm=='E5' and any(line.startswith(str(job['id'])+'|') and line.endswith('|PENDING') for line in queue.splitlines()):
                    detail=subprocess.check_output(['scontrol','show','job',str(job['id']),'-o'],text=True)
                    if 'Reason=JobHeldUser' in detail:
                        subprocess.check_call(['scontrol','update','JobId='+str(job['id']),'Dependency=afterok:'+str(smoke['id'])])
                        subprocess.check_call(['scontrol','release',str(job['id'])])
                        print('RELEASED_REVIEWED_E5',job['id'])
            else:assert read(RUN/'generation'/(arm+'_complete.json'))['n']==288
        elif slots:
            submit_job(action,'submit/m12_confirm_gpu.sh',['--dependency=afterok:'+str(smoke['id'])],['generate','--arm',arm]);slots-=1;active.append(action)
        else:active.append(action)
    if active:print('WAIT_GENERATION',active);return
    if 'evaluate' not in by_action:
        if slots:submit_job('evaluate','submit/m12_confirm_gpu.sh')
        else:print('WAIT_QOS_EVALUATION_SLOT')
        return
    if states.get(by_action['evaluate']['id'])!='COMPLETED':print('WAIT_EVALUATION');return
    assert read(RUN/'metrics/complete.json')['complete']
    for action,marker in [('cpu_lpips','precision/complete.json'),('finalize','audit/finalized.json')]:
        if action not in by_action:
            submit_job(action,'submit/m12_confirm_cpu.sh');return
        if states.get(by_action[action]['id'])!='COMPLETED':print('WAIT',action);return
        assert read(RUN/marker)['complete']
    accounting()
    write(RUN/'resources/server_ready.json',dict(complete=True,utc=utc(),science_images=1728))
    print('SERVER_COMPLETE_LOCAL_REVIEW_PENDING')


if __name__=='__main__':
    with (RUN/'resources/dispatch.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        advance()
