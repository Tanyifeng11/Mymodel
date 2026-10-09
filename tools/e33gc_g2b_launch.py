"""用累计Slurm分配时长限制新GPU作业；仅支持固定单GPU协议。"""
import argparse,datetime,json,math,os,re,subprocess
from pathlib import Path
OUT=Path('output_eval/e33_gc_g2b_20261009')

def gpu_seconds():
    ids=sorted({re.search(r'job_(\d+)',p.name)[1] for p in OUT.glob('job_*.log')})
    if not ids:return 0
    result=subprocess.run(['sacct','-j',','.join(ids),'-Pn','-o','JobIDRaw,ElapsedRaw,AllocTRES'],capture_output=True,text=True,check=True)
    seconds=0
    for line in result.stdout.splitlines():
        values=line.split('|')
        if len(values)<3 or not values[0].isdigit():continue
        match=re.search(r'(?:^|,)gres/gpu=(\d+)(?:,|$)',values[2])
        if match:seconds+=int(values[1])*int(match[1])
    return seconds

def run():
    p=argparse.ArgumentParser();p.add_argument('module');p.add_argument('arguments',nargs='*');args=p.parse_args()
    pending=subprocess.run(['squeue','-h','-u',os.environ['USER'],'-n','E33GC_G2b_GPU','-o','%i'],capture_output=True,text=True,check=True)
    assert not pending.stdout.strip(),'another G2b GPU job is pending/running'
    used=gpu_seconds();remaining=math.floor((21600-used)/60)
    request=60
    profile=OUT/'G2b_smoke/smoke_audit.json'
    if profile.exists():
        seconds=json.loads(profile.read_text())['full4_identity_update_seconds']
        if args.module.endswith('train_eval'):request=math.ceil((seconds*180+900)*1.3/60)
        elif args.module.endswith('upper'):request=math.ceil((seconds*40+600)*1.3/60)
        elif args.module.endswith('eval'):request=30
    minutes=min(remaining,request)
    assert minutes>=1,'fixed6GPUh budget exhausted'
    env=dict(os.environ,G2B_MODULE=args.module)
    command=['sbatch','--time='+str(minutes),'submit/e33gc_g2b_gpu.sh',*args.arguments]
    submitted=subprocess.run(command,env=env,capture_output=True,text=True,check=True)
    record=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),module=args.module,command=command,
        GPU_seconds_used_before_submission=used,job_id=re.search(r'job (\d+)',submitted.stdout)[1])
    dest=OUT/'protocol/gpu_submissions.jsonl';dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open('a',encoding='utf-8') as f:f.write(json.dumps(record)+'\n')
    print(submitted.stdout,end='',flush=True)
    print('CUMULATIVE_ALLOCATED_GPU_SECONDS',used,'NEW_JOB_MAX_MINUTES',minutes,flush=True)

if __name__=='__main__':run()
