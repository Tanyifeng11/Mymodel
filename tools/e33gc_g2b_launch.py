"""用累计Slurm分配时长限制新GPU作业；仅支持固定单GPU协议。"""
import argparse,math,os,re,subprocess
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
    used=gpu_seconds();minutes=math.floor((21600-used)/60)
    assert minutes>=1,'fixed6GPUh budget exhausted'
    env=dict(os.environ,G2B_MODULE=args.module)
    subprocess.run(['sbatch','--time='+str(minutes),'submit/e33gc_g2b_gpu.sh',*args.arguments],env=env,check=True)
    print('CUMULATIVE_ALLOCATED_GPU_SECONDS',used,'NEW_JOB_MAX_MINUTES',minutes,flush=True)

if __name__=='__main__':run()
