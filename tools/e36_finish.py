"""资源与产物归档；结果总结仅在用户本地 docs 编写。"""
import argparse
import re
import subprocess
import tarfile
from tools.e36_protocol import *


def finish(job_ids):
    verify_sources();jobs=[]
    if job_ids:
        accounting=subprocess.check_output(['sacct','-X','-j',','.join(job_ids),'--noheader','--parsable2',
            '--format=JobID,JobName,Partition,State,ElapsedRaw,AllocTRES,ExitCode,NodeList'],text=True)
        (OUT/'audit/slurm_accounting.txt').write_text(accounting,encoding='utf-8')
        keys=['job_id','name','partition','state','seconds','resources','exit_code','nodes']
        jobs=[dict(zip(keys,line.split('|'))) for line in accounting.strip().splitlines()]
    gpu_seconds=0
    for job in jobs:
        found=re.search(r'gres/gpu=(\d+)',job['resources'])
        if found:gpu_seconds+=int(found.group(1))*int(job['seconds'])
    write(OUT/'resource_ledger.json',dict(jobs=jobs,gpu_seconds=gpu_seconds,gpu_hours=gpu_seconds/3600,
        phases={arm:read(OUT/'train'/arm/'complete.json') for arm in ARMS if (OUT/'train'/arm/'complete.json').exists()}))
    sources=list(Path('tools').glob('e36*.py'))+[Path('models/e36_guided_kernel_filter.py'),Path('submit/e36.sh'),Path('tools/e35_freeu_stats.py')]
    write(OUT/'audit/run_manifest.json',dict(commit=commit(),protocol_sha256=sha(OUT/'protocol.json'),
        code={str(p):sha(p) for p in sources},
        artifacts={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()
            and p.suffix not in ['.gz','.tmp','.log','.err'] and p.name!='run_manifest.json'},
        narrative='local docs only',decision=read(OUT/'decision.json')))
    archive=OUT/'e36_review_results.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for p in sorted(OUT.rglob('*')):
            if p.is_file() and p.suffix in ['.json','.csv','.png','.txt','.patch','.log','.err']:
                tar.add(p,arcname=str(p.relative_to(OUT)))
    print('ARCHIVE',archive,flush=True);print('SHA256',sha(archive),flush=True)
    print('DECISION',read(OUT/'decision.json'),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--jobs',nargs='*',default=[]);finish(p.parse_args().jobs)
