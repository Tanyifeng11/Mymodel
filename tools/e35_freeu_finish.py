"""保存资源账本、产物哈希和可下载归档；总结文档只在本地写。"""
import argparse
import re
import subprocess
import tarfile
from tools.e35_freeu_protocol import *

def finish(job_ids):
    verify_sources()
    jobs=[]
    if job_ids:
        text=subprocess.check_output(['sacct','-X','-j',','.join(job_ids),'--noheader','--parsable2',
            '--format=JobID,JobName,Partition,State,ElapsedRaw,AllocTRES,ExitCode,NodeList'],text=True)
        (OUT/'audit/slurm_accounting.txt').write_text(text,encoding='utf-8')
        for line in text.strip().splitlines():
            values=line.split('|');keys=['job_id','name','partition','state','seconds','resources','exit_code','nodes']
            jobs.append(dict(zip(keys,values)))
    gpu_seconds=0
    for job in jobs:
        match=re.search(r'gres/gpu=(\d+)',job['resources'])
        if match:gpu_seconds+=int(match.group(1))*int(job['seconds'])
    write(OUT/'audit/resource_ledger.json',dict(jobs=jobs,gpu_seconds=gpu_seconds,gpu_hours=gpu_seconds/3600))
    sources=list(Path('tools').glob('e35_freeu*.py'))+[Path('submit/e35_freeu.sh')]
    write(OUT/'audit/run_manifest.json',dict(git_commit=commit(),protocol_sha256=sha(OUT/'protocol.json'),
        code={str(p):sha(p) for p in sources},
        artifacts={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()
            and p.suffix not in ['.gz','.tmp','.log','.err'] and p.name!='run_manifest.json'},
        narrative='local docs only',state=read(OUT/'final_decision.json')))
    archive=OUT/'review_results.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for p in sorted(OUT.rglob('*')):
            if p.is_file() and p.suffix in ['.json','.csv','.png','.txt','.patch','.log','.err']:
                tar.add(p,arcname=str(p.relative_to(OUT)))
    print('ARCHIVE',str(archive),flush=True);print('SHA256',sha(archive),flush=True)
    print('STATE',json.dumps(read(OUT/'final_decision.json')),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--jobs',nargs='*',default=[]);args=p.parse_args();finish(args.jobs)
