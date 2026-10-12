"""结果归档，不在服务器生成叙述报告；本地 docs 独立交付。"""
import argparse
import re
import subprocess
import tarfile
from tools.e44_protocol import *


def finish(jobs):
    accounting=subprocess.check_output(['sacct','-X','-j',','.join(jobs),'--noheader','--parsable2',
        '--format=JobID,JobName,Partition,State,ElapsedRaw,AllocTRES,ExitCode,NodeList'],text=True)
    (OUT/'slurm_accounting.txt').write_text(accounting,encoding='utf-8')
    keys=['job_id','name','partition','state','seconds','resources','exit_code','nodes']
    rows=[dict(zip(keys,line.split('|'))) for line in accounting.strip().splitlines()]
    gpu_seconds=sum(int(re.search(r'gres/gpu=(\d+)',r['resources']).group(1))*int(r['seconds'])
        for r in rows if re.search(r'gres/gpu=(\d+)',r['resources']))
    write(OUT/'resource_ledger.json',dict(jobs=rows,gpu_hours=gpu_seconds/3600))
    verify_inputs()
    code=list(Path('tools').glob('e44*.py'))+list(Path('models').glob('reference_*.py'))+list(Path('submit').glob('e44*.sh'))
    write(OUT/'run_manifest.json',dict(commit=commit(),code={str(p):sha(p) for p in code},
        files={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()
            and p.suffix not in ['.gz','.tmp','.log','.err'] and p.name!='run_manifest.json'},decision=read(OUT/'decision.json')))
    archive=OUT/'e44_review_results.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for p in sorted(OUT.rglob('*')):
            if p.is_file() and p.suffix in ['.json','.csv','.png','.txt','.log','.err']:
                tar.add(p,arcname=str(p.relative_to(OUT)))
    print('ARCHIVE',str(archive),'SHA256',sha(archive),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--jobs',nargs='+',required=True)
    finish(parser.parse_args().jobs)
