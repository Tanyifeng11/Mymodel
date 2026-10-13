"""e47 产物与真实 Slurm 资源归档；叙述总结由本地 docs 承载。"""
import argparse
import re
import subprocess
import tarfile
from pathlib import Path
from tools.e47_protocol import *


def finish(jobs):
    accounting=subprocess.check_output(['sacct','-X','-j',','.join(jobs),'--noheader','--parsable2',
        '--format=JobID,JobName,Partition,State,ElapsedRaw,AllocTRES,ExitCode,NodeList'],text=True)
    (OUT/'slurm_accounting.txt').write_text(accounting,encoding='utf-8')
    keys=['job_id','name','partition','state','seconds','resources','exit_code','nodes']
    rows=[dict(zip(keys,line.split('|'))) for line in accounting.strip().splitlines()]
    gpu_seconds=sum(int(re.search(r'gres/gpu=(\d+)',r['resources']).group(1))*int(r['seconds'])
        for r in rows if re.search(r'gres/gpu=(\d+)',r['resources']))
    write(OUT/'resource_ledger.json',dict(jobs=rows,gpu_hours=gpu_seconds/3600,
        cpu_seconds=sum(int(r['seconds'])*int(re.search(r'cpu=(\d+)',r['resources']).group(1))
            for r in rows if re.search(r'cpu=(\d+)',r['resources'])),
        queue_time_included=False,smoke_updates='temporary gradient checks; candidate training starts from zero'))
    frozen=read(OUT/'manifests/input_hashes.json')
    changed=[p for p,h in frozen.items() if sha(p)!=h]
    assert not changed,changed
    assert sha(E5)==E5_SHA and sha(B1)==B1_SHA
    expected=read(OUT/'checkpoint_hashes.json')[str(E2B)]['sha256'];assert sha(E2B)==expected
    write(OUT/'manifests/final_frozen_check.json',dict(unchanged=True,files=len(frozen),historical_checkpoints_unchanged=True))
    code=list(Path('tools').glob('e47*.py'))+list(Path('submit').glob('e47*.sh'))
    write(OUT/'run_manifest.json',dict(commit=commit(),code={str(p):sha(p) for p in code},
        files={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()
            and p.suffix not in ['.gz','.tmp'] and p.name!='run_manifest.json'},decision=read(OUT/'decision.json'),
        raw_tensors='remain on server; archive contains hashes, PNG and audit results'))
    archive=OUT/'e47_review_results.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for p in sorted(OUT.rglob('*')):
            if p.is_file() and p.suffix in ['.json','.csv','.png','.txt','.log','.err','.py']:
                tar.add(p,arcname=str(p.relative_to(OUT)))
    print('ARCHIVE',str(archive),'SHA256',sha(archive),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--jobs',nargs='+',required=True)
    finish(parser.parse_args().jobs)
