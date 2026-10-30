"""计算结束后打包小型审查产物；大权重和完整恢复状态保留服务器。"""
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
from tools.t3_training_diagnosis import read, write, sha

ROOT=Path('/share/home/u2515283058/Mymodel/output_eval/t3_controlled_continue_20261030')
assert read(ROOT/'training/training_complete.json')['updates']==1000
for stage in ['C0250','C0500','C1000']:
    assert read(ROOT/'audit'/('evaluation_'+stage+'.json'))['complete']
jobs=[]
for line in (ROOT/'audit/jobs.csv').read_text().splitlines():
    parts=line.split(',')
    if len(parts)==2 and parts[1].isdigit():jobs.append(parts[1])
account=subprocess.check_output(['sacct','-j',','.join(jobs),'-X','-n','-P',
    '--format=JobID,JobName,Partition,State,ExitCode,ElapsedRaw,AllocTRES,Start,End,NodeList'],text=True)
(ROOT/'audit/resources_sacct.psv').write_text(account)
gpu_seconds=0
import re
for line in account.splitlines():
    a=line.split('|'); match=re.search(r'gres/gpu=(\d+)',a[6])
    if match:gpu_seconds+=int(a[5])*int(match[1])
assert gpu_seconds<=9000, ('GPU_BUDGET_EXCEEDED',gpu_seconds)
write(ROOT/'audit/package_resource_summary.json',dict(gpu_seconds=gpu_seconds,gpu_hours=gpu_seconds/3600,
    hard_cap_seconds=9000,packaging_job=os.environ.get('SLURM_JOB_ID')))
source=Path(__file__).resolve().parents[1]
for relative in ['tools/t3_controlled_continue_report.py','tools/t3_controlled_package.py']:
    shutil.copy2(source/relative,ROOT/'code'/Path(relative).name)
ownlog='package_'+os.environ.get('SLURM_JOB_ID','local')+'.log'
def include(p):
    return p.is_file() and not p.is_symlink() and p.suffix.lower() not in ['.pt','.bin','.pkl','.gz'] and p.name!=ownlog
files=[p for p in sorted(ROOT.rglob('*')) if include(p) and p.name!='bundle_sha256sums.txt']
manifest=ROOT/'audit/bundle_sha256sums.txt'
manifest.write_text(''.join(sha(p)+'  '+p.relative_to(ROOT).as_posix()+'\n' for p in files))
files.append(manifest)
# 临时归档放在结果目录外，防止归档把自身或不断增长的作业日志包含进去。
temporary=ROOT.parent/('t3_controlled_review_'+os.environ.get('SLURM_JOB_ID','local')+'.tar.gz')
with tarfile.open(temporary,'w:gz',compresslevel=3) as archive:
    for p in files:archive.add(p,arcname=p.relative_to(ROOT).as_posix(),recursive=False)
destination=ROOT/'review_results.tar.gz'
temporary.replace(destination)
print('BUNDLE_BYTES',destination.stat().st_size,flush=True)
print('BUNDLE_MEMBERS',len(files),flush=True)
print('BUNDLE_SHA256',sha(destination),flush=True)
print('GPU_SECONDS',gpu_seconds,flush=True)
