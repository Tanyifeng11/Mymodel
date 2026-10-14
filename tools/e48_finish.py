"""保存真实作业账本与逐文件哈希，打包可在本地复核的结果；不创建报告。"""
import argparse
import re
import subprocess
import tarfile
from tools.e48_protocol import *


def finish(phase,jobs):
    assert read(OUT/'probe/evaluation_complete.json')['complete']
    if phase=='final':assert read(OUT/'decision.json').get('final_decision')
    assert len(jobs)==len(set(jobs))
    fields='JobIDRaw,JobName,State,ExitCode,ElapsedRaw,AllocTRES,TimelimitRaw,NodeList,MaxRSS'
    command=['sacct','-j',','.join(map(str,jobs)),'-P','--format='+fields]
    text=subprocess.check_output(command,text=True)
    (OUT/('slurm_'+phase+'.txt')).write_text(text,encoding='utf-8')
    lines=text.strip().splitlines();keys=lines[0].split('|');ledger=[]
    for line in lines[1:]:
        row=dict(zip(keys,line.split('|')))
        if '.' in row['JobIDRaw']:continue
        tres=row['AllocTRES'];gpu=re.search(r'(?:^|,)gres/gpu=(\d+)(?:,|$)',tres)
        cpu=re.search(r'(?:^|,)cpu=(\d+)(?:,|$)',tres)
        seconds=int(row['ElapsedRaw']);row.update(gpu_hours=seconds*int(gpu.group(1))/3600 if gpu else 0.,
            cpu_hours=seconds*int(cpu.group(1))/3600 if cpu else 0.,maxrss_available=bool(row['MaxRSS']))
        ledger.append(row)
    assert {int(r['JobIDRaw']) for r in ledger}==set(jobs)
    assert all(r['State'] not in ['RUNNING','PENDING','COMPLETING'] for r in ledger)
    write(OUT/('resource_ledger_'+phase+'.json'),dict(jobs=ledger,
        gpu_hours=sum(r['gpu_hours'] for r in ledger),cpu_hours=sum(r['cpu_hours'] for r in ledger),
        failed_jobs_included=True,missing_maxrss='unavailable, not zero'))
    inputs=read(OUT/'manifests/input_hashes.json')
    assert all(sha(p)==expected for p,expected in inputs.items())
    code=[p for pattern in ['tools/e48*.py','models/e48*.py','submit/e48*.sh'] for p in Path('.').glob(pattern)]
    code += [Path(p) for p in ['tools/e44_runtime.py','tools/e44_train.py','tools/e44_metrics.py',
        'tools/e47_runtime.py','models/reference_attention_variants.py','models/reference_feature_writer.py',
        'garment_mask_utils.py','eval/eval_utils.py']]
    files=[]
    for p in OUT.rglob('*'):
        if not p.is_file() or p.name.endswith(('.tar.gz','.sha256','.tmp')) or p.name.startswith('run_manifest_'):continue
        relative=p.relative_to(OUT)
        if phase=='probe':
            if len(relative.parts)>1 and relative.parts[0] not in ['audit','probe','calibration','splits','manifests']:continue
            if p.suffix in ['.log','.err'] and p.stem not in ['gpu_'+str(j) for j in jobs]:continue
        files.append(p)
    hashes={str(p.relative_to(OUT)):sha(p) for p in sorted(files)}
    manifest=OUT/('run_manifest_'+phase+'.json')
    write(manifest,dict(experiment='e48',phase=phase,commit=commit(),input_hashes_unchanged=True,
        code_sha256={str(p):sha(p) for p in sorted(code)},files=hashes,
        raw_tensor_files='hashed on server; excluded from review archive, retained on server',
        decision=read(OUT/'decision.json')))
    archive=OUT/('e48_'+phase+'_review.tar.gz')
    with tarfile.open(archive,'w:gz') as tar:
        for p in sorted(files)+[manifest]:
            if p.suffix!='.pt':tar.add(p,arcname=str(p.relative_to(OUT)),recursive=False)
    digest=sha(archive);archive.with_name(archive.name+'.sha256').write_text(digest+'  '+archive.name+'\n',encoding='utf-8')
    print('E48 REVIEW',str(archive),archive.stat().st_size,digest,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['probe','final'],default='probe')
    parser.add_argument('--jobs',nargs='+',type=int,required=True);args=parser.parse_args()
    finish(args.phase,args.jobs)
