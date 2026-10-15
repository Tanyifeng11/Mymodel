"""保存全部真实作业与可下载审阅包；张量留服务器，报告只在本地。"""
import argparse
import re
import subprocess
import tarfile
from tools.e51_protocol import *


def finish(phase,jobs):
    if phase=='audit':assert read(OUT/'audit/complete.json')['passed']
    else:assert read(OUT/'decision.json').get('final_decision') in ['NO_GO','PROMISING_REQUIRES_CONFIRMATION']
    assert len(jobs)==len(set(jobs))
    fields='JobIDRaw,JobName,State,ExitCode,ElapsedRaw,AllocTRES,TimelimitRaw,NodeList,MaxRSS'
    result=subprocess.check_output(['sacct','-j',','.join(map(str,jobs)),'-P','--format='+fields],text=True)
    (OUT/('slurm_'+phase+'.txt')).write_text(result,encoding='utf-8')
    lines=result.strip().splitlines();keys=lines[0].split('|');ledger=[]
    for line in lines[1:]:
        row=dict(zip(keys,line.split('|')))
        if '.' in row['JobIDRaw']:continue
        tres=row['AllocTRES'];gpu=re.search(r'(?:^|,)gres/gpu=(\d+)(?:,|$)',tres)
        cpu=re.search(r'(?:^|,)cpu=(\d+)(?:,|$)',tres);seconds=int(row['ElapsedRaw'])
        row.update(gpu_hours=seconds*int(gpu.group(1))/3600 if gpu else 0.,
            cpu_hours=seconds*int(cpu.group(1))/3600 if cpu else 0.)
        ledger.append(row)
    assert {int(r['JobIDRaw']) for r in ledger}==set(jobs)
    assert all(r['State'] not in ['RUNNING','PENDING','COMPLETING'] for r in ledger)
    write(OUT/('resource_ledger_'+phase+'.json'),dict(jobs=ledger,
        gpu_hours=sum(r['gpu_hours'] for r in ledger),cpu_hours=sum(r['cpu_hours'] for r in ledger),
        failures_included=True,missing_maxrss='unavailable, not zero'))
    hashes=read(OUT/'manifests/input_hashes.json');assert all(sha(p)==h for p,h in hashes.items())
    code=[p for pattern in ['tools/e51*.py','models/e51*.py','submit/e51*.sh','tests/test_e51*.py'] for p in Path('.').glob(pattern)]
    code +=[Path(p) for p in ['tools/e48_eval.py','tools/e44_runtime.py','tools/e47_runtime.py',
        'models/reference_attention_variants.py','models/reference_feature_writer.py','adapter/attention_processor.py',
        'garment_mask_utils.py','tools/e44_metrics.py','eval/eval_utils.py','tools/e49_protocol.py','tools/e44_train.py','tools/e44_b1_mask.py','tools/e47_protocol.py','tools/e35_freeu_protocol.py']]
    files=[p for p in OUT.rglob('*') if p.is_file() and not p.name.endswith(('.tar.gz','.sha256','.tmp'))
        and not p.name.startswith('run_manifest_')]
    manifest=OUT/('run_manifest_'+phase+'.json')
    write(manifest,dict(experiment='e51',phase=phase,commit=commit(),input_hashes_unchanged=True,
        code_sha256={str(p):sha(p) for p in sorted(code)},files={str(p.relative_to(OUT)):sha(p) for p in sorted(files)},
        tensors='hashed, retained on server, excluded from review archive',decision=read(OUT/'decision.json')))
    archive=OUT/('e51_'+phase+'_review.tar.gz')
    with tarfile.open(archive,'w:gz') as tar:
        for p in sorted(files)+[manifest]:
            if p.suffix!='.pt':tar.add(p,arcname=str(p.relative_to(OUT)),recursive=False)
        for p in code:tar.add(p,arcname='code/'+str(p),recursive=False)
    digest=sha(archive)
    archive.with_name(archive.name+'.sha256').write_text(digest+'  '+archive.name+'\n',encoding='utf-8')
    print('E51 REVIEW',archive,archive.stat().st_size,digest,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['audit','final'],required=True)
    p.add_argument('--jobs',nargs='+',type=int,required=True);args=p.parse_args();finish(args.phase,args.jobs)
