"""作业完成后归档真实产物与 Slurm 账本；归档不包含训练权重。"""
import subprocess
import tarfile
from tools.e37_protocol import *


def run(job_ids):
    ids=','.join(job_ids)
    accounting=subprocess.check_output(['sacct','-j',ids,'--parsable2','--noheader',
        '--format=JobIDRaw,State,ExitCode,ElapsedRaw,AllocTRES,NodeList,MaxRSS'],text=True)
    write(OUT/'resource_ledger.json',dict(job_ids=job_ids,slurm_accounting=accounting,
        runtime=read(OUT/'resource_runtime.json') if (OUT/'resource_runtime.json').exists() else None,
        decision=read(OUT/'decision.json'),commit=commit()))
    # 先固化所有产物 SHA，再打包；自身与活跃归档日志不加入校验表。
    files=[p for p in sorted(OUT.rglob('*')) if p.is_file() and p.suffix not in ['.pt','.gz','.sha256']
        and p.name!='artifact_hashes.json' and not p.name.startswith('archive_')]
    write(OUT/'artifact_hashes.json',{str(p.relative_to(OUT)):sha(p) for p in files})
    archive=OUT/'e37_review_results.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for p in files+[OUT/'artifact_hashes.json']:tar.add(p,arcname=str(p.relative_to(OUT)))
    digest=sha(archive)
    (OUT/'e37_review_results.tar.gz.sha256').write_text(digest+'  '+archive.name+'\n')
    print('E37_ARCHIVE_SHA256='+digest,flush=True)


if __name__=='__main__':
    import sys
    run(sys.argv[1:])
