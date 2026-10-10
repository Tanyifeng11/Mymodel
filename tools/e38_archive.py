"""只归档 E38 输入副本和原始产物，不上传本地报告。"""
import sys
import tarfile
from tools.e38_protocol import *


def run(ids):
    accounting=subprocess.check_output(['sacct','-j',','.join(ids),'--noheader','--parsable2',
        '--format=JobIDRaw,State,ExitCode,ElapsedRaw,AllocTRES,NodeList,MaxRSS'],text=True)
    write(OUT/'resource_ledger.json',dict(job_ids=ids,slurm_accounting=accounting,
          runtime=read(OUT/'resource_runtime.json') if (OUT/'resource_runtime.json').exists() else None,commit=commit()))
    files=[p for p in sorted(OUT.rglob('*')) if p.is_file() and p.suffix not in ['.gz','.sha256']
           and p.name!='artifact_hashes.json' and not p.name.startswith('archive_')]
    write(OUT/'artifact_hashes.json',{str(p.relative_to(OUT)):sha(p) for p in files})
    archive=OUT/'e38_review_results.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for p in files+[OUT/'artifact_hashes.json']:tar.add(p,arcname=str(p.relative_to(OUT)))
    digest=sha(archive);(OUT/(archive.name+'.sha256')).write_text(digest+'  '+archive.name+'\n')
    print('E38_ARCHIVE_SHA256='+digest,flush=True)


if __name__=='__main__':run(sys.argv[1:])
