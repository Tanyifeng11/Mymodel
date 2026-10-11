"""冻结已结束的E41作业账本，再索引与打包，避免账本更新使哈希过期。"""
import subprocess
import tarfile
from tools.e41_protocol import *


def run():
    assert (OUT/'E41_attribute_trajectory.json').exists()
    ids=(OUT/'jobs.txt').read_text().split()
    assert len(ids)>=4 and all(x.isdigit() for x in ids)
    accounting=subprocess.check_output(['sacct','-j',','.join(ids),'-X','-P','-o',
        'JobID,JobName,State,ElapsedRaw,AllocTRES,ExitCode,NodeList,Start,End'],text=True)
    (OUT/'slurm_final_accounting.txt').write_text(accounting,encoding='utf-8')
    write(OUT/'audit/finalization.json',dict(complete=True,jobs=ids,new_generations=0,
        note='Final accounting and manifest collected after analysis; packing job itself is separately tracked locally.'))
    files={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file()
        and p.name!='artifact_hashes.json' and not any(s in p.name for s in ['.tar.gz','job_'])}
    write(OUT/'audit/artifact_hashes.json',files)
    archive=OUT/'e41_review_results.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for path in OUT.rglob('*'):
            if path.is_file() and not path.name.endswith('.npz') and not any(s in path.name for s in ['.tar.gz','job_']):
                tar.add(path,arcname=str(path.relative_to(OUT)),recursive=False)
    archive.with_suffix(archive.suffix+'.sha256').write_text(sha(archive)+'  '+str(archive)+'\n',encoding='utf-8')
    print('E41 FINALIZED',len(files),flush=True)


if __name__=='__main__':run()
