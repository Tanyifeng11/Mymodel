"""计算作业结束后冻结账本与产物索引；不重新计算实验指标。"""
import shutil
import subprocess
from tools.e40_protocol import OUT, read, write, sha


def run():
    jobs='116775,116776,116777,116778,116779,116780,116781,116783'
    text=subprocess.check_output(['sacct','-j',jobs,'-X','-P','-o',
        'JobID,JobName,State,ElapsedRaw,AllocTRES,ExitCode,NodeList,Start,End'],text=True)
    assert all('|COMPLETED|' in row and '|0:0|' in row for row in text.splitlines()[1:] if row.strip())
    (OUT/'slurm_final_accounting.txt').write_text(text,encoding='utf-8')
    manifest=OUT/'audit/artifact_hashes.json';old=OUT/'audit/pre_finalize_manifest.json'
    if not old.exists():shutil.copyfile(manifest,old)
    write(OUT/'audit/finalization.json',dict(complete=True,scientific_jobs=jobs.split(','),
        scientific_jobs_completed=True,metrics_recomputed=False,new_generations=0,
        reason='Intermediate slurm_accounting.txt was updated by submit script after analysis manifest; final index is taken after all scientific jobs.'))
    write(manifest,{str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file()
        and p!=manifest and not any(s in p.name for s in ['.tar.gz','job_'])})
    print('E40 ARTIFACTS FINALIZED',flush=True)


if __name__=='__main__':run()
