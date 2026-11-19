"""准备、预检和两组顺序对照，最多占用双 GPU。"""
import json
from pathlib import Path
import subprocess

ROOT = Path('/share/home/u2515283058/Mymodel/output_eval/e1_bfm_reference_region_20261119')
ROOT.joinpath('logs').mkdir(parents=True,exist_ok=True)
path=ROOT/'submitted_jobs.json'
jobs=json.loads(path.read_text()) if path.exists() else {}


def submit(stage,gpu,dependency=None):
    if stage in jobs:return jobs[stage]
    args=['sbatch','--parsable','-p','gpu' if gpu else 'cpu','-J','e1rr_'+stage,
        '-N1','-n1','-c8' if gpu else '-c4','--mem','128G' if gpu else '64G',
        '-t',('01:00:00' if stage=='preflight' else '12:00:00') if gpu else '02:00:00',
        '-o',str(ROOT/'logs'/(stage+'_%j.log'))]
    if gpu:args+=['--gres=gpu:2']
    if dependency:args+=['--dependency=afterok:'+dependency]
    args+=[str(ROOT/'source/scripts/e1_bfm_reference_region.sbatch'),stage]
    job=subprocess.check_output(args,text=True).strip()
    assert job.isdigit(),job
    jobs[stage]=job;path.write_text(json.dumps(jobs,indent=2)+'\n')
    print(stage,job,flush=True)
    return job


parent=submit('prepare',False)
parent=submit('preflight',True,parent)
parent=submit('control',True,parent)
submit('texture',True,parent)
