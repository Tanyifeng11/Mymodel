"""一个双卡作业依次完成五轮训练、保存、评测和汇总。"""
import json
from pathlib import Path
import subprocess

ROOT = Path('/share/home/u2515283058/Mymodel/output_eval/e1_bfm_continue_20261117')
ROOT.joinpath('logs').mkdir(parents=True, exist_ok=True)
path = ROOT / 'submitted_jobs.json'
jobs = json.loads(path.read_text()) if path.exists() else {}


def submit(key, stage, gpu, dependency=None, epoch=None):
    if key in jobs:
        return jobs[key]
    args = ['sbatch', '--parsable', '-p', 'gpu' if gpu else 'cpu', '-J', 'e1ct_' + key,
        '-N1', '-n1', '-c', '8' if gpu == 2 else '4', '--mem', '128G' if gpu == 2 else '64G',
        '-t', '7-00:00:00', '-o', str(ROOT / 'logs' / (key + '_%j.log'))]
    if gpu:
        args += ['--gres=gpu:' + str(gpu)]
    if dependency:
        args += ['--dependency=afterok:' + dependency]
    args += [str(ROOT / 'source/scripts/e1_bfm_continue.sbatch'), stage]
    if epoch is not None:
        args += ['--epoch', str(epoch)]
    job = subprocess.check_output(args, text=True).strip()
    assert job.isdigit(), job
    jobs[key] = job
    path.write_text(json.dumps(jobs, indent=2) + '\n')
    print(key, job, flush=True)
    return job


dependency = submit('prepare', 'prepare', 0)
submit('pipeline', 'run', 2, dependency)
