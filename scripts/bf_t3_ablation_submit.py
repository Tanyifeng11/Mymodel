"""逐次保存Slurm编号；重试不重复提交已登记的作业。"""
import json
from pathlib import Path
import subprocess

ROOT = Path('/share/home/u2515283058/Mymodel/output_eval/bf_t3_ablation_20261102')
JOB = ROOT / 'source/scripts/bf_t3_ablation.sbatch'
ROOT.joinpath('logs').mkdir(parents=True, exist_ok=True)
path = ROOT / 'submitted_jobs.json'
jobs = json.loads(path.read_text()) if path.exists() else {}


def submit(key, mode, dependency=(), arm=None, gpu=0):
    if key in jobs: return jobs[key]
    args = ['sbatch', '--parsable', '-p', 'gpu' if gpu else 'cpu', '-J', 'bfa_' + key,
            '-N1', '-n1', '-c', '8' if gpu == 2 else '4', '--mem', '96G' if gpu == 2 else '48G',
            '-t', '7-00:00:00', '-o', str(ROOT / 'logs' / (key + '_%j.log'))]
    if gpu: args += ['--gres=gpu:' + str(gpu)]
    if dependency: args += ['--dependency=afterok:' + ':'.join(map(str, dependency))]
    args += [str(JOB), mode] + ([arm] if arm else [])
    job = subprocess.check_output(args, text=True).strip()
    assert job.isdigit(), job
    jobs[key] = job
    tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(jobs, indent=2) + '\n'); tmp.replace(path)
    print(key, job, flush=True)
    return job


prepare = submit('prepare', 'prepare')
check = submit('preflight', 'preflight', [prepare], gpu=1)
no_body = submit('no_body', 'train', [check], 'BF-SD-T3-no-body', 2)
no_rest = submit('no_rest', 'train', [check], 'BF-SD-T3-no-rest', 2)
reused = submit('reused', 'reused', [check], gpu=1)
evaluate = submit('evaluate', 'evaluate', [no_body, no_rest, reused], gpu=1)
package = submit('package', 'package', [evaluate])
print(json.dumps(jobs, indent=2))
