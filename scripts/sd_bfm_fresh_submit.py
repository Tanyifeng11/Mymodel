"""一个正式训练作业；依赖成功后才执行生成与统一评测。"""
import json
from pathlib import Path
import re
import subprocess

ROOT = Path('/share/home/u2515283058/Mymodel/output_eval/sd_bfm_fresh_20261113')
ROOT.joinpath('logs').mkdir(parents=True, exist_ok=True)
path = ROOT / 'submitted_jobs.json'
jobs = json.loads(path.read_text()) if path.exists() else {}


def submit(key, gpu, dependency=None):
    if key in jobs:
        return jobs[key]
    partition = 'gpu' if gpu else 'cpu'
    info = subprocess.check_output(['scontrol', 'show', 'partition', partition, '-o'], text=True)
    maximum = re.search(r'\bMaxTime=(\S+)', info).group(1)
    args = ['sbatch', '--parsable', '-p', partition, '-J', 'sdbf_' + key, '-N1', '-n1',
            '-c', '8' if gpu == 2 else '4', '--mem', '128G' if gpu == 2 else '64G',
            '-t', '0' if maximum == 'UNLIMITED' else maximum,
            '-o', str(ROOT / 'logs' / (key + '_%j.log'))]
    if gpu:
        args += ['--gres=gpu:' + str(gpu)]
    if dependency:
        args += ['--dependency=afterok:' + dependency]
    args += [str(ROOT / 'source/scripts/sd_bfm_fresh.sbatch'), key]
    result = subprocess.check_output(args, text=True).strip()
    assert result.isdigit(), result
    jobs[key] = result
    path.write_text(json.dumps(jobs, indent=2) + '\n')
    print(key, result, flush=True)
    return result


prepare = submit('prepare', 0)
train = submit('train', 2, prepare)
submit('evaluate', 1, train)
