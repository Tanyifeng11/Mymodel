"""保留原作业记录，幂等提交第 8 轮评测及第 9—10 轮恢复作业。"""
import json
from pathlib import Path
import subprocess

ROOT = Path('/share/home/u2515283058/Mymodel/output_eval/e1_bfm_continue_20261117')
path = ROOT / 'submitted_jobs.json'
jobs = json.loads(path.read_text())
if 'recovery' not in jobs:
    job = subprocess.check_output(['sbatch', '--parsable', '-p', 'gpu', '-J', 'e1ct_recovery',
        '-N1', '-n1', '-c8', '--mem', '128G', '-t', '7-00:00:00', '--gres=gpu:2',
        '-o', str(ROOT / 'logs/recovery_%j.log'),
        str(ROOT / 'source/scripts/e1_bfm_continue_recover.sbatch')], text=True).strip()
    assert job.isdigit(), job
    jobs['recovery'] = job
    path.write_text(json.dumps(jobs, indent=2) + '\n')
print(json.dumps(jobs), flush=True)
