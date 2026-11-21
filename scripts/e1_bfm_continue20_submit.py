"""一个双卡作业完成准备、预检、十轮训练和评测，适应提交数量限制。"""
import json
from pathlib import Path
import subprocess

ROOT = Path('/share/home/u2515283058/Mymodel/output_eval/e1_bfm_continue20_20261121')


def main():
    (ROOT/'logs').mkdir(parents=True, exist_ok=True)
    path = ROOT/'submitted_jobs.json'
    jobs = json.loads(path.read_text()) if path.exists() else {}
    if 'pipeline' in jobs:
        print('existing pipeline',jobs['pipeline'],flush=True)
        return
    args = ['sbatch','--parsable','-p','gpu','-J','e1e20_pipeline',
        '-N1','-n1','-c','8','--mem','128G','-t','7-00:00:00','--gres=gpu:2',
        '-o',str(ROOT/'logs/pipeline_%j.log'),
        str(ROOT/'source/scripts/e1_bfm_continue20.sbatch'),'run']
    job = subprocess.check_output(args,text=True).strip()
    assert job.isdigit(),job
    jobs['pipeline'] = job
    path.write_text(json.dumps(jobs,indent=2)+'\n')
    print('pipeline',job,flush=True)


if __name__ == '__main__':
    main()
