"""只提交一个单卡诊断作业，不创建优化器或续训作业。"""
import json
from pathlib import Path
import subprocess

ROOT = Path('/share/home/u2515283058/Mymodel/output_eval/e1_cfg_validate_20261123')


def main():
    path = ROOT/'submitted_jobs.json'
    if path.exists():
        print('EXISTING',path.read_text(),flush=True)
        return
    (ROOT/'logs').mkdir(parents=True,exist_ok=True)
    args = ['sbatch','--parsable','-p','gpu','-J','e1_cfg_validate','-N1','-n1','-c','4',
        '--mem','64G','--gres=gpu:1','-t','04:00:00','-o',str(ROOT/'logs/diagnostic_%j.log'),
        str(ROOT/'source/scripts/e1_cfg_validate.sbatch')]
    job = subprocess.check_output(args,text=True).strip()
    assert job.isdigit(),job
    path.write_text(json.dumps(dict(diagnostic=job),indent=2)+'\n')
    print('SUBMITTED',job,flush=True)


if __name__ == '__main__':
    main()
