"""CPU准备后，以一个双卡作业顺序执行预检、十轮训练、评测和汇总。"""
import json
from pathlib import Path
import subprocess

ROOT = Path('/share/home/u2515283058/Mymodel/output_eval/e1_bfm_continue20_20261121')


def main():
    (ROOT/'logs').mkdir(parents=True, exist_ok=True)
    path = ROOT/'submitted_jobs.json'
    jobs = json.loads(path.read_text()) if path.exists() else {}
    for key,stage,gpus,dependency in [('prepare','prepare',0,None),('pipeline','run',2,'prepare')]:
        if key in jobs:
            continue
        args = ['sbatch','--parsable','-p','gpu' if gpus else 'cpu','-J','e1e20_'+key,
            '-N1','-n1','-c','8' if gpus else '4','--mem','128G' if gpus else '64G',
            '-t','7-00:00:00' if gpus else '01:00:00','-o',str(ROOT/'logs'/(key+'_%j.log'))]
        if gpus:
            args += ['--gres=gpu:2']
        if dependency:
            args += ['--dependency=afterok:'+jobs[dependency]]
        args += [str(ROOT/'source/scripts/e1_bfm_continue20.sbatch'),stage]
        job = subprocess.check_output(args,text=True).strip()
        assert job.isdigit(),job
        jobs[key] = job
        path.write_text(json.dumps(jobs,indent=2)+'\n')
        print(key,job,flush=True)


if __name__ == '__main__':
    main()
