"""依赖成功才启动下一阶段；提交记录防止重复作业。"""
import argparse
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.tg_tfm_experiment import ROOT,ARMS,read,write


def submit(stage,arm=None,gpus=0,dependency=None,retry=False):
    path=ROOT/'submitted_jobs.json'
    jobs=read(path) if path.exists() else {}
    key=stage+('_'+arm if arm else '')
    previous=jobs.get(key,[])
    if previous and not retry:return previous[-1]
    (ROOT/'logs').mkdir(parents=True,exist_ok=True)
    command=['sbatch','--parsable','-p','gpu' if gpus else 'cpu','-J','tgtfm_'+key,
             '-N1','-n1','-c',str(max(4,gpus*4)),'--mem','64G' if gpus else '16G',
             '-t','7-00:00:00' if stage=='train' else '04:00:00',
             '-o',str(ROOT/'logs'/(key+'_%j.log'))]
    if gpus:command+=['--gres=gpu:'+str(gpus)]
    if dependency:command+=['--dependency=afterok:'+str(dependency)]
    command+=[str(ROOT/'source/scripts/tg_tfm.sbatch'),stage]
    if arm:command+=[arm,str(gpus)]
    job=subprocess.check_output(command,text=True).strip()
    assert job.isdigit(),job
    jobs[key]=previous+[job];write(path,jobs)
    print('SUBMITTED',key,job,flush=True)
    return job


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus',type=int,choices=[1,2,4,8],default=1)
    parser.add_argument('--retry',action='store_true')
    options=parser.parse_args()
    prepare=submit('prepare',retry=options.retry)
    check=submit('preflight',gpus=1,dependency=prepare,retry=options.retry)
    train=[submit('train',arm,gpus=options.gpus,dependency=check,retry=options.retry) for arm in ARMS]
    submit('evaluate',gpus=1,dependency=':'.join(train),retry=options.retry)
