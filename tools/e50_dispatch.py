"""一次性提交依赖图；已有提交记录时只显示，禁止重复占用资源。"""
import subprocess
from tools.e50_protocol import *


def dispatch():
    path=OUT/'job_ids.json'
    if path.exists():
        print('EXISTING E50 JOBS',read(path),flush=True);return
    prepare();jobs={};write(path,jobs)
    def submit(name,args,minutes,dependencies=()):
        command=['sbatch','--parsable','--job-name=e50_'+name,'--time=00:%02d:00'%minutes]
        if dependencies:command +=['--dependency=afterok:'+':'.join(map(str,dependencies))]
        job=int(subprocess.check_output(command+['submit/e50_gpu.sh']+args,text=True).strip().split(';')[0])
        jobs[name]=job;write(path,jobs);print('E50 SUBMITTED',name,job,flush=True);return job
    audit=submit('audit',['audit'],25)
    arms=[submit(a,['generate','--arm',a],25,[audit]) for a in ARMS[2:]]
    submit('evaluate',['evaluate'],15,arms)
    decision(submitted_jobs=jobs)


if __name__=='__main__':dispatch()
