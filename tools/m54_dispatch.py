"""只通过Slurm执行；每次提交/失败留痕，先门禁后科研训练。"""
import argparse
import subprocess
from datetime import datetime,timezone
from tools.m54_protocol import *


def submit(name,action,gpu,dependencies=(),extra=(),slurm_options=(),logical_name=None):
    path=OUT/'resources/jobs.json';jobs=read(path) if path.exists() else []
    existing=next((x for x in jobs if x['name']==name),None)
    if existing:return existing['id']
    cmd=['sbatch','--parsable','--job-name=M54_'+name]
    if action in ['train','batch_generate','generate']:cmd+=['--time=12:00:00']
    cmd+=list(slurm_options)
    if dependencies:cmd+=['--dependency=afterok:'+':'.join(map(str,dependencies))]
    cmd+=['submit/m54_'+('gpu' if gpu else 'cpu')+'.sh',action]+list(extra)
    result=subprocess.run(cmd,text=True,capture_output=True)
    if result.returncode:
        p=OUT/'resources/submission_failures.json';failures=read(p) if p.exists() else []
        failures.append(dict(command=cmd,stdout=result.stdout,stderr=result.stderr,utc=datetime.now(timezone.utc).isoformat()));write(p,failures)
        raise RuntimeError(result.stderr)
    identity=int(result.stdout.strip().split(';')[0]);record=dict(name=name,id=identity,action=action,gpu=gpu,
        dependencies=list(dependencies),extra=list(extra),slurm_options=list(slurm_options),commit=commit(),utc=datetime.now(timezone.utc).isoformat())
    if logical_name:record['logical_name']=logical_name
    jobs.append(record)
    write(path,jobs);print('M54 SUBMITTED',name,identity,flush=True);return identity


def status():
    path=OUT/'resources/jobs.json';jobs=read(path) if path.exists() else []
    if not jobs:print('NO_M54_JOBS');return
    output=subprocess.check_output(['sacct','-X','-P','-j',','.join(str(j['id']) for j in jobs),
        '--format=JobIDRaw,JobName,State,ExitCode,ElapsedRaw,AllocCPUS,AllocTRES,Start,End'],text=True)
    (OUT/'resources/accounting_current.psv').write_text(output,encoding='utf-8');print(output,flush=True)
    if (OUT/'decision.json').exists():print('DECISION',read(OUT/'decision.json'),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['gates','status','advance','input_candidates','recheck_modules','repair_round0','accounting']);args=p.parse_args()
    (OUT/'resources').mkdir(parents=True,exist_ok=True)
    if args.action=='gates':
        prepare=submit('PREPARE','prepare',False)
        submit('ROUND0','audit',True,[prepare])
    elif args.action=='advance':advance()
    elif args.action=='input_candidates':submit('INPUT_CANDIDATES_R1','input_candidates',False,
        slurm_options=['--cpus-per-task=16','--mem=32G','--time=04:00:00'])
    elif args.action=='recheck_modules':
        original=next(x for x in read(OUT/'resources/jobs.json') if x['name']=='ROUND0')
        protocol=submit('PROTOCOL_R1','prepare',False,[original['id']])
        m11=submit('ROUND0_M11_R1','audit',True,[protocol],['--arm','M11'],logical_name='ROUND0')
        submit('ROUND0_M9_R1','audit',True,[m11],['--arm','M9'],logical_name='ROUND0')
    elif args.action=='repair_round0':
        latest={j.get('logical_name',j['name']):j for j in read(OUT/'resources/jobs.json')}
        failed=latest['ROUND0']['id']
        protocol=submit('PREPARE_REPAIR_R1','prepare_retry',False,extra=['--failed-job',str(failed)])
        submit('ROUND0_R1','audit',True,[protocol],logical_name='ROUND0')
    elif args.action=='accounting':
        from tools.m54_finalize import accounting
        accounting()
    status()


def advance():
    import fcntl
    with (OUT/'resources/dispatch.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        jobs=read(OUT/'resources/jobs.json')
        raw=subprocess.check_output(['sacct','-X','-n','-P','-j',','.join(str(j['id']) for j in jobs),'--format=JobIDRaw,State'],text=True)
        states={int(s.split('|')[0]):s.split('|')[1].split()[0] for s in raw.splitlines() if s.split('|')[0].isdigit()}
        latest={j.get('logical_name',j['name']):j for j in jobs}
        failed=[j for j in latest.values() if states.get(j['id']) in ['FAILED','TIMEOUT','OUT_OF_MEMORY','CANCELLED','NODE_FAIL']]
        if failed:print('ENGINEERING_REPAIR_REQUIRED',[(j['name'],j['id']) for j in failed]);return
        if not (OUT/'audit/round0_complete.json').exists():return
        assert read(OUT/'audit/round0_complete.json')['passed']
        if states.get(latest['ROUND0']['id'])!='COMPLETED':return
        if not (OUT/'audit/input_review/pilot_approval.json').exists():print('ASSISTANT_PILOT_INPUT_REVIEW_PENDING');return
        require_input_review('pilot')
        slots=[max(0,4-sum(states.get(j['id'],'PENDING')!='COMPLETED' for j in latest.values()))]
        def enqueue(name,action,gpu,extra=()):
            if name not in latest:
                if slots[0]<=0:return False
                identity=submit(name,action,gpu,extra=extra)
                latest[name]=dict(id=identity,name=name);states[identity]='PENDING';slots[0]-=1
            return states.get(latest[name]['id'])=='COMPLETED'
        def generate_stage(prefix,stage,arms,updates=800,seed=42):
            folder=result_stage(stage,updates,seed);complete=True
            for arm in arms:
                done=OUT/folder/(arm+'_complete.json')
                job=prefix+'_'+arm
                if done.exists() and job not in latest:continue  # 已核验复用结果，无虚构新作业。
                ready=enqueue(job,'generate',True,['--stage',stage,'--arm',arm,'--updates',str(updates),'--seed',str(seed)])
                complete=ready and done.exists() and complete
            return complete
        def evaluate_stage(name,stage):
            # CLIP/LPIPS均使用CUDA，评价必须申请真实GPU，不能投纯CPU分区。
            ready=enqueue(name,'evaluate',True,['--stage',stage])
            return ready and (OUT/stage/'evaluation_complete.json').exists()
        round1=generate_stage('R1','round1',['E5','B1','SAFE']+ZERO_ARMS)
        if not round1:return
        diagnostic=generate_stage('ORACLE','oracle_diagnostic',['E5','B1','ORACLE'])
        if not diagnostic:return
        if not evaluate_stage('R1_EVAL','round1'):return
        choice=OUT/'round1/selection.json'
        if not choice.exists():
            gates=read(OUT/'round1/gates.json');means=read(OUT/'round1/means.json')
            possible=[a for a in ['M8_T100','M8_T085','M8_T070'] if gates[a]['passed']]
            winner=min(possible,key=lambda a:(means[a]['lpips_gt'],a)) if possible else 'M8_T100'
            freeze(choice,dict(M1_temperature={'M8_T100':1.,'M8_T085':.85,'M8_T070':.70}[winner],
                M8_selected=winner,selection='minimum LPIPS among six-gate passers; otherwise tau1',before_training=True))
        trained=True
        for arm in TRAIN_ARMS:
            ready=enqueue('TRAIN_'+arm,'train',True,['--arm',arm])
            trained=ready and (OUT/'training'/arm/'complete0800.json').exists() and trained
        if not trained:decision(R1='COMPLETE',R2='TRAINING_800');return
        from tools.m54_selection import share_results,freeze_top4,freeze_combinations,freeze_passers
        share_results('round1','pilot',['E5','B1','SAFE']+ZERO_ARMS)
        pilots=True
        for arm in TRAIN_ARMS:
            recipes=['M5_S075','M5_S100','M5_S125'] if arm=='M5' else [arm]
            pilots=generate_stage('PILOT','pilot',recipes) and pilots
        if not pilots:decision(R2='PILOT_GENERATION');return
        if not evaluate_stage('PILOT_EVAL','pilot'):return
        if not (OUT/'selection/top4.json').exists():
            chosen=freeze_top4()
            if not chosen['ready']:
                decision(R2='PILOT_RULE_INTERPRETATION_REQUIRED',pilot_ambiguous=chosen['ambiguous']);return
        top=read(OUT/'selection/top4.json')['selected']
        if not top:
            decision(R2='COMPLETE',fallback='CONCRETE_DESIGN_REQUIRED_ALL_LIGHTWEIGHT_REJECTED',complete=False);return
        deeper=True
        for arm in top:
            if family(arm) in TRAIN_ARMS:
                ready=enqueue('DEEPEN_'+family(arm),'train',True,['--arm',family(arm),'--updates','2400'])
                deeper=ready and (OUT/'training'/family(arm)/'complete2400.json').exists() and deeper
        if not deeper:decision(R3='DEEPENING_TO_TOTAL_2400');return
        if not generate_stage('DEV','dev32',['E5','B1']+top,2400):decision(R3='COMPLETE',dev32='GENERATING');return
        if not evaluate_stage('DEV_EVAL','dev32_u2400'):return
        if not (OUT/'selection/combinations.json').exists():freeze_combinations()
        combos=list(read(OUT/'selection/combinations.json')['recipes'])
        if combos:
            share_results('dev32_u2400','dev32_combo_u2400',['E5','B1'])
            if not generate_stage('COMBO','dev32_combo',combos,2400):decision(combinations='GENERATING');return
            if not evaluate_stage('COMBO_EVAL','dev32_combo_u2400'):return
        lockfile=OUT/'selection/lock_candidates.json'
        if not lockfile.exists():
            qualified=freeze_passers('dev32_u2400','dev_passers.json')['selected']
            if combos:qualified+=freeze_passers('dev32_combo_u2400','combo_passers.json')['selected']
            freeze(lockfile,dict(selected=qualified,six_dev_gates_required=True,selection_inputs='dev32 only',lock_results_used=False))
        lockarms=read(lockfile)['selected']
        if not lockarms:
            decision(dev32='COMPLETE',combinations='COMPLETE' if combos else 'NO_ELIGIBLE_PAIR',
                fallback='CONCRETE_DESIGN_REQUIRED_NO_DEV_GO',complete=False);return
        if not (OUT/'audit/input_review/lock_approval.json').exists():
            if not (OUT/'m54_lock_input_review.tar.gz').exists():
                from tools.m54_finalize import input_review_package
                input_review_package('lock')
            decision(lock='ACTUAL_INPUT_VISUAL_REVIEW_REQUIRED');return
        require_input_review('lock')
        if not generate_stage('LOCK','lock',['E5','B1']+lockarms,2400):decision(lock='GENERATING');return
        if not evaluate_stage('LOCK_EVAL','lock_u2400'):return
        if not (OUT/'selection/natural_candidates.json').exists():freeze_passers('lock_u2400','natural_candidates.json',2)
        natural=read(OUT/'selection/natural_candidates.json')['selected']
        if not natural:decision(lock='COMPLETE_NO_GO',fallback='CONCRETE_DESIGN_REQUIRED_NO_LOCK_GO',complete=False);return
        if not generate_stage('NATURAL','natural',['E5','B1']+natural,2400):decision(natural='GENERATING');return
        if not evaluate_stage('NATURAL_EVAL','natural_u2400'):return
        if not (OUT/'selection/best1.json').exists():freeze_passers('natural_u2400','best1.json',1)
        best=read(OUT/'selection/best1.json')['selected']
        if not best:decision(natural='COMPLETE_NO_GO',fallback='CONCRETE_DESIGN_REQUIRED_NO_NATURAL_GO',complete=False);return
        seeds_ready=True
        for seed in [123,456]:
            stage=result_stage('natural',2400,seed)
            ready=generate_stage('SEED'+str(seed),'natural',['E5','B1']+best,2400,seed)
            if ready:ready=evaluate_stage('SEED'+str(seed)+'_EVAL',stage)
            seeds_ready=ready and seeds_ready
        if not seeds_ready:decision(seeds='REPLICATION_RUNNING');return
        decision(R2='COMPLETE',R3='COMPLETE',dev32='COMPLETE',combinations='COMPLETE' if combos else 'NO_ELIGIBLE_PAIR',
            lock='COMPLETE',natural='COMPLETE_GO',seeds='COMPLETE',best1=best[0],fallback='NOT_TRIGGERED',complete=False)
        if enqueue('FINAL','finalize',False) and (OUT/'audit/final_complete.json').exists():
            decision(server_experiments='COMPLETE',local_delivery='PENDING_DOWNLOAD_AND_VERIFICATION',complete=False)


if __name__=='__main__':main()
