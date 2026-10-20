"""独立确认：先审计/烟测，再冻结生成，最后一次性评分和归档。"""
import argparse
import csv
import gc
import json
import os
import platform
import subprocess
import time
from datetime import datetime,timezone
from tools.m12_gain_confirm_protocol import *
from tools.m12_protocol import write_csv

SOURCES=['tools/m12_gain_confirm.py','tools/m12_gain_confirm_protocol.py','tools/m12_gain_confirm_audit.py','tools/m12_gain_confirm_dispatch.py',
    'tools/m12_gain_confirm_eval.py','tools/m12_gain_runtime.py','tools/m12_runtime.py','tools/m54_runtime.py',
    'tools/m54_m10_gain_runtime.py','tools/e44_runtime.py','tools/e47_runtime.py','tools/m54_eval.py',
    'tools/e44_metrics.py','tools/e35_freeu_protocol.py','tools/e33tmoc_appearance_eval.py',
    'tools/m54_input_candidates.py','tools/e48_holdout.py','tools/e47_audit.py',
    'models/m12_gain_dual_path.py','models/m12_fixed_dual_path.py','models/m54_candidates.py',
    'models/m54_m10_gain.py','models/reference_attention_variants.py','models/reference_feature_writer.py',
    'garment_mask_utils.py','eval/eval_utils.py','eval/metrics.py','adapter/attention_processor.py',
    'submit/m12_confirm_cpu.sh','submit/m12_confirm_gpu.sh','tests/test_m12_gain_confirm.py']


def utc():return datetime.now(timezone.utc).isoformat()


def require():
    protocol=read(RUN/'protocol/protocol.json')
    assert protocol['settings']==SETTINGS
    assert commit()==protocol['commit'],'CODE_CHANGED_AFTER_FREEZE'
    assert all(sha(path)==value for path,value in protocol['code_sha256'].items())
    assert sha(RUN/'protocol/identities.json')==protocol['identities_sha256']
    return protocol


def prepare():
    import torch
    from tools.m12_gain_runtime import load_gain
    from tools.m54_protocol import E5,E5_SHA,B1,B1_SHA
    from tools.e44_runtime import digest
    from tools.m12_gain_confirm_audit import run_audit
    if (RUN/'protocol/protocol.json').exists():require();return
    selected=run_audit()
    old=read(M12/'protocol.json')['inference']
    assert {k:old[k] for k in SETTINGS['inference']}==SETTINGS['inference']
    checkpoints={}
    for name,path,expected in [('E5',E5,E5_SHA),('B1',B1,B1_SHA),('M10_GAIN',GAIN_CHECKPOINT,
            'a5c6912d81b1861e7e74e29142e13b036ca6d25a8ce1ab049e915b63b67f5464')]:
        assert sha(path)==expected
        checkpoints[name]=dict(path=str(path.resolve()),sha256=expected)
    gain=load_gain(GAIN_CHECKPOINT,'cpu');values=gain(torch.arange(1000))
    checkpoints['learned_gain']=dict(parameters=1280,step=2400,frozen=True,finite=True,
        state_sha256={k:digest(v) for k,v in gain.state_dict().items()},gamma_min=float(values.min()),gamma_max=float(values.max()))
    models=read(M54/'manifests/model_file_hashes.json')
    assert all(sha(path)==expected for path,expected in models.items())
    write(RUN/'protocol/checkpoint_manifest.json',checkpoints);write(RUN/'protocol/model_file_hashes.json',models)
    historical_costs={}
    for name,base in [('M54',M54),('M10_GAIN',M54/'m10_gain_r1'),('M12',M12),('M12_GAIN_DEV32',OLD)]:
        path=base/'resources/jobs.json'
        if path.exists():
            jobs=read(path)
            raw=subprocess.check_output(['sacct','-X','-P','-j',','.join(str(j['id']) for j in jobs),
                '--format=JobIDRaw,JobName,State,ExitCode,ElapsedRaw,AllocCPUS,AllocTRES,Start,End'],text=True)
            historical_costs[name]=dict(jobs=jobs,jobs_sha256=sha(path),actual_allocations=list(csv.DictReader(raw.splitlines(),delimiter='|')))
    write(RUN/'protocol/historical_cost_snapshot.json',dict(experiments=historical_costs,
        learned_gain_training_updates=2400,new_training_updates=0,
        limitations='historical allocations include audits, generation, training and failures; allocation hours are not training-only or utilization hours; full E5/B1 pretraining cost not recovered here'))
    old_records=read(OLD/'metrics/per_image.json');index={(r['variant'],r['id']):r for r in old_records}
    ids=[r['id'] for r in old_records if r['variant']=='E5'];power=[]
    from scipy.stats import norm
    for base,key,effect in [('E5','lpips_gt',.005),('M10_GAIN','lpips_gt',.002),('B1','contour_f1',.015)]:
        differences=np.asarray([index['M12_GAIN',i][key]-index[base,i][key] for i in ids]);sd=float(differences.std(ddof=1))
        power.append(dict(base=base,metric=key,minimum_effect=effect,old_sd=sd,
            normal_approx_power_96=float(norm.cdf(effect/(sd/np.sqrt(96))-norm.ppf(.975))),
            n80=int(np.ceil(((norm.ppf(.975)+norm.ppf(.8))*sd/effect)**2))))
    write(RUN/'protocol/power_analysis.json',dict(comparisons=power,decision=SETTINGS['sample_decision'],
        limitations='old development seed42 variance; three-seed identity covariance unknown; approximate planning, not achieved power',
        old_metrics_sha256=sha(OLD/'metrics/per_image.json')))
    write(RUN/'protocol/smoke_identities.json',[next(r for r in read(OLD/'protocol/dev32.json') if r['id']==identity) for identity in SMOKE_IDS])
    write(RUN/'protocol/environment_prepare.json',dict(python=platform.python_version(),pytorch=torch.__version__,
        cuda=torch.version.cuda,pillow=__import__('PIL').__version__,numpy=np.__version__,diffusers=__import__('diffusers').__version__))
    tracked=subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],text=True)
    assert not tracked.strip(),'TRACKED_WORKTREE_DIRTY'
    protocol=dict(settings=SETTINGS,commit=commit(),code_sha256={p:sha(p) for p in SOURCES},
        source_plan_sha256=PLAN_SHA,identities_sha256=sha(RUN/'protocol/identities.json'),
        input_manifest_sha256=sha(RUN/'protocol/input_manifest.json'),checkpoint_manifest_sha256=sha(RUN/'protocol/checkpoint_manifest.json'),
        smoke_ids=SMOKE_IDS,smoke_identity_exposure='old dev32, excluded from confirmation',frozen_utc=utc(),tracked_worktree_status=tracked)
    write(RUN/'protocol/protocol.json',protocol)
    write(RUN/'protocol/frozen_config.yaml',SETTINGS)
    print('CONFIRM_PREPARED',len(selected),protocol['commit'],power,flush=True)


def runtime(arm,namespace,endpoint=None):
    from tools import e44_runtime as module
    from models import e33tm_generation_wrapper as loader
    from tools.e47_runtime import load_base
    from tools.m54_protocol import LAYER
    from tools.m54_runtime import M54Runtime
    from tools.m12_runtime import DualRuntime
    from tools.m12_gain_runtime import GainDualRuntime,load_gain
    from tools.m54_m10_gain_runtime import GainRuntime
    previous=loader.load_e5,module.OUT,module.LAYER
    loader.load_e5=lambda:load_base('B');module.OUT,module.LAYER=RUN/namespace,LAYER
    try:
        if arm=='M12_GAIN':r=GainDualRuntime(GAIN_CHECKPOINT,*(endpoint or (.25,.5)))
        elif arm=='M12':r=DualRuntime(arm,*(endpoint or (.25,.5)))
        elif arm=='M10_GAIN':
            r=GainRuntime();r.load(GAIN_CHECKPOINT);r.gain.eval().requires_grad_(False)
        else:r=M54Runtime(arm)
    finally:loader.load_e5,module.OUT,module.LAYER=previous
    r.effects_enabled=False;r.m54_runtime_out,r.m54_layer=RUN/namespace,LAYER;r.pipe.use_tcpm_lite=True
    assert not r.parameters() or all(not p.requires_grad for p in r.parameters())
    return r


def dispose(r):
    from tools.m54_runtime import release
    release(r)


def trace_generate(r,row,path,seed):
    import torch
    from tools.e44_runtime import digest
    path=Path(path);counts=dict(unet=0,writer=0)
    def unet_count(*args):counts['unet']+=1
    def writer_count(*args):counts['writer']+=1
    handles=[r.pipe.unet.register_forward_hook(unet_count)]
    if r.writer:handles.append(r.writer.unet.register_forward_hook(writer_count))
    try:
        if r.arm in ['M12','M12_GAIN']:
            meta,_,_=r.generate(row,path,seed=seed)
        else:
            steps=[];original=r.pipe.scheduler.step
            def capture(*args,**kw):
                result=original(*args,**kw);steps.append(result[0].detach().cpu());return result
            r.pipe.scheduler.step=capture
            try:meta,_,_=r.generate(row,path,seed=seed)
            finally:r.pipe.scheduler.step=original
            trace=torch.load(path.with_suffix('.pt'),map_location='cpu',weights_only=False)
            trace['step_latents']=torch.stack(steps);torch.save(trace,path.with_suffix('.pt'))
    finally:
        for h in handles:h.remove()
    trace=torch.load(path.with_suffix('.pt'),map_location='cpu',weights_only=False)
    fields=['initial_latent','final_latent','conditional_epsilon','unconditional_epsilon','step_latents']
    assert all(torch.isfinite(trace[k]).all() for k in fields)
    assert trace['conditional_epsilon'].shape[0]==trace['unconditional_epsilon'].shape[0]==trace['step_latents'].shape[0]==50
    meta.update(diagnostics_sha256=sha(path.with_suffix('.pt')),trace_tensor_sha256={k:digest(trace[k]) for k in fields},
        forward_counts=counts,scientific_updates=0,commit=commit())
    assert counts['unet']==100
    assert counts['writer']==(50 if r.writer and r.active and meta['mask']['valid'] else 0)
    write(path.with_suffix('.json'),meta);return meta


def compare(left,right):
    a,b=[read(p.with_suffix('.json')) for p in [left,right]]
    checks=dict(png=sha(left)==sha(right),traces=a['trace_tensor_sha256']==b['trace_tensor_sha256'])
    assert all(checks.values()),dict(left=str(left),right=str(right),checks=checks)
    return checks


def smoke():
    import torch
    require();examples=read(RUN/'protocol/smoke_identities.json');replays=[]
    for arm,kind,endpoint in [('E5','M12_GAIN',(0.,0.)),('B1','M12_GAIN',(1.,0.)),('M10','M12',(0.,1.)),('M10_GAIN','M12_GAIN',(0.,1.))]:
        r=runtime(arm,'smoke/runtime/native/'+arm)
        try:
            for row in examples:
                path=RUN/'smoke/native'/arm/(row['id']+'.png');trace_generate(r,row,path,42)
                historical=OLD/'outputs/M12_GAIN'/row['id']/'seed42.png' if arm=='M12_GAIN' else M12/'dev32/images'/arm/(row['id']+'.png')
                assert sha(path)==sha(historical),('NATIVE_REPLAY_CHANGED',arm,row['id'])
        finally:dispose(r)
        del r;gc.collect();torch.cuda.empty_cache()
        r=runtime(kind,'smoke/runtime/endpoint/'+arm,endpoint)
        try:
            for row in examples:
                path=RUN/'smoke/endpoints'/arm/(row['id']+'.png');trace_generate(r,row,path,42)
                replays.append(dict(arm=arm,id=row['id'],**compare(path,RUN/'smoke/native'/arm/(row['id']+'.png'))))
        finally:dispose(r)
        del r;gc.collect();torch.cuda.empty_cache()
    fixed=[]
    for arm in ['M12','M12_GAIN']:
        r=runtime(arm,'smoke/runtime/fixed/'+arm)
        try:
            for row in examples:
                path=RUN/'smoke/fixed'/arm/(row['id']+'.png');meta=trace_generate(r,row,path,42)
                records=r.processor.records
                assert len(records)==400 and all(rec['fusion_energy']==0 for rec in records if rec['branch']=='unconditional')
                assert r.path_hashes()==r.source_hashes and all(not p.requires_grad for p in r.processor.parameters())
                if not meta['mask']['valid']:assert sha(path)==sha(RUN/'smoke/native/E5'/(row['id']+'.png'))
                historical=M12/'dev32/images/R025_050'/(row['id']+'.png') if arm=='M12' else OLD/'outputs/M12_GAIN'/row['id']/'seed42.png'
                assert sha(path)==sha(historical)
                fixed.append(dict(arm=arm,id=row['id'],gate=meta['mask'],unconditional_zero=True,source_unchanged=True))
            r.active=False;path=RUN/'smoke/off'/arm/'seed42.png';trace_generate(r,examples[0],path,42)
            compare(path,RUN/'smoke/native/E5'/(examples[0]['id']+'.png'))
        finally:dispose(r)
        del r;gc.collect();torch.cuda.empty_cache()
    write(RUN/'smoke/endpoint_checks.json',dict(passed=True,replays=replays,fixed=fixed))
    write(RUN/'smoke/complete.json',dict(passed=True,commit=commit(),ids=SMOKE_IDS,endpoint_pairs=len(replays),utc=utc()))
    print('CONFIRM_SMOKE_PASSED',len(replays),flush=True)


def profile(r,arm):
    import torch
    counts=dict(unet=0,writer=0)
    def count_unet(*args):counts['unet']+=1
    def count_writer(*args):counts['writer']+=1
    handles=[r.pipe.unet.register_forward_hook(count_unet)]
    if r.writer:handles.append(r.writer.unet.register_forward_hook(count_writer))
    processors=[r.processor]+([r.processor.native_m10] if hasattr(r.processor,'native_m10') else [])
    observers=[]
    for processor in processors:
        if processor is None:continue
        processor.record_enabled=False
        if hasattr(processor,'diagnostic_observer'):
            observers.append((processor,processor.diagnostic_observer));processor.diagnostic_observer=None
    examples=read(RUN/'protocol/smoke_identities.json');records=[]
    try:
        r.profile(examples[0],RUN/'production_profile'/arm/'warmup.png',seed=42)
        for index,row in enumerate(examples):
            counts.update(unet=0,writer=0)
            record=r.profile(row,RUN/'production_profile'/arm/(row['id']+'.png'),seed=42)
            record['forward_counts']=dict(counts);records.append(record)
            assert counts['unet']==100
    finally:
        for h in handles:h.remove()
        for processor in processors:
            if processor is not None:processor.record_enabled=True
        for processor,observer in observers:processor.diagnostic_observer=observer
        if hasattr(r.processor,'reset'):r.processor.reset()
    write(RUN/'production_profile'/(arm+'_profile.json'),dict(arm=arm,n=len(records),records=records,method=SETTINGS['profiling']))


def generate(arm):
    import torch
    p=require();assert read(RUN/'smoke/complete.json')['passed']
    assert arm in ARMS
    freeze=RUN/'generation'/('freeze_'+arm+'.json')
    write(freeze,dict(commit=commit(),protocol_sha256=sha(RUN/'protocol/protocol.json'),smoke_sha256=sha(RUN/'smoke/complete.json')))
    r=runtime(arm,'generation/runtime/'+arm);completed=[]
    try:
        if not (RUN/'production_profile'/(arm+'_profile.json')).exists():profile(r,arm)
        for row in rows():
            for seed in SEEDS:
                path=image(arm,row['id'],seed)
                if path.with_suffix('.json').exists():
                    meta=read(path.with_suffix('.json'))
                    assert meta.get('confirmation_complete') and sha(path)==meta['png_sha256'] and meta['commit']==p['commit'],'PARTIAL_OUTPUT_REQUIRES_AUDITED_RETRY'
                    completed.append(dict(id=row['id'],seed=seed,reused_after_engineering_retry=True));continue
                assert not path.exists(),'PARTIAL_PNG_REQUIRES_AUDITED_RETRY'
                meta=trace_generate(r,row,path,seed)
                meta.update(input_sha256={k:sha(row[k]) for k in ['gt','sketch','reference']},
                    caption_sha256=hashlib.sha256(row['caption'].encode()).hexdigest(),
                    checkpoint_manifest_sha256=sha(RUN/'protocol/checkpoint_manifest.json'),
                    protocol_sha256=sha(RUN/'protocol/protocol.json'),confirmation_complete=True)
                write(path.with_suffix('.json'),meta);completed.append(dict(id=row['id'],seed=seed,reused_after_engineering_retry=False))
                print('CONFIRM_GENERATED',arm,len(completed),row['id'],seed,flush=True)
    finally:dispose(r)
    del r;gc.collect();torch.cuda.empty_cache()
    write(RUN/'generation'/(arm+'_complete.json'),dict(arm=arm,n=len(completed),records=completed,commit=commit(),utc=utc()))


def verify_outputs():
    import torch
    require();outputs=[]
    for arm in ARMS:assert read(RUN/'generation'/(arm+'_complete.json'))['n']==288
    for row in rows():
        for seed in SEEDS:
            initial=[]
            for arm in ARMS:
                path=image(arm,row['id'],seed);meta=read(path.with_suffix('.json'))
                assert sha(path)==meta['png_sha256'] and sha(path.with_suffix('.pt'))==meta['diagnostics_sha256']
                assert meta['confirmation_complete'] and meta['seed']==seed and not meta['gt_used_in_inference']
                assert meta['commit']==commit() and meta['protocol_sha256']==sha(RUN/'protocol/protocol.json')
                assert meta['input_sha256']=={k:sha(row[k]) for k in ['gt','sketch','reference']}
                initial.append(meta['initial_sha256'])
                if not meta['mask']['valid']:assert sha(path)==sha(image('E5',row['id'],seed))
                outputs.append(dict(arm=arm,id=row['id'],seed=seed,path=str(path),png_sha256=sha(path),
                    metadata_sha256=sha(path.with_suffix('.json')),trace_sha256=meta['diagnostics_sha256'],gate_valid=meta['mask']['valid']))
            assert len(set(initial))==1,'UNPAIRED_INITIAL_LATENTS'
    write(RUN/'audit/output_hashes.json',outputs);return outputs


def finalize():
    from tools.m54_finalize import package
    require();verify_outputs();assert read(RUN/'metrics/complete.json')['complete']
    frozen=list(RUN.glob('**/manifests/frozen_*.json'))
    assert frozen and all(read(p)['unchanged'] for p in frozen)
    files={}
    assert read(RUN/'precision/complete.json')['complete']
    for folder in ['protocol','smoke','metrics','precision','audit','inputs','generation','production_profile','resources']:
        for path in (RUN/folder).rglob('*'):
            if path.is_file() and path.suffix not in ['.pt','.log','.err']:
                files[str(path.relative_to(RUN))]=path
    for path in (RUN/'outputs').rglob('*'):
        if path.is_file() and path.suffix!='.pt':files[str(path.relative_to(RUN))]=path
    for path in SOURCES:files['code/'+path]=Path(path)
    package(RUN/'m12_gain_confirm_review.tar.gz',files)
    write(RUN/'audit/finalized.json',dict(complete=True,utc=utc(),images=1728,full_traces='retained server-side; every tensor and file SHA recorded',commit=commit()))
    print('CONFIRM_PACKAGED',sha(RUN/'m12_gain_confirm_review.tar.gz'),flush=True)


def accounting():
    from tools.m54_finalize import package
    jobs=read(RUN/'resources/jobs.json')
    raw=subprocess.check_output(['sacct','-X','-P','-j',','.join(str(j['id']) for j in jobs),
        '--format=JobIDRaw,JobName,State,ExitCode,ElapsedRaw,AllocCPUS,AllocTRES,Start,End'],text=True)
    (RUN/'resources/accounting.psv').write_text(raw,encoding='utf-8')
    ledger=list(csv.DictReader(raw.splitlines(),delimiter='|'))
    terminal={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED'}
    assert ledger and all(r['State'].split()[0] in terminal for r in ledger),'JOBS_STILL_ACTIVE'
    write_csv(RUN/'audit/runtime_resource_ledger.csv',ledger)
    write_csv(RUN/'audit/failed_jobs.csv',[r for r in ledger if r['State']!='COMPLETED'])
    files={str(p.relative_to(RUN)):p for folder in ['resources','audit'] for p in (RUN/folder).glob('*') if p.is_file()}
    package(RUN/'m12_gain_confirm_ledger.tar.gz',files)
    print('CONFIRM_ACCOUNTED',[(r['JobIDRaw'],r['State']) for r in ledger],flush=True)


def submit_job(action,script,options=None,arguments=None):
    path=RUN/'resources/jobs.json';jobs=read(path) if path.exists() else []
    assert not any(j['action']==action for j in jobs),'ALREADY_SUBMITTED_NO_DUPLICATE'
    command=['sbatch','--parsable','--job-name=m12c_'+action]+(options or [])+[script]+(arguments or [action])
    job=int(subprocess.check_output(command,text=True).strip().split(';')[0])
    jobs.append(dict(id=job,action=action,command=command,submitted_utc=utc()))
    write(path,jobs);print('CONFIRM_SUBMITTED',job,action,flush=True);return job


def submit_prepare():
    submit_job('prepare','submit/m12_confirm_cpu.sh')


def submit_pipeline():
    require()
    smoke_job=submit_job('smoke','submit/m12_confirm_gpu.sh')
    generation=submit_job('generate','submit/m12_confirm_gpu.sh',
        ['--dependency=afterok:'+str(smoke_job),'--array=0-5%6'])
    evaluation=submit_job('evaluate','submit/m12_confirm_gpu.sh',['--dependency=afterok:'+str(generation)])
    precision=submit_job('cpu_lpips','submit/m12_confirm_cpu.sh',['--dependency=afterok:'+str(evaluation)])
    submit_job('finalize','submit/m12_confirm_cpu.sh',['--dependency=afterok:'+str(precision)])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','smoke','generate','evaluate','cpu_lpips','finalize','accounting','submit_prepare','submit_pipeline'])
    parser.add_argument('--arm',choices=ARMS);args=parser.parse_args();(RUN/'resources').mkdir(parents=True,exist_ok=True)
    if args.action=='generate':generate(args.arm or ARMS[int(os.environ['SLURM_ARRAY_TASK_ID'])])
    elif args.action in ['evaluate','cpu_lpips']:
        from tools import m12_gain_confirm_eval as evaluator
        getattr(evaluator,args.action)()
    else:globals()[args.action]()
