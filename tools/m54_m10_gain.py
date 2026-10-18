"""已批准的单臂M10_GAIN实验：800→合格才累计2400→dev32，禁止扩展路线。"""
import argparse
import csv
import math
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from tools.m54_protocol import OUT as ORIGINAL, CONFIG, E5, E5_SHA, B1, B1_SHA
from tools.m54_protocol import read, write, sha, commit, split_path, require_input_review, freeze

RUN=ORIGINAL/'m10_gain_r1'
ARM='M10_GAIN'
CORE=['lpips_gt','contour_f1','iou','leak']
GPU_SH='submit/m54_m10_gain_gpu.sh'
CPU_SH='submit/m54_m10_gain_cpu.sh'
TRAIN=RUN/'training'


def utc():return datetime.now(timezone.utc).isoformat()
def rows(stage):return read(split_path(stage))
def image(stage,arm,identity):return RUN/stage/'images'/arm/(identity+'.png')
def bases(stage):return ['E5','B1','M10']+(['M10_MEAN'] if stage=='pilot' else [])


def require_protocol():
    p=read(RUN/'protocol.json')
    assert p['arm']==ARM and p['maximum_additional_updates']==2400 and p['parameters']==1280
    assert p['before_training'] and p['before_candidate_outputs']
    assert p['source_policy_sha256']==sha(ORIGINAL/'selection/pilot_policy.json')
    for stage,h in p['split_hashes'].items():assert sha(split_path(stage))==h
    assert sha(ORIGINAL/'splits/train1024_records.json')==p['training_records_sha256']
    assert sha(ORIGINAL/'manifests/e5_training_meta.json')==p['e5_training_meta_sha256']
    assert sha(E5)==E5_SHA and sha(B1)==B1_SHA
    return p


def prepare():
    (RUN/'resources').mkdir(parents=True,exist_ok=True)
    if (RUN/'protocol.json').exists():require_protocol();return
    assert not list(RUN.rglob('*.pt')) and not list(RUN.glob('*/images/*/*.png'))
    require_input_review('pilot')
    assert [len(rows(s)) for s in ['train1024','pilot','dev32']]==[1024,24,32]
    ids=[{r['id'] for r in rows(s)} for s in ['train1024','pilot','dev32']]
    assert not (ids[0]&ids[1] or ids[0]&ids[2] or ids[1]&ids[2])
    hashes={**read(ORIGINAL/'manifests/input_hashes.json'),**read(ORIGINAL/'manifests/input_hashes_r1.json')}
    for stage in ['train1024','pilot','dev32']:
        for row in rows(stage):
            for key in ['gt','sketch','reference']:assert sha(row[key])==hashes[row[key]]
    assert sha(E5)==E5_SHA and sha(B1)==B1_SHA
    p=dict(experiment='M54_M10_GAIN_R1',authorization='开始实施：确认单训练臂、最多新增2400有效更新',
        arm=ARM,parameters=1280,initial_baseline='original frozen M10, NOT zero-output E5',
        formula='alpha_c(t)=2*sigmoid(a_c+b_c*(2*t/999-1)); a=b=0; multiply original full statistics residual before frozen output projection',
        strength_range=[0.,2.],channels=640,layer=CONFIG['layer'],mean_std_coupled=True,new_spatial_mask=False,
        frozen=['E5','B1','Writer','reference projections','output projection','original B1 gain'],
        original_b1_joint='OFF',maximum_additional_updates=2400,
        training={**CONFIG['training'],'updates':800,'deepening_total_updates':2400,
            'teacher':'same-state frozen E5','loss':'diffusion MSE + 0.1 outside E5 keep',
            'checkpoint_selection':'fixed 800/2400 final; no dev choice'},
        inference=dict(size=[384,512],steps=50,seed=42,cfg=7.),
        pilot_rules=dict(mean_delta_lpips_lt=0.,mean_delta_f1_ge=-.05,tail_events_lt=3),
        six_gates=CONFIG['safety'],bootstrap=dict(unit='identity',samples=10000,seed=32042),
        source_policy_sha256=sha(ORIGINAL/'selection/pilot_policy.json'),
        source_hashes={str(E5):E5_SHA,str(B1):B1_SHA},
        split_hashes={s:sha(split_path(s)) for s in ['train1024','pilot','dev32']},
        training_records_sha256=sha(ORIGINAL/'splits/train1024_records.json'),
        e5_training_meta_sha256=sha(ORIGINAL/'manifests/e5_training_meta.json'),
        scope='one arm; Pilot24; qualifying arm resumes800to2400; dev32; no Lock/Natural/seeds/other training',
        development_only=True,independent_human_review='NOT_RUN',semantic_oracle_proven=False,
        before_training=True,before_candidate_outputs=True,frozen_utc=utc(),commit=commit())
    freeze(RUN/'protocol.json',p)
    for stage in ['train1024','pilot','dev32']:freeze(RUN/'splits'/(stage+'.json'),rows(stage))
    for name in ['input_hashes.json','input_hashes_r1.json','model_file_hashes.json']:
        shutil.copy2(ORIGINAL/'manifests'/name,RUN/'splits'/name)
    print('M10_GAIN PROTOCOL FROZEN',sha(RUN/'protocol.json'),flush=True)


def runtime(namespace):
    from tools.m54_m10_gain_runtime import make_runtime
    return make_runtime(namespace)


def loss_forward(r,args,kw,noise,mask):
    # 与已有M54完全相同的扩散/区域外保护项；不增加新科学loss。
    from tools.e44_train import loss_forward as original_loss
    return original_loss(r,args,kw,noise,mask)


def audit():
    import torch
    from tools.m54_runtime import make_runtime,release
    from tools.m54_run import Inputs,seed_all
    from tools.e47_runtime import exact_trace
    require_protocol();examples=rows('pilot')[:2];replays=[]
    for arm in ['E5','B1','M10']:
        r=make_runtime(arm,'m10_gain_r1/audit/replay/'+arm)
        try:
            for row in examples:
                target=image('audit/replay',arm,row['id']);r.generate(row,target)
                original=ORIGINAL/'pilot/images'/arm/(row['id']+'.png')
                assert sha(target)==sha(original),('BASELINE_REPLAY',arm,row['id'])
                oldtrace=original.with_suffix('.pt')
                if not oldtrace.exists():oldtrace=ORIGINAL/'m10_preflight_r1/audit/replay'/arm/(row['id']+'.pt')
                check=exact_trace(target.with_suffix('.pt'),oldtrace);assert check['bit_exact']
                replays.append(dict(arm=arm,id=row['id'],png_exact=True,trace=check))
        finally:release(r)
    r=runtime('audit/initial')
    try:
        for row in examples:
            target=image('audit/initial',ARM,row['id']);r.generate(row,target)
            original=image('audit/replay','M10',row['id'])
            assert sha(target)==sha(original)
            assert exact_trace(target.with_suffix('.pt'),original.with_suffix('.pt'))['bit_exact']
            r.effects_enabled=False;p=image('audit/profile',ARM,row['id']);r.profile(row,p)
            assert sha(p)==sha(target);r.effects_enabled=True
        r.active=False;row=examples[0];p=image('audit/off',ARM,row['id']);r.generate(row,p)
        assert sha(p)==sha(image('audit/replay','E5',row['id']))
        assert exact_trace(p.with_suffix('.pt'),image('audit/replay','E5',row['id']).with_suffix('.pt'))['bit_exact']
    finally:release(r)
    # 独立训练实例，避免inference_mode缓存进入反向图；不执行任何optimizer更新。
    seed_all();r=runtime('audit/gradient')
    try:
        data=Inputs(r);seed_all()
        for _ in range(32):
            args,kw,noise,mask,record=data.next(disable_dropout=True)
            if record['mask_valid'] and record['t'] not in [499,500]:break
        else:raise AssertionError('NO_VALID_TRAIN_INPUT')
        assert list(args[0].shape)==[1,4,64,48]
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
            r.processor.mode='M10';original=r.pipe.unet(*args,**kw).sample
            r.processor.mode=ARM;initial=r.pipe.unet(*args,**kw).sample
        assert torch.equal(original,initial),'TRAIN_INITIAL_NOT_EXACT_M10'
        loss,_=loss_forward(r,args,kw,noise,mask);(loss*1024).backward()
        gradients={n:dict(norm=float(p.grad.float().norm()/1024),finite=bool(torch.isfinite(p.grad).all()),
            live_elements=int(p.grad.count_nonzero()),elements=p.numel()) for n,p in r.gain.named_parameters() if p.grad is not None}
        assert set(gradients)=={'a','b'} and all(x['finite'] and x['norm']>0 and x['live_elements']==x['elements'] for x in gradients.values())
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
            r.active=False;off=r.pipe.unet(*args,**kw).sample
            r.active=True;r.allow_reference=False;ref_off=r.pipe.unet(*args,**kw).sample
        assert torch.equal(off,ref_off)
    finally:release(r)
    write(RUN/'audit/round0_complete.json',dict(passed=True,replays=replays,initial_full50_exact_M10=True,
        production_profiles=2,off_full50_exact_E5=True,training_initial_exact_M10=True,reference_off_exact=True,
        input_shape=[1,4,64,48],gradients=gradients,parameters=1280,scientific_updates=0,
        protocol_sha256=sha(RUN/'protocol.json'),commit=commit()))
    print('M10_GAIN ROUND0 PASS',flush=True)


def train(total=800):
    import random,time
    import numpy as np
    import torch
    from tools.m54_run import Inputs,seed_all
    from tools.m54_runtime import release
    require_protocol();assert total in [800,2400] and read(RUN/'audit/round0_complete.json')['passed']
    if total==2400:assert read(RUN/'selection.json')['selected']==[ARM]
    done=TRAIN/('complete%04d.json'%total)
    if done.exists():assert read(done)['updates']==total;return
    seed_all();r=runtime('training/runtime');data=Inputs(r);seed_all()
    optimizer=torch.optim.AdamW(r.parameters(),lr=1e-4,weight_decay=.01)
    scaler=torch.amp.GradScaler('cuda');step=0;history=[];sequence=[];started=time.perf_counter()
    live={n:torch.zeros_like(p,dtype=torch.bool) for n,p in r.gain.named_parameters()}
    try:
        if total==2400:
            state=r.load(TRAIN/'checkpoints/step0800.pt');assert state['step']==800
            optimizer.load_state_dict(state['optimizer']);scaler.load_state_dict(state['scaler']);step=800
            resume=state['resume'];data.data.order=resume['order'];data.data.rng.setstate(resume['data_rng']);data.count=resume['count']
            random.setstate(resume['python_rng']);np.random.set_state(resume['numpy_rng']);torch.set_rng_state(resume['torch_rng']);torch.cuda.set_rng_state_all(resume['cuda_rng'])
            sequence=read(TRAIN/'random_sequence0800.json');history=read(TRAIN/'history.json')
            assert data.count==len(sequence)==6400
        else:
            assert not (TRAIN/'checkpoints/step0000.pt').exists(),'PRESERVE_FAILED_ATTEMPT_BEFORE_RETRY'
            r.save(TRAIN/'checkpoints/step0000.pt',0,optimizer,scaler);seed_all()
        while step<total:
            optimizer.zero_grad(set_to_none=True);items=[]
            for _ in range(8):
                args,kw,noise,mask,record=data.next();sequence.append(record)
                loss,item=loss_forward(r,args,kw,noise,mask);assert torch.isfinite(loss)
                if loss.requires_grad:scaler.scale(loss/8).backward()
                items.append(item)
            assert any(p.grad is not None for p in r.parameters()),'EMPTY_EFFECTIVE_UPDATE'
            scaler.unscale_(optimizer);norm=float(torch.nn.utils.clip_grad_norm_(r.parameters(),1.));assert math.isfinite(norm)
            for n,p in r.gain.named_parameters():
                if p.grad is not None:
                    assert torch.isfinite(p.grad).all();live[n]|=p.grad!=0
            before=scaler.get_scale();scaler.step(optimizer);scaler.update();assert scaler.get_scale()>=before,'OPTIMIZER_UPDATE_SKIPPED'
            step+=1
            if step in [1,5,20] or step%25==0:
                item=dict(step=step,gradient_norm=norm,seconds=time.perf_counter()-started,
                    **{k:sum(x[k] for x in items)/8 for k in items[0]})
                history.append(item);write(TRAIN/'history.json',history);print('M10_GAIN TRAIN',step,item,flush=True)
            if step in [100,400,800,2400]:
                path=TRAIN/'checkpoints'/('step%04d.pt'%step);r.save(path,step,optimizer,scaler)
                state=torch.load(path,map_location='cpu',weights_only=False)
                state['resume']=dict(order=data.data.order,data_rng=data.data.rng.getstate(),count=data.count,
                    python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
                torch.save(state,path);write(path.with_suffix('.json'),dict(step=step,sha256=sha(path),commit=commit()))
        assert all(bool(v.all()) for v in live.values()) and len(sequence)==8*total
        write(TRAIN/('random_sequence%04d.json'%total),sequence)
        write(done,dict(updates=total,additional_updates=800 if total==800 else 1600,parameters=1280,
            live_gradient_elements={n:int(v.sum()) for n,v in live.items()},sequence_sha256=sha(TRAIN/('random_sequence%04d.json'%total)),
            seconds=time.perf_counter()-started,peak_memory_gib=torch.cuda.max_memory_allocated()/2**30,commit=commit()))
    finally:release(r)


def verify_training(total):
    import torch
    done=read(TRAIN/('complete%04d.json'%total));seq=read(TRAIN/('random_sequence%04d.json'%total))
    assert done['updates']==total and done['parameters']==1280 and len(seq)==8*total
    assert done['live_gradient_elements']=={'a':640,'b':640}
    assert sha(TRAIN/('random_sequence%04d.json'%total))==done['sequence_sha256']
    history=read(TRAIN/'history.json');assert history[-1]['step']==total and all(math.isfinite(v) for x in history for v in x.values())
    hashes={}
    for step in ([0,100,400,800] if total==800 else [800,2400]):
        path=TRAIN/'checkpoints'/('step%04d.pt'%step);assert sha(path)==read(path.with_suffix('.json'))['sha256']
        state=torch.load(path,map_location='cpu',weights_only=False)
        assert state['step']==step and state['arm']==ARM and state['protocol']==require_protocol()
        assert set(state['state_dict'])=={'a','b'} and all(v.shape==(640,) and torch.isfinite(v).all() for v in state['state_dict'].values())
        assert state['optimizer'] is not None and state['scaler'] is not None
        if step:
            assert state['resume']['count']==8*step and all(k in state['resume'] for k in ['order','data_rng','python_rng','numpy_rng','torch_rng','cuda_rng'])
            assert all(float(x['step'])==step for x in state['optimizer']['state'].values())
            assert state['optimizer']['param_groups'][0]['lr']==1e-4
        hashes[str(step)]=sha(path)
    if total==2400:
        assert seq[:6400]==read(TRAIN/'random_sequence0800.json')
        assert hashes['800']==read(RUN/'audit/training800.json')['checkpoints']['800']
    value=dict(passed=True,updates=total,exposures=len(seq),checkpoints=hashes,sequence_sha256=done['sequence_sha256'],
        parameters=1280,all_gradient_elements_live=True,all_history_finite=True,commit=commit())
    write(RUN/'audit'/('training'+str(total)+'.json'),value);return value


def reuse(stage):
    copied=[]
    for arm in bases(stage):
        source=ORIGINAL/('pilot' if stage=='pilot' else 'dev32_u2400')
        if arm=='M10_MEAN':source=ORIGINAL/'m10_preflight_r1/pilot'
        paths=list((source/'images'/arm).glob('*'))+[source/(arm+'_complete.json'),source/(arm+'_profile.json')]
        for src in paths:
            if src.suffix=='.pt':continue
            dst=RUN/stage/src.relative_to(source);dst.parent.mkdir(parents=True,exist_ok=True)
            if not dst.exists():shutil.copy2(src,dst)
            assert sha(dst)==sha(src);copied.append(dict(path=str(dst.relative_to(RUN)),sha256=sha(src)))
    write(RUN/'audit'/(stage+'_baseline_reuse.json'),dict(passed=True,baseline_new_outputs=0,files=copied))


def verify_generation(stage,full=False):
    from tools import m54_m10_preflight as checks
    checks.RUN=RUN;checks.BASES=bases(stage);checks.rows=rows;checks.image=image
    result=checks.verify_generation(stage,[ARM],full)
    total=800 if stage=='pilot' else 2400
    c=read(RUN/stage/(ARM+'_complete.json'))
    assert c['updates']==total and c['checkpoint_sha256']==sha(TRAIN/'checkpoints'/('step%04d.pt'%total))
    assert c['protocol_sha256']==sha(RUN/'protocol.json')
    for row in rows(stage):assert read(image(stage,ARM,row['id']).with_suffix('.json'))['parameters']==1280
    return result


def generate(stage):
    from tools.m54_runtime import release
    require_protocol();total=800 if stage=='pilot' else 2400;verify_training(total)
    if stage=='dev32':assert read(RUN/'selection.json')['selected']==[ARM]
    reuse(stage);r=runtime(stage+'/runtime')
    try:
        checkpoint=TRAIN/'checkpoints'/('step%04d.pt'%total);assert r.load(checkpoint)['step']==total
        r.gain.eval().requires_grad_(False)
        for row in rows(stage):
            p=image(stage,ARM,row['id']);assert not p.exists(),'PRESERVE_FAILED_GENERATION_BEFORE_RETRY'
            r.generate(row,p);print('M10_GAIN RGB',stage,row['id'],flush=True)
        profiles=[];r.effects_enabled=False
        for row in rows(stage)[:2]:
            p=image('profiles/'+stage,ARM,row['id']);profiles.append(r.profile(row,p))
            assert sha(p)==sha(image(stage,ARM,row['id']))
        write(RUN/stage/(ARM+'_profile.json'),dict(n=2,records=profiles))
        write(RUN/stage/(ARM+'_complete.json'),dict(n=len(rows(stage)),seed=42,updates=total,
            checkpoint_sha256=sha(checkpoint),protocol_sha256=sha(RUN/'protocol.json'),split_sha256=sha(split_path(stage)),commit=commit()))
    finally:release(r)
    verify_generation(stage)


def verify_evaluation(stage):
    import numpy as np
    records=read(RUN/stage/'per_image.json');gates=read(RUN/stage/'gates.json');arms=bases(stage)+[ARM]
    assert len(records)==len(rows(stage))*len(arms) and set(r['variant'] for r in records)==set(arms)
    old=read(ORIGINAL/('pilot' if stage=='pilot' else 'dev32_u2400')/'per_image.json')
    if stage=='pilot':old+=read(ORIGINAL/'m10_preflight_r1/pilot/per_image.json')
    index={(r['variant'],r['id']):r for r in old};count=0
    for arm in arms:
        group=[r for r in records if r['variant']==arm]
        assert len(group)==len(rows(stage)) and {r['id'] for r in group}=={r['id'] for r in rows(stage)}
        assert all(math.isfinite(r[k]) for r in group for k in CORE)
        for r in group:
            meta=read(image(stage,arm,r['id']).with_suffix('.json'))
            assert r['inference_mask_valid']==meta['mask']['valid'] and sha(image(stage,arm,r['id']))==meta['png_sha256']
            if arm!=ARM:assert all(r[k]==index[arm,r['id']][k] for k in CORE)
        if arm=='E5':continue
        assert gates[arm]['regional_metrics_complete']
        for base in ['E5','B1']+(['M10'] if arm==ARM else []):
            ref={r['id']:r for r in records if r['variant']==base}
            for key in CORE:
                values=np.array([r[key]-ref[r['id']][key] for r in group],np.float64)
                rng=np.random.default_rng(32042);sample=values[rng.integers(0,len(values),(10000,len(values)))].mean(1)
                actual=dict(n=len(values),mean=float(values.mean()),ci95=np.percentile(sample,[2.5,97.5]).tolist())
                expected=gates[arm]['paired_'+base][key] if base!='M10' else read(RUN/stage/'paired_M10.json')[ARM][key]
                assert actual==expected;count+=1
        d=gates[arm]['paired_E5'];ref={r['id']:r for r in records if r['variant']=='E5'}
        tail=[r['id'] for r in group if r['contour_f1']-ref[r['id']]['contour_f1']<-.15]
        check=dict(f1=d['contour_f1']['ci95'][0]>=-.020,iou=d['iou']['ci95'][0]>=-.010,
            leak=d['leak']['ci95'][1]<=.005,tail=not tail,lpips=d['lpips_gt']['ci95'][1]<0,
            retention=gates[arm]['baseline_B1_gain']>0 and -d['lpips_gt']['mean']>=.5*gates[arm]['baseline_B1_gain'])
        assert check==gates[arm]['checks'] and all(check.values())==gates[arm]['passed']
        assert set(tail)=={r['id'] for r in gates[arm]['extreme_events']}
    value=dict(passed=True,records=len(records),paired_n=len(rows(stage)),independent_bootstrap_exact=count,
        baseline_core_exact=True,all_core_finite=True,fixed_six_gates_exact=True,commit=commit())
    write(RUN/'audit'/(stage+'_evaluation.json'),value);return value


def evaluate(stage):
    from tools import m54_eval as evaluator
    from tools.e44_metrics import paired
    verify_generation(stage);evaluator.OUT=RUN;evaluator.image_path=image;evaluator.split_path=split_path
    evaluator.evaluate(stage)
    records=read(RUN/stage/'per_image.json')
    write(RUN/stage/'paired_M10.json',{ARM:{k:paired(records,ARM,'M10',k) for k in CORE}})
    verify_evaluation(stage)


def select():
    require_protocol();verify_evaluation('pilot');g=read(RUN/'pilot/gates.json')[ARM];d=g['paired_E5']
    criteria=dict(lpips=d['lpips_gt']['mean']<0,f1=d['contour_f1']['mean']>=-.05,tail=len(g['extreme_events'])<3)
    return freeze(RUN/'selection.json',dict(selected=[ARM] if all(criteria.values()) else [],criteria=criteria,
        policy_sha256=sha(ORIGINAL/'selection/pilot_policy.json'),protocol_sha256=sha(RUN/'protocol.json'),maximum_total_updates=2400))


def finalize():
    import torch
    from tools.e44_runtime import digest
    from tools.m54_finalize import package
    require_protocol();verify_training(800);stages=['pilot'];selected=read(RUN/'selection.json')['selected']
    if selected:verify_training(2400);stages+=['dev32']
    checks={}
    for stage in stages:checks[stage]=verify_generation(stage,True);verify_evaluation(stage)
    audit_count=0
    for path in (RUN/'audit').rglob('*.pt'):
        raw=torch.load(path,map_location='cpu',weights_only=False);meta=read(path.with_suffix('.json'))
        assert sha(path)==meta['diagnostics_sha256'] and digest(raw['initial_latent'])==meta['initial_sha256'] and digest(raw['final_latent'])==meta['final_sha256']
        assert all(len(raw[k])==50 and torch.isfinite(raw[k]).all() for k in ['conditional_epsilon','unconditional_epsilon']);audit_count+=1
    frozen=list(RUN.rglob('frozen_*.json'));assert frozen and all(read(p)['unchanged'] for p in frozen)
    total=2400 if selected else 800
    for stage in stages:
        for row in rows(stage):
            for key in ['gt','sketch','reference']:
                src=Path(row[key]);dest=RUN/'inputs'/row['id']/(key+src.suffix);dest.parent.mkdir(parents=True,exist_ok=True)
                if not dest.exists():shutil.copy2(src,dest)
                assert sha(src)==sha(dest)
    write(RUN/'audit/final_complete.json',dict(passed=True,stages=checks,audit_full50_traces=audit_count,
        scientific_updates=total,maximum_2400_respected=True,dev_passers=[ARM] if selected and read(RUN/'dev32/gates.json')[ARM]['passed'] else [],
        source_freeze_records=len(frozen),lock_natural_seeds='NOT_RUN_OUTSIDE_SCOPE',independent_human_review='NOT_RUN',semantic_oracle_proven=False,commit=commit()))
    files={str(p.relative_to(RUN)):p for p in RUN.rglob('*') if p.is_file() and p.suffix in ['.json','.png','.jpg','.csv','.log','.err','.psv'] and '.tar.gz' not in p.name}
    package(RUN/'m54_m10_gain_review.tar.gz',files)
    print('M10_GAIN SERVER FINAL PASS',flush=True)


def submit(name,action,gpu=True,stage=None,total=None):
    jobs=read(RUN/'resources/jobs.json') if (RUN/'resources/jobs.json').exists() else []
    if any(j['name']==name for j in jobs):return
    cmd=['sbatch','--parsable','--job-name=M10G_'+name,GPU_SH if gpu else CPU_SH,action]
    if stage:cmd+=['--stage',stage]
    if total:cmd+=['--updates',str(total)]
    out=subprocess.run(cmd,text=True,capture_output=True);assert out.returncode==0,out.stderr
    j=dict(name=name,id=int(out.stdout.strip().split(';')[0]),gpu=gpu,action=action,stage=stage,updates=total,commit=commit(),utc=utc())
    jobs.append(j);write(RUN/'resources/jobs.json',jobs);print('M10_GAIN SUBMITTED',j,flush=True)


def accounting():
    from tools.m54_finalize import package
    if (RUN/'m54_m10_gain_ledger.tar.gz').exists():return
    jobs=read(RUN/'resources/jobs.json')
    raw=subprocess.check_output(['sacct','-X','-P','-j',','.join(str(j['id']) for j in jobs),
        '--format=JobIDRaw,JobName,State,ExitCode,ElapsedRaw,AllocCPUS,AllocTRES,Start,End'],text=True)
    (RUN/'resources/accounting.psv').write_text(raw,encoding='utf-8')
    write(RUN/'resources/accounting.json',dict(utc=utc(),records=list(csv.DictReader(raw.splitlines(),delimiter='|')),
        scientific_updates=read(RUN/'audit/final_complete.json')['scientific_updates'],allocated_resources_only=True))
    package(RUN/'m54_m10_gain_ledger.tar.gz',{n:RUN/'resources'/n for n in ['jobs.json','accounting.psv','accounting.json']})


def advance():
    import fcntl
    prepare()
    with (RUN/'resources/dispatch.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        jobs=read(RUN/'resources/jobs.json') if (RUN/'resources/jobs.json').exists() else []
        if not jobs:submit('ROUND0','audit');return
        raw=subprocess.check_output(['sacct','-X','-n','-P','-j',','.join(str(j['id']) for j in jobs),'--format=JobIDRaw,State,ExitCode'],text=True)
        status={int(x.split('|')[0]):x.split('|')[1:] for x in raw.splitlines() if x.split('|')[0].isdigit()}
        print('M10_GAIN STATES',[(j['name'],j['id'],status.get(j['id'])) for j in jobs],flush=True)
        if any(status.get(j['id'],['PENDING'])[0].split()[0] in ['FAILED','TIMEOUT','OUT_OF_MEMORY','CANCELLED','NODE_FAIL'] for j in jobs):
            print('ENGINEERING_REPAIR_REQUIRED',flush=True);return
        complete=lambda n:any(j['name']==n and status.get(j['id'],[''])[0]=='COMPLETED' and status[j['id']][1]=='0:0' for j in jobs)
        if not complete('ROUND0'):return
        assert read(RUN/'audit/round0_complete.json')['passed'];submit('TRAIN800','train',total=800)
        if not complete('TRAIN800'):return
        verify_training(800);submit('PILOT','generate',stage='pilot')
        if not complete('PILOT'):return
        verify_generation('pilot');submit('PILOT_EVAL','evaluate',stage='pilot')
        if not complete('PILOT_EVAL'):return
        selected=select()['selected']
        if selected:
            submit('DEEPEN2400','train',total=2400)
            if not complete('DEEPEN2400'):return
            verify_training(2400);submit('DEV32','generate',stage='dev32')
            if not complete('DEV32'):return
            verify_generation('dev32');submit('DEV_EVAL','evaluate',stage='dev32')
            if not complete('DEV_EVAL'):return
            verify_evaluation('dev32')
        submit('FINAL','finalize',False)
        if complete('FINAL'):accounting();print('M10_GAIN SERVER_COMPLETE_LOCAL_REVIEW_PENDING',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','advance','audit','train','generate','evaluate','finalize','accounting'])
    p.add_argument('--stage',choices=['pilot','dev32'],default='pilot');p.add_argument('--updates',type=int,choices=[800,2400],default=800);args=p.parse_args()
    if os.environ.get('SLURM_JOB_ID'):
        import torch
        torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK','2')))
        files=['models/m54_m10_gain.py','models/m54_candidates.py','tools/m54_m10_gain.py','tools/m54_m10_gain_runtime.py',
            'tools/m54_m10_preflight.py','tools/m54_runtime.py','tools/m54_run.py','tools/e44_train.py','tools/m54_eval.py',GPU_SH,CPU_SH]
        write(RUN/'resources/code_entries'/(os.environ['SLURM_JOB_ID']+'.json'),dict(commit=commit(),arguments=vars(args),source_files={f:sha(f) for f in files}))
    if args.action=='train':train(args.updates)
    elif args.action in ['generate','evaluate']:globals()[args.action](args.stage)
    else:globals()[args.action]()


if __name__=='__main__':main()
