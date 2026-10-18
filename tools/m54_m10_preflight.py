"""M10已确认的零训练前置实验；与已交付M54结果隔离，绝不调用旧advance。"""
import argparse
import csv
import json
import math
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from tools.m54_protocol import OUT as ORIGINAL, CONFIG, E5, E5_SHA, B1, B1_SHA
from tools.m54_protocol import read, write, sha, commit, split_path, require_input_review

RUN = ORIGINAL / 'm10_preflight_r1'
ARMS = ['M10_MEAN', 'M10_STD', 'M10_MASK']
BASES = ['E5', 'B1', 'M10']
CORE = ['lpips_gt', 'contour_f1', 'iou', 'leak']
GPU_SH = 'submit/m54_m10_preflight_gpu.sh'
CPU_SH = 'submit/m54_m10_preflight_cpu.sh'


def utc():
    return datetime.now(timezone.utc).isoformat()


def rows(stage):
    return read(split_path(stage))


def image(stage, arm, identity):
    return RUN/stage/'images'/arm/(identity+'.png')


def protocol():
    value = read(RUN/'protocol.json')
    assert value['arms'] == ARMS and value['scientific_updates'] == 0
    for stage, expected in value['splits_sha256'].items():
        assert sha(split_path(stage)) == expected
    assert sha(ORIGINAL/'selection/pilot_policy.json') == value['original_policy_sha256']
    return value


def prepare():
    (RUN/'resources').mkdir(parents=True,exist_ok=True)
    if (RUN/'protocol.json').exists():
        protocol(); return
    assert not list(RUN.glob('*/images/M10_*/*.png'))
    require_input_review('pilot')
    assert sha(E5) == E5_SHA and sha(B1) == B1_SHA
    inputs={**read(ORIGINAL/'manifests/input_hashes.json'),**read(ORIGINAL/'manifests/input_hashes_r1.json')}
    for stage in ['pilot','dev32']:
        for row in rows(stage):
            for key in ['gt','sketch','reference']:
                assert sha(Path(row[key]))==inputs[row[key]],('INPUT_SHA',row['id'],key)
    value = dict(experiment='M54_M10_PREFLIGHT_R1', authorization='先完成这个前置实验',
        arms=ARMS, scientific_updates=0, pilot_new_outputs=72, maximum_dev_candidates=1,
        inference=dict(size=[384,512],steps=50,seed=42,cfg=7.,original_b1_joint='OFF'),
        mechanisms=dict(M10_MEAN='仅(mu_reference-mu_target)，原冻结gain和输出投影',
            M10_STD='仅标准差迁移残差，target均值保持，原冻结gain和输出投影',
            M10_MASK='原M10完整残差乘原有Sketch softmask，不改mask阈值或valid门'),
        original_M10='原全空间均值+标准差迁移对照，保持代码运算顺序；不修改旧产物',
        pilot_rules=dict(mean_delta_lpips_lt=0.,mean_delta_f1_ge=-.05,tail_events_lt=3),
        ranking='tail count; F1 safety/lower CI; IoU safety/lower CI; LPIPS; parameters; latency; VRAM; arm',
        six_gates=CONFIG['safety'], bootstrap=dict(unit='identity',samples=10000,seed=32042),
        splits_sha256={s:sha(split_path(s)) for s in ['pilot','dev32']},
        original_policy_sha256=sha(ORIGINAL/'selection/pilot_policy.json'),
        source_hashes={str(E5):E5_SHA,str(B1):B1_SHA},
        development_only='Pilot/dev已用于M54及本轮方向选择，不冒充独立确认集',
        lock_natural='本前置实验不推进Lock/Natural/三seed或任何训练',
        independent_human_review='NOT_RUN', semantic_oracle_proven=False,
        before_candidate_outputs=True, before_evaluation=True, frozen_utc=utc(), commit=commit())
    write(RUN/'protocol.json',value)
    for stage in ['pilot','dev32']:
        write(RUN/'splits'/(stage+'.json'),rows(stage))
    for name in ['model_file_hashes.json','input_hashes.json','input_hashes_r1.json']:
        shutil.copy2(ORIGINAL/'manifests'/name,RUN/'splits'/name)
    print('M10 PROTOCOL FROZEN',sha(RUN/'protocol.json'),flush=True)


def reuse(stage):
    source = 'pilot' if stage == 'pilot' else 'dev32_u2400'
    copied=[]
    for arm in BASES:
        paths=list((ORIGINAL/source/'images'/arm).glob('*'))
        paths += [ORIGINAL/source/(arm+'_complete.json'),ORIGINAL/source/(arm+'_profile.json')]
        for src in paths:
            if src.suffix == '.pt': continue
            dest=RUN/stage/src.relative_to(ORIGINAL/source)
            dest.parent.mkdir(parents=True,exist_ok=True)
            if not dest.exists(): shutil.copy2(src,dest)
            assert sha(dest)==sha(src)
            copied.append(dict(path=str(dest.relative_to(RUN)),sha256=sha(src)))
    write(RUN/'audit'/(stage+'_baseline_reuse.json'),dict(passed=True,source=source,
        source_split_sha256=sha(split_path(stage)),files=copied,
        baseline_new_generation=0,raw_tensor_reuse=False,
        deleted_old_M10_trace='原删除前真实full50审计留证；本次不重建全部旧M10张量'))


def runtime(arm, namespace):
    from tools.m54_runtime import make_runtime
    r=make_runtime(arm,'m10_preflight_r1/'+namespace)
    assert not r.parameters()
    return r


def audit():
    import torch
    from tools.m54_runtime import release
    from tools.e47_runtime import exact_trace
    protocol(); examples=rows('pilot')[:2];checks=[]
    for arm in BASES:
        r=runtime(arm,'audit/runtime/'+arm)
        try:
            for row in examples:
                p=image('audit/replay',arm,row['id']);r.generate(row,p,seed=42)
                original=ORIGINAL/'pilot/images'/arm/(row['id']+'.png')
                assert sha(p)==sha(original),('PNG_REPLAY',arm,row['id'])
                trace=None
                if original.with_suffix('.pt').exists():
                    trace=exact_trace(p.with_suffix('.pt'),original.with_suffix('.pt'))
                    assert trace['bit_exact']
                checks.append(dict(arm=arm,id=row['id'],png_exact=True,full50_trace_exact=trace))
        finally: release(r)
    # 独立分支OFF必须沿完整50步回到E5；mask全1/全0只作工程契约测试。
    for arm in ARMS:
        r=runtime(arm,'audit/runtime/'+arm)
        try:
            row=examples[0];r.active=False
            off=image('audit/off',arm,row['id']);r.generate(row,off,seed=42)
            assert sha(off)==sha(ORIGINAL/'pilot/images/E5'/(row['id']+'.png'))
            assert exact_trace(off.with_suffix('.pt'),image('audit/replay','E5',row['id']).with_suffix('.pt'))['bit_exact']
            r.active=True
            r.generate(row,image('audit/active',arm,row['id']),seed=42)
        finally: release(r)
    for fill,baseline in [(1.,'M10'),(0.,'E5')]:
        r=runtime('M10_MASK','audit/mask_contract/'+str(int(fill)))
        original_hook=r.before_unet
        def hook(module,args,kw):
            original_hook(module,args,kw)
            if r.processor.enabled:r.processor.mask=torch.full_like(r.processor.mask,fill)
        r.unet_handle.remove();r.unet_handle=r.pipe.unet.register_forward_pre_hook(hook,with_kwargs=True)
        try:
            row=examples[0];p=image('audit/mask_'+str(int(fill)),'M10_MASK',row['id']);r.generate(row,p,seed=42)
            assert sha(p)==sha(image('audit/replay',baseline,row['id']))
            assert exact_trace(p.with_suffix('.pt'),image('audit/replay',baseline,row['id']).with_suffix('.pt'))['bit_exact']
        finally: release(r)
    write(RUN/'audit/round0_complete.json',dict(passed=True,replays=checks,off_exact_e5=True,
        mask_one_exact_m10=True,mask_zero_exact_e5=True,scientific_updates=0,protocol_sha256=sha(RUN/'protocol.json'),commit=commit()))
    print('M10 GPU ROUND0 PASS',flush=True)


def generate(arm, stage):
    from tools.m54_runtime import release
    protocol();assert arm in ARMS and read(RUN/'audit/round0_complete.json')['passed']
    assert not (RUN/stage/(arm+'_complete.json')).exists()
    if stage=='dev32':assert read(RUN/'selection.json')['selected']==[arm]
    r=runtime(arm,stage+'/'+arm+'/runtime')
    try:
        for row in rows(stage):
            p=image(stage,arm,row['id']);assert not p.exists(),'保留失败尝试后才能重试'
            meta,_,_=r.generate(row,p,seed=42)
            print('M10 RGB',stage,arm,row['id'],meta['seconds'],flush=True)
        r.effects_enabled=False;profile=[]
        for row in rows(stage)[:2]:
            p=image('profiles/'+stage,arm,row['id']);record=r.profile(row,p,seed=42)
            assert sha(p)==sha(image(stage,arm,row['id']))
            profile.append(record)
        write(RUN/stage/(arm+'_profile.json'),dict(n=2,records=profile))
        write(RUN/stage/(arm+'_complete.json'),dict(n=len(rows(stage)),seed=42,updates=0,
            arm=arm,split_sha256=sha(split_path(stage)),protocol_sha256=sha(RUN/'protocol.json'),commit=commit()))
    finally:release(r)


def verify_generation(stage, arms, full=False):
    import torch
    from tools.e44_runtime import digest
    n=0;live=0;gateoff={}
    for arm in BASES+arms:
        gateoff[arm]=[]
        assert read(RUN/stage/(arm+'_complete.json'))['n']==len(rows(stage))
        for row in rows(stage):
            p=image(stage,arm,row['id']);m=read(p.with_suffix('.json'));base=read(image(stage,'E5',row['id']).with_suffix('.json'))
            assert sha(p)==m['png_sha256'] and m['seed']==42
            assert m['initial_sha256']==base['initial_sha256'] and m['steps']==base['steps'] and len(m['steps'])==50
            assert not m['gt_used_in_inference']
            ef=read(p.with_name(p.stem+'_effects.json'));assert len(ef)==50
            if arm in arms:
                assert read(RUN/stage/(arm+'_complete.json'))['split_sha256']==sha(split_path(stage))
                assert sha(p.with_suffix('.pt'))==m['diagnostics_sha256']
                if not m['mask']['valid']:
                    gateoff[arm].append(row['id']);assert m['png_sha256']==base['png_sha256']
                if full:
                    raw=torch.load(p.with_suffix('.pt'),map_location='cpu',weights_only=False)
                    assert digest(raw['initial_latent'])==m['initial_sha256'] and digest(raw['final_latent'])==m['final_sha256']
                    for k in ['conditional_epsilon','unconditional_epsilon']:
                        assert len(raw[k])==50 and torch.isfinite(raw[k]).all()
                    live+=1
            n+=1
        profile=read(RUN/stage/(arm+'_profile.json'));assert profile['n']==2
        if arm in arms:
            for x in profile['records']:
                assert x['png_sha256']==sha(image(stage,arm,x['id']))
                assert sha(image('profiles/'+stage,arm,x['id']))==x['png_sha256']
    value=dict(passed=True,n=len(rows(stage)),conditions=n,new_arms=arms,gateoff_ids=gateoff,
        full50_new_traces_reloaded=live,protocol_sha256=sha(RUN/'protocol.json'),commit=commit())
    write(RUN/'audit'/(stage+'_generation'+('_full50' if full else '')+'.json'),value)
    return value


def evaluate(stage):
    from tools import m54_eval as evaluator
    arms=ARMS if stage=='pilot' else read(RUN/'selection.json')['selected']
    verify_generation(stage,arms)
    evaluator.OUT=RUN;evaluator.image_path=image;evaluator.split_path=split_path
    evaluator.evaluate(stage)
    # 另存相对原M10的配对值，不能仅看相对E5判断消融收益。
    from tools.e44_metrics import paired
    records=read(RUN/stage/'per_image.json')
    write(RUN/stage/'paired_M10.json',{a:{k:paired(records,a,'M10',k) for k in CORE} for a in arms})
    verify_evaluation(stage)


def verify_evaluation(stage):
    import numpy as np
    records=read(RUN/stage/'per_image.json');gates=read(RUN/stage/'gates.json')
    original_stage='pilot' if stage=='pilot' else 'dev32_u2400'
    old={(r['variant'],r['id']):r for r in read(ORIGINAL/original_stage/'per_image.json')}
    comparisons=0
    for arm in sorted({r['variant'] for r in records}):
        group=[r for r in records if r['variant']==arm]
        assert len(group)==len(rows(stage)) and {r['id'] for r in group}=={r['id'] for r in rows(stage)}
        assert all(math.isfinite(r[k]) for r in group for k in CORE)
        if arm in BASES:
            assert all(r[k]==old[arm,r['id']][k] for r in group for k in CORE)
        if arm=='E5':continue
        assert gates[arm]['regional_metrics_complete']
        for baseline in ['E5','B1']:
            ref={r['id']:r for r in records if r['variant']==baseline}
            for key in CORE:
                values=np.asarray([r[key]-ref[r['id']][key] for r in group],np.float64)
                rng=np.random.default_rng(32042);sample=values[rng.integers(0,len(values),(10000,len(values)))].mean(1)
                actual=dict(n=len(values),mean=float(values.mean()),ci95=np.percentile(sample,[2.5,97.5]).tolist())
                assert actual==gates[arm]['paired_'+baseline][key],(arm,baseline,key,actual)
                comparisons+=1
    value=dict(passed=True,records=len(records),paired_n=len(rows(stage)),all_core_finite=True,
        baseline_core_values_exact=len(BASES)*len(rows(stage))*4,independent_bootstrap_exact=comparisons,
        evaluation_sha256=sha(RUN/stage/'evaluation_complete.json'),commit=commit())
    write(RUN/'audit'/(stage+'_evaluation.json'),value)
    return value


def select():
    from tools.m54_selection import ranking
    p=protocol();verify_evaluation('pilot')
    g=read(RUN/'pilot/gates.json');m=read(RUN/'pilot/means.json');r=read(RUN/'pilot/per_image.json')
    eligible=[a for a in ARMS if g[a]['paired_E5']['lpips_gt']['mean']<0 and
        g[a]['paired_E5']['contour_f1']['mean']>=-.05 and len(g[a]['extreme_events'])<3]
    eligible=sorted(eligible,key=lambda a:ranking(a,m,g,r))
    selected=eligible[:1]
    value=dict(selected=selected,eligible=eligible,protocol_sha256=sha(RUN/'protocol.json'),
        original_policy_sha256=p['original_policy_sha256'],ranking=p['ranking'],development_only=True,scientific_updates=0)
    path=RUN/'selection.json'
    if path.exists():assert read(path)==value
    else:write(path,value)
    return value


def finalize():
    from tools.m54_finalize import package
    import torch
    from tools.e44_runtime import digest
    protocol();assert read(RUN/'audit/round0_complete.json')['passed']
    checks={'pilot':verify_generation('pilot',ARMS,full=True)};verify_evaluation('pilot')
    chosen=read(RUN/'selection.json')['selected'];dev_passers=[]
    if chosen:
        checks['dev32']=verify_generation('dev32',chosen,full=True);verify_evaluation('dev32')
        dev_passers=[a for a in chosen if read(RUN/'dev32/gates.json')[a]['passed']]
    audit_traces=0
    for p in (RUN/'audit').rglob('*.pt'):
        raw=torch.load(p,map_location='cpu',weights_only=False);m=read(p.with_suffix('.json'))
        assert sha(p)==m['diagnostics_sha256']
        assert digest(raw['initial_latent'])==m['initial_sha256'] and digest(raw['final_latent'])==m['final_sha256']
        for k in ['conditional_epsilon','unconditional_epsilon']:
            assert len(raw[k])==50 and torch.isfinite(raw[k]).all()
        audit_traces+=1
    for stage in checks:
        for row in rows(stage):
            for key in ['gt','sketch','reference']:
                src=Path(row[key]);dest=RUN/'inputs'/row['id']/(key+src.suffix)
                dest.parent.mkdir(parents=True,exist_ok=True)
                if not dest.exists():shutil.copy2(src,dest)
                assert sha(src)==sha(dest)
    frozen=list(RUN.glob('**/manifests/frozen_*.json'))
    assert frozen and all(read(p)['unchanged'] for p in frozen)
    value=dict(passed=True,stages=checks,audit_full50_traces=audit_traces,source_freeze_records=len(frozen),
        pilot_selected=chosen,dev_passers=dev_passers,scientific_updates=0,
        lock_natural_seeds_training='NOT_RUN_OUTSIDE_THIS_PREEXPERIMENT',
        independent_human_review='NOT_RUN',semantic_oracle_proven=False,commit=commit())
    write(RUN/'audit/final_complete.json',value)
    files={str(p.relative_to(RUN)):p for p in RUN.rglob('*') if p.is_file() and
        p.suffix in ['.json','.png','.csv','.log','.err','.psv'] and '.tar.gz' not in p.name}
    package(RUN/'m54_m10_preflight_review.tar.gz',files)
    print('M10 PREFLIGHT FINAL PASS',json.dumps(value),flush=True)


def submit(name, action, gpu, arm=None, stage=None):
    jobs=read(RUN/'resources/jobs.json') if (RUN/'resources/jobs.json').exists() else []
    if any(j['name']==name for j in jobs):return
    cmd=['sbatch','--parsable','--job-name=M10PF_'+name,GPU_SH if gpu else CPU_SH,action]
    if arm:cmd+=['--arm',arm]
    if stage:cmd+=['--stage',stage]
    out=subprocess.run(cmd,text=True,capture_output=True);assert out.returncode==0,out.stderr
    job=dict(name=name,id=int(out.stdout.strip().split(';')[0]),gpu=gpu,action=action,arm=arm,stage=stage,commit=commit(),utc=utc())
    jobs.append(job);write(RUN/'resources/jobs.json',jobs);print('M10 SUBMITTED',job,flush=True)


def accounting():
    from tools.m54_finalize import package
    jobs=read(RUN/'resources/jobs.json')
    raw=subprocess.check_output(['sacct','-X','-P','-j',','.join(str(x['id']) for x in jobs),
        '--format=JobIDRaw,JobName,State,ExitCode,ElapsedRaw,AllocCPUS,AllocTRES,Start,End'],text=True)
    (RUN/'resources/accounting.psv').write_text(raw,encoding='utf-8')
    write(RUN/'resources/accounting.json',dict(utc=utc(),records=list(csv.DictReader(raw.splitlines(),delimiter='|')),
        scientific_updates=0,allocated_resources_only=True))
    package(RUN/'m54_m10_preflight_ledger.tar.gz',{n:RUN/'resources'/n for n in ['jobs.json','accounting.psv','accounting.json']})


def advance():
    prepare();jobs=read(RUN/'resources/jobs.json') if (RUN/'resources/jobs.json').exists() else []
    if not jobs:submit('ROUND0','audit',True);return
    raw=subprocess.check_output(['sacct','-X','-n','-P','-j',','.join(str(j['id']) for j in jobs),'--format=JobIDRaw,State'],text=True)
    states={int(x.split('|')[0]):x.split('|')[1].split()[0] for x in raw.splitlines() if x.split('|')[0].isdigit()}
    print('M10 STATES',[(j['name'],j['id'],states.get(j['id'])) for j in jobs],flush=True)
    if any(states.get(j['id']) in ['FAILED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','CANCELLED'] for j in jobs):
        print('ENGINEERING_REPAIR_REQUIRED');return
    complete=lambda name:any(j['name']==name and states.get(j['id'])=='COMPLETED' for j in jobs)
    if not complete('ROUND0'):return
    assert read(RUN/'audit/round0_complete.json')['passed'];reuse('pilot')
    for arm in ARMS:submit('PILOT_'+arm,'generate',True,arm,'pilot')
    if not all(complete('PILOT_'+a) for a in ARMS):return
    verify_generation('pilot',ARMS);submit('PILOT_EVAL','evaluate',True,stage='pilot')
    if not complete('PILOT_EVAL'):return
    selected=select()['selected']
    if selected:
        reuse('dev32');arm=selected[0];submit('DEV_'+arm,'generate',True,arm,'dev32')
        if not complete('DEV_'+arm):return
        verify_generation('dev32',selected);submit('DEV_EVAL','evaluate',True,stage='dev32')
        if not complete('DEV_EVAL'):return
        verify_evaluation('dev32')
    submit('FINAL','finalize',False)
    if complete('FINAL'):accounting();print('M10 SERVER_COMPLETE_LOCAL_REVIEW_PENDING',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','advance','audit','generate','evaluate','finalize','accounting'])
    parser.add_argument('--arm',choices=ARMS);parser.add_argument('--stage',choices=['pilot','dev32'],default='pilot');args=parser.parse_args()
    if os.environ.get('SLURM_JOB_ID'):
        import torch
        torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK','2')))
        files=['tools/m54_m10_preflight.py','models/m54_candidates.py','tools/m54_runtime.py','tools/m54_eval.py',GPU_SH,CPU_SH]
        write(RUN/'resources/code_entries'/(os.environ['SLURM_JOB_ID']+'.json'),dict(commit=commit(),arguments=vars(args),source_files={p:sha(p) for p in files}))
    if args.action=='generate':generate(args.arm,args.stage)
    elif args.action=='evaluate':evaluate(args.stage)
    else:globals()[args.action]()


if __name__=='__main__':main()
