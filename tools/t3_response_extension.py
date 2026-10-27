"""用户在确认停止后明确要求的探索扩展；原确认协议与判定只读。"""
import argparse
import csv
import datetime
import hashlib
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path
import numpy as np
from tools.t3_response_common import PROJECT,SOURCE,ROOT as PARENT,SUP,ATTR,MODELS,CONFIG as METRIC_CONFIG
from tools.t3_response_common import read,write,freeze,sha,commit,csv_write

ROOT=PARENT/'exploratory_extension_20261027'
ARMS={
    'T3_no_body_s42':dict(seed=42,method='T3-no-body',body_rgb=0.,rest=.03816244801608925),
    'T3_no_rest_s42':dict(seed=42,method='T3-no-rest',body_rgb=.1,rest=0.),
    'C_s43':dict(seed=43,method='C',body_rgb=.1,boundary=.05,outside=.1),
    'T3_s43':dict(seed=43,method='T3',body_rgb=.1,rest=.03816244801608925),
    'C_s44':dict(seed=44,method='C',body_rgb=.1,boundary=.05,outside=.1),
    'T3_s44':dict(seed=44,method='T3',body_rgb=.1,rest=.03816244801608925),
}
PLAN=dict(protocol='T3-REFERENCE-RESPONSE-EXPLORATORY-EXTENSION-20261027',
    authorization='用户明确要求：先暂缓人工复核，完成后续实验',
    evidence_status='exploratory extension authorized after seeing confirmation results',
    original_confirmation_unchanged=True,original_gate_overridden_for_new_work_by_user=True,
    no_relabeling_as_independent_confirmation=True,selection_or_coefficients_not_tuned=True,
    arms=ARMS,updates=28365,effective_batch=8,world_size=2,accumulation=4,
    initialization='same frozen E1',training_data='same train_bf_texture.json',
    inference='same frozen128 identities/reference assignment, explicit identity-specific latent and DDIM50 settings',
    human_review='deferred by user; no human ratings generated',
    planned_contrasts=['T3-C seed42','T3-C seed43','T3-C seed44','T3-full minus no-body seed42','T3-full minus no-rest seed42'],
    multiplicity='Holm over all five exploratory paired DISTS contrasts',
    seed_summary='report three seed-level effects, mean and SD; never treat identities as independent training seeds',
    model_selection='final successful update only, no best checkpoint selection',
    resource_policy='up to3 submitted dual-A30 jobs, GPU QOS permits2 running; one CPU coordinator; all computations in Slurm jobs',
    failed_run_policy='preserve logs and stop scheduling on engineering failure; reviewed repair only')
TRAIN_SOURCES=['train_GAM_texture_joint.py','tools/e2a_attr_runtime.py','tools/e2a_sup_losses.py',
    'tools/t3_response_extension.py','adapter/attention_processor.py','models/bf_texture_module.py','texture_preprocess.py']


def folder(key): return ROOT/'training'/key


def checkpoint(key): return folder(key)/'checkpoint-final/joint_model.pt'


def status(**changes):
    p=ROOT/'status.json';v=read(p) if p.exists() else {}
    v.update(changes);v['updated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();write(p,v)


def prepare():
    for name in ['logs','audit','protocol','splits','metrics','statistics']:(ROOT/name).mkdir(parents=True,exist_ok=True)
    freeze(ROOT/'protocol/extension.json',PLAN)
    assert read(PARENT/'statistics/decision.json')['continue_new_training'] is False
    frozen={str(p.relative_to(PARENT)):sha(p) for p in [PARENT/'statistics/decision.json',
        PARENT/'protocol/config.json',PARENT/'splits/confirmation.json',PARENT/'protocol/model_registry.json']}
    freeze(ROOT/'protocol/original_confirmation_sha256.json',frozen)
    freeze(ROOT/'protocol/metric_config.json',METRIC_CONFIG)
    for name in ['confirmation.json','qualitative_fixed24.json']:
        freeze(ROOT/'splits'/name,read(PARENT/'splits'/name))
    historical=read(SUP/'training/formal/T3/config.json')
    assert sha(SOURCE/'train_GAM_texture_joint.py')==historical['base_source_sha256']
    assert sha(SOURCE/'tools/e2a_sup_losses.py')==historical['additions']['tools/e2a_sup_losses.py']
    assert historical['coefficient_registry']['coefficients']==dict(body_rgb=.1,rest=.03816244801608925)
    assert historical['world_size']==2 and historical['updates']==28365
    assert sha(MODELS['E1'][0])==MODELS['E1'][1]
    assert sha(PROJECT/'data/train_bf_texture.json')==read(PARENT/'audit/independence.json')['train_manifest_sha256']
    freeze(ROOT/'protocol/training_source_sha256.json',{p:sha(SOURCE/p) for p in TRAIN_SOURCES})
    freeze(ROOT/'protocol/initialization_registry.json',read(ATTR/'checkpoint_manifest.json'))
    for path,rec in read(ROOT/'protocol/initialization_registry.json').items():assert sha(path)==rec['sha256']
    write(ROOT/'audit/preparation_complete.json',dict(passed=True,commit=commit(),parent_sha256=frozen,
        world_size=2,updates=28365,effective_batch=8,original_confirmation_not_modified=True))


def verify_parent():
    for p,h in read(ROOT/'protocol/original_confirmation_sha256.json').items():assert sha(PARENT/p)==h,p


def training_source(key):
    original=(SOURCE/'train_GAM_texture_joint.py').read_text(encoding='utf-8')
    if ARMS[key]['method']=='C':return original
    replacements={
        '    runtime = AuditRuntime(args, accelerator)':'    runtime = AuditRuntime(args, accelerator)\n    EXT_HOOK.setup(locals())',
        '                if not torch.isfinite(loss.detach()).all()':
        '                loss = EXT_HOOK(locals(), globals())\n                if not torch.isfinite(loss.detach()).all()',
    }
    for before,after in replacements.items():
        assert original.count(before)==1,('HOOK_ANCHOR_CHANGED',before)
        original=original.replace(before,after)
    return original


class LossHook:
    def __init__(self,key):self.key=key;self.impl=None;self.pending=[];self.checked=False

    def setup(self,env):
        import torch
        self.acc=env['accelerator'];self.runtime=env['runtime']
        assert self.acc.num_processes==2 and env['args'].seed==ARMS[self.key]['seed']
        assert not any(p.requires_grad for p in env['vae'].parameters())
        assert env['args'].lambda_region_texture==env['args'].lambda_boundary==env['args'].lambda_leak==0
        original=self.runtime.log_update
        def log_update(step,optimizer,grad_norm):
            original(step,optimizer,grad_norm)
            if not self.pending:return
            keys=list(self.pending[0]);x=torch.tensor([np.mean([r[k] for r in self.pending]) for k in keys],device=self.acc.device)
            values=self.acc.reduce(x,reduction='mean').cpu().tolist()
            if self.acc.is_main_process:
                p=folder(self.key)/'auxiliary_log.csv';exists=p.exists()
                with p.open('a',newline='',encoding='utf-8') as f:
                    writer=csv.DictWriter(f,fieldnames=['update']+keys)
                    if not exists:writer.writeheader()
                    writer.writerow(dict(update=step,**dict(zip(keys,values))))
            self.pending.clear()
        self.runtime.log_update=log_update

    def __call__(self,env,namespace):
        import random
        import torch
        from tools.e2a_sup_losses import Supervision
        if self.impl is None:
            state=random.getstate();npstate=np.random.get_state()
            try:
                with torch.random.fork_rng(devices=[self.acc.device.index]):
                    self.impl=Supervision('T3',dict(namespace,**env))
            finally:random.setstate(state);np.random.set_state(npstate)
        with torch.cuda.amp.autocast(enabled=False):
            raw=self.impl.raw(env['decoded_loss'],env['target_loss'],env['texture_target_loss'],
                env['mask_loss'],env['timesteps'],env['texture_condition_weight'])
            weights=ARMS[self.key]
            auxiliary=sum(raw[k]*weights[k] for k in ['body_rgb','rest'])
            assert all(torch.isfinite(v).all().item() for v in raw.values())
            if not self.checked:
                # 仅使用已产生张量，不抽随机数，不修改主训练随机流。
                with torch.no_grad():
                    p,g,r,m=[env[k].detach() for k in ['decoded_loss','target_loss','texture_target_loss','mask_loss']]
                    t=env['timesteps'];w=torch.ones(p.shape[0],device=p.device)
                    a=self.impl.raw(p,g,r,m,t,w);b=self.impl.raw(p,g,-r,m,t,w)
                    assert torch.equal(a['rest'],b['rest'])
                    body,boundary,outside=self.impl.h['build_region_masks'](m,9)
                    assert torch.allclose(1-body,boundary+outside,atol=1e-6,rtol=0)
                    assert torch.equal(a['body_rgb'],self.impl.h['texture_color_stat_loss'](p,r,body,sample_weight=w))
                    assert torch.equal(a['rest'],self.impl.h['masked_l1_loss'](p,g,1-body))
                write(folder(self.key)/('loss_semantics_rank%d.json'%self.acc.process_index),dict(passed=True,
                    method=weights['method'],coefficients={k:weights[k] for k in ['body_rgb','rest']},
                    body_target='reference RGB mean/std',rest_target='GT outside eroded body',
                    base_color_loss_retained=True,main_rng_unchanged=True))
                self.checked=True
        self.pending.append(dict(body_rgb=float(raw['body_rgb'].detach()),rest=float(raw['rest'].detach()),
            weighted_aux=float(auxiliary.detach())))
        return env['loss']+auxiliary


def train(key):
    import torch
    verify_parent();assert read(ROOT/'audit/preparation_complete.json')['passed']
    spec=ARMS[key];run=folder(key);run.mkdir(parents=True,exist_ok=True)
    assert not (run/'checkpoint_verification.json').exists(),'NO_RETRAINING_COMPLETED_RUN'
    for p,h in read(ROOT/'protocol/training_source_sha256.json').items():assert sha(SOURCE/p)==h
    assert int(os.environ.get('WORLD_SIZE','1'))==2
    argv=list(read(ATTR/'manifests/historical_command.json'))
    native=spec['method']=='C'
    argv+=['--output_dir',str(run),'--attr_audit',str(run),'--attr_mask_cache',str(ATTR/'masks/train'),
        '--start_global_step','0','--seed',str(spec['seed']),'--gradient_accumulation_steps','4',
        '--max_train_steps','28365','--checkpointing_epochs','9999','--dataloader_num_workers','2',
        '--lambda_region_texture',str(.1 if native else 0),
        '--lambda_boundary',str(.05 if native else 0),'--lambda_leak',str(.1 if native else 0),
        '--report_to','none','--wandb_mode','disabled','--val_vis_steps','0','--vis_every_n_steps','0']
    transformed=training_source(key)
    cfg=dict(key=key,protocol=PLAN['protocol'],spec=spec,argv=argv,world_size=2,effective_batch=8,
        updates=28365,seed=spec['seed'],source_sha256=read(ROOT/'protocol/training_source_sha256.json'),
        transformed_source_sha256=hashlib.sha256(transformed.encode()).hexdigest())
    if int(os.environ.get('RANK','0'))==0:freeze(run/'config.json',cfg)
    hook=LossHook(key)
    namespace=dict(__name__='__main__',__file__=str(SOURCE/'train_GAM_texture_joint.py'),EXT_HOOK=hook)
    sys.argv=[namespace['__file__']]+argv
    exec(compile(transformed,namespace['__file__'],'exec'),namespace)
    if int(os.environ.get('RANK','0'))!=0:return
    count=read(run/'optimizer_update_count.json')
    assert count['actual_successful_updates']==28365 and count['skipped_updates']==0 and count['complete']
    streams=read(run/'seed_and_rng_manifest.json');assert len(streams['streams'])==2
    weights=torch.load(checkpoint(key),map_location='cpu',mmap=True,weights_only=False)
    tensors=[v for group in weights.values() if isinstance(group,dict) for v in group.values() if torch.is_tensor(v)]
    assert all(torch.isfinite(t).all().item() for t in tensors)
    write(run/'checkpoint_verification.json',dict(passed=True,key=key,seed=spec['seed'],actual_updates=28365,
        finite_tensors=len(tensors),sha256=sha(checkpoint(key)),path=str(checkpoint(key)),commit=commit()))
    print('EXTENSION_CHECKPOINT_VERIFIED',key,flush=True)


def evaluate(key):
    import fcntl
    from tools import e2a_attr_eval as evaluator
    from tools import t3_response_run as metrics
    assert read(folder(key)/'checkpoint_verification.json')['passed']
    rows=read(ROOT/'splits/confirmation.json')
    evaluator.OUT=ROOT;evaluator.model_path=lambda _:checkpoint(key)
    for stage in ['original','swapped']:
        variants=[dict(r,reference=r['reference_alternative'] if stage=='swapped' else r['reference']) for r in rows]
        evaluator.input_rows=lambda _,v=variants:v
        evaluator.generate(stage,key)
    metrics.ROOT=ROOT
    # 共用资产注册表采用单个原子写入路径，评价阶段串行避免同名临时文件竞争。
    with (ROOT/'metric.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        metrics.metric(key)
    verify_parent()


def job_state(identifier):
    x=subprocess.check_output(['sacct','-X','-n','-P','-j',str(identifier),'--format=State'],text=True).strip()
    return x.splitlines()[0].split('|')[0].split()[0] if x else 'PENDING'


def coordinate():
    import fcntl
    prepare()
    with (ROOT/'coordinator.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            verify_parent()
            jobs=read(ROOT/'jobs.json') if (ROOT/'jobs.json').exists() else []
            active=[];failed=[]
            for j in jobs:
                j['state']=job_state(j['id'])
                if j['state'] in ['PENDING','RUNNING','CONFIGURING','COMPLETING']:active.append(j)
                elif not (ROOT/f'metrics/{j["key"]}/complete.json').exists():failed.append(j)
            write(ROOT/'jobs.json',jobs)
            if failed:
                status(stage='ENGINEERING_FAILURE_REVIEW_REQUIRED',failed=failed,complete=False)
                raise RuntimeError(('PRESERVED_FAILED_JOBS',failed))
            if all((ROOT/f'metrics/{k}/complete.json').exists() for k in ARMS):
                analyse();package();status(stage='SERVER_COMPLETE_LOCAL_REVIEW_PENDING',complete=False,
                    server_execution_complete=True,human_review='DEFERRED');return
            for key in ARMS:
                if len(active)>=3:break
                if any(j['key']==key for j in jobs):continue
                command=['sbatch','--parsable','-p','gpu','-N1','-n1','-c8','--mem=48G','--gres=gpu:2',
                    '-t','7-00:00:00','-J','T3X_'+key,'-o',str(ROOT/f'logs/{key}_%j.log'),
                    str(SOURCE/'scripts/t3_response_extension.sbatch'),'train',key]
                identifier=int(subprocess.check_output(command,text=True).strip().split(';')[0])
                j=dict(id=identifier,key=key,command=command,state='PENDING',gpus=2,commit=commit())
                jobs.append(j);active.append(j);write(ROOT/'jobs.json',jobs)
                print('SUBMITTED',key,identifier,flush=True)
            status(stage='TRAINING_AND_EVALUATION',jobs=jobs,complete=False,human_review='DEFERRED')
            time.sleep(120)


def analyse():
    from tools.t3_response_run import statistics,holm
    verify_parent()
    lookup={};summary=[]
    keys={'C_s42':('C',PARENT),'T3_s42':('T3',PARENT)}
    keys.update({k:(k,ROOT) for k in ARMS})
    for key,(arm,root) in keys.items():
        assert read(root/f'metrics/{arm}/complete.json')['passed']
        recs=read(root/f'metrics/{arm}/per_identity.json');assert len(recs)==256
        if root==PARENT:freeze(ROOT/f'baseline_metrics/{key}.json',recs)
        for r in recs:lookup[key,r['stage'],r['identity']]=r
        for stage in ['original','swapped']:
            for metric in ['margin','color_margin','iou','edge_f1','leak']:
                values=[r[metric] for r in recs if r['stage']==stage and r.get(metric) is not None]
                summary.append(dict(key=key,stage=stage,metric=metric,**statistics(values)))
    pairs=[('T3_s42','C_s42'),('T3_s43','C_s43'),('T3_s44','C_s44'),
           ('T3_s42','T3_no_body_s42'),('T3_s42','T3_no_rest_s42')]
    contrasts=[];per_identity=[]
    rows=read(ROOT/'splits/confirmation.json')
    for a,b in pairs:
        diffs=[]
        for row in rows:
            i=row['id'];x=lookup[a,'swapped',i]['margin'];y=lookup[b,'swapped',i]['margin']
            ok=x is not None and y is not None
            per_identity.append(dict(a=a,b=b,identity=i,valid=ok,difference=x-y if ok else None))
            if ok:diffs.append(x-y)
        contrasts.append(dict(a=a,b=b,planned_n=128,**statistics(diffs)))
    holm(contrasts)
    # 跨 seed 只以三个真实训练差值汇总，不能把321个身份配对当321个训练重复。
    seed_effects=np.array([x['mean'] for x in contrasts[:3]])
    result=dict(exploratory=True,original_confirmatory_decision=read(PARENT/'statistics/decision.json'),
        contrasts=contrasts,seed_effects=seed_effects.tolist(),seed_effect_mean=float(seed_effects.mean()),
        seed_effect_sd=float(seed_effects.std(ddof=1)),training_seeds=[42,43,44],
        human_reviewers=0,human_blind_scores=None,human_review='DEFERRED BY USER',
        cannot_reinterpret_as_fresh_confirmation=True,coefficients_retuned=False,commit=commit())
    write(ROOT/'statistics/results.json',result);write(ROOT/'statistics/model_summary.json',summary)
    csv_write(ROOT/'statistics/contrasts.csv',contrasts);csv_write(ROOT/'statistics/per_identity.csv',per_identity)
    csv_write(ROOT/'statistics/model_summary.csv',summary)
    audits=[]
    for a,b in [('T3_s43','C_s43'),('T3_s44','C_s44')]:
        sa=read(folder(a)/'seed_and_rng_manifest.json');sb=read(folder(b)/'seed_and_rng_manifest.json')
        assert sa==sb,('PAIRED_TRAINING_STREAM_CHANGED',a,b)
        with (folder(a)/'training_log.csv').open() as f:la=list(csv.DictReader(f))
        with (folder(b)/'training_log.csv').open() as f:lb=list(csv.DictReader(f))
        assert len(la)==len(lb)==28365
        for x,y in zip(la,lb):
            assert all(x[k]==y[k] for k in ['update','lr','microbatches_per_rank','rng_rank0'])
        audits.append(dict(a=a,b=b,all_rank_final_rng_equal=True,all_update_trace_equal=True))
    for key in ['T3_no_body_s42','T3_no_rest_s42']:
        assert read(folder(key)/'seed_and_rng_manifest.json')==read(SUP/'training/formal/T3/seed_and_rng_manifest.json')
        audits.append(dict(key=key,all_rank_final_rng_equal_historical_T3=True))
    write(ROOT/'audit/paired_training_streams.json',audits)
    # 图审仍用原生成前固定24例，生成两组对照面板，不按结果挑样本。
    from PIL import Image,ImageDraw
    from tools.t3_response_report import place
    fixed=read(ROOT/'splits/qualitative_fixed24.json')
    for group,arms in [('ablation',['C_s42','T3_s42','T3_no_body_s42','T3_no_rest_s42']),
                       ('seeds',['C_s43','T3_s43','C_s44','T3_s44'])]:
        for identity in fixed:
            row=next(r for r in rows if r['id']==identity)
            canvas=Image.new('RGB',(1280,520),'white');draw=ImageDraw.Draw(canvas)
            for j,k in enumerate(['sketch','reference','reference_alternative']):
                draw.text((j*400+5,5),identity+' '+k,fill='black');place(canvas,row[k],j*400+80,25)
            for j,(key,stage) in enumerate((a,s) for a in arms for s in ['original','swapped']):
                arm,root=keys[key];draw.text((j*160+3,252),key+' '+stage,fill='black')
                place(canvas,root/f'evaluation/{stage}/{arm}/images/{identity}.png',j*160,284)
            out=ROOT/f'figures/{group}/{identity}.jpg';out.parent.mkdir(parents=True,exist_ok=True);canvas.save(out,quality=92)
    write(ROOT/'statistics/analysis_complete.json',dict(passed=True,commit=commit(),panels=48,
        human_review='deferred',original_confirmation_unchanged=True))


def package():
    assert read(ROOT/'statistics/analysis_complete.json')['passed']
    verify_parent()
    jobs=read(ROOT/'jobs.json');identifiers=[str(j['id']) for j in jobs]+[os.environ.get('SLURM_JOB_ID','')]
    accounting=subprocess.check_output(['sacct','-X','-n','-P','-j',','.join(x for x in identifiers if x),
        '--format=JobID,JobName,State,ExitCode,ElapsedRaw,AllocTRES,NodeList'],text=True)
    (ROOT/'audit/sacct.txt').write_text(accounting,encoding='utf-8')
    for key in ARMS:
        dst=ROOT/'training_records'/key;dst.mkdir(parents=True,exist_ok=True)
        for p in folder(key).iterdir():
            if p.is_file() and p.suffix in ['.json','.csv','.txt']:shutil.copy2(p,dst/p.name)
    for p in ['tools/t3_response_extension.py','scripts/t3_response_extension.sbatch']:
        dst=ROOT/'code'/p;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(SOURCE/p,dst)
    files=[p for d in ['audit','protocol','splits','metrics','baseline_metrics','statistics','figures','evaluation','training_records','logs','code']
        for p in (ROOT/d).rglob('*') if p.is_file()]
    manifest=[dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(files)]
    write(ROOT/'artifact_manifest.json',manifest)
    bundle=ROOT/'extension_results.tar.gz'
    with tarfile.open(bundle,'w:gz',compresslevel=3) as tar:
        for p in files+[ROOT/'artifact_manifest.json',ROOT/'jobs.json']:tar.add(p,arcname=p.relative_to(ROOT),recursive=False)
    write(ROOT/'bundle_info.json',dict(path=str(bundle),sha256=sha(bundle),bytes=bundle.stat().st_size,
        manifest_count=len(manifest),generated_images=1536,training_runs=6,human_review='DEFERRED',
        local_report_complete=False,original_confirmation_unchanged=True))
    print('EXTENSION_SERVER_BUNDLE_COMPLETE',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['coordinate','prepare','train','evaluate','analyse','package'])
    p.add_argument('--key',choices=list(ARMS));a=p.parse_args()
    if a.action in ['train','evaluate']:globals()[a.action](a.key)
    else:globals()[a.action]()


if __name__=='__main__':main()
