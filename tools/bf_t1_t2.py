"""新BF-SD架构T1/T2：只增加监督，保持既有T3的初始化与训练随机流。"""
import argparse
import ast
import csv
from contextlib import nullcontext
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import tarfile
import time

import numpy as np
from PIL import Image
from tools.bf_t3_ablation import core, ROOT as PARENT, core_source_path

ROOT=core.PROJECT/'output_eval/bf_t1_t2_20261103'
SOURCE=Path(__file__).resolve().parents[1]
HISTORY=core.PROJECT/'output_eval/bf_fusion_t3_three_arm_20261101'
COLOR=core.PROJECT/'output_eval/t3_cfg5_lab_20261103'
SUP=core.PROJECT/'output_eval/e2a_sup_comp_01_20261021'
NEW=['BF-SD-T1','BF-SD-T2']
RAW=['BF-SD']+NEW+['BF-SD-T3']
ARMS=RAW+[a+'_LAB' for a in RAW]
read,write,sha=core.read,core.write,core.sha


def output(arm,row):return ROOT/'outputs'/arm/row['condition']/(row['id']+'.png')


def initial_digest(network):
    h=hashlib.sha256()
    for name,value in network.trainable_state().items():h.update(name.encode()+value.numpy().tobytes())
    return h.hexdigest()


def captured_losses():
    """唯一变换是增加返回已存在的张量；基础目标和随机调用逐语句相同。"""
    original=ast.parse(inspect.getsource(core.losses));changed=ast.parse(inspect.getsource(core.losses))
    fn=changed.body[0];assert isinstance(fn.body[-1],ast.Return) and len(fn.body[-1].value.elts)==2
    cache=ast.parse("dict(decoded=decoded,predicted=predicted,x0=x0,reference=reference,mask=mask,t=t,keep=keep,gt=batch['vae_cloth'])",mode='eval').body
    fn.body[-1].value.elts.append(cache)
    normalized=ast.parse(ast.unparse(changed));normalized.body[0].body[-1].value.elts.pop()
    assert ast.dump(normalized,include_attributes=False)==ast.dump(original,include_attributes=False)
    namespace=dict(core.__dict__);exec(compile(ast.fix_missing_locations(changed),str(core.__file__),'exec'),namespace)
    return namespace['losses'],hashlib.sha256(ast.unparse(changed).encode()).hexdigest()


def prepare():
    assert not (ROOT/'protocol.json').exists(),'不覆盖冻结协议'
    config=read(PARENT/'protocol.json');audit=read(PARENT/'audit/final_comparability.json');assert audit['passed']
    for name,expected in config['core_source_sha256'].items():assert sha(core_source_path(name))==expected,name
    assert sha(core.E1)==core.E1_SHA
    registry=read(SUP/'assets/registry.json')
    for arm in ['T1','T2']:
        for name,expected in registry[arm]['files'].items():assert sha(name)==expected,name
    rows=read(PARENT/'evaluation.json');assert len(rows)==192
    assert read(COLOR/'evaluation_complete.json')['complete']
    old_color={(r['condition'],r['id']):r for r in read(COLOR/'evaluation.json')}
    for row in rows:
        assert all(row[k]==old_color[row['condition'],row['id']][k] for k in ['caption','seed','sketch_sha256','reference_sha256','gt_sha256'])
        for key in ['sketch','reference','gt']:
            source=Path(row[key]);expected=row[key+'_sha256'];assert sha(source)==expected
            dest=ROOT/'inputs'/expected;dest.parent.mkdir(parents=True,exist_ok=True)
            if not dest.exists():shutil.copy2(source,dest)
            row[key]=str(dest)
        for arm in ['BF-SD','BF-SD-T3','BF-SD-T3_LAB']:
            source=(COLOR/'outputs/CFG7_LAB' if arm.endswith('_LAB') else PARENT/'outputs'/arm)/row['condition']/(row['id']+'.png')
            meta=read(source.with_suffix('.json'));assert sha(source)==meta['output_sha256']
            dest=output(arm,row);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
            write(dest.with_suffix('.json'),dict(meta,arm=arm,caption=row['caption'],reused_from=str(source)))
    train=PARENT/'training_manifest.json';assert sha(train)==config['training_manifest_sha256']
    shutil.copy2(train,ROOT/'training_manifest.json')
    shutil.copytree(COLOR/'masks',ROOT/'masks',dirs_exist_ok=True)
    write(ROOT/'audit/masks.json',read(COLOR/'audit/masks.json'))
    write(ROOT/'audit/assets_registry.json',registry)
    write(ROOT/'audit/parent_training.json',read(PARENT/'audit/parent_training.json'))
    write(ROOT/'evaluation.json',rows)
    _,transformed_sha=captured_losses()
    write(ROOT/'protocol.json',dict(name=ROOT.name,new_training=NEW,raw_arms=RAW,arms=ARMS,
        training=dict(samples=45384,updates=28365,seed=42,world=2,effective_batch=8,
            optimizer='AdamW betas(.9,.999) eps1e-8 wd.01',peak_lr=5e-5,warmup=500,schedule='cosine',
            precision='bf16 autocast; trainable float32',trainable_parameters=101752192,
            initial_sha256=config['initial_sha256'],manifest_sha256=sha(train),core_source_sha256=config['core_source_sha256'],
            transformed_loss_ast_sha256=transformed_sha,supervision_source_sha256=sha(SOURCE/'tools/bf_t1_t2_frozen_losses.py'),
            auxiliary='T1 perceptual+GT-edge; T2 DINO CLS reference appearance+GT patch-key self-similarity; no T3 auxiliary',
            eta=249,adaptation_not_full_paper_replication=True),
        calibration=dict(samples=512,effective_batches=64,world=2,per_rank=256,optimizer_updates=0,
            selection='first256 per rank in common seed42 DistributedSampler(epoch0)',
            rule='coefficient=.1 / median positive per-sample aux/base epsilon-gradient RMS ratio',
            active='t<=249; complete batch mean; zero/positive rates retained; no test selection'),
        inference=dict(width=384,height=512,steps=50,cfg=7.,sketch_scale=.6,ipa_scale=0,texture_tokens=False),
        evaluation=dict(original=128,neutral_original=32,neutral_swapped=32,conditions_per_arm=192,
            main='four raw CFG7 arms',secondary='same fixed M12 Lab for each of four arms'),
        lab=read(COLOR/'protocol.json')['lab'],statistics=dict(bootstrap=10000,sign_flip=10000,seed=110142,
            primary='raw T3 minus T1/T2: six original128 metrics Holm12',
            response='raw T3 minus T1/T2: three neutral-swapped DISTS metrics Holm6',
            secondary='LAB T3 minus LAB T1/T2 six primary metrics Holm12; color pairs own Holm family'),
        gpu_seconds_limit=360000,planned_gpu_reservation=325800,
        reuse='BF-SD and full T3 final28365 weights; raw192 each; T3 LAB192; no retraining',
        limitations=['one seed; previously viewed128 identities; exploratory',
            'reference donor reuse; 64square patches not full motifs',
            'T1 directly uses DISTS supervision: DISTS is not independent evidence',
            'gradient ratio target does not imply equal full-objective gradients across methods',
            'Lab is postprocessing, not learned motif control'],
        source_commit=subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip()))
    print('PREPARED',45384,len(rows),flush=True)


def supervision(arm,acc):
    import torch
    from tools.bf_t1_t2_frozen_losses import Supervision
    from train_GAM_texture_joint import build_region_masks,texture_color_stat_loss,masked_l1_loss
    state=random.getstate();npstate=np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[acc.device.index]):
            impl=Supervision(arm[-2:],dict(accelerator=acc,build_region_masks=build_region_masks,
                texture_color_stat_loss=texture_color_stat_loss,masked_l1_loss=masked_l1_loss))
    finally:random.setstate(state);np.random.set_state(npstate)
    return impl


def train(arm,phase):
    import torch
    from torch.utils.data import DataLoader,DistributedSampler
    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs
    from diffusers.optimization import get_scheduler
    from train_GAM_texture_joint import JointTextureDataset,VGGGramStyleLoss
    assert arm in NEW and phase in ['calibrate','preflight','formal']
    protocol=read(ROOT/'protocol.json');config=protocol['training']
    if phase!='calibrate':assert read(ROOT/'calibration'/arm/'complete.json')['passed']
    if phase=='formal':assert read(ROOT/'preflight'/arm/'complete.json')['passed']
    assert int(os.environ.get('WORLD_SIZE','1'))==2
    acc=Accelerator(mixed_precision='bf16',gradient_accumulation_steps=4,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
    torch.cuda.set_device(acc.local_process_index)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    print('LAUNCH_ALLOCATION',phase,os.environ.get('SLURM_JOB_NODELIST'),acc.num_processes,
        acc.local_process_index,os.environ.get('CUDA_VISIBLE_DEVICES'),flush=True)
    for name,expected in config['core_source_sha256'].items():assert sha(core_source_path(name))==expected
    assert sha(SOURCE/'tools/bf_t1_t2_frozen_losses.py')==config['supervision_source_sha256']
    capture,ast_sha=captured_losses();assert ast_sha==config['transformed_loss_ast_sha256']
    pipe,network=core.load_network(acc.device);initial=initial_digest(network);assert initial==config['initial_sha256']
    params=[p for p in network.parameters() if p.requires_grad];assert sum(p.numel() for p in params)==config['trainable_parameters']
    optimizer=torch.optim.AdamW(params,lr=5e-5,betas=(.9,.999),eps=1e-8,weight_decay=.01)
    scheduler=get_scheduler('cosine',optimizer=optimizer,num_warmup_steps=500,num_training_steps=28365)
    dataset=JointTextureDataset(str(ROOT/'training_manifest.json'),pipe.tokenizer,str(core.PROJECT.parent/'datasets/BF/training'),
        width=384,height=512,texture_preprocess_mode='plain_resize',mask_cache=str(core.MASKS))
    sampler=DistributedSampler(dataset,num_replicas=2,rank=acc.process_index,seed=42,drop_last=False)
    loader=DataLoader(dataset,batch_size=1,sampler=sampler,num_workers=2,pin_memory=True)
    assert len(dataset)==45384 and len(loader)%4==0
    style=VGGGramStyleLoss().to(acc.device).eval().requires_grad_(False)
    impl=supervision(arm,acc)
    folder=ROOT/('calibration' if phase=='calibrate' else 'preflight' if phase=='preflight' else 'training')/arm
    folder.mkdir(parents=True,exist_ok=True);assert not (folder/'complete.json').exists(),'不重训已完成阶段'
    coefficients=None if phase=='calibrate' else read(ROOT/'calibration'/arm/'complete.json')['coefficients']
    rng=torch.Generator(device=acc.device).manual_seed(42000+acc.process_index);stream='0'*64
    step=0;resume=folder/'latest.pt'
    if phase=='formal' and resume.exists():
        state=torch.load(resume,map_location='cpu');assert state['initial_sha256']==initial and state['world']==2 and state['arm']==arm
        assert state['coefficients']==coefficients
        network.load_trainable(state['network']);optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
        rng.set_state(state['rngs'][acc.process_index]);stream=state['streams'][acc.process_index];step=state['step']
        if acc.is_main_process:
            log=folder/'training_log.csv';old=list(csv.DictReader(log.open()))
            if len(old)>step:
                shutil.copy2(log,folder/('before_resume_job'+os.environ['SLURM_JOB_ID']+'.csv'))
                with log.open('w',newline='') as f:
                    writer=csv.DictWriter(f,fieldnames=list(old[0]));writer.writeheader();writer.writerows(old[:step])
    network,optimizer=acc.prepare(network,optimizer);network.train()
    total=3 if phase=='preflight' else 28365
    updates_per_epoch=len(loader)//4;epoch,skip=step//updates_per_epoch,(step%updates_per_epoch)*4
    cal_rows=[];grad_probes={};semantic_checked=False;started=time.monotonic()

    def save(final=False):
        states=[None]*2;chains=[None]*2
        torch.distributed.all_gather_object(states,rng.get_state().cpu());torch.distributed.all_gather_object(chains,stream)
        if acc.is_main_process:
            value=dict(network=acc.unwrap_model(network).trainable_state(),optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),rngs=states,streams=chains,step=step,world=2,arm=arm,
                initial_sha256=initial,coefficients=coefficients)
            tmp=folder/'latest.tmp';torch.save(value,tmp);tmp.replace(resume)
            if final or step%updates_per_epoch==0:torch.save(dict(network=value['network'],step=step,arm=arm,initial_sha256=initial),folder/('step%05d.pt'%step))

    def rms(g):return float(g.detach().float().square().mean().sqrt())
    while step<total:
        sampler.set_epoch(epoch)
        for micro,batch in enumerate(loader):
            if micro<skip:continue
            batch={k:v.to(acc.device) for k,v in batch.items()}
            with acc.accumulate(network), (acc.no_sync(network) if phase=='calibrate' else nullcontext()):
                if not cal_rows and phase=='calibrate':
                    # 同样输入/权重/RNG比较原函数和仅扩展返回值的函数，不改变正式流。
                    saved=rng.get_state()
                    with torch.no_grad(),acc.autocast():baseline,base_token=core.losses(pipe,network,batch,rng,style)
                    rng.set_state(saved)
                with acc.autocast():raw,token,ctx=capture(pipe,network,batch,rng,style,phase=='preflight')
                if not cal_rows and phase=='calibrate':
                    assert token==base_token and all(torch.equal(raw[k].detach(),baseline[k]) for k in raw)
                    del baseline
                stream=hashlib.sha256(stream.encode()+json.dumps(token,sort_keys=True).encode()).hexdigest()
                with torch.autocast('cuda',enabled=False):
                    auxiliary=impl.raw(ctx['decoded'].float(),ctx['gt'].float(),ctx['reference'].float(),ctx['mask'].float(),
                        ctx['t'],torch.full_like(ctx['t'],float(ctx['keep']),dtype=torch.float32))
                assert all(torch.isfinite(v).item() for v in [raw['base'],*auxiliary.values()])
                if not semantic_checked and int(ctx['t'][0])<=249:
                    with torch.no_grad():
                        flipped=impl.raw(ctx['decoded'].float(),ctx['gt'].float(),-ctx['reference'].float(),ctx['mask'].float(),ctx['t'],torch.ones_like(ctx['t']))
                        if arm.endswith('T1'):assert all(torch.equal(auxiliary[k],flipped[k]) for k in auxiliary)
                        else:
                            assert torch.equal(auxiliary['structure'],flipped['structure'])
                            assert not torch.allclose(auxiliary['appearance'],flipped['appearance'],rtol=0,atol=1e-7)
                        inactive=impl.raw(ctx['decoded'],ctx['gt'],ctx['reference'],ctx['mask'],ctx['t']*0+999,torch.ones_like(ctx['t']))
                        assert all(v.item()==0 for v in inactive.values())
                    assert not impl.network.training and not any(p.requires_grad for p in impl.network.parameters())
                    semantic_checked=True
                if phase=='calibrate':
                    base_rms=rms(torch.autograd.grad(raw['base'],ctx['predicted'],retain_graph=True)[0]);assert base_rms>0
                    rec=dict(rank=acc.process_index,microbatch=len(cal_rows)+1,base_rms=base_rms,**token,stream=stream)
                    for name,value in auxiliary.items():
                        g,gx=torch.autograd.grad(value,[ctx['predicted'],ctx['x0']],retain_graph=True)
                        assert torch.isfinite(g).all() and torch.isfinite(gx).all()
                        rec[name+'_ratio']=rms(g)/base_rms;rec[name+'_x0_rms']=rms(gx)
                        if rms(g)>0 and name not in grad_probes:
                            probe=acc.unwrap_model(network).adapter.projections[0].weight
                            pg=torch.autograd.grad(value,probe,retain_graph=True,allow_unused=True)[0]
                            assert pg is not None and torch.isfinite(pg).all() and rms(pg)>0
                            grad_probes[name]=rms(pg)
                    cal_rows.append(rec)
                    # 不step/scheduler，只释放共享计算图；formal在全新进程重新初始化。
                    acc.backward(raw['base']);optimizer.zero_grad(set_to_none=True)
                else:
                    weighted=sum(auxiliary[k]*v for k,v in coefficients.items());loss=raw['base']+weighted
                    if phase=='preflight' and step==2 and micro%4==0:
                        bare=acc.unwrap_model(network);assert bare.adapter.injection_counts==[3,2,2,3]
                        for name,value in auxiliary.items():
                            pg=torch.autograd.grad(value,bare.adapter.projections[0].weight,retain_graph=True)[0]
                            assert torch.isfinite(pg).all() and rms(pg)>0;grad_probes[name]=rms(pg)
                        fg=torch.autograd.grad(loss,bare.adapter.fusion.downconv1[0].weight,retain_graph=True)[0]
                        assert torch.isfinite(fg).all() and rms(fg)>0;grad_probes['original_fusion']=rms(fg)
                    acc.backward(loss)
                    if acc.sync_gradients:
                        grad=acc.clip_grad_norm_(params,1.);assert torch.isfinite(grad).item()
                    optimizer.step()
                    if acc.sync_gradients:scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
            acc.unwrap_model(network).adapter.clear()
            if phase=='calibrate':
                if len(cal_rows)==256:break
            elif acc.sync_gradients:
                assert not acc.optimizer_step_was_skipped
                step+=1
                record=dict(step=step,seconds=time.monotonic()-started,lr=scheduler.get_last_lr()[0],grad=float(grad),
                    peak_gib=torch.cuda.max_memory_allocated()/2**30,total=float(loss.detach()),weighted_aux=float(weighted.detach()),
                    **{k:float(v.detach()) for k,v in raw.items()},**{k:float(v.detach()) for k,v in auxiliary.items()})
                if acc.is_main_process:
                    path=folder/'training_log.csv';exists=path.exists()
                    with path.open('a',newline='') as f:
                        writer=csv.DictWriter(f,fieldnames=list(record))
                        if not exists:writer.writeheader()
                        writer.writerow(record)
                    if step<=3 or step%25==0:write(folder/'progress.json',record);print('TRAIN',arm,json.dumps(record),flush=True)
                if phase=='formal' and (step%1000==0 or step==total):save(step==total)
                if step>=total:break
            del ctx,auxiliary,raw
        if phase=='calibrate':break
        skip=0;epoch+=1
    assert semantic_checked
    if phase=='calibrate':
        assert len(cal_rows)==256 and not optimizer.optimizer.state and initial_digest(acc.unwrap_model(network))==initial
        write(folder/('rank%d.json'%acc.process_index),dict(rows=cal_rows,gradient_probes=grad_probes,passed=True,
            base_exact=True,optimizer_updates=0,initial_sha256=initial))
        acc.wait_for_everyone()
        if acc.is_main_process:
            all_rows=sum([read(folder/('rank%d.json'%r))['rows'] for r in range(2)],[])
            names=[k[:-6] for k in all_rows[0] if k.endswith('_ratio')];coef={};evidence={}
            assert len(all_rows)==512
            for name in names:
                values=np.array([r[name+'_ratio'] for r in all_rows]);positive=values[values>0];assert len(positive)>0
                assert all(name in read(folder/('rank%d.json'%r))['gradient_probes'] for r in range(2))
                coef[name]=.1/float(np.median(positive))
                evidence[name]=dict(median_positive_ratio=float(np.median(positive)),positive_n=len(positive),active_rate=float((values>0).mean()))
            write(folder/'complete.json',dict(passed=True,coefficients=coef,evidence=evidence,samples=512,optimizer_updates=0,
                gradient_target=.1,initial_sha256=initial,seconds=time.monotonic()-started))
            print('CALIBRATED',arm,coef,flush=True)
    else:
        streams=[None]*2;torch.distributed.all_gather_object(streams,stream)
        if phase=='formal':assert streams==read(ROOT/'audit/parent_training.json')['BF-SD']['stream_hashes']
        write(folder/('rank%d_gradients.json'%acc.process_index),grad_probes)
        if acc.is_main_process:
            checkpoint=folder/'step28365.pt'
            write(folder/'complete.json',dict(passed=True,arm=arm,updates=step,world=2,effective_batch=8,seed=42,
                initial_sha256=initial,stream_hashes=streams,coefficients=coefficients,trainable_parameters=sum(p.numel() for p in params),
                seconds=time.monotonic()-started,checkpoint_sha256=sha(checkpoint) if phase=='formal' else None,
                source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                log_scope='rank0 final microbatch per update; not global batch mean'))
    acc.wait_for_everyone();print('COMPLETE',arm,phase,flush=True)


def generate(arm):
    import torch
    assert arm in NEW
    folder=ROOT/'training'/arm;complete=read(folder/'complete.json');checkpoint=folder/'step28365.pt'
    assert complete['updates']==28365 and sha(checkpoint)==complete['checkpoint_sha256']
    pipe,network=core.load_network('cuda',checkpoint);network.eval().requires_grad_(False)
    pipe.unet.half();pipe.vae.half();network.adapter.half();pipe.image_encoder.to('cuda');pipe.bf_texture_conditioner.to('cuda')
    assert pipe._execution_device==next(pipe.text_encoder.parameters()).device==next(pipe.unet.parameters()).device
    with torch.inference_mode():
        for row in read(ROOT/'evaluation.json'):
            dest=output(arm,row);base=read(output('BF-SD-T3',row).with_suffix('.json'))
            if dest.exists():
                meta=read(dest.with_suffix('.json'));assert sha(dest)==meta['output_sha256'] and meta['checkpoint_sha256']==complete['checkpoint_sha256'];continue
            sketch=core.tensor(row['sketch']).cuda().half();reference=core.tensor(row['reference']).cuda().half()
            network.adapter.condition(sketch,reference)
            latent=torch.randn((1,4,64,48),device='cuda',dtype=torch.float16,generator=torch.Generator(device='cuda').manual_seed(row['seed']))
            before=core.digest(latent);assert before==base['initial_latent_sha256'];started=time.monotonic()
            im=pipe(ref_image=sketch,prompt=row['caption'],texture_clip_image=None,texture_embeds=None,null_prompt='',
                negative_prompt=' worst quality, low quality',width=384,height=512,num_images_per_prompt=1,
                guidance_scale=7.,sketch_scale=.6,ipa_scale=0.,generator=torch.Generator(device='cuda').manual_seed(row['seed']),
                latents=latent,num_inference_steps=50)[0]
            dest.parent.mkdir(parents=True,exist_ok=True);im.save(dest);network.adapter.clear()
            write(dest.with_suffix('.json'),dict(base,arm=arm,caption=row['caption'],checkpoint=str(checkpoint),
                checkpoint_sha256=complete['checkpoint_sha256'],output_sha256=sha(dest),reused_from=None,seconds=time.monotonic()-started))
            print('GENERATED',arm,row['condition'],row['id'],flush=True)
    write(ROOT/'generation'/arm/'complete.json',dict(n=192,checkpoint_sha256=complete['checkpoint_sha256']))


def package():
    assert read(ROOT/'evaluation_complete.json')['complete']
    jobs=read(ROOT/'submitted_jobs.json')
    sacct=subprocess.check_output(['sacct','-X','-j',','.join(map(str,jobs.values())),'-P',
        '--format=JobID,JobName,State,ExitCode,ElapsedRaw,AllocTRES,AllocCPUS,NodeList'],text=True)
    (ROOT/'audit/sacct_at_packaging.psv').write_text(sacct)
    files=[p for name in ['audit','calibration','preflight','training','generation','inputs','masks','outputs','metrics','figures','logs']
        for p in (ROOT/name).rglob('*') if p.is_file() and p.suffix not in ['.pt','.tmp']]
    files += [ROOT/name for name in ['protocol.json','evaluation.json','submitted_jobs.json','job_reservations.json','evaluation_complete.json']]
    files += [p for pattern in ['tools/bf_t1_t2*.py','scripts/bf_t1_t2*'] for p in SOURCE.glob(pattern)]
    write(ROOT/'bundle_manifest.json',{str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in files})
    with tarfile.open(ROOT/'review_results.tar.gz','w:gz') as tar:
        for p in files+[ROOT/'bundle_manifest.json']:tar.add(p,arcname=str(p.relative_to(ROOT)))
    write(ROOT/'bundle_info.json',dict(bytes=(ROOT/'review_results.tar.gz').stat().st_size,sha256=sha(ROOT/'review_results.tar.gz'),members=len(files)+1))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','calibrate','preflight','formal','generate','process','evaluate','package']);p.add_argument('--arm',choices=NEW)
    args=p.parse_args()
    if args.action in ['calibrate','preflight','formal']:train(args.arm,args.action)
    elif args.action=='generate':generate(args.arm)
    elif args.action in ['process','evaluate']:
        from tools import bf_t1_t2_report
        getattr(bf_t1_t2_report,args.action)()
    else:globals()[args.action]()
