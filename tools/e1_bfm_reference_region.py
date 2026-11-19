"""从 E1 第10轮起点进行两轮参考纹理与区域约束对照。"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tarfile
import time

import numpy as np
from PIL import Image, ImageDraw
from tools import e1_bfm_joint as previous
from tools.e1_bfm_continue import trim_log

core = previous.core
PROJECT, SOURCE = core.PROJECT, Path(__file__).resolve().parents[1]
ROOT = PROJECT / 'output_eval/e1_bfm_reference_region_20261119'
PARENT = PROJECT / 'output_eval/e1_bfm_continue_20261117/epochs/epoch_10'
PARENT_ARM = 'E1-BFM-T3-epoch10'
PARENT_FILE = PARENT / 'training' / PARENT_ARM / 'final.pt'
PARENT_SHA = '40beb650de16ee648e6339318e7471f853ee076104490286eba80ea1b97dc213'
PARENT_UPDATES, EPOCH_UPDATES, UPDATES = 56730, 5673, 11346
LRS = (2e-6, 1e-5)
ARMS = ('control', 'texture')
BASELINES = ['BF-Fashion', PARENT_ARM]
SOURCE_FILES = ['tools/e1_bfm_reference_region.py', 'models/reference_region_loss.py',
    'tools/e1_bfm_joint.py', 'tools/bf_t3_frozen_core.py', 'models/bf_sd_skip.py',
    'train_GAM_texture_joint.py', 'pipelines/IMAGGarment_pipeline.py', 'adapter/attention_processor.py',
    'tools/e1_bfm_continue.py', 'models/bf_original_fusion.py', 'texture_preprocess.py']
read, write, sha, digest = core.read, core.write, core.sha, core.digest


def source_hashes():
    return {name: sha(SOURCE / name) for name in SOURCE_FILES}


def name(arm, epoch):
    return 'E1-RR-%s-e%02d' % (arm, epoch)


def folder(arm, epoch):
    return ROOT / 'arms' / arm / ('epoch_%02d' % epoch)


def checkpoint(arm, epoch):
    return folder(arm, epoch) / 'training' / name(arm, epoch) / 'final.pt'


def prepare():
    import torch
    assert sha(PARENT_FILE) == PARENT_SHA
    parent = torch.load(PARENT_FILE, map_location='cpu', weights_only=False)
    assert parent['total_updates'] == PARENT_UPDATES and parent['world'] == 2
    assert parent['optimizer']['state'] and len(parent['rngs']) == len(parent['streams']) == 2
    write(ROOT / 'audit/parent.json', dict(checkpoint=str(PARENT_FILE), sha256=PARENT_SHA,
        total_updates=PARENT_UPDATES, complete_optimizer_and_rngs=True, streams=parent['streams']))
    del parent
    protocol = dict(parent=str(PARENT_FILE), parent_sha256=PARENT_SHA, parent_updates=PARENT_UPDATES,
        arms=list(ARMS), additional_epochs=2, final_epoch=12, primary_final_epoch=12,
        samples=45384, epoch_updates=EPOCH_UPDATES, updates_per_arm=UPDATES,
        total_updates_per_arm=PARENT_UPDATES+UPDATES, effective_batch=8, world=2, seed=42,
        unet_lr=LRS[0], condition_lr=LRS[1], warmup=0, schedule='one cosine across two epochs',
        optimizer='same parent AdamW8bit moments; reset only LR and scheduler identically in both arms',
        trainable='generation UNet, BFM, four projections, GAM; same as parent',
        frozen='VAE, text encoder, reference UNet, unused image token branch',
        common_change='VGG input [-1,1] to [0,1], clamp, ImageNet mean/std; original relu3_4/relu4_4 and L1 Gram retained',
        common_losses='MSE + calibrated GT Gram + .05 edge + .1 garment RGB + .1 body RGB + .03816244801608925 rest GT L1',
        treatment='edge only near silhouette and sketch structure lines; add grayscale multi-scale interior reference contextual loss at t<=400 when reference kept',
        calibration='8 training-only cases; Gram noise-prediction gradient matched to historical Gram; new reference gradient 10% of epsilon MSE; median ratios',
        reference_loss=dict(patches=[32,64], maximum_generated_patches=8, mask_occupancy=.99,
            layers=['relu2_2','relu3_2'], bandwidth=.5, vector_limits=[256,512]),
        structure=dict(inner_erosion=9, sketch_edge_threshold=.12, line_dilation=5),
        inference=dict(size=[384,512], steps=50, cfg=7., sketch_scale=.6, texture_tokens=False),
        selection='fixed second epoch; validation is monitoring only, no test-set checkpoint selection',
        safety_margins=dict(leak=.002, edge_f1=.01, iou=.005),
        texture_gate='gray DISTS significantly lower versus parent and control after Holm; gray margin must improve in direction versus both',
        limitations=['single training seed; historical primary test identities previously inspected',
            'primary test retains old captions and inherited CFG mismatch for historical comparability',
            'official validation identities are disjoint from training and primary test, previously used for development',
            'treatment combines reference supervision and structure-region change; cannot isolate their separate causal effects',
            'BF-Fashion has a different training budget, VQ backbone, resolution and 200-step sampling'])
    protocol['source_hashes'] = source_hashes()
    protocol['source_commit'] = read(ROOT / 'deployment.json')['source_commit']
    if (ROOT / 'protocol.json').exists():
        assert read(ROOT / 'protocol.json') == protocol
    else:
        write(ROOT / 'protocol.json', protocol)
    train = read(PROJECT / 'output_eval/e1_bfm_continue_20261117/training_manifest.json')
    assert len(train) == 45384
    write(ROOT / 'training_manifest.json', train)
    rows = read(PARENT / 'evaluation.json')
    assert len(rows) == 192
    for row in rows:
        meta = read((PARENT / 'outputs' / PARENT_ARM / row['condition'] / (row['id']+'.png')).with_suffix('.json'))
        row['initial_latent_sha256'] = meta['initial_latent_sha256']
        for key in ['sketch','reference','gt']:
            assert sha(row[key]) == row[key+'_sha256']
    write(ROOT / 'evaluation.json', rows)
    train_ids = {Path(row['cloth']).stem.split('_')[0] for row in train}
    test_ids = {row['id'].split('_')[0] for row in rows}
    candidates = read(ROOT / 'validation_candidates.json')
    assert len(candidates) == 64
    originals = []
    for row in candidates:
        identity = Path(row['cloth']).stem
        assert identity not in train_ids | test_ids
        item = dict(id=identity, condition='original', split='validation64',
            seed=int(hashlib.sha256(('RR-val/'+identity).encode()).hexdigest()[:8],16) % (2**31-1),
            caption='a garment following the outline and structural details in the sketch.',
            pattern_group=row['pattern_group'], GT_used_in_inference=False)
        for key, field in [('sketch','sketch'),('reference','texture'),('gt','cloth')]:
            source = PROJECT.parent / 'datasets/BF/validation' / row[field]
            target = ROOT / 'validation_inputs' / sha(source)
            target.parent.mkdir(exist_ok=True)
            if not target.exists(): shutil.copy2(source,target)
            item[key], item[key+'_sha256'] = str(target), sha(target)
        originals.append(item)
    originals.sort(key=lambda row: hashlib.sha256(row['id'].encode()).hexdigest())
    assert len({row['id'] for row in originals}) == 64
    shift = next(s for s in range(1,64) if all(originals[i]['reference_sha256'] != originals[(i+s)%64]['reference_sha256'] for i in range(64)))
    swaps = [dict(row, condition='swapped', reference=originals[(i+shift)%64]['reference'],
        reference_sha256=originals[(i+shift)%64]['reference_sha256'],
        donor_id=originals[(i+shift)%64]['id']) for i,row in enumerate(originals)]
    write(ROOT / 'validation.json', originals+swaps)
    write(ROOT / 'audit/splits.json', dict(training=45384, primary_identities=len(test_ids),
        validation_identities=64, overlap=0, donor_shift=shift,
        validation_caption='geometry-only common prompt; primary captions unchanged',
        training_sha256=sha(ROOT/'training_manifest.json'), primary_sha256=sha(ROOT/'evaluation.json'),
        validation_sha256=sha(ROOT/'validation.json')))
    # 校准样本只来自训练集；不使用验证或测试结果调节损失。
    indices = sorted(range(len(train)), key=lambda i: hashlib.sha256(('RR-cal/'+train[i]['cloth']).encode()).hexdigest())[:32]
    write(ROOT / 'calibration_indices.json', indices)
    print('PREPARED', json.dumps(protocol), flush=True)


def load_network(device, path=PARENT_FILE, training=False):
    pipe, network = previous.load_network(device, checkpoint=path, training=training)
    if not training:
        pipe.register_modules(image_encoder=None)
        assert pipe._execution_device == pipe.unet.device
    return pipe, network


def calibrate():
    import torch
    from models.reference_region_loss import ReferenceRegionLoss, training_losses
    from train_GAM_texture_joint import JointTextureDataset
    assert read(ROOT/'protocol.json')['source_hashes'] == source_hashes()
    if (ROOT/'calibration.json').exists():
        assert read(ROOT/'calibration.json')['source_hashes'] == source_hashes()
        return
    pipe, network = load_network('cuda')
    network.eval().requires_grad_(False)
    # UNet 只提供预测值；以预测噪声为叶子，经 VAE/VGG 校准辅助梯度。
    style = ReferenceRegionLoss().cuda().eval()
    data = JointTextureDataset(str(ROOT/'training_manifest.json'),pipe.tokenizer,
        str(PROJECT.parent/'datasets/BF/training'),width=384,height=512,
        texture_preprocess_mode='plain_resize',mask_cache=str(core.MASKS))
    observations = []
    for i in read(ROOT/'calibration_indices.json'):
        batch={k:v[None].cuda() for k,v in data[i].items()}
        rng=torch.Generator(device='cuda').manual_seed(931000+i)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            raw, token, audit=training_losses(pipe,network,batch,rng,style,'control',dict(gram=1.,reference=0.),True)
            old=style.gt_gram(audit['decoded'],batch['vae_cloth'],batch['garment_mask'],normalized=False)
        predicted=audit['predicted']
        norms={key:float(torch.autograd.grad(loss,predicted,retain_graph=True)[0].float().norm())
            for key,loss in [('mse',raw['mse']),('old_gram',old),('gram',raw['gram']),('reference',raw['reference_texture'])]}
        if all(np.isfinite(x) and x>0 for x in norms.values()):
            observations.append(dict(index=i,token=token,gradients=norms,patch_counts=audit['patch_counts']))
        network.adapter.clear()
        del raw,token,audit,predicted,old,batch
        if len(observations)==8: break
    assert len(observations)==8,observations
    gram=float(np.median([r['gradients']['old_gram']/r['gradients']['gram'] for r in observations]))
    reference=float(np.median([.1*r['gradients']['mse']/r['gradients']['reference'] for r in observations]))
    assert 0<gram<100 and 0<reference<100,(gram,reference)
    write(ROOT/'calibration.json',dict(gram=gram,reference=reference,source_hashes=source_hashes(),
        observations=observations,training_only=True,reference_gradient_target=.1))
    print('CALIBRATED',gram,reference,flush=True)


def train(arm, epoch, smoke=False):
    import torch
    import bitsandbytes as bnb
    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs
    from diffusers.optimization import get_scheduler
    from torch.utils.data import DataLoader,DistributedSampler
    from train_GAM_texture_joint import JointTextureDataset
    from models.reference_region_loss import ReferenceRegionLoss,training_losses
    config,weights=read(ROOT/'protocol.json'),read(ROOT/'calibration.json')
    assert config['source_hashes']==weights['source_hashes']==source_hashes()
    assert int(os.environ.get('WORLD_SIZE','1'))==2
    work=ROOT/('preflight' if smoke else 'training')/arm
    work.mkdir(parents=True,exist_ok=True)
    target=3 if smoke else (epoch-10)*EPOCH_UPDATES
    epoch_folder=checkpoint(arm,epoch).parent
    if not smoke and (epoch_folder/'complete.json').exists():
        complete=read(epoch_folder/'complete.json')
        assert complete['updates']==target and sha(epoch_folder/'final.pt')==complete['checkpoint_sha256']
        return
    if not smoke: assert read(ROOT/'audit/preflight_passed.json')['passed']
    acc=Accelerator(mixed_precision='bf16',gradient_accumulation_steps=4,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
    torch.cuda.set_device(acc.local_process_index)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    resume=work/'latest.pt'
    pipe,network=load_network(acc.device,resume if resume.exists() and not smoke else PARENT_FILE,True)
    backbone,condition=[],[]
    for key,param in network.named_parameters():
        if param.requires_grad:
            (condition if key.startswith('adapter.') or '.processor.' in key else backbone).append(param)
    parameters=backbone+condition
    optimizer=bnb.optim.AdamW8bit([dict(params=backbone,lr=LRS[0]),dict(params=condition,lr=LRS[1])],
        betas=(.9,.999),eps=1e-8,weight_decay=.01)
    scheduler=get_scheduler('cosine',optimizer=optimizer,num_warmup_steps=0,num_training_steps=UPDATES)
    state=torch.load(resume if resume.exists() and not smoke else PARENT_FILE,map_location='cpu',weights_only=False)
    optimizer.load_state_dict(state['optimizer'])
    rng=torch.Generator(device=acc.device)
    rng.set_state(state['rngs'][acc.process_index])
    stream=state['streams'][acc.process_index]
    if resume.exists() and not smoke:
        assert state['source_hashes']==source_hashes() and state['world']==2 and state['arm']==arm
        step=state['step']; scheduler.load_state_dict(state['scheduler'])
    else:
        assert state['total_updates']==PARENT_UPDATES and state['world']==2
        step=0
        for group,lr in zip(optimizer.param_groups,LRS):group['lr']=group['initial_lr']=lr
    del state
    assert smoke or target-EPOCH_UPDATES<=step<=target
    expected=[lr*.5*(1+math.cos(math.pi*step/UPDATES)) for lr in LRS]
    assert scheduler.last_epoch==step and np.allclose([g['lr'] for g in optimizer.param_groups],expected,rtol=1e-10,atol=1e-15)
    if acc.is_main_process:
        trim_log(work/'training_log.csv',step)
        write(work/('start_epoch%02d.json'%epoch),dict(step=step,total_updates=PARENT_UPDATES+step,
            parent_sha256=PARENT_SHA,optimizer_moments_restored=True,private_rngs_restored=True,
            sampler_epoch=10+step//EPOCH_UPDATES,unet_lr=optimizer.param_groups[0]['lr'],
            condition_lr=optimizer.param_groups[1]['lr'],backbone_parameters=sum(p.numel() for p in backbone),
            condition_parameters=sum(p.numel() for p in condition),gpu=torch.cuda.get_device_name()))
    acc.wait_for_everyone()
    data=JointTextureDataset(str(ROOT/'training_manifest.json'),pipe.tokenizer,
        str(PROJECT.parent/'datasets/BF/training'),width=384,height=512,
        texture_preprocess_mode='plain_resize',mask_cache=str(core.MASKS))
    sampler=DistributedSampler(data,num_replicas=2,rank=acc.process_index,seed=42,drop_last=False)
    loader=DataLoader(data,batch_size=1,sampler=sampler,num_workers=2,pin_memory=True)
    assert len(loader)//4==EPOCH_UPDATES and len(loader)%4==0
    sampler.set_epoch(10+step//EPOCH_UPDATES)
    skip=(step%EPOCH_UPDATES)*4
    style=ReferenceRegionLoss().to(acc.device).eval()
    network,optimizer=acc.prepare(network,optimizer)
    network.train(); started=time.monotonic()

    def save(final=False):
        rngs,streams=[None]*2,[None]*2
        torch.distributed.all_gather_object(rngs,rng.get_state().cpu())
        torch.distributed.all_gather_object(streams,stream)
        if acc.is_main_process:
            value=dict(network={k:v.detach().cpu() for k,v in acc.unwrap_model(network).state_dict().items()},
                optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),rngs=rngs,streams=streams,
                step=step,total_updates=PARENT_UPDATES+step,world=2,source_hashes=source_hashes(),
                parent_checkpoint_sha256=PARENT_SHA,arm=arm)
            temporary=work/'latest.tmp';torch.save(value,temporary);temporary.replace(resume)
            if final:
                epoch_folder.mkdir(parents=True,exist_ok=True)
                assert not (epoch_folder/'final.pt').exists()
                os.link(resume,epoch_folder/'final.pt')
                write(epoch_folder/'complete.json',dict(updates=step,total_updates=PARENT_UPDATES+step,
                    epoch=epoch,world=2,effective_batch=8,checkpoint_sha256=sha(epoch_folder/'final.pt'),
                    parent_checkpoint_sha256=PARENT_SHA,source_hashes=source_hashes(),streams=streams))
        acc.wait_for_everyone()

    for micro,batch in enumerate(loader):
        if micro<skip:continue
        if step>=target:break
        batch={k:v.to(acc.device) for k,v in batch.items()}
        with acc.accumulate(network):
            with acc.autocast():raw,token,audit=training_losses(pipe,network,batch,rng,style,arm,weights)
            assert all(torch.isfinite(value).item() for value in raw.values()),token
            acc.backward(raw['total'])
            if acc.sync_gradients:
                if step%EPOCH_UPDATES==0:
                    model=acc.unwrap_model(network)
                    groups=dict(unet=backbone,bfm=list(model.adapter.fusion.parameters()),
                        projections=list(model.adapter.projections.parameters()),
                        gam=[p for k,p in model.unet.named_parameters() if '.processor.' in k and p.requires_grad])
                    norms={key:float(torch.stack([p.grad.float().square().sum() for p in group if p.grad is not None]).sum().sqrt()) for key,group in groups.items()}
                    assert all(np.isfinite(v) and v>0 for v in norms.values()),norms
                    assert model.adapter.injection_counts==[3,2,2,3]
                    if acc.is_main_process:write(work/('joint_gradients_epoch%02d.json'%epoch),norms)
                grad=acc.clip_grad_norm_(parameters,1.)
                assert torch.isfinite(grad).item()
            optimizer.step()
            if acc.sync_gradients:scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        stream=hashlib.sha256(stream.encode()+json.dumps(token,sort_keys=True).encode()).hexdigest()
        acc.unwrap_model(network).adapter.clear()
        if acc.sync_gradients:
            assert not acc.optimizer_step_was_skipped
            step+=1
            record=dict(step=step,total_updates=PARENT_UPDATES+step,epoch=epoch,
                seconds=time.monotonic()-started,unet_lr=scheduler.get_last_lr()[0],
                condition_lr=scheduler.get_last_lr()[1],grad=float(grad),
                peak_gib=torch.cuda.max_memory_allocated()/2**30,
                **{k:float(v.detach()) for k,v in raw.items()})
            if acc.is_main_process:
                path=work/'training_log.csv';exists=path.exists()
                with path.open('a',encoding='utf-8',newline='') as handle:
                    writer=csv.DictWriter(handle,fieldnames=list(record))
                    if not exists:writer.writeheader()
                    writer.writerow(record)
                if step<=3 or step%25==0 or step==target:
                    write(work/'progress.json',record);print('TRAIN',arm,json.dumps(record),flush=True)
            if not smoke and (step%1000==0 or step==target):save(step==target)
        del audit
    assert step==target
    if smoke:
        streams=[None]*2
        torch.distributed.all_gather_object(streams,stream)
        if acc.is_main_process:write(work/'passed.json',dict(passed=True,steps=step,streams=streams))


def confirm_smoke():
    a,b=[read(ROOT/'preflight'/arm/'passed.json') for arm in ARMS]
    assert a['passed'] and b['passed'] and a['streams']==b['streams']
    write(ROOT/'audit/preflight_passed.json',dict(passed=True,paired_data_noise_streams=True,
        jointly_updated=['UNet','BFM','projections','GAM'],streams=a['streams']))


def copy_outputs(source,target,arm,rows):
    for row in rows:
        old=source/'outputs'/arm/row['condition']/(row['id']+'.png')
        meta=read(old.with_suffix('.json'));assert sha(old)==meta['output_sha256']
        dest=target/'outputs'/arm/row['condition']/old.name
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(old,dest);shutil.copy2(old.with_suffix('.json'),dest.with_suffix('.json'))


def generate(where,arm,path,rows,expected_updates):
    import torch
    complete=read(path.parent/'complete.json')
    assert complete['updates']==expected_updates and sha(path)==complete['checkpoint_sha256']
    pipe,network=load_network('cuda',path)
    active={key:str(next(module.parameters()).device) for key,module in dict(unet=pipe.unet,
        vae=pipe.vae,text=pipe.text_encoder,reference=pipe.reference_unet,adapter=network.adapter).items()}
    assert set(active.values())=={'cuda:0'} and str(pipe._execution_device)=='cuda:0',active
    write(where/'audit'/('generation_device_'+arm+'.json'),active)
    network.eval().requires_grad_(False);pipe.unet.half();pipe.vae.half();network.adapter.half()
    with torch.inference_mode():
        for row in rows:
            dest=where/'outputs'/arm/row['condition']/(row['id']+'.png')
            if dest.exists():
                meta=read(dest.with_suffix('.json'))
                assert sha(dest)==meta['output_sha256'] and meta['checkpoint_sha256']==complete['checkpoint_sha256']
                continue
            sketch=core.tensor(row['sketch']).cuda().half();reference=core.tensor(row['reference']).cuda().half()
            network.adapter.condition(sketch,reference)
            latent=torch.randn((1,4,64,48),device='cuda',dtype=torch.float16,
                generator=torch.Generator(device='cuda').manual_seed(row['seed']))
            latent_sha=digest(latent)
            if 'initial_latent_sha256' in row:assert latent_sha==row['initial_latent_sha256']
            output=pipe(ref_image=sketch,prompt=row['caption'],texture_clip_image=None,texture_embeds=None,
                null_prompt='',negative_prompt=' worst quality, low quality',width=384,height=512,
                num_images_per_prompt=1,guidance_scale=7.,sketch_scale=.6,ipa_scale=0.,
                generator=torch.Generator(device='cuda').manual_seed(row['seed']),latents=latent,num_inference_steps=50)[0]
            dest.parent.mkdir(parents=True,exist_ok=True);output.save(dest)
            write(dest.with_suffix('.json'),dict(id=row['id'],arm=arm,condition=row['condition'],split=row['split'],
                seed=row['seed'],initial_latent_sha256=latent_sha,output_sha256=sha(dest),
                checkpoint_sha256=complete['checkpoint_sha256'],GT_used_in_inference=False,
                input_sketch_sha256=row['sketch_sha256'],input_reference_sha256=row['reference_sha256'],
                steps=50,cfg=7.,native_size=[384,512]))
            network.adapter.clear();print('GENERATED',arm,row['id'],row['condition'],flush=True)
    write(where/('generation_complete_'+arm+'.json'),dict(complete=True,outputs=len(rows)))


def evaluate(where,arms,rows,primary_split):
    import torch
    from eval.eval_utils import prepare_evaluation_masks
    from eval.metrics import compute_structure_preservation,compute_texture_color_fidelity,compute_texture_leakage
    from tools.e2a_attr_eval import frozen_lpips
    from tools import t3_response_run as response
    from tools.t3_response_common import patches
    from tools.bf_t3_three_arm_report import gray,paired,table
    response.ROOT=where;dists=response.load_dists('cuda');lpips,assets=frozen_lpips()
    write(where/'audit/lpips_assets.json',assets)
    originals={r['id']:r for r in rows if r['condition']=='original'}
    cache={}
    def features(parts):
        value=torch.from_numpy(np.stack(parts).transpose(0,3,1,2).copy()).float().cuda()/255
        return dists.forward_once(value)
    def reference(path,is_gray):
        key=(path,is_gray)
        if key not in cache:
            image=np.asarray(Image.open(path).convert('RGB').resize((384,512),Image.Resampling.LANCZOS))
            parts=[image[y:y+64,x:x+64] for y in np.linspace(0,448,4).round().astype(int) for x in np.linspace(0,320,4).round().astype(int)]
            cache[key]=features([gray(p) for p in parts] if is_gray else parts)
        return cache[key]
    def distance(a,b):
        matrix=response.distances(dists,a,b)
        return float(.5*(matrix.min(0).values.mean()+matrix.min(1).values.mean()))
    records=[]
    with torch.inference_mode():
        for row in rows:
            masks=prepare_evaluation_masks((384,512),sketch_path=row['sketch'],mask_policy='sketch_only')
            coords,_=patches(masks['garment']);latents=[]
            for arm in arms:
                path=where/'outputs'/arm/row['condition']/(row['id']+'.png')
                meta=read(path.with_suffix('.json'));assert sha(path)==meta['output_sha256']
                assert meta['input_sketch_sha256']==row['sketch_sha256'] and meta['input_reference_sha256']==row['reference_sha256']
                if arm!='BF-Fashion':latents.append(meta['initial_latent_sha256'])
                kwargs=dict(gen_path=str(path),sketch_path=row['sketch'],target_path=row['gt'],mask_bundle=masks)
                structure=compute_structure_preservation(**kwargs)
                item=dict(arm=arm,id=row['id'],split=row['split'],condition=row['condition'],
                    edge_f1=structure.get('edge_f1'),iou=structure.get('sketch_iou'),edge_l1=structure.get('edge_l1'),
                    leak=compute_texture_leakage(**kwargs).get('leak_colored_frac'),lpips=None,tcf_lab=None,
                    dists_margin=None,gray_margin=None,gray_dists=None,patch_n=len(coords))
                if row['condition']=='original':
                    item['lpips']=float(lpips(core.tensor(path).cuda(),core.tensor(row['gt']).cuda()).mean())
                    item['tcf_lab']=compute_texture_color_fidelity(texture_path=row['reference'],**kwargs).get('tcf_lab_delta')
                if row['condition']=='swapped' and len(coords)>=4:
                    image=np.asarray(Image.open(path).convert('RGB'));parts=[image[y:y+64,x:x+64] for x,y in coords]
                    for is_gray in (False,True):
                        generated=features([gray(p) for p in parts] if is_gray else parts)
                        new=distance(generated,reference(row['reference'],is_gray))
                        old=distance(generated,reference(originals[row['id']]['reference'],is_gray))
                        item['gray_margin' if is_gray else 'dists_margin']=old-new
                        if is_gray:item['gray_dists']=new
                records.append({k:None if isinstance(v,float) and not np.isfinite(v) else v for k,v in item.items()})
            assert len(set(latents))==1
            print('EVALUATED',where.name,row['id'],row['condition'],flush=True)
    assert len(records)==len(rows)*len(arms)
    metrics=['edge_f1','iou','edge_l1','leak','lpips','tcf_lab','dists_margin','gray_margin','gray_dists']
    summary=[]
    for arm in arms:
        for split,condition in sorted({(r['split'],r['condition']) for r in rows}):
            selected=[r for r in records if r['arm']==arm and r['split']==split and r['condition']==condition]
            for metric in metrics:
                values=[r[metric] for r in selected if r[metric] is not None]
                summary.append(dict(arm=arm,split=split,condition=condition,metric=metric,n=len(values),
                    planned_n=len(selected),mean=float(np.mean(values)) if values else None))
    lookup={(r['arm'],r['id'],r['condition']):r for r in records};comparisons=[]
    for newer in arms[1:]:
        for other in arms[:arms.index(newer)]:
            family=[]
            for metric in metrics:
                condition='swapped' if metric in metrics[6:] else 'original'
                selected=[r for r in rows if r['condition']==condition and (condition=='swapped' or r['split']==primary_split)]
                values=[]
                for row in selected:
                    a,b=lookup[newer,row['id'],condition][metric],lookup[other,row['id'],condition][metric]
                    if a is not None and b is not None:values.append(a-b)
                family.append(dict(comparison=newer+' minus '+other,metric=metric,**paired(values)))
            running=0.;ordered=sorted([r for r in family if r['p'] is not None],key=lambda r:r['p'])
            for i,item in enumerate(ordered):
                running=max(running,min(1.,item['p']*(len(ordered)-i)));item['p_holm']=running
            comparisons.extend(family)
    for filename,value in [('per_image',records),('summary',summary),('paired_statistics',comparisons)]:
        write(where/'metrics'/(filename+'.json'),value)
        table(where/'metrics'/(filename+'.csv'),[dict(x,ci95=json.dumps(x['ci95'])) if 'ci95' in x else x for x in value])
    write(where/'evaluation_complete.json',dict(complete=True,rows_per_arm=len(rows),arms=arms,records=len(records)))


def validation(arm,epoch):
    rows=read(ROOT/'validation.json');where=folder(arm,epoch)/'validation'
    baseline=ROOT/'validation_baseline'
    if not (baseline/('generation_complete_'+PARENT_ARM+'.json')).exists():
        generate(baseline,PARENT_ARM,PARENT_FILE,rows,28365)
    copy_outputs(baseline,where,PARENT_ARM,rows)
    generate(where,name(arm,epoch),checkpoint(arm,epoch),rows,(epoch-10)*EPOCH_UPDATES)
    evaluate(where,[PARENT_ARM,name(arm,epoch)],rows,'validation64')


def formal():
    rows=read(ROOT/'evaluation.json');where=ROOT/'final_comparison'
    for arm in BASELINES:copy_outputs(PARENT,where,arm,rows)
    for arm in ARMS:
        generate(where,name(arm,12),checkpoint(arm,12),rows,UPDATES)
    evaluate(where,BASELINES+[name(arm,12) for arm in ARMS],rows,'heldout128')


def report():
    from tools.bf_t3_three_arm_report import table
    where=ROOT/'final_comparison'
    assert read(where/'evaluation_complete.json')['complete']
    parent={(r['arm'],r['split'],r['condition'],r['metric']):r for r in read(PARENT/'metrics/summary.json')}
    for row in read(where/'metrics/summary.json'):
        if row['arm'] in BASELINES and row['mean'] is not None:
            old=parent[row['arm'],row['split'],row['condition'],row['metric']]
            assert row['n']==old['n'] and abs(row['mean']-old['mean'])<1e-6
    for epoch in (11,12):
        a,b=[read(checkpoint(arm,epoch).parent/'complete.json') for arm in ARMS]
        assert a['streams']==b['streams'],'paired data/noise streams diverged'
    rows=read(where/'metrics/per_image.json');lookup={(r['arm'],r['id'],r['condition']):r for r in rows}
    gate=[]
    for condition,split in [('original','heldout128'),('swapped','swap32')]:
        selected=[r for r in rows if r['arm']==PARENT_ARM and r['condition']==condition and r['split']==split]
        for metric,margin in read(ROOT/'protocol.json')['safety_margins'].items():
            sign=1 if metric=='leak' else -1
            deltas=np.array([sign*(lookup[name('texture',12),r['id'],condition][metric]-r[metric]) for r in selected],float)
            assert np.isfinite(deltas).all()
            rng=np.random.default_rng(190112)
            bootstrap=deltas[rng.integers(len(deltas),size=(10000,len(deltas)))].mean(1)
            bound=float(np.quantile(bootstrap,1-.05/6))
            gate.append(dict(condition=condition,metric=metric,worse_direction_mean=float(deltas.mean()),
                simultaneous_one_sided_upper=bound,tolerance=margin,passed=bound<=margin))
    write(ROOT/'metrics/safety_gate.json',gate);table(ROOT/'metrics/safety_gate.csv',gate)
    paired=read(where/'metrics/paired_statistics.json')
    texture_tests=[r for r in paired if r['metric']=='gray_dists' and r['comparison'] in
        [name('texture',12)+' minus '+other for other in [PARENT_ARM,name('control',12)]]]
    texture_pass=all(r['mean']<0 and r['p_holm']<.05 for r in texture_tests) and len(texture_tests)==2
    response_tests=[r for r in paired if r['metric']=='gray_margin' and r['comparison'] in
        [name('texture',12)+' minus '+other for other in [PARENT_ARM,name('control',12)]]]
    texture_pass=texture_pass and len(response_tests)==2 and all(r['mean']>0 for r in response_tests)
    write(ROOT/'complete.json',dict(complete=True,arms=list(ARMS),epochs_per_arm=2,
        updates_per_arm=UPDATES,total_updates_per_arm=PARENT_UPDATES+UPDATES,parent_sha256=PARENT_SHA,
        primary_final_epoch=12,safety_pass=all(r['passed'] for r in gate),texture_improvement_pass=texture_pass,
        formal_new_outputs=384,validation_new_outputs=512,validation_baseline_outputs=128,total_new_outputs=1024,
        accepted=all(r['passed'] for r in gate) and texture_pass,
        interpretation='report outcome regardless of success; no automatic promotion or extension'))
    files=[p for p in ROOT.rglob('*') if p.is_file() and 'source' not in p.relative_to(ROOT).parts
        and p.suffix in ['.json','.csv','.png'] and p.name not in ['bundle_info.json','bundle_manifest.json']]
    write(ROOT/'bundle_manifest.json',{str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in files})
    with tarfile.open(ROOT/'review_results.tar.gz','w:gz') as archive:
        for path in files+[ROOT/'bundle_manifest.json']:archive.add(path,arcname=str(path.relative_to(ROOT)))
    write(ROOT/'bundle_info.json',dict(bytes=(ROOT/'review_results.tar.gz').stat().st_size,sha256=sha(ROOT/'review_results.tar.gz')))
    print('COMPLETED',json.dumps(read(ROOT/'complete.json')),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['prepare','calibrate','train','smoke','confirm_smoke','validation','formal','report'])
    parser.add_argument('--arm',choices=ARMS)
    parser.add_argument('--epoch',type=int,choices=(11,12),default=11)
    args=parser.parse_args()
    if args.stage in ['train','smoke']:
        assert args.arm;train(args.arm,args.epoch,args.stage=='smoke')
    elif args.stage=='validation':
        assert args.arm;validation(args.arm,args.epoch)
    else:globals()[args.stage]()
