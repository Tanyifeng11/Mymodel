"""三组受控比较；已有 BF 只读复用，新两臂只改变 T3 监督开关。"""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import time

import numpy as np
from PIL import Image

SOURCE = Path(__file__).resolve().parents[1]
PROJECT = Path('/share/home/u2515283058/Mymodel')
ROOT = PROJECT/'output_eval/bf_fusion_t3_three_arm_20261101'
BF_REPO = PROJECT.parent/'BF-Fashion-'
E1 = PROJECT/'output/phase1_e1_grouped_e5/checkpoint-final/joint_model.pt'
E1_SHA = 'e51d71dbf17c424fb639293ab99cfa83b9658451e127e6a61e8263e15bd5e39a'
BF_SHA = '1c57330eb6447a6161816ad2138234f74e0506d85d7577ae1375d65af24d594c'
ARMS = ['BF-Fashion', 'BF-SD', 'BF-SD-T3']
MASKS = PROJECT/'output_eval/e2a_attr_01_20261019/masks/train'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def tensor(path, size=(384,512)):
    import torch
    a = np.asarray(Image.open(path).convert('RGB').resize(size, Image.Resampling.BILINEAR), np.float32)
    return torch.from_numpy(a.transpose(2,0,1).copy())[None]/127.5-1


def prepare():
    import torch
    assert not (ROOT/'protocol.json').exists(), '不覆盖冻结协议'
    assert sha(E1) == E1_SHA
    upstream = BF_REPO/'ldm/modules/diffusionmodules/Fusion.py'
    copied = SOURCE/'models/bf_original_fusion.py'
    assert upstream.read_text() == copied.read_text(), '完整BF源码必须原样一致'
    bf = read(PROJECT/'output_eval/t3_p0_p1_20261028/audit/BF_provenance.json')
    assert sha(bf['checkpoint_path']) == BF_SHA
    state = torch.load(bf['checkpoint_path'], map_location='cpu')['state_dict']
    fusion = {k[len('cond_stage_model.'):]:v for k,v in state.items() if k.startswith('cond_stage_model.')}
    from models.bf_sd_skip import BFSkipFusion
    adapter = BFSkipFusion()
    adapter.fusion.load_state_dict(fusion, strict=True)
    del state, adapter
    (ROOT/'initialization').mkdir(parents=True)
    torch.save(fusion, ROOT/'initialization/bf_fusion.pt')
    train_path = PROJECT/'data/train_bf_texture.json'
    train = read(train_path)
    train_roots = {Path(r['cloth']).stem.split('_')[0] for r in train}
    cases = []
    for old_name, label in [('garment_test128_20261031','heldout128'), ('t3_p0_p1_20261028','swap32')]:
        old = PROJECT/'output_eval'/old_name
        rows = read(old/'protocol/input_manifest.json')
        swap_ids = {r['id'] for r in rows if r['condition']=='swapped'}
        rows = rows if label=='heldout128' else [r for r in rows if r['id'] in swap_ids]
        for row in rows:
            assert row['id'].split('_')[0] not in train_roots
            row = dict(row, split=label)
            for key in ['sketch','reference','gt']:
                source = Path(row[key])
                assert source.is_file()
                target = ROOT/'inputs'/sha(source)
                if not target.exists(): shutil.copy2(source,target)
                row[key] = str(target)
                row[key+'_sha256'] = sha(target)
            output = old/'outputs/BF-Fashion'/row['condition']/(row['id']+'.png')
            meta = read(output.with_suffix('.json'))
            assert meta['output_sha256'] == sha(output)
            assert meta['input_reference_sha256'] == row['reference_sha256']
            assert meta['input_sketch_sha256'] == row['sketch_sha256']
            assert meta['checkpoint'] == bf['checkpoint_path']
            dest = ROOT/'outputs/BF-Fashion'/row['condition']/(row['id']+'.png')
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(output,dest)
            write(dest.with_suffix('.json'),dict(meta,reused_from=str(output),split=label))
            cases.append(row)
    assert len(cases)==192 and len({(r['id'],r['condition']) for r in cases})==192
    write(ROOT/'evaluation.json',cases)
    write(ROOT/'training_manifest.json',train)
    write(ROOT/'audit/bf_provenance.json',bf)
    write(ROOT/'protocol.json',dict(name=ROOT.name,arms=ARMS,seed=42,updates=28365,
        epochs=5,samples=len(train),effective_batch=8,learning_rate=5e-5,warmup=500,
        optimizer='AdamW betas(.9,.999) eps1e-8 wd.01',precision='bf16 autocast; trainable parameters float32',
        initialization='same E1 + existing trained original BF fusion + zero channel projections',
        e1_sha256=E1_SHA,bf_checkpoint_sha256=BF_SHA,fusion_sha256=sha(ROOT/'initialization/bf_fusion.pt'),
        bf_source_sha256=sha(upstream),copied_source_sha256=sha(copied),train_sha256=sha(train_path),
        source_commit=subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip(),
        architecture='original bFMFusion ngf144 depth100 resblocks0; four SD decoder skip scales; original channel-mismatch skip rule; no 16-token texture branch',
        trainable='original BF fusion, four channel projections, existing GAM self-attention processors; SD original backbone and reference UNet frozen',
        objectives=dict(base='epsilon MSE + 1 GT masked Gram + .05 sketch edge + .1 reference RGB garment',
                        T3='base + .1 reference RGB body + .03816244801608925 GT L1 rest'),
        dropout=dict(text_only=.2,reference_only=.05,both=.05),
        inference=dict(width=384,height=512,steps=50,cfg=7.,sketch_scale=.6,texture_tokens=False),
        evaluation=dict(original_primary=128,reference_swap_identities=32,rows_per_arm=192,
                        paired_initial_noise='same within new SD arms and reference conditions; BF native noise differs'),
        limitations=['BF is a historical trained baseline with different budget/native resolution',
                     'previously inspected test sets; exploratory single training seed',
                     'copied fusion weights have prior training exposure; current test product-root overlap zero',
                     'keep original T3 coefficients; no new gradient-ratio calibration on this architecture']))
    print('PREPARED',len(train),len(cases),flush=True)


def load_network(device, checkpoint=None):
    import torch
    from types import SimpleNamespace
    from models.bf_sd_skip import BFSkipFusion, BFSDNetwork
    from adapter.attention_processor import IPAttnProcessor2_0
    spec=importlib.util.spec_from_file_location('bf_t3_inference',SOURCE/'inference_IMAGGarment-1.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    sd=PROJECT/'models/stable-diffusion-v1-5'
    args=SimpleNamespace(GAM_model_ckpt=str(E1),texture_ckpt=str(E1),base_model_path=str(sd),
        vae_model_path=str(sd/'vae'),image_encoder_path=str(PROJECT/'models/clip'),device=str(device),
        width=384,height=512,texture_num_tokens=16,force_texture_num_tokens_override=False,
        texture_condition_mode='token',alpha1=2.,alpha2=2.,alpha3=1.5,alpha4=1.)
    pipe,_=module.prepare(args)
    for m in [pipe.unet,pipe.reference_unet,pipe.vae,pipe.text_encoder,pipe.image_encoder,pipe.bf_texture_conditioner]:
        m.eval().requires_grad_(False)
    pipe.set_layer_group_enabled(True);pipe._setup_layer_groups();pipe.set_progress_bar_config(disable=True)
    pipe.unet.to(device=device,dtype=torch.float32)
    pipe.vae.to(device=device,dtype=torch.float32)
    pipe.reference_unet.to(device)
    pipe.text_encoder.to(device)
    pipe.set_texture_token_enabled(False)
    pipe.image_encoder.cpu()
    pipe.bf_texture_conditioner.cpu()
    for name,processor in pipe.unet.attn_processors.items():
        if not isinstance(processor,IPAttnProcessor2_0):
            for p in processor.parameters(): p.requires_grad_(True)
    torch.manual_seed(42)
    adapter = BFSkipFusion().to(device)
    adapter.fusion.load_state_dict(torch.load(ROOT/'initialization/bf_fusion.pt',map_location='cpu'),strict=True)
    network = BFSDNetwork(pipe.unet,adapter)
    if checkpoint:
        network.load_trainable(torch.load(checkpoint,map_location='cpu')['network'])
    return pipe,network


def sketch_states(pipe, sketch_latent, timestep):
    import torch
    with torch.no_grad():
        pipe.reference_unet(sketch_latent,torch.zeros_like(timestep),None,return_dict=False)
    return {name:p.cache['hidden_states'] for name,p in pipe.reference_unet.attn_processors.items()
            if 'attn1' in name and hasattr(p,'cache')}


def losses(pipe, network, batch, rng, style, audit_mode=False):
    import torch
    import torch.nn.functional as F
    from train_GAM_texture_joint import masked_edge_l1, texture_color_stat_loss, build_region_masks, masked_l1_loss
    from diffusers import DDIMScheduler
    scheduler = DDIMScheduler(beta_start=.00085,beta_end=.012,beta_schedule='scaled_linear',num_train_timesteps=1000)
    device = batch['vae_cloth'].device
    with torch.no_grad():
        posterior = pipe.vae.encode(batch['vae_cloth']).latent_dist
        latent = (posterior.mean + posterior.std*torch.randn(posterior.mean.shape,device=device,generator=rng))*pipe.vae.config.scaling_factor
        sketch_latent = pipe.vae.encode(batch['vae_sketch']).latent_dist.mean*pipe.vae.config.scaling_factor
        drop = float(torch.rand((),device=device,generator=rng))
        if audit_mode: drop=0.
        text_ids = batch['input_ids']
        if drop < .2 or .25 <= drop < .3:
            text_ids = pipe.tokenizer('',padding='max_length',truncation=True,max_length=77,return_tensors='pt').input_ids.to(device)
        text = pipe.text_encoder(text_ids)[0]
        noise = torch.randn(latent.shape,device=device,dtype=latent.dtype,generator=rng)
        t = torch.randint(0,1000,(len(latent),),device=device,generator=rng)
        if audit_mode: t.fill_(100)
        sa = sketch_states(pipe,sketch_latent.to(pipe.reference_unet.dtype),t)
    reference = batch['texture_image']
    keep = not (.2 <= drop < .3)
    conditioned_reference = reference if keep else torch.zeros_like(reference)
    noisy = scheduler.add_noise(latent,noise,t)
    predicted = network(noisy,t,text,batch['vae_sketch'],conditioned_reference,sa)
    mse = F.mse_loss(predicted.float(),noise.float())
    alpha = scheduler.alphas_cumprod.to(device)[t].reshape(-1,1,1,1)
    x0 = (noisy.float()-(1-alpha).sqrt()*predicted.float())/alpha.sqrt()
    # VAE 冻结但保留输入梯度；两臂都走同样的解码与基础风格监督。
    decoded = pipe.vae.decode(x0/pipe.vae.config.scaling_factor).sample.float()
    mask = batch['garment_mask']
    gram = style(decoded,batch['vae_cloth'],mask=mask)
    edge = masked_edge_l1(decoded,batch['vae_sketch'],mask=mask)
    color = texture_color_stat_loss(decoded,reference,mask) if keep else predicted.sum()*0
    body,_,_ = build_region_masks(mask,9)
    body_rgb = texture_color_stat_loss(decoded,reference,body) if keep else predicted.sum()*0
    rest = masked_l1_loss(decoded,batch['vae_cloth'],1-body)
    return dict(base=mse+gram+.05*edge+.1*color,body=body_rgb,rest=rest,mse=mse,gram=gram,edge=edge,color=color), dict(
        index=int(batch['dataset_index'][0]),t=int(t[0]),drop=drop,noise_sha256=digest(noise))


def train(arm, preflight=False):
    import torch
    from torch.utils.data import DataLoader,DistributedSampler
    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs
    from diffusers.optimization import get_scheduler
    from train_GAM_texture_joint import JointTextureDataset,VGGGramStyleLoss
    config = read(ROOT/'protocol.json')
    if not preflight: assert read(ROOT/'audit/preflight.json')['passed']
    world = int(os.environ.get('WORLD_SIZE','1'))
    assert world in [1,2,4,8]
    accumulation = 8//world
    acc = Accelerator(mixed_precision='bf16',gradient_accumulation_steps=accumulation,
                      kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
    torch.cuda.set_device(acc.device)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    pipe,network = load_network(acc.device)
    params = [p for p in network.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params,lr=5e-5,betas=(.9,.999),eps=1e-8,weight_decay=.01)
    lr = get_scheduler('cosine',optimizer=optimizer,num_warmup_steps=500,num_training_steps=config['updates'])
    dataset = JointTextureDataset(str(ROOT/'training_manifest.json'),pipe.tokenizer,str(PROJECT.parent/'datasets/BF/training'),
                                 width=384,height=512,texture_preprocess_mode='plain_resize',mask_cache=str(MASKS))
    sampler = DistributedSampler(dataset,num_replicas=world,rank=acc.process_index,seed=42,drop_last=False)
    loader = DataLoader(dataset,batch_size=1,sampler=sampler,num_workers=2,pin_memory=True)
    assert len(loader)%accumulation==0
    style = VGGGramStyleLoss().to(acc.device).eval().requires_grad_(False)
    folder = ROOT/('audit/preflight_training' if preflight else 'training/'+arm)
    folder.mkdir(parents=True,exist_ok=True)
    source_hashes={p:sha(SOURCE/p) for p in ['models/bf_original_fusion.py','models/bf_sd_skip.py',
                   'tools/bf_t3_three_arm.py','train_GAM_texture_joint.py','adapter/attention_processor.py']}
    initial = hashlib.sha256()
    for name,value in network.trainable_state().items(): initial.update(name.encode()+value.numpy().tobytes())
    initial_sha = initial.hexdigest()
    rng = torch.Generator(device=acc.device).manual_seed(42000+acc.process_index)
    stream = '0'*64
    step,start = 0,time.monotonic()
    resume = folder/'latest.pt'
    if resume.exists() and not preflight:
        state = torch.load(resume,map_location='cpu')
        assert state['world']==world
        network.load_trainable(state['network'])
        optimizer.load_state_dict(state['optimizer'])
        lr.load_state_dict(state['scheduler'])
        rng.set_state(state['rngs'][acc.process_index])
        stream=state['stream_hashes'][acc.process_index]
        step = state['step']
    network,optimizer = acc.prepare(network,optimizer)
    network.train()
    total = 3 if preflight else config['updates']
    updates_per_epoch = len(loader)//accumulation
    epoch = step//updates_per_epoch
    skip_batches = (step%updates_per_epoch)*accumulation
    def save(step,final=False):
        states = [None]*world
        chains = [None]*world
        if world>1: torch.distributed.all_gather_object(states,rng.get_state().cpu())
        else: states[0]=rng.get_state().cpu()
        if world>1: torch.distributed.all_gather_object(chains,stream)
        else: chains[0]=stream
        if acc.is_main_process:
            value = dict(network=acc.unwrap_model(network).trainable_state(),optimizer=optimizer.state_dict(),
                         scheduler=lr.state_dict(),rngs=states,stream_hashes=chains,step=step,world=world,initial_sha256=initial_sha)
            tmp=folder/'latest.tmp'
            torch.save(value,tmp);tmp.replace(resume)
            if final or step%updates_per_epoch==0:
                torch.save(dict(network=value['network'],step=step,initial_sha256=initial_sha),folder/('step%05d.pt'%step))
    logs=[]
    while step<total:
        sampler.set_epoch(epoch)
        for micro,batch in enumerate(loader):
            if micro<skip_batches: continue
            batch={k:v.to(acc.device) for k,v in batch.items()}
            with acc.accumulate(network):
                with acc.autocast(): raw,token=losses(pipe,network,batch,rng,style,preflight)
                stream=hashlib.sha256(stream.encode()+json.dumps(token,sort_keys=True).encode()).hexdigest()
                aux=.1*raw['body']+.03816244801608925*raw['rest']
                loss=raw['base']+(aux if arm=='BF-SD-T3' else 0)
                assert torch.isfinite(loss).item(),token
                if preflight and step==2 and micro%accumulation==0:
                    assert acc.unwrap_model(network).adapter.injection_counts==[3,2,2,3]
                    probe=acc.unwrap_model(network).adapter.projections[0].weight
                    gradients={k:float(torch.autograd.grad(raw[k],probe,retain_graph=True)[0].float().norm()) for k in ['body','rest']}
                    assert all(np.isfinite(v) and v>0 for v in gradients.values()),gradients
                    original=acc.unwrap_model(network).adapter.fusion.downconv1[0].weight
                    gradients['original_fusion']=float(torch.autograd.grad(raw['base']+aux,original,retain_graph=True)[0].float().norm())
                    assert np.isfinite(gradients['original_fusion']) and gradients['original_fusion']>0
                    if acc.is_main_process: write(ROOT/'audit/auxiliary_gradients.json',gradients)
                acc.backward(loss)
                if acc.sync_gradients:
                    grad=acc.clip_grad_norm_(params,1.)
                    assert torch.isfinite(grad).item()
                optimizer.step()
                if acc.sync_gradients: lr.step()
                optimizer.zero_grad(set_to_none=True)
            acc.unwrap_model(network).adapter.clear()
            if acc.sync_gradients:
                step+=1
                record=dict(step=step,seconds=time.monotonic()-start,lr=lr.get_last_lr()[0],grad=float(grad),
                            peak_gib=torch.cuda.max_memory_allocated()/2**30,
                            **{k:float(v.detach()) for k,v in raw.items()})
                logs.append(record)
                if acc.is_main_process:
                    log_path=folder/'training_log.csv'
                    exists=log_path.exists()
                    with log_path.open('a',encoding='utf-8',newline='') as f:
                        writer=csv.DictWriter(f,fieldnames=list(record))
                        if not exists:writer.writeheader()
                        writer.writerow(record)
                if acc.is_main_process and (step<=3 or step%25==0):
                    write(folder/'progress.json',record)
                    print('TRAIN',arm,json.dumps(record),flush=True)
                if step%1000==0 or step==total: save(step,step==total)
                if step>=total: break
        skip_batches=0
        epoch+=1
    acc.wait_for_everyone()
    stream_values=[None]*world
    if world>1: torch.distributed.all_gather_object(stream_values,stream)
    else: stream_values[0]=stream
    if acc.is_main_process:
        write(folder/'complete.json',dict(arm=arm,updates=step,initial_sha256=initial_sha,
              trainable_parameters=sum(p.numel() for p in params),stream_hashes=stream_values,
              source_sha256=source_hashes,
              seconds=time.monotonic()-start,world=world,effective_batch=8,logs=logs[-10:],
              checkpoint_sha256=sha(folder/('step%05d.pt'%step))))
        if preflight:
            write(ROOT/'audit/preflight.json',dict(passed=True,updates=step,
                  source_exact=True,auxiliary_gradients=read(ROOT/'audit/auxiliary_gradients.json'),
                  initialization_sha256=initial_sha,peak_gib=torch.cuda.max_memory_allocated()/2**30))
    print('COMPLETE',arm,step,flush=True)


def generate(arm,smoke=False):
    import torch
    checkpoint=ROOT/'training'/arm/'step28365.pt'
    pipe,network=load_network('cuda',checkpoint)
    network.eval().requires_grad_(False)
    pipe.unet.half(); pipe.vae.half(); network.adapter.half()
    rows=read(ROOT/'evaluation.json')
    if smoke: rows=rows[:4]
    with torch.inference_mode():
        for row in rows:
            dest=ROOT/'outputs'/arm/row['condition']/(row['id']+'.png')
            if dest.exists():
                assert read(dest.with_suffix('.json'))['output_sha256']==sha(dest)
                continue
            sketch=tensor(row['sketch']).cuda().half()
            reference=tensor(row['reference']).cuda().half()
            network.adapter.condition(sketch,reference)
            latent=torch.randn((1,4,64,48),device='cuda',dtype=torch.float16,
                               generator=torch.Generator(device='cuda').manual_seed(row['seed']))
            before=digest(latent)
            image=pipe(ref_image=sketch,prompt=row['caption'],texture_clip_image=None,texture_embeds=None,
                null_prompt='',negative_prompt=' worst quality, low quality',width=384,height=512,
                num_images_per_prompt=1,guidance_scale=7.,sketch_scale=.6,ipa_scale=0.,
                generator=torch.Generator(device='cuda').manual_seed(row['seed']),latents=latent,num_inference_steps=50)[0]
            dest.parent.mkdir(parents=True,exist_ok=True)
            image.save(dest)
            write(dest.with_suffix('.json'),dict(id=row['id'],arm=arm,condition=row['condition'],split=row['split'],
                seed=row['seed'],initial_latent_sha256=before,output_sha256=sha(dest),
                checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),text_used=True,
                GT_used_in_inference=False,input_sketch_sha256=row['sketch_sha256'],
                input_reference_sha256=row['reference_sha256'],steps=50,cfg=7.,native_size=[384,512]))
            network.adapter.clear()
            print('GENERATED',arm,row['condition'],row['id'],flush=True)
    write(ROOT/'generation'/arm/'complete.json',dict(n=len(rows),smoke=smoke))


def evaluate():
    from tools import bf_t3_three_arm_report
    bf_t3_three_arm_report.evaluate()


def package():
    import tarfile
    files=[]
    for folder in ['audit','generation','metrics','figures','outputs','inputs']:
        files.extend(p for p in (ROOT/folder).rglob('*') if p.is_file())
    files += [ROOT/'protocol.json',ROOT/'evaluation.json']
    manifest={str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in files}
    write(ROOT/'bundle_manifest.json',manifest)
    with tarfile.open(ROOT/'review_results.tar.gz','w:gz') as tar:
        for p in files+[ROOT/'bundle_manifest.json']:tar.add(p,arcname=str(p.relative_to(ROOT)))
    write(ROOT/'bundle_info.json',dict(bytes=(ROOT/'review_results.tar.gz').stat().st_size,
          sha256=sha(ROOT/'review_results.tar.gz'),members=len(files)+1))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['prepare','preflight','train','generate','evaluate','package'])
    parser.add_argument('--arm',choices=ARMS[1:])
    args=parser.parse_args()
    if args.action=='preflight':train('BF-SD-T3',True)
    elif args.action=='train':train(args.arm)
    elif args.action=='generate':generate(args.arm)
    else:globals()[args.action]()
