"""冻结选中CN，训练原生标准IPA；完整保存数据游标、RNG、优化器与Scaler。"""
import random
import time
import numpy as np
import torch
from PIL import Image
from diffusers import DDPMScheduler
from cn_ipa.common import *
from cn_ipa import runtime as rt


def selected_cn():
    return Path(read(ROOT/'selection/cn_checkpoint.json')['path'])


def tensors(pipe,row):
    # 与C0/C1管线完全相同的CLIP预处理；只把独立reference交给图像编码器。
    sketch,ref=rt.inputs(row)
    pixels=pipe.feature_extractor(images=ref,return_tensors='pt').pixel_values.to('cuda',torch.float16)
    ids=pipe.tokenizer(row['caption'],padding='max_length',truncation=True,
        max_length=pipe.tokenizer.model_max_length,return_tensors='pt').input_ids.to('cuda')
    with torch.no_grad():
        emb=pipe.image_encoder(pixels).image_embeds[:,None]
        text=pipe.text_encoder(ids)[0]
    sketch=torch.from_numpy(np.asarray(sketch,np.float32).transpose(2,0,1)/255)[None].to('cuda',torch.float16)
    return text,emb,sketch


def enable_training(pipe):
    parameters=rt.trainable_parameters(pipe.unet)
    assert parameters
    for parameter in parameters.values():parameter.data=parameter.data.float();parameter.requires_grad_(True)
    pipe.unet.train();pipe.unet.enable_gradient_checkpointing()
    assert all(module.p==0 for module in pipe.unet.modules() if isinstance(module,torch.nn.Dropout))
    return parameters


def smoke_training():
    report=ROOT/'audit/ip_training_smoke.json'
    if report.exists():assert read(report)['passed'];return
    pipe=rt.load(selected_cn());rows=read(ROOT/'manifests/dev.json');row=rows[0]
    meta,capture=rt.generate(pipe,row,ROOT/'smoke/IPA/parity_pipeline.png',capture=True)
    sketch,_=rt.inputs(row)
    condition=pipe.control_image_processor.preprocess(sketch,height=512,width=384).to('cuda',torch.float16).repeat(2,1,1,1)
    with torch.inference_mode():
        direct=rt.forward(pipe,capture['noisy'],capture['t'],capture['text'],capture['reference'],condition)
    parity=float((direct-capture['prediction']).abs().max())
    assert parity<=1e-5,('TRAIN_INFERENCE_FORWARD_MISMATCH',parity)
    # 以保存的同一初始latent再次运行包装forward，验证终图逐像素一致。
    original=pipe.unet.forward
    def equivalent(sample,timestep,encoder_hidden_states,**kwargs):
        from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput
        pipe.unet.forward=original
        try:
            prediction=rt.forward(pipe,sample,timestep,encoder_hidden_states,
                kwargs['added_cond_kwargs']['image_embeds'][0],condition)
        finally:pipe.unet.forward=equivalent
        return UNet2DConditionOutput(sample=prediction) if kwargs.get('return_dict',True) else (prediction,)
    pipe.unet.forward=equivalent
    try:rt.generate(pipe,row,ROOT/'smoke/IPA/parity_replay.png')
    finally:pipe.unet.forward=original
    assert sha(ROOT/'smoke/IPA/parity_pipeline.png')==sha(ROOT/'smoke/IPA/parity_replay.png')
    exported=ROOT/'smoke/IPA/initial_export.bin';rt.save_adapter(pipe.unet,exported)
    frozen_before=rt.frozen_hashes(pipe)
    params=enable_training(pipe)
    before={name:rt.digest(p) for name,p in params.items()}
    text,ref,sketch=tensors(pipe,row);_,other_ref,other_sketch=tensors(pipe,rows[1])
    pipe.set_ip_adapter_scale(.65)
    generator=torch.Generator(device='cuda').manual_seed(42)
    latent=torch.randn((1,4,64,48),generator=generator,device='cuda',dtype=torch.float16)
    target=torch.randn((1,4,64,48),generator=generator,device='cuda',dtype=torch.float16)
    t=torch.tensor([500],device='cuda')
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
        a=rt.forward(pipe,latent,t,text,ref,sketch)
        b=rt.forward(pipe,latent,t,text,other_ref,sketch)
        c=rt.forward(pipe,latent,t,text,ref,other_sketch)
        off_a=rt.forward(pipe,latent,t,text,ref,sketch,0.)
        off_b=rt.forward(pipe,latent,t,text,ref,other_sketch,0.)
        zero_ref=rt.forward(pipe,latent,t,text,torch.zeros_like(ref),sketch)
    assert not torch.equal(a,b) and not torch.equal(a,c) and not torch.equal(a,zero_ref)
    assert torch.equal(off_a,off_b)
    optimizer=torch.optim.AdamW(list(params.values()),lr=5e-5,weight_decay=.01)
    with torch.autocast('cuda',dtype=torch.float16):
        prediction=rt.forward(pipe,latent,t,text,ref,sketch)
        loss=torch.nn.functional.mse_loss(prediction.float(),target.float())
    loss.backward()
    gradients={name:dict(finite=bool(p.grad is not None and torch.isfinite(p.grad).all()),
        norm=float(p.grad.float().norm()) if p.grad is not None else None) for name,p in params.items()}
    assert all(r['finite'] and r['norm']>0 for r in gradients.values()),gradients
    assert all(p.grad is None for name,p in pipe.unet.named_parameters() if name not in params)
    for module in [pipe.vae,pipe.controlnet,pipe.text_encoder,pipe.image_encoder]:
        assert all(p.grad is None for p in module.parameters())
    optimizer.step();optimizer.zero_grad(set_to_none=True)
    assert any(before[name]!=rt.digest(p) for name,p in params.items())
    frozen_after=rt.frozen_hashes(pipe);assert frozen_before==frozen_after
    rt.release(pipe);del optimizer,params;torch.cuda.empty_cache()
    # 原生官方格式的导出文件要真的重新加载，并验证图像一致。
    roundtrip=rt.load(selected_cn(),exported)
    rt.generate(roundtrip,row,ROOT/'smoke/IPA/initial_export_roundtrip.png')
    assert sha(ROOT/'smoke/IPA/parity_pipeline.png')==sha(ROOT/'smoke/IPA/initial_export_roundtrip.png')
    rt.release(roundtrip)
    write(report,dict(passed=True,forward_max_abs_delta=parity,final_image_exact_replay=True,
        official_export_roundtrip_exact=True,all_trainable_gradients=gradients,
        frozen_before=frozen_before,frozen_after=frozen_after,loss=float(loss.detach()),
        reference_forward_mae=float((a-b).abs().mean()),sketch_forward_mae=float((a-c).abs().mean()),
        zero_embedding_forward_mae=float((a-zero_ref).abs().mean()),CN_off_sketch_invariant=True,
        reference_input='独立texture；零化embedding与CFG负分支一致，投影偏置保留',code_commit=commit()))


class Stream:
    def __init__(self):
        self.rows=read(ROOT/'manifests/train.json');self.rng=random.Random(42)
        self.order=[];self.samples=0

    def next(self):
        if not self.order:self.order=list(range(len(self.rows)));self.rng.shuffle(self.order)
        i=self.order.pop();self.samples+=1
        value=self.rng.random()
        drop='image' if value<.05 else 'text' if value<.10 else 'both' if value<.15 else 'none'
        return i,self.rows[i],drop

    def state_dict(self):return dict(order=self.order,rng=self.rng.getstate(),samples=self.samples)
    def load_state_dict(self,state):self.order=state['order'];self.rng.setstate(state['rng']);self.samples=state['samples']


def cache_inputs(pipe):
    path=ROOT/'checkpoints/ip_fashion/input_cache.pt'
    identity=sha(ROOT/'manifests/train.json')
    if path.exists():
        cache=torch.load(path,map_location='cpu',weights_only=False)
        assert cache['manifest_sha256']==identity
        return cache
    cache=dict(manifest_sha256=identity,items=[])
    signatures=read(ROOT/'manifests/source_hashes.json')
    for i,row in enumerate(read(ROOT/'manifests/train.json')):
        for key in ['gt','sketch','reference']:assert sha(row[key])==signatures[row[key]]['file_sha256']
        target=Image.open(row['gt']).convert('RGB').resize(SIZE,Image.Resampling.BILINEAR)
        target=torch.from_numpy(np.asarray(target,np.float32).transpose(2,0,1)/127.5-1)[None].to('cuda',torch.float16)
        with torch.no_grad():
            posterior=pipe.vae.encode(target).latent_dist
            text,ref,_=tensors(pipe,row)
            cache['items'].append(dict(mean=posterior.mean.cpu(),std=posterior.std.cpu(),text=text.cpu(),ref=ref.cpu()))
        if i%100==0:print('CNIPA_CACHE',i,flush=True)
    path.parent.mkdir(parents=True,exist_ok=True);torch.save(cache,path)
    return cache


def train(steps=3000,checkpoint_every=500,output=None):
    require_audit();assert read(ROOT/'audit/ip_training_smoke.json')['passed']
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    folder=Path(output) if output else ROOT/'checkpoints/ip_fashion';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'complete.json').exists():assert read(folder/'complete.json')['updates']==steps;return
    pipe=rt.load(selected_cn());pipe.set_ip_adapter_scale(1.)
    frozen_before=rt.frozen_hashes(pipe);cache=cache_inputs(pipe);params=enable_training(pipe)
    initial={name:rt.digest(p) for name,p in params.items()}
    stream=Stream();optimizer=torch.optim.AdamW(list(params.values()),lr=5e-5,weight_decay=.01)
    scaler=torch.amp.GradScaler('cuda')
    schedule=DDPMScheduler.from_pretrained(str(SD),subfolder='scheduler',local_files_only=True)
    assert schedule.config.prediction_type=='epsilon'
    empty=pipe.tokenizer('',padding='max_length',max_length=pipe.tokenizer.model_max_length,return_tensors='pt').input_ids.to('cuda')
    with torch.no_grad():null=pipe.text_encoder(empty)[0]
    checkpoints=sorted(folder.glob('checkpoint-*/training_state.pt'),key=lambda p:int(p.parent.name.split('-')[-1]))
    step=0;skipped=0;previous_seconds=0.
    if checkpoints:
        saved=torch.load(checkpoints[-1],map_location='cpu',weights_only=False)
        assert saved['manifest_sha256']==cache['manifest_sha256'] and saved['cn_sha256']==sha(selected_cn()/'diffusion_pytorch_model.safetensors')
        for name,p in params.items():p.data.copy_(saved['parameters'][name].to(p.device))
        optimizer.load_state_dict(saved['optimizer']);scaler.load_state_dict(saved['scaler']);stream.load_state_dict(saved['stream'])
        random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng']);torch.set_rng_state(saved['torch_rng'])
        torch.cuda.set_rng_state_all(saved['cuda_rng']);step=saved['step'];skipped=saved['skipped'];previous_seconds=saved['seconds']
    started=time.perf_counter();sequence=[]
    while step<steps:
        optimizer.zero_grad(set_to_none=True);losses=[]
        for micro in range(4):
            index,row,drop=stream.next();item=cache['items'][index]
            mean,std,text,ref=[item[k].to('cuda') for k in ['mean','std','text','ref']]
            latent=(mean+std*torch.randn_like(mean))*pipe.vae.config.scaling_factor
            noise=torch.randn_like(latent);t=torch.randint(0,1000,(1,),device='cuda')
            noisy=schedule.add_noise(latent.float(),noise.float(),t).half()
            if drop in ['text','both']:text=null
            if drop in ['image','both']:ref=torch.zeros_like(ref)
            sketch=Image.open(row['sketch']).convert('RGB').resize(SIZE,Image.Resampling.BILINEAR)
            sketch=torch.from_numpy(np.asarray(sketch,np.float32).transpose(2,0,1)/255)[None].to('cuda',torch.float16)
            with torch.autocast('cuda',dtype=torch.float16):
                prediction=rt.forward(pipe,noisy,t,text,ref,sketch)
                loss=torch.nn.functional.mse_loss(prediction.float(),noise.float())
            assert torch.isfinite(loss)
            scaler.scale(loss/4).backward();losses.append(float(loss.detach()))
            sequence.append(dict(sample=stream.samples,id=row['id'],drop=drop,t=int(t),noise_sha256=rt.digest(noise)))
        scaler.unscale_(optimizer)
        grad_norm=float(torch.nn.utils.clip_grad_norm_(list(params.values()),1.))
        before_scale=scaler.get_scale();scaler.step(optimizer);scaler.update()
        if scaler.get_scale()<before_scale:skipped+=1;continue
        assert np.isfinite(grad_norm)
        step+=1;seconds=previous_seconds+time.perf_counter()-started
        if step<=5 or step%25==0:
            record=dict(step=step,loss=float(np.mean(losses)),grad_norm=grad_norm,seconds=seconds,
                skipped_updates=skipped,samples_consumed=stream.samples,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
            with (folder/'history.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(record)+'\n')
            print('CNIPA_TRAIN_IP',record,flush=True)
        if step%checkpoint_every==0 or step==steps:
            checkpoint=folder/f'checkpoint-{step}';checkpoint.mkdir(exist_ok=True)
            rt.save_adapter(pipe.unet,checkpoint/'ip-adapter.bin')
            torch.save(dict(step=step,parameters={n:p.detach().cpu() for n,p in params.items()},
                optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),stream=stream.state_dict(),
                python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all(),manifest_sha256=cache['manifest_sha256'],
                cn_sha256=sha(selected_cn()/'diffusion_pytorch_model.safetensors'),seconds=seconds,
                skipped=skipped,code_commit=commit()),checkpoint/'training_state.pt')
            with (folder/'sample_sequence.jsonl').open('a',encoding='utf-8') as f:
                for record in sequence:f.write(json.dumps(record)+'\n')
            sequence=[]
    frozen_after=rt.frozen_hashes(pipe);assert frozen_before==frozen_after
    assert all(p.grad is None for n,p in pipe.unet.named_parameters() if n not in params)
    changed=[name for name,p in params.items() if rt.digest(p)!=initial[name]]
    assert len(changed)==len(params)
    write(folder/'complete.json',dict(updates=step,samples_consumed=stream.samples,skipped_updates=skipped,
        effective_batch=4,seconds=seconds,trainable_parameters=sum(p.numel() for p in params.values()),
        changed_parameters=changed,frozen_before=frozen_before,frozen_after=frozen_after,
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
        gpu=torch.cuda.get_device_name(),code_commit=commit(),training_CN_scale=1.,training_IP_scale=1.,
        dropout=dict(image=.05,text=.05,both=.05)))
    rt.release(pipe)


def select():
    from cn_ipa.run import dev_run
    from cn_ipa.scoring import selection_key,csv_write
    path=ROOT/'selection/ip_checkpoint.json'
    if path.exists():return
    cn_step=read(ROOT/'selection/cn_checkpoint.json')['step']
    baseline=read(ROOT/'dev/cn_checkpoints'/str(cn_step)/'means.json');scores=[]
    for step in CONFIG['ipa_checkpoints']:
        adapter=ROOT/'checkpoints/ip_fashion'/f'checkpoint-{step}'/'ip-adapter.bin'
        pipe=rt.load(selected_cn(),adapter)
        metrics=dev_run(pipe,ROOT/'dev/ip_checkpoints'/str(step),f'IPA_{step}')
        key=selection_key(metrics,baseline,step)
        scores.append(dict(step=step,path=str(adapter),weights_sha256=sha(adapter),
            eligible=key[0]==0,violation=key[1],**metrics))
        rt.release(pipe)
    choice=min(scores,key=lambda r:(not r['eligible'],r['violation'],r['lpips_gt'],r['step']))
    write(ROOT/'selection/ip_checkpoint_scores.json',scores);csv_write(ROOT/'selection/ip_checkpoint_scores.csv',scores)
    write(path,choice)


if __name__=='__main__':train()
