"""固定4身份/160更新，原G2 last8链与三个最终RGB损失；不选最佳checkpoint。"""
import argparse,time,hashlib,json
import cv2,numpy as np,torch
from torch import nn
from torch.nn import functional as F
from PIL import Image
from torchvision.transforms.functional import to_tensor
from tools.e33gc_g2b_protocol import *
from data.e33gc_g2b_renderer import load,input_dir
from tools.e33gc_smoke import fields,png
from tools.e33gc_sampling import prefix,final8,tensor_sha
from models.e33gc_adapter import CausalResidual,TextureInjection
from models.e33tm_generation_wrapper import load_e5
from models.apacc_features import load_dino
from tools.e33rf_common import build,WEIGHTS
from tools.e22_4_generation import module_hashes
from tools.e33tm_weight_audit import effective_hashes
from tools.e33tmoc_appearance_eval import model_hash

def tensors(value):
    return {k:torch.from_numpy(value[k].astype(np.float32))[None,None].cuda() for k in ['mask','inner','falloff']}

def orientation(rgb,scale):
    gray=(rgb*rgb.new_tensor([.299,.587,.114])[None,:,None,None]).sum(1,keepdim=True)
    k=gray.new_tensor([[-1,0,1],[-2,0,2],[-1,0,1]])[None,None]/8
    gx=F.conv2d(gray,k,padding=1);gy=F.conv2d(gray,k.transpose(-1,-2),padding=1)
    raw=-torch.cat([F.avg_pool2d(gx.square()-gy.square(),scale),F.avg_pool2d(2*gx*gy,scale)],1)
    return F.normalize(raw,dim=1,eps=1e-8),F.avg_pool2d(gx.square()+gy.square(),scale)

def losses(images,targets,baseline,masks):
    inner=masks['inner'];protect=1-masks['falloff']
    def charbonnier(a,b,m):return (((a-b).square()+1e-6).sqrt()*m).sum()/m.expand_as(a).sum().clamp_min(1)
    rgb=torch.stack([charbonnier(a,b,inner) for a,b in zip(images,targets)]).mean()
    outside=torch.stack([charbonnier(a,b,protect) for a,b in zip(images,baseline)]).mean()
    pairs=[]
    for scale in [8,16]:
        ori=[orientation(a,scale)[0] for a in images]
        # 可靠性只由冻结训练目标提供，属于loss标签；输出不能改变支持分母。
        energy=orientation(targets[0],scale)[1].detach()
        weight=F.interpolate(inner,energy.shape[-2:],mode='area')*(energy/(energy+1e-4))
        r90=(1+(ori[0]*ori[1]).sum(1,keepdim=True))*weight
        r180=(1-(ori[0]*ori[2]).sum(1,keepdim=True))*weight
        pairs.append((r90.sum()+.5*r180.sum())/weight.sum().clamp_min(1))
    pair=torch.stack(pairs).mean()
    total=rgb+.5*pair+.25*outside
    return total,dict(rgb=rgb,orientation_pair=pair,outside=outside,total=total)

def gradient_stats(adapter):
    values={n:dict(present=p.grad is not None,norm=float(p.grad.norm()) if p.grad is not None else None,
        finite=bool(torch.isfinite(p.grad).all()) if p.grad is not None else None) for n,p in adapter.named_parameters()}
    assert all(v['finite'] for v in values.values() if v['present'])
    assert sum(v['norm'] or 0 for v in values.values())>0
    return values

class TracedInjection(TextureInjection):
    def __init__(self,pipe,adapter):
        super().__init__(pipe,adapter);self.tracing=False;self.residual_trace=[]
    def transform(self,texture):
        result=super().transform(texture)
        if self.tracing:
            difference=(result-texture).detach().float();m=F.interpolate(self.falloff,difference.new_zeros(1,1,64,48).shape[-2:],mode='area').flatten(2).transpose(1,2)
            self.residual_trace.append(dict(conditional=self.conditional,enabled=self.enabled,index=self.index,
                residual_rms=float(difference.square().mean().sqrt()),
                outer_residual_rms=float((difference.square()*(1-m)).sum().div(((1-m).sum()*320).clamp_min(1)).sqrt()),
                shape=list(texture.shape)))
        return result

def cache_case(row,pipe,ns,adapter,injection,rf,dino):
    inputs=load(row,False);m=tensors(inputs);geom=fields(inputs,rf,dino).detach();sid=row['id']
    folder=OUT/'G2b_smoke/cache'/sid;folder.mkdir(parents=True,exist_ok=True)
    states=[];baseline=[];proof=[]
    geom_np=geom.cpu().numpy();np.savez_compressed(folder/'RF2_fields.npz',geometry=geom_np)
    for i,arm in enumerate(ARMS):
        ref_path=input_dir(row)/(arm+'_reference.png')
        key=dict(id=sid,arm=arm,reference=sha(ref_path),sketch=sha(input_dir(row)/'sketch.png'),
            caption=hashlib.sha256(row['caption'].encode()).hexdigest(),E5=sha(E5),RF2=sha(RF/'seed42/RF2/checkpoint_final.pt'),
            noise_seed=42,dtype=str(pipe.unet.dtype),cfg=7.,steps=50,scheduler=dict(pipe.scheduler.config))
        keyhash=hashlib.sha256(json.dumps(key,sort_keys=True).encode()).hexdigest()
        dest=folder/(arm+'_prefix.pt');injection.geometry=geom[i:i+1];injection.falloff=m['falloff']
        if dest.exists():
            cached=torch.load(dest,map_location='cpu');assert cached['cache_key_hash']==keyhash
            image=Image.open(folder/(arm+'_A0.png')).convert('RGB')
        else:
            cached,image=prefix(pipe,ns,row['caption'],inputs['sketch'],inputs['references'][i],Image.fromarray(inputs['mask']*255),injection)
            cached.update(cache_key=key,cache_key_hash=keyhash);torch.save(cached,dest);image.save(folder/(arm+'_A0.png'))
        with torch.no_grad():
            injection.tracing=True;injection.residual_trace=[]
            rgb,trace=final8(pipe,ns,row['caption'],inputs['sketch'],inputs['references'][i],m['mask'],cached,injection,False,True)
            injection.tracing=False
        exact=np.array_equal(np.asarray(image),np.asarray(png(rgb)));assert exact
        png(rgb).save(folder/(arm+'_step0.png'))
        assert cached['timesteps'].tolist()[-8:]==[141,121,101,81,61,41,21,1]
        proof.append(dict(arm=arm,exact=exact,baseline_sha256=sha(folder/(arm+'_A0.png')),
            step0_sha256=sha(folder/(arm+'_step0.png')),cache_key=key,cache_key_hash=keyhash,
            residual_trace=injection.residual_trace,CFG_trace=trace,noise_sha256=cached['generation']['noise_sha256']))
        states.append(cached);baseline.append(rgb.detach())
    assert len({p['noise_sha256'] for p in proof})==1
    write(folder/'proof.json',dict(id=sid,arms=proof,RF_shape=list(geom.shape),attention_sequence=[3072,320],
        RF_axis_interpolation='cos2theta/sin2theta then normalize',inference_target_read=False))
    targets=[to_tensor(a)[None].cuda() for a in load(row,True)['targets']]
    return dict(row=row,inputs=inputs,masks=m,geom=geom,states=states,baseline=baseline,targets=targets)

def render_case(case,pipe,ns,injection,checkpointed=True,constant=False):
    images=[]
    for i in range(3):
        injection.geometry=case['geom'][0:1] if constant else case['geom'][i:i+1]
        injection.falloff=case['masks']['falloff']
        if hasattr(injection.adapter,'arm'):injection.adapter.arm=i
        rgb,_=final8(pipe,ns,case['row']['caption'],case['inputs']['sketch'],case['inputs']['references'][i],
            case['masks']['mask'],case['states'][i],injection,checkpointed)
        images.append(rgb)
    return images

def diagnostic(cases,pipe,ns,injection,step,label='G2b_train',constant=False):
    folder=OUT/label/('constant_RF' if constant else 'step%d'%step);folder.mkdir(parents=True,exist_ok=True)
    records=[]
    with torch.no_grad():
        for case in cases:
            images=render_case(case,pipe,ns,injection,False,constant)
            total,parts=losses(images,case['targets'],case['baseline'],case['masks'])
            dest=folder/case['row']['id'];dest.mkdir(parents=True,exist_ok=True)
            for arm,rgb in zip(ARMS,images):png(rgb).save(dest/(arm+'.png'))
            records.append(dict(id=case['row']['id'],label=case['row']['label'],loss={k:float(v) for k,v in parts.items()}))
    write(folder/'losses.json',records)

def run(action):
    init();torch.manual_seed(42);torch.set_num_threads(2);cv2.setNumThreads(1)
    contract=read(OUT/'protocol/input_target_contract.json');assert contract['approved_for_AI_amended_training']
    assert all(sha(p)==h for p,h in contract['files'].items()),'frozen input/target files changed'
    rows=read(OUT/'protocol/g2b_fit_probe_ids.json')['fit'];assert len(rows)==4
    pipe,modules,size,ns=load_e5();assert size==(384,512)
    rf,_=build(42,checkpoint=RF/'seed42/RF2/checkpoint_final.pt');rf.eval().requires_grad_(False)
    dino,_=load_dino('cuda',WEIGHTS);dino.eval().requires_grad_(False)
    before=module_hashes(modules);effective=effective_hashes(modules);rf_hash=model_hash(rf);dino_hash=model_hash(dino)
    adapter=CausalResidual().cuda();assert sum(p.numel() for p in adapter.parameters())==196064
    injection=TracedInjection(pipe,adapter);torch.cuda.reset_peak_memory_stats();began=time.monotonic()
    cases=[cache_case(row,pipe,ns,adapter,injection,rf,dino) for row in rows]
    if action=='smoke':
        diagnostic(cases,pipe,ns,injection,0)
        opt=torch.optim.AdamW(adapter.parameters(),lr=1e-4,weight_decay=1e-4);opt.zero_grad(set_to_none=True)
        step_start=time.monotonic();values=[]
        for case in cases:
            images=render_case(case,pipe,ns,injection);total,parts=losses(images,case['targets'],case['baseline'],case['masks'])
            assert all(torch.isfinite(v) for v in parts.values());(total/4).backward()
            values.append({k:float(v.detach()) for k,v in parts.items()})
        first=gradient_stats(adapter);torch.nn.utils.clip_grad_norm_(adapter.parameters(),1.);opt.step()
        seconds=time.monotonic()-step_start
        opt.zero_grad(set_to_none=True);images=render_case(cases[0],pipe,ns,injection)
        total,parts=losses(images,cases[0]['targets'],cases[0]['baseline'],cases[0]['masks']);total.backward()
        second=gradient_stats(adapter);assert second['body.0.weight']['norm']>0
        assert all(p.grad is None for m in list(modules.values())+[rf,dino] for p in m.parameters())
        # 将三loss各自反传到末层，新增outside项在非零更新后验证。
        components={}
        for name in ['rgb','orientation_pair','outside']:
            adapter.zero_grad(set_to_none=True);im=render_case(cases[0],pipe,ns,injection)
            _,pieces=losses(im,cases[0]['targets'],cases[0]['baseline'],cases[0]['masks']);pieces[name].backward()
            components[name]=dict(value=float(pieces[name].detach()),gradients=gradient_stats(adapter))
        projected=(seconds*160+seconds*.5*80+seconds*10+time.monotonic()-began)/3600
        peak=torch.cuda.max_memory_allocated()/2**30
        audit=dict(pass_autograd=True,full4_identity_update_seconds=seconds,peak_memory_gib=peak,first_gradients=first,
            after_update_gradients=second,losses=values,component_gradients=components,projected_GPU_hours=projected,
            cap_GPU_hours=6.,budget_pass=projected<=6,exact_arms=12,formal_updates=0,
            cache_hits=len(cases)*3,checkpointed=True,code_commit=commit(),freeze_before=before)
        write(OUT/'G2b_smoke/smoke_audit.json',audit);assert projected<=6,'budget exceeds pre-registered6GPUh'
        decision(g2b_step0_exact_match=True,g2b_loss_autograd_pass=True,g2b_peak_memory_gib=peak,next_route='fixed160_training')
    else:
        smoke=read(OUT/'G2b_smoke/smoke_audit.json');assert smoke['budget_pass'] and smoke['pass_autograd']
        opt=torch.optim.AdamW(adapter.parameters(),lr=1e-4,weight_decay=1e-4)
        log=OUT/'G2b_train/updates.jsonl';log.parent.mkdir(parents=True,exist_ok=True)
        assert not (OUT/'G2b_train/checkpoint_step160.pt').exists(),'endpoint already exists; do not retrain'
        for step in range(1,161):
            opt.zero_grad(set_to_none=True);t=time.monotonic();values=[]
            for case in cases:
                images=render_case(case,pipe,ns,injection);total,parts=losses(images,case['targets'],case['baseline'],case['masks'])
                assert all(torch.isfinite(v) for v in parts.values());(total/4).backward()
                values.append(dict(id=case['row']['id'],**{k:float(v.detach()) for k,v in parts.items()}))
            norm=torch.nn.utils.clip_grad_norm_(adapter.parameters(),1.);assert torch.isfinite(norm);opt.step()
            record=dict(update=step,identities=values,grad_norm_before_clip=float(norm),peak_memory_gib=torch.cuda.max_memory_allocated()/2**30,
                seconds=time.monotonic()-t,elapsed_seconds=time.monotonic()-began)
            with log.open('a',encoding='utf-8') as f:f.write(json.dumps(record)+'\n')
            print('UPDATE',step,'loss',sum(v['total'] for v in values)/4,'seconds',round(record['seconds'],2),flush=True)
            assert (time.monotonic()-began)/3600<6,'GPU budget exhausted; do not extend'
            if step in [40,80,160]:
                torch.save(dict(adapter=adapter.state_dict(),step=step,config=CONFIG),OUT/'G2b_train'/('checkpoint_step%d.pt'%step))
                diagnostic(cases,pipe,ns,injection,step)
        diagnostic(cases,pipe,ns,injection,160,constant=True)
        probe_rows=read(OUT/'protocol/g2b_fit_probe_ids.json')['probe']
        if len(probe_rows)==4:
            trained=injection.adapter;injection.adapter=CausalResidual().cuda()
            probes=[cache_case(row,pipe,ns,injection.adapter,injection,rf,dino) for row in probe_rows]
            injection.adapter=trained
            diagnostic(probes,pipe,ns,injection,160,label='G2b_probe')
        write(OUT/'G2b_train/training_complete.json',dict(updates=160,elapsed_seconds=time.monotonic()-began,
            checkpoint_sha256=sha(OUT/'G2b_train/checkpoint_step160.pt'),code_commit=commit(),endpoint_only=True))
        decision(g2b_training_complete=True,next_route='endpoint_evaluation_and_two_AI_blind_reviews')
    assert before==module_hashes(modules) and effective==effective_hashes(modules)
    assert rf_hash==model_hash(rf) and dino_hash==model_hash(dino)
    write(OUT/('G2b_smoke' if action=='smoke' else 'G2b_train')/'frozen_modules.json',dict(before=before,after=module_hashes(modules),
        RF2=rf_hash,DINO=dino_hash,effective_before=effective,effective_after=effective_hashes(modules)))
    injection.close();verify_frozen();bundle(action)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['smoke','train']);a=p.parse_args()
    try:run(a.action)
    except Exception:
        import traceback,os
        write(OUT/'failures'/('job_'+os.environ.get('SLURM_JOB_ID','local')+'.json'),dict(traceback=traceback.format_exc(),action=a.action,commit=commit()))
        raise
