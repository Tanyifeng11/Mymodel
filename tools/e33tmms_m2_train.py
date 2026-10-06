"""按最后固定endpoint训练M2；Stage-A失败即停止，不训练Stage-B。"""
import argparse,math,random,time
import cv2,numpy as np,torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from PIL import Image
from data.e33tmms_learned_data import CounterfactualRGBDataset
from data.e33r_group_dataset import encode_reference
from models.apacc_features import load_dino
from models.e33tmms_m2 import AppearanceGeometryDecoder
from tools.e33rf_common import build,OUT as RF,WEIGHTS
from tools.e33tmms_protocol import *
from tools.e33tmoc_appearance_eval import model_hash,patches,measure
from tools.e33tmoc_geometry_audit import extract
from tools.e33tmif_metrics import pair,arm_geometry,summarize,bootstrap
from tools.e33tmms_learned_losses import LearnedLoss
from data.e32_target_pseudogt import upsample_geometry

def stream(loader):
    while True:yield from loader

def inputs_audit():
    prepare();assert read(OUT/'decision_summary.json')['M1_carrier_pass'] is False
    cf=read(RF/'controlled_manifest.json');real=read(RF/'split_manifest.json')
    held=set(sum([read(OUT/'splits'/(s+'.json')) for s in ['diagnostic64','confirmation64','remaining126']],[]))
    checks={domain:set(r['id'] for r in data['train']).isdisjoint(held) for domain,data in [('controlled',cf),('real',real)]}
    assert all(checks.values()) and len(cf['train'])==25272 and len(cf['dev'])==128 and len(real['train'])==45126
    value=dict(train_counts=dict(controlled=len(cf['train']),real=len(real['train'])),heldout_count=len(held),
        train_heldout_disjoint=checks,files={str(p):sha(p) for p in [RF/'controlled_manifest.json',RF/'split_manifest.json']},
        supervision='controlled original self-reference bbox crop rotations; real R0 reconstruction only; target/GT never passed to encoder/decoder')
    write(OUT/'protocol/M2_training_identity_audit.json',value)
    write(OUT/'protocol/M2_implementation.json',dict(PROTOCOL['M2'],seed=42,
        data_source_sha256=sha('data/e33tmms_learned_data.py'),
        geometry_q=.25,RGB_size=[384,512],gradient_clip=1.,microbatch_identities=1,
        effective_batch_identities=8,checkpoint_selection='last scheduled step only',
        controlled_metric='all original available dev128, original M_cf support; unreadable failure',
        loss_proxy='Sobel structure tensor averaged8; soft8-bin differentiable D65Lab histogram; global/local16 color statistics; frozen E5 CNN stage1..3 mean/std embedding',
        lpips='verified official AlexNet plus LPIPS v0.1; masked support resized256x192, differentiable'))
    return cf,real

class FrozenField:
    def __init__(self):
        self.model,_=build(42,checkpoint=RF/'seed42/RF2/checkpoint_final.pt')
        self.model.eval().requires_grad_(False);self.dino,_=load_dino('cuda',WEIGHTS)
        self.dino.eval().requires_grad_(False)
        self.before={'RF2':model_hash(self.model),'DINO':model_hash(self.dino)}
    @torch.no_grad()
    def __call__(self,case,domain):
        sid=case['id'][0];path=OUT/'frozen_fields'/domain/(sid+'.npz')
        if path.exists():
            with np.load(path) as z:return torch.from_numpy(z['orientation']).cuda(),torch.from_numpy(z['confidence']).cuda()
        reference=encode_reference(case['pixels'],case['aux'],self.dino) if bool(case['controlled'][0]) else case['reference_features'].cuda()
        structure=case['structure'].cuda().expand(3,-1,-1,-1)
        with torch.autocast('cuda',dtype=torch.bfloat16):pred=self.model(reference.flatten(0,1),structure)
        ori=pred['orientation'].float();conf=pred['confidence_logits'].float().sigmoid()
        path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,orientation=ori.cpu().numpy(),confidence=conf.cpu().numpy())
        return ori,conf
    def verify(self):
        after={'RF2':model_hash(self.model),'DINO':model_hash(self.dino)}
        assert self.before==after and all(p.grad is None for m in [self.model,self.dino] for p in m.parameters())
        return dict(before=self.before,after=after,**{'pass':True})

def geometry(case,field,domain):
    ori,q=field(case,domain);size=case['mask'].shape[-2:]
    dense=F.interpolate(torch.cat([ori,q],1),size,mode='bilinear',align_corners=False)
    dense[:,:2]=F.normalize(dense[:,:2],dim=1)
    return torch.cat([dense,case['mask'].cuda().expand(3,-1,-1,-1)],1)

def fixed_appearance(model,references):
    code,tokens=model.appearance(references)
    fixed=(code[:1].expand(3,-1),tokens[:1].expand(3,-1,-1))
    return (code,tokens),fixed

def loader(rows,controlled,seed,shuffle):
    return DataLoader(CounterfactualRGBDataset(rows,controlled),batch_size=1,shuffle=shuffle,
        num_workers=2,multiprocessing_context='spawn',pin_memory=True,generator=torch.Generator().manual_seed(seed))

@torch.no_grad()
def controlled_eval(model,field,criterion,rows):
    dest=OUT/'M2/controlled/dev128';direction=[];app=[];model.eval()
    for case in loader(rows,True,32042,False):
        sid=case['id'][0];folder=dest/sid;folder.mkdir(parents=True,exist_ok=True)
        geom=geometry(case,field,'controlled_dev');refs=case['references'][0].cuda()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            _,appearance=fixed_appearance(model,refs);output=model.decode(appearance,geom).float().cpu().numpy()
        support=case['support'][0].numpy();gt=case['gt'][0].numpy();geos={};arms={};apps=[]
        inner=cv2.resize(support.astype(np.uint8),(384,512),interpolation=cv2.INTER_NEAREST)>0
        for i,arm in enumerate(['R0','R90','R180']):
            raw=np.uint8(np.clip(np.round(output[i].transpose(1,2,0)*255),0,255));raw[np.asarray(case['mask'][0,0])==0]=255
            image=Image.fromarray(raw);image.save(folder/(arm+'.png'))
            reference=Image.fromarray(np.uint8(np.round(case['references'][0,i].numpy().transpose(1,2,0)*255)))
            source_mask=case['source_masks'][0,i,0].numpy()>0
            tb=patches(reference,inner,sid,'shared','target');sb=patches(reference,source_mask,sid,arm,'source')
            a=measure(image,reference,inner,source_mask,tb,sb,criterion.lpips.cpu());criterion.lpips.cuda()
            apps.append(a);write(folder/(arm+'_appearance.json'),a)
            g=upsample_geometry(extract(image)).transpose(2,0,1);geos[arm]=g;arms[arm]=arm_geometry(g,support,gt[3])
            np.savez_compressed(folder/(arm+'_orientation.npz'),geometry=g,support_mask=support&(g[3]>=.25))
        row=dict(id=sid,stage='S2',arms=arms,**pair(geos,support,gt[3]));direction.append(row)
        values=[a['texture_score'] for a in apps if a['texture_score'] is not None]
        app.append(dict(id=sid,texture_score=float(np.mean(values)) if values else None))
        write(folder/'case.json',dict(id=sid,direction=row,appearance=app[-1]));print('[MS M2 controlled dev]',sid,flush=True)
    summary=dict(direction=summarize(direction),appearance=dict(texture_score=bootstrap([r['texture_score'] for r in app])))
    write(dest/'summary.json',summary)
    stats=summary['direction']['statistics'];tex=summary['appearance']['texture_score']['mean']
    checks=dict(R90=stats['r90_success']['mean']>=.70,R180=stats['r180_success']['mean']>=.80,Texture=tex is not None and tex>=.18)
    gate=dict(checks=checks,**{'pass':all(checks.values())},case_count=128)
    write(OUT/'M2/controlled/gate.json',gate)
    decision(M2_run=True,M2_controlled_pass=gate['pass'],M2_controlled_representation_fail=not gate['pass'],
        next_route='M2_stage_B' if gate['pass'] else 'M3')
    print('[MS M2 StageA Gate]',gate,summary,flush=True)

def train(phase):
    cf,real=inputs_audit();dest=OUT/'M2'/('controlled' if phase=='A' else 'real_mixed');dest.mkdir(parents=True,exist_ok=True)
    if phase=='B':assert read(OUT/'decision_summary.json')['M2_controlled_pass']
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    torch.set_num_threads(2);model=AppearanceGeometryDecoder().cuda();field=FrozenField();criterion=LearnedLoss()
    if phase=='B':model.load_state_dict(torch.load(OUT/'M2/controlled/checkpoint_final.pt',map_location='cpu')['model'])
    model.train();steps=4000 if phase=='A' else 2000;optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    control=stream(loader(cf['train'],True,42,True));actual=stream(loader(real['train'],False,1042,True)) if phase=='B' else None
    history=[];start=time.monotonic()
    resume=dest/'checkpoint_resume.pt'
    first=1
    if resume.exists():
        state=torch.load(resume,map_location='cpu');model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        first=state['step']+1;history=read(dest/'history.json')
        # 仅从已保存同一epoch游标继续，不从dev重新挑checkpoint。
        for _ in range(state['controlled_seen']):next(control)
        if actual:
            for _ in range(state['real_seen']):next(actual)
        random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng']);torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
    # 恢复实际消耗量，包含为保证每批身份互异而跳过的样本。
    control_seen=state['controlled_seen'] if first>1 else 0
    real_seen=state['real_seen'] if first>1 else 0
    write(dest/'training_protocol.json',dict(phase=phase,steps=steps,seed=42,trainable=sum(p.numel() for p in model.parameters()),
        effective_batch=8,microbatch=1,lr=1e-4,weight_decay=1e-4,warmup=200,optimizer='fresh phase AdamW; cosine',
        target_groundtruth='supervision only',appearance='A(R0) held fixed across geometry arms; A(R90/R180) invariance loss',git_commit=commit()))
    decision(M2_run=True,next_route='M2_stage_'+phase+'_running')
    for step in range(first,steps+1):
        factor=step/200 if step<=200 else .5*(1+math.cos(math.pi*(step-200)/(steps-200)))
        for group in optimizer.param_groups:group['lr']=1e-4*factor
        optimizer.zero_grad(set_to_none=True);metrics={};seen=set()
        for j in range(8):
            controlled=phase=='A' or j<4
            case=next(control if controlled else actual);sid=case['id'][0]
            if controlled:control_seen+=1
            else:real_seen+=1
            while sid in seen:
                case=next(control if controlled else actual);sid=case['id'][0]
                if controlled:control_seen+=1
                else:real_seen+=1
            seen.add(sid)
            geom=geometry(case,field,'controlled_train' if controlled else 'real_train');refs=case['references'][0].cuda()
            with torch.autocast('cuda',dtype=torch.bfloat16):
                appearance,fixed=fixed_appearance(model,refs);output=model.decode(fixed,geom).float()
            loss,parts=criterion(output,refs,case['targets'][0].cuda(),geom,case['support'][0].cuda(),
                case['rec_arms'][0].cuda(),appearance,case['source_masks'][0].cuda())
            (loss/8).backward()
            for k,v in dict(total=float(loss.detach()),**parts).items():metrics[k]=metrics.get(k,0)+v/8
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        if step==first:write(dest/'gradient_check.json',dict(distinct_identities=len(seen),finite=True,
            encoder_gradient=float(model.encoder.cnn[0].weight.grad.norm()),frozen_gradients_absent=all(p.grad is None for m in [field.model,field.dino,criterion.lpips,criterion.texture] for p in m.parameters())))
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        if step%25==0 or step==first:
            history.append(dict(step=step,elapsed_seconds=time.monotonic()-start,lr=1e-4*factor,**metrics));write(dest/'history.json',history)
            print('[MS M2 train]',phase,history[-1],flush=True)
        if step%250==0:
            temporary=resume.with_suffix('.tmp');torch.save(dict(step=step,model=model.state_dict(),optimizer=optimizer.state_dict(),
                controlled_seen=control_seen,real_seen=real_seen,python_rng=random.getstate(),numpy_rng=np.random.get_state(),
                torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),temporary);temporary.replace(resume)
    checkpoint=dest/'checkpoint_final.pt';torch.save(dict(model=model.state_dict(),phase=phase,steps=steps,seed=42,git_commit=commit()),checkpoint)
    write(dest/'training_complete.json',dict(steps=steps,checkpoint_sha256=sha(checkpoint),**{'complete':True}))
    if phase=='A':controlled_eval(model,field,criterion,cf['dev'])
    else:decision(M2_run=True,next_route='M2_real_diagnostic')
    write(dest/'frozen_modules.json',dict(field=field.verify(),metrics=criterion.verify()))
    frozen_check()
    from tools.e33tmms_benchmark import bundle
    bundle('M2_stage_'+phase)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['A','B']);train(parser.parse_args().phase)
