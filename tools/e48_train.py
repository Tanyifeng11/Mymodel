"""仅在诊断证据冻结后运行最多两次800更新；同数据随机序列、同冻结E5教师。"""
import argparse
import time
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image
from tools.e48_protocol import *
from tools.e34_sarr_protocol import seed_all
from tools.e44_train import TrainingInputs,gradients
from models.e48_reference_attention import configure

TRAIN_ARMS=['DECOUPLED','BOUNDARY']


def freeze():
    assert read(OUT/'probe/evaluation_complete.json')['complete']
    assert read(OUT/'audit/complete.json')['passed']
    value=dict(experiment='e48',arms=TRAIN_ARMS,training=CONFIG['training'],
        source_diagnosis_sha256=sha(OUT/'mechanism_summary.json'),
        parent_B1_sha256=B1_SHA,base_sha256=E5_SHA,
        initialization='fresh B1 seed42; DECOUPLED same parameters but separate reference softmax',
        boundary='Sketch external contour dilation/erosion 19x19; distill same-state E5 epsilon',
        train_split_sha256=sha(OUT/'splits/train1024.json'),
        decisions='fixed step800; no best checkpoint selection; two runs maximum')
    path=OUT/'training/protocol.json'
    if path.exists():assert read(path)==value
    else:write(path,value)


def make_runtime(arm):
    from tools import e47_runtime as module
    previous=module.OUT;module.OUT=OUT
    try:r=module.runtime('B',True,'training/'+arm+'/runtime')
    finally:module.OUT=previous
    configure(r,arm)
    return r


def inputs(r):
    from tools import e44_train as module
    previous=module.OUT;module.OUT=OUT
    try:data=TrainingInputs(r,'train1024')
    finally:module.OUT=previous
    return data


def forward_loss(r,args,kw,noise,mask,arm,boundary):
    r.active=False
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):teacher=r.pipe.unet(*args,**kw).sample
    r.active=True
    with torch.autocast('cuda',dtype=torch.float16):pred=r.pipe.unet(*args,**kw).sample
    diff=F.mse_loss(pred.float(),noise.float())
    delta=(pred.float()-teacher.float()).square()
    outside=1-mask.float();keep=(delta*outside).sum()/outside.sum().clamp_min(1e-8)
    edge=(delta*boundary).sum()/boundary.sum().clamp_min(1e-8)
    total=diff+.1*keep+(.1*edge if arm=='BOUNDARY' else 0.)
    return total,dict(diffusion=float(diff.detach()),keep=float(keep.detach()),boundary=float(edge.detach()))


def train(arm):
    from tools.e47_runtime import close
    from tools.e44_runtime import sketch_masks
    from tools.e44_b1_mask import region_masks,weights_at
    from tools.e22_4_generation import module_hashes
    freeze();folder=OUT/'training'/arm
    assert not (folder/'complete.json').exists(),'禁止重复训练选择结果'
    seed_all(42);r=make_runtime(arm);data=inputs(r);seed_all(42)
    assert sum(p.numel() for p in r.parameters())==820481
    # 只从 Sketch 构造权重，不使用 GT 分割；缓存不消耗训练 RNG。
    boundaries={}
    for row in read(OUT/'splits/train1024.json'):
        raw,_,info=sketch_masks(Image.open(row['sketch']).convert('RGB').resize(SIZE,Image.Resampling.BILINEAR),'cpu')
        maps=region_masks(raw)
        boundaries[row['id']]=weights_at(maps,(64,48),r.pipe.device)['boundary'] if info['valid'] else torch.zeros(1,1,64,48,device=r.pipe.device)
    optimizer=torch.optim.AdamW(r.parameters(),lr=1e-4,weight_decay=.01)
    scaler=torch.amp.GradScaler('cuda');sequence=[];history=[];step=0;empty=0
    started=time.perf_counter();before=r.before
    r.save(folder/'checkpoints/step0000.pt',0,optimizer,scaler)
    try:
        while step<800:
            optimizer.zero_grad(set_to_none=True);items=[]
            for _ in range(8):
                args,kw,noise,mask,record=data.next();sequence.append(record)
                loss,item=forward_loss(r,args,kw,noise,mask,arm,boundaries[record['id']])
                assert torch.isfinite(loss)
                if loss.requires_grad:scaler.scale(loss/8).backward()
                items.append(item)
            if not any(p.grad is not None for p in r.parameters()):empty+=1;continue
            scaler.unscale_(optimizer);norm=float(torch.nn.utils.clip_grad_norm_(r.parameters(),1.))
            assert np.isfinite(norm)
            grad=gradients(r) if step+1 in [1,5,20,800] else None
            if step==0:assert grad['gate'] is not None and grad['gate']>0
            if step+1 in [5,20]:assert grad['to_k_ref.weight']>0 and grad['to_v_ref.weight']>0
            old=scaler.get_scale();scaler.step(optimizer);scaler.update();assert scaler.get_scale()>=old
            step+=1
            if step in [1,5,20] or step%25==0:
                item=dict(step=step,grad_norm=norm,gradients=grad,seconds=time.perf_counter()-started,
                    **{k:sum(x[k] for x in items)/8 for k in items[0]})
                history.append(item);write(folder/'history.json',history)
                print('E48 TRAIN',arm,step,item['diffusion'],item['boundary'],flush=True)
            if step in [100,400,800]:r.save(folder/'checkpoints'/('step%04d.pt'%step),step,optimizer,scaler)
        write(folder/'random_sequence.json',sequence)
        expected=read('output_eval/e44_mimicbrush_drra_20261012/p1/B1/complete.json')['sequence_sha256']
        assert sha(folder/'random_sequence.json')==expected
        # 新 processor 是 U-Net 子模块；比较冻结基线前须暂时移除可训练分支。
        target=dict(r.pipe.unet.named_modules())[LAYER[:-10]]
        target.set_processor(r.original)
        try:assert module_hashes(r.modules)==before
        finally:target.set_processor(r.processor)
        write(folder/'complete.json',dict(arm=arm,updates=step,empty_windows=empty,parameters=820481,
            sequence_sha256=expected,teacher='frozen E5 with new branch OFF',seconds=time.perf_counter()-started,
            peak_memory_gib=torch.cuda.max_memory_allocated()/2**30,frozen_base_unchanged=True,commit=commit()))
    finally:close(r)


def generate(arm,stage):
    from tools.e48_run import run_case,release
    from tools.e48_protocol import image_path
    r=make_runtime(arm);folder=OUT/'training'/arm
    assert read(folder/'complete.json')['updates']==800
    r.load(folder/'checkpoints/step0800.pt');r.processor.eval().requires_grad_(False)
    rows=read(OUT/'splits'/('probe.json' if stage=='probe' else 'dev32.json'))
    try:
        for row in rows:run_case(r,row,arm,stage,compare=False)
    finally:release(r)
    write(OUT/stage/(arm+'_complete.json'),dict(arm=arm,n=len(rows),checkpoint_sha256=sha(folder/'checkpoints/step0800.pt'),commit=commit()))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','train','generate','evaluate'])
    parser.add_argument('--arm',choices=TRAIN_ARMS);parser.add_argument('--stage',default='probe',choices=['probe','dev32'])
    args=parser.parse_args();torch.set_num_threads(2)
    if args.action=='freeze':freeze()
    elif args.action=='train':train(args.arm)
    elif args.action=='generate':generate(args.arm,args.stage)
    else:
        from tools.e48_eval import evaluate
        extra=COMPONENT_ARMS if args.stage=='probe' else []
        evaluate(args.stage,arms=ARMS+extra+TRAIN_ARMS)


if __name__=='__main__':main()
