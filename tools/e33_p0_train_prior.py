"""E33 P0：通过integrity后训练固定sketch prior，共8000步。"""
import math
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from data.e32_field_dataset import FieldDataset
from models.e33_counterfactual_field import SketchPrior
from tools.e32_a_geometry_train import geometry_loss
from tools.e32_common import read,write,sha,finish_frozen,git_commit
from tools.e33_protocol import OUT,E32,PROTOCOL


def main():
    assert read(OUT/'decision_summary.json')['transform_integrity_pass'], 'integrity Gate未通过，禁止训练'
    split=read(OUT/'split_manifest.json');folder=OUT/'P0_prior';cfg=PROTOCOL['p0']
    assert not (folder/'checkpoint_final.pt').exists(), 'P0已完成，禁止覆盖'
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    dataset=FieldDataset(E32,split['train'])
    loader=DataLoader(dataset,batch_size=8,shuffle=True,num_workers=2,pin_memory=True,drop_last=True,
                      generator=torch.Generator().manual_seed(42))
    model=SketchPrior().cuda().train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler=torch.cuda.amp.GradScaler(enabled=dtype==torch.float16)
    history=[];iterator=iter(loader)
    for step in range(1,8001):
        try:batch=next(iterator)
        except StopIteration:iterator=iter(loader);batch=next(iterator)
        batch={k:v.cuda(non_blocking=True) for k,v in batch.items() if torch.is_tensor(v)}
        rate=min(step/500,1.) if step<=500 else .5*(1+math.cos(math.pi*(step-500)/7500))
        for group in optimizer.param_groups:group['lr']=1e-4*rate
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=dtype):pred=model(batch['structure'])
        # E32 loss仅读取前两个confidence头，且BCE固定在float32。
        loss,parts=geometry_loss(pred,batch,torch.zeros(len(batch['structure']),device='cuda',dtype=torch.bool))
        assert torch.isfinite(loss)
        scaler.scale(loss).backward();scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.);scaler.step(optimizer);scaler.update()
        if step%100==0:
            history.append(dict(step=step,loss=float(loss.detach()),**parts))
            write(folder/'training_history.json',history);print('[E33 P0]',history[-1],flush=True)
    torch.save(dict(model=model.state_dict(),steps=8000,seed=42,git_commit=git_commit()),folder/'checkpoint_final.pt')
    write(folder/'training_protocol.json',dict(**cfg,checkpoint_sha256=sha(folder/'checkpoint_final.pt'),
        input='sketch-only7 channels',prior_frozen_in_following_phases=True,mixed_precision=str(dtype),loss='E32 fixed geometry loss, no reference dropout'))
    write(folder/'status.json',dict(status='complete',steps=8000));finish_frozen(OUT)


if __name__=='__main__':main()
