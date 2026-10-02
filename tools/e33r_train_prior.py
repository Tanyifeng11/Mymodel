"""E33-R P0，方向专用6000步；正式dev Gate后才允许sanity。"""
import math,random
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from data.e32_field_dataset import cache_path
from models.e33r_rotation_control import SketchPrior
from tools.e33r_common import *

class PriorDataset(Dataset):
    def __init__(self,records):self.records=records
    def __len__(self):return len(self.records)
    def __getitem__(self,index):
        row=self.records[index]
        with np.load(cache_path(E32,row['id'])) as f:
            values={k:torch.from_numpy(np.asarray(f[k],np.float32)) for k in ('structure','supervision_geometry','supervision_interior')}
        values['id']=row['id'];return values

def axial_map(a,b):return torch.rad2deg(torch.acos((a*b).sum(1).clamp(-1,1)))/2

def main():
    assert (OUT/'R0_integrity/summary.json').exists()
    chosen=read(OUT/'selected_manifest.json');split=read(OUT/'split_manifest.json');folder=OUT/'P0_prior';folder.mkdir(parents=True,exist_ok=True)
    assert not (folder/'checkpoint_final.pt').exists(),'禁止覆盖已完成prior'
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    assert torch.cuda.is_bf16_supported(),'按固定协议需要bf16 GPU'
    loader=DataLoader(PriorDataset(chosen['train']),batch_size=8,shuffle=True,num_workers=2,pin_memory=True,drop_last=True,
                      generator=torch.Generator().manual_seed(42))
    model=SketchPrior().cuda().train();optim=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    iterator=iter(loader);history=[]
    for step in range(1,6001):
        try:batch=next(iterator)
        except StopIteration:iterator=iter(loader);batch=next(iterator)
        structure=batch['structure'].cuda(non_blocking=True);gt=batch['supervision_geometry'].cuda(non_blocking=True)
        interior=batch['supervision_interior'].cuda(non_blocking=True)
        rate=min(step/400,1.) if step<=400 else .5*(1+math.cos(math.pi*(step-400)/5600))
        for group in optim.param_groups:group['lr']=1e-4*rate
        optim.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):pred=model(structure)['orientation']
        weight=(interior>=.95)*(gt[:,3]>=.25)*gt[:,3]
        loss=((1-(pred*gt[:,:2]).sum(1))*weight).sum()/weight.sum().clamp_min(1e-8)
        assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optim.step()
        if step%100==0:
            history.append(dict(step=step,loss=float(loss.detach())));write(folder/'history.json',history)
            print('[E33R P0]',history[-1],flush=True)
    torch.save(dict(model=model.state_dict(),steps=6000,seed=42,git_commit=git_commit()),folder/'checkpoint_final.pt')
    model.eval();rows=[];expected=0
    with torch.inference_mode():
        for row in split['dev']:
            case=PriorDataset([row])[0];gt=case['supervision_geometry'];interior=case['supervision_interior']
            valid=(interior>=.95)&(gt[3]>=.25);expected+=int(valid.any())
            with torch.autocast('cuda',dtype=torch.bfloat16):pred=model(case['structure'][None].cuda())['orientation'].cpu()
            finite=bool(torch.isfinite(pred).all());error=None
            if valid.any():error=float((axial_map(pred,gt[:2][None])[0][valid]*gt[3][valid]).sum()/gt[3][valid].sum())
            rows.append(dict(id=row['id'],finite=finite,orientation_error=error,readable_cells=int(valid.sum())))
    error=bootstrap([r['orientation_error'] for r in rows if r['orientation_error'] is not None])
    baseline=read(E32/'A_geometry/seed42/dev/summary.json')['arms']['matched']['orientation_error']['n']
    checks=dict(all_finite=all(r['finite'] for r in rows),readable_cases_unchanged=expected==baseline,
                orientation=error['mean'] is not None and error['mean']<=15)
    passed=all(checks.values());write(folder/'dev_rows.json',rows)
    write(folder/'summary.json',dict(orientation_error=error,readable_cases=expected,E32_readable_cases=baseline,
         checks=checks,**{'pass':passed},checkpoint_sha256=sha(folder/'checkpoint_final.pt')))
    write(folder/'training_protocol.json',dict(**PROTOCOL['p0'],device=torch.cuda.get_device_name(),
         loss='confidence-weighted orientation cosine only',reference_input=False,period_loss=False))
    update_decision(prior_pass=passed,prior_orientation_error_deg=error['mean'],
                    next_route='sanity' if passed else 'prior_accuracy_failed_no_control_training')
    frozen=read(OUT/'frozen_check.json');frozen['training_steps']=6000;write(OUT/'frozen_check.json',frozen)
    finish_frozen(OUT);print('[E33R P0 Gate]',checks,error,flush=True)

if __name__=='__main__':main()
