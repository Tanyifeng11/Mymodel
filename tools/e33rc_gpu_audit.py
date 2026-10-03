"""真实缓存与完整controlled arms前后向审计；不执行optimizer.step。"""
import numpy as np
import torch
from torch.utils.data import DataLoader
from models.apacc_features import load_dino
from data.e33r_group_dataset import GroupDataset,encode_reference
from data.e33rc_real_pair_dataset import RealPairDataset
from tools.e33r_losses import group_loss
from tools.e33rc_losses import real_losses,retention_loss
from tools.e33rc_train_curriculum import forward_group
from tools.e33rc_common import *

def gpu_audit():
    if (OUT/'audits/gpu_integration.json').exists():
        assert read(OUT/'audits/gpu_integration.json')['pass'];return
    torch.manual_seed(33042);np.random.seed(33042)
    model,source_sha=load_control(42);teacher,_=load_control(42);teacher.eval().requires_grad_(False)
    dino,_=load_dino('cuda',WEIGHTS);model.train()
    before={k:v.cpu().clone() for k,v in model.prior.state_dict().items()}
    cf=next(iter(DataLoader(GroupDataset(read(OUT/'teacher_probes.json')[:2],True),batch_size=2,num_workers=0)))
    reference=encode_reference(cf['pixels'],cf['aux'],dino)
    reference=torch.cat([reference,torch.zeros_like(reference[:,:1])],1)
    pred=forward_group(model,reference,cf['structure'].cuda())
    controlled,_=group_loss(pred,cf['gt'].cuda(),cf['support'].cuda())
    with torch.no_grad():target=forward_group(teacher,reference[:,:3],cf['structure'].cuda())['orientation']
    retain=retention_loss(pred['orientation'][:,:3],target,cf['support'].float().cuda()*cf['gt'][:,3].cuda())
    real=next(iter(DataLoader(RealPairDataset(read(OUT/'split_manifest.json')['train'][:2],read(OUT/'real_wrong_train.json')),batch_size=2,num_workers=0)))
    realpred=forward_group(model,real['reference'].cuda(),real['structure'].cuda())
    reconstruct,rank=real_losses(realpred,real['gt'].cuda(),real['support'].cuda())
    loss=.75*controlled+.25*(reconstruct+.5*rank)+.5*retain;loss.backward()
    checks=dict(finite_loss=bool(torch.isfinite(loss)),
        finite_gradients=all(torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None),
        prior_no_gradient=all(p.grad is None for p in model.prior.parameters()),
        prior_unchanged=all(torch.equal(v.cpu(),before[k]) for k,v in model.prior.state_dict().items()),
        teacher_no_gradient=all(p.grad is None for p in teacher.parameters()),
        dino_no_gradient=all(p.grad is None for p in dino.parameters()),
        Q_gradient=float(model.q_head.weight.grad.norm())>0,
        projection_gradient=float(model.reference_projection[0].weight.grad.norm())>0,
        confidence_gradient=float(model.confidence_head.weight.grad.norm())>0,
        real_reference_dimension=real['reference'].shape[-1]==394,
        controlled_group_arms=reference.shape[1]==7,
        checkpoint_unchanged=sha(source_checkpoint(42))==source_sha)
    write(OUT/'audits/gpu_integration.json',dict(checks=checks,**{'pass':all(checks.values())},training_steps=0,
        losses=dict(controlled=float(controlled.detach()),real=float(reconstruct.detach()),rank=float(rank.detach()),retain=float(retain.detach()))))
    assert all(checks.values());print('[E33RC GPU integration]',checks,flush=True)
    del model,teacher,dino;torch.cuda.empty_cache()
