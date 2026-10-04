"""真实输入前后向审计，无optimizer.step，不计训练预算。"""
import torch
from data.e33rf_real_rotation_dataset import reference_group
from tools.e33rc_train_curriculum import forward_group
from tools.e33rf_common import *
from tools.e33rf_losses import losses

def audit(dino):
    output=OUT/'audits/gpu_integration.json'
    if output.exists():assert read(output)['pass'];return
    row=read(OUT/'split_manifest.json')['dev'][0];case=reference_group(row,'RF2')
    model,params=build(42);model.train()
    ref=case['reference'][None].cuda();structure=case['structure'][None].cuda()
    pred=forward_group(model,ref,structure)
    with torch.no_grad():
        base=forward_group(model.backbone,ref,structure)
    checks=dict(zero_init_exact=torch.equal(pred['orientation'],base['orientation']),
        reference_projection_frozen=not any(p.requires_grad for p in model.backbone.parameters()),
        backbone_eval=not model.backbone.training,small_adapter=params['ratio']<.05,finite_outputs=all(torch.isfinite(v).all().item() for v in pred.values()))
    loss,_=losses(pred,case['gt'][None].cuda(),case['support'][None].cuda(),'RF2');loss.backward()
    checks.update(adapter_gradient=float(model.adapter.net[-1].weight.grad.norm())>0,
        backbone_no_gradient=all(p.grad is None for p in model.backbone.parameters()),
        dino_frozen=not any(p.requires_grad or p.grad is not None for p in dino.parameters()),
        finite_gradients=all(torch.isfinite(p.grad).all().item() for p in model.adapter.parameters() if p.grad is not None))
    write(output,dict(checks=checks,parameters=params,training_steps=0,**{'pass':all(checks.values())}));assert all(checks.values())
    print('[E33RF GPU audit]',checks,flush=True)
