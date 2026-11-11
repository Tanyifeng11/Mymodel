import copy

import pytest
import torch

from models.bf_original_fusion import bFMFusion, affine_transformation
from models.m1_patch_retrieval import M1Fusion, StructureToTexturePatchAttention, attach_m1


@pytest.mark.parametrize('c,h,w,stride', [(288,64,48,2),(576,32,24,2),(1152,16,12,1),(1152,8,6,1)])
def test_shapes_identity_and_second_backward(c,h,w,stride):
    torch.set_num_threads(2)
    module=StructureToTexturePatchAttention(c,patch_stride=stride)
    s,t=torch.randn(1,c,h,w),torch.randn(1,c,h,w)
    optimizer=torch.optim.AdamW(module.parameters(),lr=5e-5)
    assert torch.equal(module(s,t),s)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss=module(s,t).square().mean();loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())
        optimizer.step()
    for name in ['q_proj','patch_embed','kv_proj','out_proj']:
        assert getattr(module,name).weight.grad.norm()>0
    state=copy.deepcopy(module.state_dict());opt=copy.deepcopy(optimizer.state_dict())
    restored=StructureToTexturePatchAttention(c,patch_stride=stride)
    restored.load_state_dict(state,strict=True)
    other=torch.optim.AdamW(restored.parameters(),lr=5e-5);other.load_state_dict(opt)
    assert torch.equal(module(s,t),restored(s,t))
    for a,b in zip(optimizer.state.values(),other.state.values()):
        assert all(torch.equal(a[k],b[k]) for k in a)


def test_reverse_film_uses_pre_fusion_structure_and_shared_keys_strict():
    torch.set_num_threads(2)
    old=bFMFusion(1,3,ngf=4,bottleneck_depth=3,num_resblocks=0)
    model=M1Fusion(ngf=4,bottleneck_depth=3)
    incompatible=model.load_state_dict(old.state_dict(),strict=False)
    assert set(incompatible.missing_keys)=={k for k in model.state_dict() if k.startswith('s2t_patch_attn')}
    before={};after={};handles=[]
    for i in range(2,6):
        handles.append(getattr(model,'s2t_patch_attn%d'%i).register_forward_pre_hook(
            lambda m,args,i=i: before.update({i:args[0].detach().clone()})))
        handles.append(getattr(model,'bottleneck_alpha_%d'%i).register_forward_pre_hook(
            lambda m,args,i=i: after.update({i:args[0].detach().clone()})))
        torch.nn.init.normal_(getattr(model,'s2t_patch_attn%d'%i).out_proj.weight,std=.1)
    s,t=torch.randn(1,1,128,96),torch.randn(1,3,128,96)
    result=model(s,t)
    assert all(torch.equal(before[i],after[i]) for i in range(2,6))
    assert [list(v.shape) for v in result]==[[1,8,16,12],[1,16,8,6],[1,32,4,3],[1,32,2,1]]
    for handle in handles:handle.remove()
    missing=copy.deepcopy(old.state_dict());missing.pop(next(iter(missing)))
    bad=model.load_state_dict(missing,strict=False)
    assert any(not key.startswith('s2t_patch_attn') for key in bad.missing_keys)


def test_m0_freezing_unused_parameters_is_bitwise_invariant():
    model=torch.nn.Module();model.adapter=torch.nn.Module()
    model.adapter.fusion=bFMFusion(1,3,ngf=4,bottleneck_depth=3,num_resblocks=0)
    s,t=torch.randn(1,1,128,96),torch.randn(1,3,128,96)
    before=[v.detach().clone() for v in model.adapter.fusion(s,t)]
    attach_m1(model,enabled=False)
    after=model.adapter.fusion(s,t)
    assert all(torch.equal(a,b) for a,b in zip(before,after))
    assert not any(p.requires_grad for p in model.adapter.fusion.bottleneck_alpha_5.parameters())


def test_spectral_control_identical_stripes_and_constant_failures():
    import numpy as np
    from tools.m1_evaluation import spectral_metrics
    x=np.indices((512,384))[1]
    im=np.repeat(np.where(x%12<6,220,30)[...,None],3,2).astype('uint8')
    coordinates=[(32,32),(96,96),(160,160),(224,224)]
    value=spectral_metrics(im,im,coordinates,'stripes')
    assert value['fft_peak_distance']==0
    assert value['autocorrelation_period_log2_error']==0
    assert value['orientation_error_degrees']==0
    value=spectral_metrics(np.full_like(im,127),np.full_like(im,127),coordinates,'stripes')
    assert value['fft_peak_distance'] is None
    assert value['autocorrelation_period_log2_error'] is None
    assert value['orientation_error_degrees'] is None
