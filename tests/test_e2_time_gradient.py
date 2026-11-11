from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from tools.e2_time_gradient_common import (FullTrace, predicted_x0, gradient_pair,
    schedule_record, descriptive, same_patch_objective)


def test_x0_difference_is_known_schedule_scaled_epsilon_difference():
    torch.manual_seed(7);z=torch.randn(1,4,8,6);a=torch.randn_like(z);b=torch.randn_like(z)
    for alpha in [.0001,.2,.99]:
        delta=predicted_x0(z,b,alpha)-predicted_x0(z,a,alpha)
        expected=-np.sqrt((1-alpha)/alpha)*(b-a)
        assert torch.allclose(delta,expected,atol=3e-5,rtol=1e-5)


def test_gradient_conflict_and_zero_are_distinct_and_weight_is_not_squared():
    a=torch.ones(1,3,2,2);b=-2*a
    result=gradient_pair(a,b,torch.full((1,1,4,4),.25))
    assert result['cosine']==-1 and np.isclose(result['first_norm'],np.sqrt(3))
    assert result['dot']==-6 and result['first_along_second']<0
    assert gradient_pair(a,a*0)['cosine'] is None
    assert gradient_pair(a,None)['second_connected']==False


def test_gradient_cosine_matches_finite_directional_difference():
    x=torch.tensor([.4,-.3],requires_grad=True)
    first=((x-torch.tensor([1.,0.]))**2).sum();second=((x-torch.tensor([-1.,0.]))**2).sum()
    a=torch.autograd.grad(first,x,retain_graph=True)[0];b=torch.autograd.grad(second,x)[0]
    h=1e-3;direction=-b/b.norm()
    fn=lambda y:((y-torch.tensor([1.,0.]))**2).sum()
    numerical=float((fn(x.detach()+h*direction)-fn(x.detach()-h*direction))/(2*h))
    assert np.isclose(numerical,float((a*direction).sum()),atol=1e-4)
    assert gradient_pair(a,b)['cosine']<0


def test_full_trace_does_not_change_sampling_or_leave_hooks():
    adapter=SimpleNamespace(injection_counts=[0]*4)
    class Unet(nn.Module):
        def forward(self,x,t,**kwargs):
            cond='sa_hidden_states' in (kwargs.get('cross_attention_kwargs') or {})
            adapter.injection_counts=[3,2,2,3] if cond else [0]*4
            return (x*.1+int(cond),)
    class Scheduler:
        def step(self,p,t,z,**kwargs):return (z-p*.01,)
    pipe=SimpleNamespace(unet=Unet(),scheduler=Scheduler())
    def sample():
        x=torch.ones(1,4,3,3)
        for i in range(50):
            t=torch.tensor(981-20*i)
            c=pipe.unet(x,t,cross_attention_kwargs={'sa_hidden_states':{}})[0]
            u=pipe.unet(x,t)[0];x=pipe.scheduler.step(u+7*(c-u),t,x,eta=0.,return_dict=False)[0]
        return x
    original=sample()
    with FullTrace(pipe,adapter) as tap:observed=sample()
    assert torch.equal(original,observed) and len(tap.states)==50
    assert torch.equal(tap.states[49]['next_sample'],original)
    assert all(torch.equal(tap.states[i]['next_sample'],tap.states[i+1]['sample']) for i in range(49))
    assert 'step' not in pipe.scheduler.__dict__ and not pipe.unet._forward_hooks


def test_schedule_final_previous_alpha_matches_legacy_set_alpha_false():
    s=SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000,clip_sample=False),num_inference_steps=50,
        alphas_cumprod=torch.linspace(.999,.01,1000),final_alpha_cumprod=torch.tensor(.999))
    r=schedule_record(s,1)
    assert r['previous_t']==-19 and r['previous_alpha']==float(s.final_alpha_cumprod)
    assert r['epsilon_to_x0']>0 and r['epsilon_to_next']<0


def test_same_gt_local_descriptor_control_has_prediction_gradient_only():
    x=torch.randn(1,3,8,8,requires_grad=True);target=torch.randn_like(x,requires_grad=True)
    descriptor=lambda a:a.mean((2,3))
    value=same_patch_objective(x,target,{'4':[(0,0),(4,4)]},descriptor)
    a,b=torch.autograd.grad(value,[x,target],allow_unused=True)
    assert a is not None and a.norm()>0 and b is None
    assert descriptive([1,None,3])['n']==2


def test_frozen_patch_stream_separates_legacy_gate_from_counterfactual():
    from models.e2_local_fft import LocalFFTLoss
    from tools.e2_time_gradient_gradients import FixedCoordinates
    torch.manual_seed(3);x=torch.rand(1,3,96,96,requires_grad=True)*2-1;ref=torch.rand_like(x)*2-1
    mask=torch.ones(1,1,96,96);mask[:,:,:8]=0;mask[:,:,-8:]=0
    fixed={'32':dict(body=[(16,16)],reference=[(0,0)],body_candidates=2000)}
    old=LocalFFTLoss(sizes=(32,),count=1,t_max=249);new=LocalFFTLoss(sizes=(32,),count=1,t_max=999)
    old.sampler=FixedCoordinates(fixed);new.sampler=FixedCoordinates(fixed)
    a,sa=old(x,ref,mask,torch.tensor([100]),torch.tensor([4]))
    b,sb=new(x,ref,mask,torch.tensor([100]),torch.tensor([4]))
    assert torch.equal(a,b) and sa['valid_n']==sb['valid_n']==1
    a,sa=old(x,ref,mask,torch.tensor([250]),torch.tensor([4]))
    b,sb=new(x,ref,mask,torch.tensor([250]),torch.tensor([4]))
    assert a==0 and sa['active_n']==0 and b>0 and sb['valid_n']==1
    zero,stats=new(x,ref,mask,torch.tensor([100]),torch.tensor([4]),reference_keep=False)
    assert zero==0 and stats['active_n']==0


def test_gradient_targets_resolve_actual_fusion_convolution_parameters():
    from models.bf_original_fusion import bFMFusion
    from tools.e2_time_gradient_gradients import targets_for
    network=nn.Module();network.adapter=nn.Module()
    network.adapter.fusion=bFMFusion(1,3,ngf=2,bottleneck_depth=2,num_resblocks=0)
    network.adapter.projections=nn.ModuleList([nn.Conv2d(1,1,1) for _ in range(4)])
    network.unet=nn.Linear(1,1)
    value=torch.ones(1,1,2,2,requires_grad=True)
    names,targets,metadata=targets_for(network,dict(epsilon=value,decoded=value,features=[value]*4))
    assert metadata['texture_conv2']['parameter_name']=='adapter.fusion.G_downconv2.1.weight'
    assert targets[names.index('texture_conv2')] is network.adapter.fusion.G_downconv2[1].weight
    assert all(t.requires_grad for t in targets)
