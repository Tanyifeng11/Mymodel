from types import SimpleNamespace
import numpy as np
import torch
from torch import nn

from models.bf_original_fusion import bFMFusion
from models.e2_texture_diagnostics import (VARIANTS, PATTERNS, reference_control, color_signature,
    spectrum_signature, FeatureTap, TrajectoryTap, feature_mix, grouped_ridge, group_accuracy, region_change)


def test_reference_controls_preserve_joint_rgb_histogram_and_independent_rng():
    a = np.random.default_rng(4).integers(0, 256, (64, 96, 3), dtype=np.uint8)
    def sorted_pixels(x):
        p=x.reshape(-1,3);return p[np.lexsort(p.T)]
    state=torch.get_rng_state().clone()
    for name in VARIANTS:
        b=reference_control(a,name,5)
        assert np.array_equal(sorted_pixels(a),sorted_pixels(b))
        assert np.array_equal(color_signature(a),color_signature(b))
    assert torch.equal(state,torch.get_rng_state())


def test_control_period_direction_positive_control_and_shift_invariance():
    a=np.arange(64*96*3,dtype=np.uint8).reshape(64,96,3)
    for i,name in enumerate(PATTERNS):
        original=spectrum_signature(reference_control(a,name,42))
        shifted=spectrum_signature(reference_control(a,name+'_shift',42))
        assert original.argmax()==i
        assert np.allclose(original,shifted,rtol=0,atol=1e-7)


def test_feature_tap_is_observational_and_reaches_returned_fused_maps():
    torch.set_num_threads(2);torch.manual_seed(9)
    class Adapter(nn.Module):
        def __init__(self):
            super().__init__();self.fusion=bFMFusion(1,3,ngf=4,bottleneck_depth=2,num_resblocks=0)
            self.projections=nn.ModuleList([nn.Conv2d(c,c,1) for c in [8,16,32,32]])
        def forward(self,s,r):return [p(x) for p,x in zip(self.projections,self.fusion(s,r))]
    m=Adapter().eval();s=torch.randn(1,1,128,128);r=torch.randn(1,3,128,128)
    before=m(s,r);state=torch.get_rng_state().clone()
    with FeatureTap(m) as tap: after=m(s,r)
    assert all(torch.equal(a,b) for a,b in zip(before,after)) and torch.equal(state,torch.get_rng_state())
    assert {'ref2_pure','ref3_mixed','sketch2_before_mod','film5_alpha','fused5','projection3'} <= set(tap.values)
    assert all(np.isfinite(v).all() for x in tap.values.values() for v in x.values())
    assert all(not module._forward_hooks for module in m.modules())


def test_trajectory_tap_preserves_sampling_and_restores_scheduler_method():
    adapter=SimpleNamespace(injection_counts=[0]*4)
    class Unet(nn.Module):
        def forward(self,x,t,**kw):
            on='sa_hidden_states' in (kw.get('cross_attention_kwargs') or {})
            adapter.injection_counts=[3,2,2,3] if on else [0]*4
            return (x*.1+(1 if on else 0),)
    class Scheduler:
        def step(self,p,t,s,**kw):return (s-.01*p,)
    pipe=SimpleNamespace(unet=Unet(),scheduler=Scheduler())
    def run():
        x=torch.ones(1,4,4,4)
        for i in range(50):
            t=torch.tensor(999-i*20);a=pipe.unet(x,t,cross_attention_kwargs={'sa_hidden_states':{}},return_dict=False)[0]
            b=pipe.unet(x,t,return_dict=False)[0];x=pipe.scheduler.step(b+7*(a-b),t,x,return_dict=False)[0]
        return x
    before=run();rng=torch.get_rng_state().clone()
    with TrajectoryTap(pipe,adapter) as tap:after=run()
    assert torch.equal(before,after) and torch.equal(rng,torch.get_rng_state())
    assert set(tap.states)=={0,10,25,40,49} and tap.scheduler_calls==50 and tap.condition_calls==50
    assert all(x['conditional']['counts']==[3,2,2,3] and x['unconditional']['counts']==[0]*4 for x in tap.states.values())
    assert 'step' not in pipe.scheduler.__dict__ and not pipe.unet._forward_hooks


def test_feature_mix_exact_endpoints_and_no_mutation():
    a=[torch.randn(1,3,4,4).half()];b=[torch.randn_like(a[0])];old=a[0].clone();new=b[0].clone()
    assert feature_mix(a,b,0) is a and feature_mix(a,b,1) is b
    assert torch.equal(feature_mix(a,b,2)[0],(old.float()+2*(new.float()-old.float())).half())
    assert torch.equal(a[0],old) and torch.equal(b[0],new)


def test_grouped_readout_cannot_use_identity_but_can_read_signal():
    groups=np.repeat(np.arange(12),4);labels=np.tile(np.arange(4),12)
    identity=np.eye(12)[groups];fold={i:i%4 for i in range(12)}
    a,b=grouped_ridge(identity,identity,labels,groups,fold,4)
    assert (a==labels).mean()==.25 and (b==labels).mean()==.25
    signal=np.c_[identity,np.eye(4)[labels]*2]
    a,b=grouped_ridge(signal,signal,labels,groups,fold,4)
    result=group_accuracy(a,labels,groups)
    assert result['accuracy']==1 and result['groups']==12 and result['samples']==48
    assert np.array_equal(a,b) and result['descriptive_ci95']==[1,1]


def test_region_change_uses_soft_weight_once_and_empty_region_is_null():
    a=torch.full((1,2,2,2),3.);b=torch.full_like(a,2.)
    r=region_change(a,b,{'soft':torch.full((1,1,4,4),.25),'empty':torch.zeros(1,1,4,4)})
    assert r['soft']['rms']==1 and r['soft']['relative_rms']==.5
    assert r['empty']['rms'] is None


def test_probe_pairs_shift_by_identity_reference_and_filters_input_only():
    from tools.e2_texture_bottleneck_eval import probe_arrays
    rows=[]
    for case in ['a','b']:
        for condition in ['old','new']:
            for variant in list(PATTERNS)+[x+'_shift' for x in PATTERNS]:
                rows.append(dict(case_id=case,identity_group=case,condition=condition,variant=variant,
                    readout_eligible=not(case=='b' and condition=='new')))
    array=np.arange(len(rows))[:,None]
    chosen,shifted,labels,groups,values=probe_arrays(rows,{'signal':array})
    assert len(chosen)==12 and len(set(groups))==2 and len(labels)==12
    for a,b in zip(chosen,shifted):
        assert rows[a]['case_id']==rows[b]['case_id'] and rows[a]['condition']==rows[b]['condition']
        assert rows[b]['variant']==rows[a]['variant']+'_shift'
    assert np.array_equal(values['signal'][0],array[chosen])


def test_diagnostic_response_pairs_and_effective_denominator():
    from tools.e2_texture_bottleneck_eval import response_rows,descriptive
    records=[]
    for condition,value in [('old',.2),('new',.5)]:
        records.append(dict(group='baseline',case_id='a',identity_group='a',condition=condition,
            scores={mode:dict(strict_margin=value,tolerant_margin=value*2) for mode in ['gray','rgb']}))
    row=response_rows(records)[0]
    assert np.isclose(row['gray_strict_response_change'],.3) and np.isclose(row['rgb_tolerant_response_change'],.6)
    assert descriptive([None,float('nan'),1,3])['n']==2
    assert descriptive([])==dict(n=0,mean=None,descriptive_ci95=None)
