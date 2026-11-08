from types import SimpleNamespace

import pytest
import torch
from torch import nn

from models.e2_local_fft import LocalFFTLoss
from models.e2_local_texture import GaborDescriptor, ReferenceGramDescriptor, LocalTextureObjective


@pytest.fixture(autouse=True)
def threads():
    old=torch.get_num_threads();torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def image(size=96,period=8,theta=0.):
    y,x=torch.meshgrid(torch.arange(size),torch.arange(size),indexing='ij')
    value=.4*torch.sin(2*torch.pi/period*(x*torch.cos(torch.tensor(theta))+y*torch.sin(torch.tensor(theta))))
    return value[None,None].expand(1,3,size,size).clone()


def inputs():
    y=image(period=8).requires_grad_();r=image(period=4)
    mask=torch.zeros(1,1,96,96);mask[:,:,8:-8,8:-8]=1
    return y,r,mask,torch.tensor([100]),torch.tensor([1])


@pytest.mark.parametrize('size',[32,64])
def test_gabor_period_orientation_and_finite_gradient(size):
    d=GaborDescriptor();x=(image(size)+1)/2;x.requires_grad_()
    a=d(x);different=d((image(size,period=4)+1)/2);rotated=d((image(size,theta=torch.pi/2)+1)/2)
    assert a.shape==(1,32 if size==32 else 48)
    assert (a-different).abs().mean()>.01 and (a-rotated).abs().mean()>.01
    (a-different).abs().mean().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum()>0
    response=a[0,:8];assert int(response.argmax())==0


@pytest.mark.parametrize('level',[0.,.5,1.])
def test_gabor_constant_and_kernel_energy(level):
    d=GaborDescriptor();x=torch.full((1,3,32,32),level,requires_grad=True)
    d(x).sum().backward();assert torch.isfinite(x.grad).all()
    k=d.kernel(4,x.device).flatten(1)
    assert torch.allclose(k.sum(1),torch.zeros(16),atol=2e-6)
    assert torch.allclose(k.square().sum(1).reshape(8,2).sum(1),torch.ones(8),atol=1e-6)


def test_fft_only_objective_keeps_original_value_rng_and_sampling():
    args=inputs();legacy=LocalFFTLoss(seed=5);new=LocalTextureObjective({'fft':.7},seed=5)
    rng=torch.get_rng_state().clone()
    old,stats=legacy(*args);value,other=new(*args)
    assert torch.equal(value,.7*old)
    assert stats['samples']==other['samples']
    assert torch.equal(torch.get_rng_state(),rng)
    assert torch.equal(legacy.sampling_state()['rng'],new.sampling_state()['fft']['rng'])


def test_joint_shares_patch_stream_and_restores_state():
    args=inputs();d=LocalTextureObjective({'fft':.4,'gabor':.03},seed=8)
    state=d.sampling_state();a,stats=d(*args)
    assert set(stats['raw_components'])=={'fft','gabor'}
    assert torch.allclose(a,.4*stats['_component_tensors']['aux_fft']+.03*stats['_component_tensors']['aux_gabor'])
    a.backward();assert torch.isfinite(args[0].grad).all() and args[0].grad.abs().sum()>0
    d.load_sampling_state(state);b,other=d(*args)
    assert torch.equal(a,b) and stats['samples']==other['samples']
    assert torch.equal(d.sampling_state()['fft']['rng'],d.sampling_state()['gabor']['rng'])


@pytest.mark.parametrize('reason',['high_t','reference_dropout','empty_mask','solid_reference'])
def test_invalid_auxiliary_keeps_connected_finite_zero(reason):
    args=list(inputs());kwargs={}
    if reason=='high_t':args[3].fill_(900)
    if reason=='reference_dropout':kwargs['reference_keep']=False
    if reason=='empty_mask':args[2].zero_()
    if reason=='solid_reference':args[1].zero_()
    loss,stats=LocalTextureObjective({'gabor':1.})(*args,**kwargs)
    assert float(loss.detach())==0 and stats['valid_n']==0
    loss.backward();assert torch.isfinite(args[0].grad).all()


def test_gram_matches_spatial_statistics_without_pair_alignment():
    style=SimpleNamespace(l3=nn.Identity(),l4=nn.AvgPool2d(2))
    d=ReferenceGramDescriptor(style)
    x=torch.rand(2,3,32,32,requires_grad=True)
    feature=d(x);permuted=d(x.flip((-2,-1)))
    assert torch.allclose(feature,permuted,atol=1e-6)
    feature.square().mean().backward();assert x.grad.abs().sum()>0


def test_gram_layers_have_equal_weight_with_different_channel_counts():
    layer=nn.Conv2d(3,5,1).requires_grad_(False)
    style=SimpleNamespace(l3=nn.Identity(),l4=layer)
    d=ReferenceGramDescriptor(style)
    x=torch.rand(2,3,32,32);y=torch.rand_like(x)
    def grams(image,module):
        mean=image.new_tensor([.485,.456,.406])[None,:,None,None]
        std=image.new_tensor([.229,.224,.225])[None,:,None,None]
        f=module((image-mean)/std).flatten(2)
        return (f@f.transpose(1,2)/(f.shape[1]*f.shape[2]+1e-6)).mean(0)
    expected=sum((grams(x,m)-grams(y,m)).abs().mean() for m in [style.l3,style.l4])/2
    actual=(d(x).mean(0)-d(y).mean(0)).abs().mean()
    assert torch.allclose(actual,expected,atol=1e-7)
