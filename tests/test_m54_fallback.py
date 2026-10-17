"""后备机制CPU契约；不替代真实E5/GPU门禁。"""
import copy
import unittest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from models.m54_fallback import CatBranch,WarpBranch,concat_inputs,warp


class TinyAttention(nn.Module):
    def __init__(self):
        super().__init__();self.to_q=nn.Linear(8,8,bias=False);self.to_k=nn.Linear(8,8,bias=False)
        self.to_v=nn.Linear(8,8,bias=False);self.to_out=nn.ModuleList([nn.Linear(8,8)])
    def forward(self,x):
        return self.to_out[0](F.scaled_dot_product_attention(self.to_q(x)[None],self.to_k(x)[None],self.to_v(x)[None])[0])


class TinyUNet(nn.Module):
    def __init__(self):
        super().__init__();self.conv_in=nn.Conv2d(4,8,3,1,1);self.block=nn.Module()
        self.block.attn1=TinyAttention();self.conv_out=nn.Conv2d(8,4,3,1,1);self.checkpointing=False
    def enable_gradient_checkpointing(self):self.checkpointing=True
    def forward(self,sample,timestep,**kwargs):
        hidden=self.conv_in(sample);shape=hidden.shape
        def attention(x):return self.block.attn1(x.flatten(2).transpose(1,2)).transpose(1,2).reshape(shape)
        hidden=checkpoint(attention,hidden,use_reentrant=False) if self.training and self.checkpointing and torch.is_grad_enabled() else attention(hidden)
        return (self.conv_out(hidden),)


class FallbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)
    def test_cat_nine_channels_and_target_reference_spatial_halves(self):
        x=torch.ones(1,4,8,6);m=torch.ones(1,1,8,6)
        joined=concat_inputs(x,2*x,3*x,4*x,m)
        self.assertEqual(tuple(joined.shape),(1,9,16,6))
        self.assertTrue(torch.equal(joined[:,:4,:8],x));self.assertTrue(torch.equal(joined[:,:4,8:],2*x))
        self.assertEqual(float(joined[:,4,8:].abs().sum()),0)

    def test_cat_zero_identity_live_checkpointed_gradients_and_frozen_backbone(self):
        original=TinyUNet();source=copy.deepcopy(original.state_dict());candidate=CatBranch(copy.deepcopy(original))
        x=torch.randn(1,4,8,6);m=torch.ones(1,1,8,6)
        delta=candidate(x,torch.randn_like(x),torch.randn_like(x),torch.randn_like(x),m,torch.tensor([42]),None,{'sa_hidden_states':{'fixture':x}})
        self.assertTrue(torch.equal(delta,torch.zeros_like(delta)))
        (delta-x).square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm()>0 for p in candidate.parameters()))
        torch.optim.AdamW(candidate.parameters(),lr=1e-4).step();candidate.verify_frozen()
        self.assertTrue(all(torch.equal(original.state_dict()[k],v) for k,v in source.items()))
        self.assertFalse(any(k.startswith('backbone.') or k.startswith('initial_') for k in candidate.state_dict()))

    def test_flow_identity_grid_and_zero_residual_full_latent_resolution(self):
        model=WarpBranch();x=torch.randn(1,4,64,48);mask=torch.ones(1,1,64,48)
        self.assertTrue(torch.allclose(warp(x,torch.zeros(1,2,64,48)),x,atol=2e-5))
        out=model(x,torch.randn_like(x),torch.randn_like(x),mask,torch.tensor([500]))
        self.assertTrue(torch.equal(out,torch.zeros_like(out)));self.assertEqual(float(model.smoothness.detach()),0.)
        out.square().sum().backward();self.assertIsNotNone(model.output.weight.grad)

    def test_flow_and_residual_interior_have_live_gradients_after_zero_heads_open(self):
        model=WarpBranch()
        with torch.no_grad():
            model.output.weight.normal_(std=.01)
            for head in model.flow.heads:head[-1].weight.normal_(std=.01)
        x=torch.randn(1,4,32,24);out=model(x,torch.randn_like(x),torch.randn_like(x),torch.ones(1,1,32,24),torch.tensor([42]))
        ((out-x).square().mean()+.01*model.smoothness).backward()
        missing=[n for n,p in model.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all() or p.grad.norm()==0]
        self.assertEqual(missing,[])

    def test_cat_compact_state_reload_preserves_trained_residual(self):
        source=TinyUNet();candidate=CatBranch(copy.deepcopy(source))
        with torch.no_grad():candidate.trainable[0].add_(.01)
        restored=CatBranch(copy.deepcopy(source));restored.load_state_dict(candidate.state_dict())
        x=torch.randn(1,4,8,6);args=(x,torch.randn_like(x),torch.randn_like(x),torch.randn_like(x),torch.ones(1,1,8,6),42,None,{})
        candidate.eval();restored.eval()
        with torch.no_grad():self.assertTrue(torch.equal(candidate(*args),restored(*args)))
        restored.verify_frozen()


if __name__=='__main__':unittest.main()
