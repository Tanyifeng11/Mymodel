"""科研有效性门禁：防止零初始化恒等失败或双零导致永久死梯度。"""
import unittest
import torch
from torch import nn
from models.m54_candidates import Candidate,M54Processor


class Original(nn.Module):
    def forward(self,attn,x,**kwargs):return x


class Attention(nn.Module):
    def __init__(self):
        super().__init__();self.to_q=nn.Linear(16,16,bias=False);self.to_k=nn.Linear(16,16,bias=False)
        self.to_v=nn.Linear(16,16,bias=False);self.to_out=nn.ModuleList([nn.Linear(16,16,bias=False),nn.Identity()])
        self.heads=4;self.spatial_norm=None;self.group_norm=None;self.rescale_output_factor=1.


class M54Tests(unittest.TestCase):
    def test_zero_output_and_live_gradient_paths(self):
        torch.manual_seed(42)
        x=torch.randn(1,16,16);ref=torch.randn(1,256,16);q=torch.randn_like(x);geo=torch.rand(1,3,4,4)
        for arm in ['M1','M2','M3','M4','M5','M6','M9','M11']:
            with self.subTest(arm=arm):
                module=Candidate(arm,16);prior=torch.rand(1,16,256)
                result=module(x,ref,q,4,geo,torch.tensor([500]),prior)
                self.assertTrue(torch.equal(result,torch.zeros_like(x)))
                (result*torch.randn_like(x)).sum().backward()
                self.assertGreater(float(module.output.weight.grad.norm()),0)
                module.zero_grad(set_to_none=True)
                with torch.no_grad():
                    module.output.weight.normal_(std=.01)
                    if hasattr(module,'global_output'):module.global_output.weight.normal_(std=.01)
                (module(x,ref,q,4,geo,torch.tensor([500]),prior)*torch.randn_like(x)).sum().backward()
                for name,p in module.named_parameters():
                    self.assertIsNotNone(p.grad,(arm,name));self.assertTrue(torch.isfinite(p.grad).all(),(arm,name))
                    self.assertGreater(float(p.grad.norm()),0,(arm,name))

    def test_frozen_b1_and_M9_identity(self):
        torch.manual_seed(42);attn=Attention();x=torch.randn(1,16,16)
        p=M54Processor(Original(),'layer',16,16,'B1');p.features=torch.randn(1,256,16)
        p.mask=torch.ones(1,1,4,4);p.enabled=True;p.mode='B1';p.gate.data.fill_(.2)
        kw=dict(sa_hidden_states={});b1=p(attn,x,**kw)
        p.mode='M8_T100';self.assertTrue(torch.equal(b1,p(attn,x,**kw)))
        p.mode='M9';p.candidate=Candidate('M9',16);p.new_branch_active=True
        p.geometry=torch.rand(1,3,4,4);p.timestep=torch.tensor([500])
        self.assertTrue(torch.equal(b1,p(attn,x,**kw)))
        p.enabled=False;self.assertTrue(torch.equal(x,p(attn,x,**kw)))


if __name__=='__main__':unittest.main()
