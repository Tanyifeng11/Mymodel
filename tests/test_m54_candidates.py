"""科研有效性门禁：防止零初始化恒等失败或双零导致永久死梯度。"""
import unittest
import inspect
import torch
from torch import nn
from models.m54_candidates import Candidate,CandidateCombination,M54Processor,high_frequency


class Original(nn.Module):
    def forward(self,attn,x,**kwargs):return x+(.25 if kwargs.get('sa_hidden_states') is not None else 0.)


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
        p.mask=torch.ones(1,1,4,4);p.enabled=True;p.mode='B1';p.gate.data.fill_(.2);p.last_delta={}
        kw=dict(sa_hidden_states={});b1=p(attn,x,**kw)
        p.mode='M8_T100';self.assertTrue(torch.equal(b1,p(attn,x,**kw)))
        p.mode='M9';p.candidate=Candidate('M9',16);p.new_branch_active=True
        p.geometry=torch.rand(1,3,4,4);p.timestep=torch.tensor([500])
        self.assertTrue(torch.equal(b1,p(attn,x,**kw)))
        p.enabled=False;self.assertTrue(torch.equal(x+.25,p(attn,x,**kw)))

    def test_diffusers_keyword_dispatch_preserves_existing_sketch_path(self):
        torch.manual_seed(42);attn=Attention();x=torch.randn(1,16,16)
        p=M54Processor(Original(),'layer',16,16,'B1');p.mode='M2';p.enabled=True
        p.features=torch.randn(1,256,16);p.candidate=Candidate('M2',16)
        p.geometry=torch.ones(1,3,4,4);p.timestep=torch.tensor([500]);p.spatial_prior=None
        # 复现外部Attention.forward的签名过滤约定；zero模块仍须保留原Sketch贡献。
        incoming={'sa_hidden_states':{},'tcpm_garment_mask':torch.ones(1,1,4,4)}
        accepted={k:v for k,v in incoming.items() if k in inspect.signature(p.__call__).parameters}
        self.assertEqual(set(accepted),set(incoming))
        self.assertTrue(torch.equal(p(attn,x,**accepted),x+.25))

    def test_temperature_changes_routing_without_reference_mass_shrink(self):
        torch.manual_seed(42);attn=Attention();x=torch.randn(1,16,16)
        p=M54Processor(Original(),'layer',16,16,'B1');p.features=torch.randn(1,256,16)
        p.mask=torch.ones(1,1,4,4);p.enabled=True;p.gate.data.fill_(.2)
        p.mode='M8_T085';a=p(attn,x,sa_hidden_states={});first=dict(p.last_attention)
        p.mode='M8_T070';b=p(attn,x,sa_hidden_states={});second=dict(p.last_attention)
        self.assertEqual(first['p_ref'],second['p_ref'])
        self.assertLess(second['entropy'],first['entropy'])
        self.assertFalse(torch.equal(a,b))

    def test_high_frequency_branch_does_not_invent_constant_reference_content(self):
        module=Candidate('M11',16)
        with torch.no_grad():module.output.weight.normal_(std=.1)
        reference=torch.ones(1,256,16)
        self.assertTrue(torch.equal(high_frequency(reference),torch.zeros_like(reference)))
        output=module(torch.randn(1,16,16),reference,torch.randn(1,16,16),4,torch.ones(1,3,4,4),torch.tensor([500]))
        self.assertTrue(torch.equal(output,torch.zeros_like(output)))

    def test_combination_preserves_learned_paths_and_adds_no_parameters(self):
        torch.manual_seed(54)
        appearance=Candidate('M1',16);protector=Candidate('M9',16)
        with torch.no_grad():
            appearance.output.weight.normal_(std=.02);protector.output.weight.normal_(std=.02)
        inputs=(torch.randn(1,16,16),torch.randn(1,256,16),torch.randn(1,16,16),4,
                torch.ones(1,3,4,4),torch.tensor([500]),None)
        wrapper=CandidateCombination([appearance,protector],[.75,1.]).requires_grad_(False)
        expected=appearance(*inputs)*.75+protector(*inputs)
        self.assertTrue(torch.equal(wrapper(*inputs),expected))
        self.assertEqual(sum(p.numel() for p in wrapper.parameters()),sum(p.numel() for p in appearance.parameters())+sum(p.numel() for p in protector.parameters()))
        self.assertFalse(any(p.requires_grad for p in wrapper.parameters()))


if __name__=='__main__':unittest.main()
