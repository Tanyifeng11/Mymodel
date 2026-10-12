"""低成本核验：零初始化、参考 OFF、CFG 与两阶段梯度。"""
import unittest
from types import SimpleNamespace
import torch
from torch import nn
from torch.nn import functional as F
from models.reference_attention_variants import ReferenceAttentionProcessor


class Original(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_k_ref=nn.Linear(8,8,bias=False)
        self.to_v_ref=nn.Linear(8,8,bias=False)

    def __call__(self,attn,x,sa_hidden_states=None,**kwargs):
        q,k,v=[layer(x).reshape(1,-1,2,4).transpose(1,2) for layer in [attn.to_q,attn.to_k,attn.to_v]]
        a=F.scaled_dot_product_attention(q,k,v).transpose(1,2).reshape_as(x)
        if sa_hidden_states is not None:
            k,v=[layer(sa_hidden_states['layer']).reshape(1,-1,2,4).transpose(1,2)
                for layer in [self.to_k_ref,self.to_v_ref]]
            a=a+.6*F.scaled_dot_product_attention(q,k,v).transpose(1,2).reshape_as(x)
        return attn.to_out[0](a)


class TestReferenceAttention(unittest.TestCase):
    def test_fixed_weight_spatial_intervention(self):
        torch.manual_seed(42)
        attn=SimpleNamespace(spatial_norm=None,group_norm=None,heads=2,
            to_q=nn.Linear(8,8),to_k=nn.Linear(8,8),to_v=nn.Linear(8,8),
            to_out=nn.ModuleList([nn.Linear(8,8),nn.Identity()]),rescale_output_factor=1.)
        original=Original().requires_grad_(False)
        p=ReferenceAttentionProcessor(original,'layer',8,5,'B1')
        p.gate.data.fill_(.2);p.enabled=True;p.features=torch.randn(1,4,5)
        p.mask=torch.tensor([[[[0.,1.,0.],[1.,0.,1.]]]])
        x=torch.randn(1,6,8);sa={'layer':torch.randn(1,6,8)}
        base=original(attn,x,sa_hidden_states=sa)
        global_output=p(attn,x,sa_hidden_states=sa)
        observed=[]
        p.diagnostic_observer=lambda base,raw,applied:observed.append((raw.clone(),applied.clone()))
        self.assertTrue(torch.equal(global_output,p(attn,x,sa_hidden_states=sa)))
        self.assertTrue(torch.equal(observed[-1][0],observed[-1][1]))
        p.spatial_intervention='interior'
        masked=p(attn,x,sa_hidden_states=sa)
        selected=p.mask.flatten().bool()
        self.assertTrue(torch.equal(masked[:,~selected],base[:,~selected]))
        self.assertTrue(torch.equal(masked[:,selected],global_output[:,selected]))
        self.assertEqual(float(observed[-1][1][:,~selected].abs().max()),0.)
        self.assertTrue(torch.equal(p(attn,x),original(attn,x)))

    def test_zero_cfg_and_gradients(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        attn=SimpleNamespace(spatial_norm=None,group_norm=None,heads=2,
            to_q=nn.Linear(8,8),to_k=nn.Linear(8,8),to_v=nn.Linear(8,8),
            to_out=nn.ModuleList([nn.Linear(8,8),nn.Identity()]),rescale_output_factor=1.)
        for layer in [attn.to_q,attn.to_k,attn.to_v,*attn.to_out]:layer.requires_grad_(False)
        x=torch.randn(1,6,8);sa={'layer':torch.randn(1,6,8)}
        original=Original().requires_grad_(False)
        base=original(attn,x,sa_hidden_states=sa)
        unconditional=original(attn,x)
        self.assertFalse(torch.equal(base,unconditional))
        for variant in ['B1','B2','B3']:
            p=ReferenceAttentionProcessor(original,'layer',8,5,variant)
            p.enabled=True;p.features=torch.randn(1,4,5);p.mask=torch.ones(1,1,2,3)
            self.assertTrue(torch.equal(p(attn,x,sa_hidden_states=sa),base))
            optimizer=torch.optim.AdamW(p.parameters(),lr=.01)
            for step in range(3):
                optimizer.zero_grad();p(attn,x,sa_hidden_states=sa).square().mean().backward()
                first=p.gate if variant=='B1' else p.output.weight
                self.assertGreater(float(first.grad.norm()),0)
                if step==2:self.assertGreater(float(p.to_k_ref.weight.grad.norm()),0)
                optimizer.step()
            self.assertTrue(torch.equal(p(attn,x,sa_hidden_states=None),unconditional))
            if variant!='B1':
                p.mask[0,0,0,0]=0
                output=p(attn,x,sa_hidden_states=sa)
                self.assertTrue(torch.equal(output[:,0],base[:,0]))
                self.assertGreater(float((output[:,1:]-base[:,1:]).abs().max()),0)
            p.enabled=False
            self.assertTrue(torch.equal(p(attn,x,sa_hidden_states=sa),base))


if __name__=='__main__':unittest.main()
