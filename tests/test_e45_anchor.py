"""E45 数学语义核验；真实层序、CFG、终图回放交给 GPU S0。"""
import unittest
from types import SimpleNamespace
import torch
from torch import nn
from models.e45_pnp_anchor import feature_anchor, equal_energy, attention_delta, strength


class TestAnchor(unittest.TestCase):
    def test_energy_is_norm_not_squared_ratio(self):
        delta=torch.ones(1,4,2)
        mask=torch.tensor([[[[1.,0.],[0.,0.]]]])
        result,audit=equal_energy(delta,mask)
        self.assertEqual(audit['amplitude'],.5)
        self.assertEqual(float(result.square().sum()),2.)
        self.assertEqual(audit['relative_energy_error'],0.)
        self.assertEqual(float(equal_energy(delta,mask*0)[0].abs().max()),0.)

    def test_zero_is_exact_bypass_and_schedule(self):
        x=torch.randn(1,2,4,3);teacher=torch.randn_like(x)
        self.assertIs(feature_anchor(x,teacher,0),x)
        self.assertTrue(torch.allclose(feature_anchor(x,teacher,1),teacher,atol=1e-6))
        self.assertEqual(strength(0),.25);self.assertEqual(strength(14),.25)
        self.assertEqual(strength(24),0.);self.assertEqual(strength(49),0.)

    def test_qk_preserves_student_values_and_no_output_bias(self):
        torch.manual_seed(42)
        attn=SimpleNamespace(heads=2,to_out=nn.ModuleList([nn.Linear(8,8),nn.Identity()]),rescale_output_factor=2.)
        attn.to_out.requires_grad_(False)
        student={k:torch.randn(1,6,8) for k in ['q','k','v']}
        self.assertIsNone(attention_delta(attn,student,student,0))
        self.assertEqual(float(attention_delta(attn,student,student,.25).abs().max()),0.)
        teacher={k:torch.randn(1,6,8) for k in ['q','k']}
        d=attention_delta(attn,teacher,student,.25)
        # self 项两端插值投影之差的偏置必须抵消；Sketch 项与外层残差未进入这个增量。
        attn.to_out[0].bias.data.fill_(100)
        self.assertTrue(torch.equal(d,attention_delta(attn,teacher,student,.25)))
        self.assertGreater(float(d.abs().max()),0.)


if __name__=='__main__':unittest.main()
