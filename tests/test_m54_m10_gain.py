"""CPU验证真实参数化契约；完整采样和冻结核验由GPU门禁完成。"""
import unittest
import torch
from models.m54_m10_gain import M10Gain


class GainContract(unittest.TestCase):
    def test_initial_exact_and_parameter_budget(self):
        g=M10Gain()
        self.assertEqual(sum(p.numel() for p in g.parameters()),1280)
        alpha=g(torch.tensor([0,1,500,999]))
        self.assertTrue(torch.equal(alpha,torch.ones(4,1,640)))
        x=torch.randn(4,768,640)
        self.assertTrue(torch.equal(x*alpha,x))

    def test_time_channel_independence_and_bounds(self):
        g=M10Gain(3)
        with torch.no_grad():g.a.copy_(torch.tensor([-2.,0.,2.]));g.b.fill_(1)
        alpha=g(torch.tensor([0,999]))
        self.assertTrue(((alpha>=0)&(alpha<=2)).all())
        self.assertTrue((alpha[0]<alpha[1]).all())
        self.assertTrue((alpha[:,:,0]<alpha[:,:,2]).all())

    def test_all_parameters_have_live_finite_gradient(self):
        g=M10Gain()
        (g(torch.tensor([0,100]))*torch.arange(1,641)).square().mean().backward()
        for p in g.parameters():
            self.assertTrue(torch.isfinite(p.grad).all())
            self.assertEqual(p.grad.count_nonzero(),p.numel())

    def test_production_processor_initial_and_off_contract(self):
        from test_m54_candidates import Attention, Original
        from models.m54_candidates import M54Processor
        torch.manual_seed(42)
        attn=Attention().requires_grad_(False);x=torch.randn(1,16,16)
        p=M54Processor(Original(),'layer',16,16,'B1').requires_grad_(False)
        p.features=torch.randn(1,256,16);p.enabled=True;p.gate.data.fill_(.2);p.mode='M10'
        original=p(attn,x,sa_hidden_states={})
        p.mode='M10_GAIN';p.m10_gain=M10Gain(16);p.timestep=torch.tensor([100])
        initial=p(attn,x,sa_hidden_states={})
        self.assertTrue(torch.equal(original,initial))
        initial.square().mean().backward()
        for v in p.m10_gain.parameters():self.assertGreater(float(v.grad.norm()),0)
        p.enabled=False
        self.assertTrue(torch.equal(p(attn,x,sa_hidden_states={}),x+.25))


if __name__=='__main__':unittest.main()
