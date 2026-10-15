"""验证零初始化、时间消融和幅度约束；整条 GPU 轨迹另由 audit 验收。"""
import unittest
import torch
from models.e51_residual_controller import ConditionalResidualController


class ControllerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42);torch.set_num_threads(2)
        self.model=ConditionalResidualController(8,8)
        self.h=torch.randn(1,8,4,3);self.r=torch.randn_like(self.h)
        self.sketch=torch.rand(1,3,4,3)

    def test_zero_output_finite_first_gradient(self):
        output,gate,cap=self.model(self.h,self.r,self.sketch,torch.tensor([900]),True)
        self.assertTrue(torch.equal(output,torch.zeros_like(output)))
        self.assertTrue(torch.equal(gate,torch.full_like(gate,.5)))
        output.sum().backward()
        grad=self.model.output.weight.grad
        self.assertTrue(torch.isfinite(grad).all());self.assertGreater(float(grad.norm()),0)
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in self.model.parameters()))

    def test_notime_invariant_but_timed_effective(self):
        with torch.no_grad():self.model.output.weight.normal_(0,.01)
        a=self.model(self.h,self.r,self.sketch,torch.tensor([900]),False)[0]
        b=self.model(self.h,self.r,self.sketch,torch.tensor([100]),False)[0]
        self.assertTrue(torch.equal(a,b))
        c=self.model(self.h,self.r,self.sketch,torch.tensor([900]),True)[0]
        d=self.model(self.h,self.r,self.sketch,torch.tensor([100]),True)[0]
        self.assertGreater(float((c-d).detach().abs().max()),1e-5)

    def test_projection_cap_only_shrinks(self):
        with torch.no_grad():self.model.output.weight.fill_(100)
        output,gate,cap=self.model(self.h,self.r,self.sketch,torch.tensor([500]),True)
        self.assertLessEqual(float(output.detach().square().mean()),float(self.r.square().mean())+1e-5)
        self.assertGreater(float(cap.detach()),0);self.assertLess(float(cap.detach()),1)
        self.assertTrue(((gate>=0)&(gate<=1)).all())

    def test_zero_reference_has_zero_applied_residual(self):
        with torch.no_grad():self.model.output.weight.fill_(1)
        output,_,_=self.model(self.h,torch.zeros_like(self.r),self.sketch,torch.tensor([500]),True)
        self.assertTrue(torch.equal(output,torch.zeros_like(output)))

    def test_condition_and_gate_backpropagate_after_output_head(self):
        with torch.no_grad():self.model.output.weight.normal_(0,.001)
        output,gate,_=self.model(self.h,self.r,self.sketch,torch.tensor([500]),True)
        (output*gate).square().mean().backward()
        named=dict(self.model.named_parameters())
        for n in ['target.weight','reference.weight','mix.weight','time.0.weight','time.2.weight','gating.weight']:
            self.assertTrue(torch.isfinite(named[n].grad).all(),n)
            self.assertGreater(float(named[n].grad.norm()),0,n)


if __name__=='__main__':unittest.main()
