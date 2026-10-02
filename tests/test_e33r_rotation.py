"""验证轴向旋转、冻结梯度与原坐标监督，避免符号/泄漏错误。"""
import unittest
import numpy as np
import torch
from torch.nn.functional import normalize
from models.e33r_rotation_control import SketchPrior,RotationControl,compose
from tools.e33r_common import cf_support
from tools.e33r_r180_integrity import measure

class RotationTests(unittest.TestCase):
    def test_composition_and_gradient(self):
        torch.set_num_threads(2);torch.manual_seed(42)
        p=normalize(torch.randn(2,2,4,3),dim=1)
        ident=torch.zeros_like(p);ident[:,0]=1
        self.assertTrue(torch.allclose(compose(p,ident),p,atol=1e-6))
        self.assertTrue(torch.allclose(compose(p,-ident),-p,atol=1e-6))
        model=RotationControl(SketchPrior()).train()
        before={k:v.clone() for k,v in model.prior.state_dict().items()}
        result=model(torch.randn(1,16,16,394),torch.rand(1,7,16,16))
        (1+result['q'][:,0]).mean().backward()
        self.assertGreater(float(model.q_head.weight.grad.norm()),0)
        self.assertTrue(all(p.grad is None for p in model.prior.parameters()))
        self.assertFalse(model.prior.training)
        self.assertTrue(all(torch.equal(v,model.prior.state_dict()[k]) for k,v in before.items()))
    def test_original_bbox_intersection(self):
        gt=np.zeros((4,64,48));gt[3]=1;interior=np.ones((64,48))
        gt[3,4,3]=0;interior[5,4]=0
        support=cf_support(gt,interior,[16,24,80,88])
        self.assertEqual(int(support.sum()),62)
        self.assertFalse(support[4,3]);self.assertFalse(support[5,4])
        self.assertFalse(support[:3].any());self.assertFalse(support[:,10:].any())
    def test_native_pixel_rotation(self):
        yy,xx=np.indices((96,96));stripe=((xx//5)%2*180+30).astype(np.uint8)
        rgb=np.repeat(stripe[...,None],3,axis=2)
        for k in (1,2):
            for seed in (None,42,43):self.assertTrue(measure(rgb,k,seed)['valid'])

if __name__=='__main__':unittest.main()
