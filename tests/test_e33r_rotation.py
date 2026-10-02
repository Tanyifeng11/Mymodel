"""验证轴向旋转、冻结梯度与原坐标监督，避免符号/泄漏错误。"""
import unittest
import numpy as np
import torch
from torch.nn.functional import normalize
from models.e33r_rotation_control import SketchPrior,RotationControl,compose
from tools.e33r_common import cf_support
from tools.e33r_r180_integrity import measure
from tools.e33r_losses import group_loss
from tools.e33r_evaluate import ratio_bootstrap,case_metrics,summarize

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
    def test_complete_group_targets_and_paired_statistics(self):
        gt=torch.zeros(2,4,3,4);gt[:,0]=1;gt[:,3]=1
        support=torch.ones(2,3,4,dtype=torch.bool)
        q=gt[:,:2,None].transpose(1,2).expand(-1,7,-1,-1,-1).clone()
        q[:,1]=-q[:,1];q[:,4]=-q[:,4]
        logits=torch.full((2,7,1,3,4),50.);logits[:,6]=-50
        pred=dict(orientation=q,q=q,confidence_logits=logits)
        loss,parts=group_loss(pred,gt,support)
        self.assertLess(float(loss),1e-6)
        pred['orientation'][:,:,:,0,0]=0;support[:,0,0]=False
        self.assertLess(float(group_loss(pred,gt,support)[0]),1e-6)
        ratio=ratio_bootstrap([1.,3.],[1.,1.])
        self.assertEqual(ratio['mean'],2.)
        # 严格按case聚合：grid、branch和noise均不扩大n。
        p=np.zeros((11,2,3,4));p[:,0]=1;p[1]=-p[1];p[4]=-p[4];p[7]=-p[7]
        qp=p.copy();qp[10]=p[0]
        p[10,0]=np.cos(np.deg2rad(2));p[10,1]=np.sin(np.deg2rad(2))
        data=dict(orientation=p,q=qp,frozen_prior=p[0],confidence_logits=np.zeros((11,1,3,4)))
        row=case_metrics(dict(id='test',strict=True),data,gt[0].numpy(),support[0].numpy(),{})
        summary=summarize([row])
        self.assertEqual(summary['clean_r90_success']['n'],1)
        self.assertTrue(summary['gate_pass'])

if __name__=='__main__':unittest.main()
